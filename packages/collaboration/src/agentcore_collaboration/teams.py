"""Dynamic, immutable teams.yaml projection."""

from __future__ import annotations

import hashlib
import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Any

import yaml
from yaml.constructor import ConstructorError

from agentcore.collaboration.errors import CollaborationConfigError

DEFAULT_TEAMS_PATH = Path("/var/run/agentcore/agent/teams.yaml")
_MAX_FILE_BYTES = 1024 * 1024
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TEAM_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_ROLES = {"leader", "worker", "manager", "admin", "human", "member"}
logger = logging.getLogger("agentcore.collaboration.teams")
_FileSignature = tuple[int, int, int]


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found duplicate key",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping,
)


@dataclass(frozen=True)
class TeamMemberSnapshot:
    type: str
    name: str
    runtime_name: str | None
    role: str
    matrix_user_id: str | None
    personal_room_id: str | None


@dataclass(frozen=True)
class TeamSnapshot:
    name: str
    room_id: str | None
    role: str
    members: tuple[TeamMemberSnapshot, ...]


@dataclass(frozen=True)
class TeamsSnapshot:
    runtime_name: str
    self_name: str
    self_matrix_user_id: str | None
    self_personal_room_id: str | None
    matrix_token_env: str | None
    default_team_name: str | None
    teams: Mapping[str, TeamSnapshot]


class TeamsProvider:
    """Read the latest complete teams.yaml and retain the last valid update."""

    def __init__(self, path: str | Path = DEFAULT_TEAMS_PATH) -> None:
        self.path = Path(path)
        self._last_good: TeamsSnapshot | None = None
        self._observed_signature: _FileSignature | None = None
        self._observed_digest: bytes | None = None
        self._lock = Lock()

    def snapshot(self) -> TeamsSnapshot | None:
        with self._lock:
            return self._snapshot()

    def _snapshot(self) -> TeamsSnapshot | None:
        signature: _FileSignature | None = None
        digest: bytes | None = None
        try:
            signature = self._file_signature()
            if signature == self._observed_signature:
                return self._last_good
            data, signature = self._read_stable()
            digest = hashlib.sha256(data).digest()
            if digest == self._observed_digest:
                self._observed_signature = signature
                return self._last_good
            snapshot = parse_teams_config(data)
        except FileNotFoundError:
            self._last_good = None
            self._observed_signature = None
            self._observed_digest = None
            return None
        except CollaborationConfigError as exc:
            if self._last_good is not None:
                if signature is not None and digest is not None:
                    self._observed_signature = signature
                    self._observed_digest = digest
                logger.warning(
                    "agentcore.collaboration.teams.update_ignored path=%s error_type=%s",
                    self.path,
                    type(exc).__name__,
                )
                return self._last_good
            raise
        self._last_good = snapshot
        self._observed_signature = signature
        self._observed_digest = digest
        return snapshot

    def _file_signature(self) -> _FileSignature:
        try:
            return _signature(self.path.stat())
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise CollaborationConfigError("cannot stat teams.yaml") from exc

    def _read_stable(self) -> tuple[bytes, _FileSignature]:
        try:
            before = self.path.stat()
            data = self.path.read_bytes()
            after = self.path.stat()
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise CollaborationConfigError("cannot read teams.yaml") from exc
        before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_id != after_id or len(data) != after.st_size:
            raise CollaborationConfigError("teams.yaml changed while reading")
        return data, _signature(after)


def _signature(value: os.stat_result) -> _FileSignature:
    return value.st_size, value.st_mtime_ns, value.st_ctime_ns


def parse_teams_config(data: bytes) -> TeamsSnapshot:
    root = _load_yaml(data)
    if root.get("apiVersion") != "agentteams.io/v1alpha1" or root.get("kind") != "TeamsConfig":
        raise CollaborationConfigError("unsupported teams.yaml apiVersion or kind")
    metadata = _mapping(root.get("metadata"), "metadata")
    spec = _mapping(root.get("spec"), "spec")
    self_identity = _mapping(spec.get("self"), "spec.self")
    runtime_name = _required_text(metadata.get("runtimeName"), "metadata.runtimeName")
    self_runtime_name = _required_text(
        self_identity.get("runtimeName"),
        "spec.self.runtimeName",
    )
    if self_runtime_name != runtime_name:
        raise CollaborationConfigError("spec.self.runtimeName must match metadata.runtimeName")
    self_name = _required_text(self_identity.get("name"), "spec.self.name")
    self_matrix_user_id = _optional_text(
        self_identity.get("matrixUserId"),
        "spec.self.matrixUserId",
    )
    self_personal_room_id = _optional_text(
        self_identity.get("personalRoomId"),
        "spec.self.personalRoomId",
    )

    token_env = None
    if spec.get("matrix") is not None:
        matrix = _mapping(spec["matrix"], "spec.matrix")
        token_env = _required_text(matrix.get("tokenEnv"), "spec.matrix.tokenEnv")
        if _ENV_NAME.fullmatch(token_env) is None:
            raise CollaborationConfigError("spec.matrix.tokenEnv is invalid")
        if self_matrix_user_id is None:
            raise CollaborationConfigError(
                "spec.self.matrixUserId is required when spec.matrix is configured"
            )

    raw_teams = spec.get("teams")
    if not isinstance(raw_teams, list):
        raise CollaborationConfigError("spec.teams must be a list")
    teams: dict[str, TeamSnapshot] = {}
    room_ids: set[str] = set()
    for index, raw_team in enumerate(raw_teams):
        team = _mapping(raw_team, f"spec.teams[{index}]")
        name = _required_text(team.get("name"), f"spec.teams[{index}].name")
        if _TEAM_NAME.fullmatch(name) is None:
            raise CollaborationConfigError(f"spec.teams[{index}].name must be mount-safe")
        if name in teams:
            raise CollaborationConfigError("spec.teams contains duplicate name")
        membership = _mapping(team.get("membership"), f"spec.teams[{index}].membership")
        role = _required_text(membership.get("role"), f"spec.teams[{index}].membership.role")
        if role not in _ROLES:
            raise CollaborationConfigError(f"spec.teams[{index}].membership.role is unsupported")
        room_id = _optional_text(team.get("teamRoomId"), f"spec.teams[{index}].teamRoomId")
        if room_id is not None:
            if room_id in room_ids:
                raise CollaborationConfigError("spec.teams contains duplicate teamRoomId")
            room_ids.add(room_id)
        raw_members = team.get("members")
        if not isinstance(raw_members, list):
            raise CollaborationConfigError(f"spec.teams[{index}].members must be a list")
        members = tuple(
            _parse_member(raw_member, f"spec.teams[{index}].members[{member_index}]")
            for member_index, raw_member in enumerate(raw_members)
        )
        _validate_roster(members, self_identity=(self_name, self_runtime_name, self_matrix_user_id))
        teams[name] = TeamSnapshot(
            name=name,
            room_id=room_id,
            role=role,
            members=members,
        )

    default_team_name = _optional_text(spec.get("defaultTeamName"), "spec.defaultTeamName")
    if default_team_name is not None and default_team_name not in teams:
        raise CollaborationConfigError("spec.defaultTeamName must reference a configured Team")
    return TeamsSnapshot(
        runtime_name=runtime_name,
        self_name=self_name,
        self_matrix_user_id=self_matrix_user_id,
        self_personal_room_id=self_personal_room_id,
        matrix_token_env=token_env,
        default_team_name=default_team_name,
        teams=MappingProxyType(teams),
    )


def _parse_member(value: object, path: str) -> TeamMemberSnapshot:
    member = _mapping(value, path)
    member_type = _required_text(member.get("type"), f"{path}.type")
    if member_type not in {"agent", "human"}:
        raise CollaborationConfigError(f"{path}.type is unsupported")
    runtime_name = _optional_text(member.get("runtimeName"), f"{path}.runtimeName")
    if member_type == "agent" and runtime_name is None:
        raise CollaborationConfigError(f"{path}.runtimeName is required for an agent")
    if member_type == "human" and runtime_name is not None:
        raise CollaborationConfigError(f"{path}.runtimeName is not allowed for a human")
    role = _required_text(member.get("role"), f"{path}.role")
    if role not in _ROLES:
        raise CollaborationConfigError(f"{path}.role is unsupported")
    return TeamMemberSnapshot(
        type=member_type,
        name=_required_text(member.get("name"), f"{path}.name"),
        runtime_name=runtime_name,
        role=role,
        matrix_user_id=_optional_text(member.get("matrixUserId"), f"{path}.matrixUserId"),
        personal_room_id=_optional_text(
            member.get("personalRoomId"),
            f"{path}.personalRoomId",
        ),
    )


def _validate_roster(
    members: tuple[TeamMemberSnapshot, ...],
    *,
    self_identity: tuple[str, str, str | None],
) -> None:
    self_name, self_runtime_name, self_matrix_user_id = self_identity
    names: set[str] = set()
    runtime_names: set[str] = set()
    matrix_user_ids: set[str] = set()
    for member in members:
        if member.name in names:
            raise CollaborationConfigError("spec.teams members contain duplicate name")
        names.add(member.name)
        if member.runtime_name is not None:
            if member.runtime_name in runtime_names:
                raise CollaborationConfigError("spec.teams members contain duplicate runtimeName")
            runtime_names.add(member.runtime_name)
        if member.matrix_user_id is not None:
            if member.matrix_user_id in matrix_user_ids:
                raise CollaborationConfigError("spec.teams members contain duplicate matrixUserId")
            matrix_user_ids.add(member.matrix_user_id)
        if (
            member.name == self_name
            or member.runtime_name == self_runtime_name
            or (
                self_matrix_user_id is not None
                and member.matrix_user_id == self_matrix_user_id
            )
        ):
            raise CollaborationConfigError("spec.teams member identity conflicts with spec.self")


def _load_yaml(data: bytes) -> Mapping[str, Any]:
    if not data or len(data) > _MAX_FILE_BYTES:
        raise CollaborationConfigError("teams.yaml is empty or exceeds 1 MiB")
    try:
        text = data.decode("utf-8")
        for token in yaml.scan(text):
            if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)):
                raise CollaborationConfigError("teams.yaml aliases and anchors are not allowed")
        loaded = yaml.load(text, Loader=_UniqueKeyLoader)
    except UnicodeDecodeError as exc:
        raise CollaborationConfigError("teams.yaml must be UTF-8") from exc
    except yaml.YAMLError as exc:
        raise CollaborationConfigError("invalid teams.yaml") from exc
    return _mapping(loaded, "teams.yaml")


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise CollaborationConfigError(f"{path} must be an object")
    return value


def _required_text(value: object, path: str) -> str:
    text = _optional_text(value, path)
    if text is None:
        raise CollaborationConfigError(f"{path} is required")
    return text


def _optional_text(value: object, path: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CollaborationConfigError(f"{path} must be a non-empty string")
    return value.strip()
