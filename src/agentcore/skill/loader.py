"""Materialize local and AgentCore-managed Skills without source ambiguity."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
import shutil
import tempfile
import zipfile
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from threading import Lock
from typing import Protocol

import anyio
import yaml

from agentcore.controlplane import SkillArtifact
from agentcore.errors import ConfigError, ResourceNotConfiguredError

_FRONTMATTER = re.compile(r"^---\s*\n(.*?)\n---(?:\s*\n|$)", re.DOTALL)
_MAX_LOCAL_SKILL_BYTES = 10 * 1024 * 1024
_DEFAULT_MANAGED_SKILL_WORKSPACE = Path(".skills") / ".agentcore-managed"
_MATERIALIZE_LOCK = Lock()
logger = logging.getLogger(__name__)


class SkillProvider(Protocol):
    async def get_skill(self, name: str, version: str | None = None) -> SkillArtifact: ...


SkillRuntimeProvider = Callable[[], Awaitable[tuple[SkillProvider | None, str | None]]]


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    version: str
    instruction: str
    root: Path
    files: tuple[str, ...]
    source: str
    digest: str | None = None
    workspace_id: str | None = None


@dataclass(frozen=True)
class _PreparedSkill:
    version: str
    files: Mapping[str, bytes]
    digest: str


class AsyncSkills:
    def __init__(
        self,
        control_plane: SkillProvider | None,
        workspace_id: str | None,
        *,
        workspace_dir: str | Path | None = None,
        _runtime_provider: SkillRuntimeProvider | None = None,
    ) -> None:
        self._control_plane = control_plane
        self._workspace_id = workspace_id
        self._runtime_provider = _runtime_provider
        configured_workspace = workspace_dir or os.getenv("AGENTCORE_SKILL_WORKSPACE_DIR")
        self._workspace = Path(configured_workspace or _DEFAULT_MANAGED_SKILL_WORKSPACE)
        self._pins: dict[tuple[str, str, str | None], Skill] = {}
        self._lock = anyio.Lock()

    async def local(self, root: str | Path) -> list[Skill]:
        logger.info("agentcore.skill.local.load.started root=%s", root)
        try:
            skills = await anyio.to_thread.run_sync(_load_local_skills, Path(root))
        except Exception as exc:
            logger.warning(
                "agentcore.skill.local.load.failed root=%s error_type=%s",
                root,
                type(exc).__name__,
            )
            raise
        logger.info(
            "agentcore.skill.local.load.succeeded root=%s skill_count=%s",
            root,
            len(skills),
        )
        return skills

    async def managed(self, name: str, *, version: str | None = None) -> Skill:
        control_plane = self._control_plane
        workspace_id = self._workspace_id
        if self._runtime_provider is not None:
            control_plane, workspace_id = await self._runtime_provider()
        if control_plane is None or workspace_id is None:
            raise ResourceNotConfiguredError("managed Skill runtime is not configured")
        key = (workspace_id, name, version)
        pinned = self._pins.get(key)
        if pinned is not None:
            logger.debug(
                "agentcore.skill.managed.cache.hit workspace_id=%s name=%s version=%s",
                workspace_id,
                name,
                pinned.version,
            )
            return pinned
        async with self._lock:
            pinned = self._pins.get(key)
            if pinned is not None:
                logger.debug(
                    "agentcore.skill.managed.cache.hit workspace_id=%s name=%s version=%s",
                    workspace_id,
                    name,
                    pinned.version,
                )
                return pinned
            logger.info(
                "agentcore.skill.managed.load.started workspace_id=%s name=%s version=%s",
                workspace_id,
                name,
                version or "latest",
            )
            try:
                artifact = await control_plane.get_skill(name, version)
                prepared = await anyio.to_thread.run_sync(_prepare_artifact, name, artifact)
                root = await anyio.to_thread.run_sync(
                    _materialize,
                    self._workspace,
                    workspace_id,
                    name,
                    prepared,
                )
                skill = await anyio.to_thread.run_sync(
                    _load_managed_skill_directory,
                    root,
                    prepared,
                    workspace_id,
                    name,
                )
            except Exception as exc:
                logger.warning(
                    "agentcore.skill.managed.load.failed workspace_id=%s name=%s "
                    "version=%s error_type=%s",
                    workspace_id,
                    name,
                    version or "latest",
                    type(exc).__name__,
                )
                raise
            self._pins[key] = skill
            logger.info(
                "agentcore.skill.managed.load.succeeded workspace_id=%s name=%s "
                "version=%s digest=%s",
                workspace_id,
                name,
                skill.version,
                skill.digest,
            )
            return skill


def _prepare_artifact(skill_name: str, artifact: SkillArtifact) -> _PreparedSkill:
    _validate_directory_component(skill_name, "name")
    _validate_directory_component(artifact.version, "version")
    if not artifact.archive or len(artifact.archive) > _MAX_LOCAL_SKILL_BYTES:
        raise ConfigError("Skill package exceeds the 10 MiB size limit")
    archived_files: dict[str, bytes] = {}
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(artifact.archive)) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                path = _safe_resource_path(entry.filename)
                if path.as_posix() in archived_files:
                    raise ConfigError("managed Skill package contains a duplicate path")
                if entry.file_size > _MAX_LOCAL_SKILL_BYTES - total:
                    raise ConfigError("Skill content exceeds the 10 MiB size limit")
                content = archive.read(entry)
                total += len(content)
                if total > _MAX_LOCAL_SKILL_BYTES:
                    raise ConfigError("Skill content exceeds the 10 MiB size limit")
                archived_files[path.as_posix()] = content
    except (zipfile.BadZipFile, RuntimeError) as exc:
        raise ConfigError("managed Skill package is not a valid ZIP archive") from exc
    files = _strip_skill_archive_root(skill_name, archived_files)
    resource_paths = {PurePosixPath(name) for name in files}
    if any(
        parent in resource_paths
        for resource_path in resource_paths
        for parent in resource_path.parents
    ):
        raise ConfigError("managed Skill contains a colliding resource path")
    digest = hashlib.sha256()
    for name, content in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return _PreparedSkill(version=artifact.version, files=files, digest=digest.hexdigest())


def _materialize(
    workspace: Path,
    workspace_id: str,
    skill_name: str,
    artifact: _PreparedSkill,
) -> Path:
    _validate_directory_component(workspace_id, "workspace ID")
    target = workspace / workspace_id / skill_name / artifact.version
    metadata = {
        "source": "agentcore",
        "workspaceId": workspace_id,
        "name": skill_name,
        "version": artifact.version,
        "digest": artifact.digest,
    }
    with _MATERIALIZE_LOCK:
        if _materialized_artifact_matches(target, metadata):
            return target
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
        temporary = Path(tempfile.mkdtemp(prefix=".skill-", dir=workspace))
        try:
            for resource_name, content in artifact.files.items():
                path = temporary.joinpath(*PurePosixPath(resource_name).parts)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
            temporary.joinpath(".agentcore-skill.json").write_text(
                json.dumps(metadata, sort_keys=True),
                encoding="utf-8",
            )
            _load_skill_directory(
                temporary,
                "agentcore",
                artifact.version,
                artifact.digest,
            )
            try:
                os.replace(temporary, target)
            except OSError:
                if not _materialized_artifact_matches(target, metadata):
                    raise
            return target
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)


def _materialized_artifact_matches(target: Path, expected: Mapping[str, str]) -> bool:
    if target.is_symlink() or not target.is_dir():
        return False
    try:
        metadata = json.loads(target.joinpath(".agentcore-skill.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(metadata, dict)
        and len(metadata) == len(expected)
        and all(metadata.get(key) == value for key, value in expected.items())
    )


def _load_managed_skill_directory(
    directory: Path,
    artifact: _PreparedSkill,
    workspace_id: str,
    resource_name: str,
) -> Skill:
    files = ("SKILL.md", *sorted(name for name in artifact.files if name != "SKILL.md"))
    for name in files:
        path = directory.joinpath(*PurePosixPath(name).parts)
        if path.is_symlink() or not path.is_file():
            raise ConfigError("managed Skill workspace is missing a source file")
    instruction = directory.joinpath("SKILL.md").read_text(encoding="utf-8")
    metadata = _frontmatter(instruction)
    return Skill(
        name=resource_name,
        description=metadata.get("description") or "",
        version=artifact.version,
        instruction=instruction,
        root=directory,
        files=files,
        source="agentcore",
        digest=artifact.digest,
        workspace_id=workspace_id,
    )


def _safe_resource_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        not name
        or "\0" in name
        or "\\" in name
        or path.is_absolute()
        or path.as_posix() != name
        or ".." in path.parts
        or "." in path.parts
        or path.parts == (".agentcore-skill.json",)
    ):
        raise ConfigError("managed Skill contains an unsafe resource path")
    return path


def _strip_skill_archive_root(
    skill_name: str,
    archived_files: Mapping[str, bytes],
) -> dict[str, bytes]:
    expected_skill_md = f"{skill_name}/SKILL.md"
    if expected_skill_md not in archived_files:
        raise ConfigError(f"managed Skill package does not contain {expected_skill_md}")
    files: dict[str, bytes] = {}
    for name, content in archived_files.items():
        path = PurePosixPath(name)
        if len(path.parts) < 2 or path.parts[0] != skill_name:
            raise ConfigError("managed Skill package must use the Skill name as its root directory")
        relative = _safe_resource_path(PurePosixPath(*path.parts[1:]).as_posix())
        files[relative.as_posix()] = content
    return files


def _validate_directory_component(value: str, label: str) -> None:
    if (
        not value
        or not value.strip()
        or "\0" in value
        or "\\" in value
        or PurePosixPath(value).parts != (value,)
        or value in {".", ".."}
    ):
        raise ConfigError(f"managed Skill {label} is invalid")


def _load_local_skills(root: Path) -> list[Skill]:
    if not root.is_dir():
        raise ConfigError("local Skill root does not exist or is not a directory")
    directories = [root] if root.joinpath("SKILL.md").is_file() else sorted(root.iterdir())
    result: list[Skill] = []
    for directory in directories:
        if directory != root and directory.is_symlink():
            raise ConfigError("local Skill root must not contain symbolic-link Skills")
        if directory.is_dir() and directory.joinpath("SKILL.md").is_file():
            result.append(_load_skill_directory(directory, "local", None, None))
    return result


def _load_skill_directory(
    directory: Path,
    source: str,
    forced_version: str | None,
    digest: str | None,
) -> Skill:
    files: list[str] = []
    total = 0
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ConfigError("Skill directories must not contain symbolic links")
        if not path.is_file() or path.name == ".agentcore-skill.json":
            continue
        total += path.stat().st_size
        if total > _MAX_LOCAL_SKILL_BYTES:
            raise ConfigError("Skill content exceeds the 10 MiB size limit")
        files.append(path.relative_to(directory).as_posix())
    instruction = directory.joinpath("SKILL.md").read_text(encoding="utf-8")
    metadata = _frontmatter(instruction)
    name = metadata.get("name") or directory.name
    description = metadata.get("description") or ""
    version = forced_version or metadata.get("version") or ""
    return Skill(
        name=name,
        description=description,
        version=version,
        instruction=instruction,
        root=directory,
        files=tuple(files),
        source=source,
        digest=digest,
    )


def _frontmatter(content: str) -> dict[str, str]:
    match = _FRONTMATTER.match(content)
    if match is None:
        return {}
    try:
        value = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ConfigError("SKILL.md contains invalid YAML frontmatter") from exc
    if not isinstance(value, dict):
        raise ConfigError("SKILL.md frontmatter must be an object")
    result: dict[str, str] = {}
    for key in ("name", "description", "version"):
        item = value.get(key)
        if item is not None:
            if not isinstance(item, str):
                raise ConfigError(f"SKILL.md frontmatter {key} must be a string")
            result[key] = item
    return result
