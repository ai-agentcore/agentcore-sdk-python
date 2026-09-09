from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pytest
from agentcore_collaboration import teams as teams_module

from agentcore.collaboration import TeamsProvider
from agentcore.collaboration.errors import CollaborationConfigError


def _teams_yaml(*, role: str = "worker") -> str:
    return f"""
apiVersion: agentteams.io/v1alpha1
kind: TeamsConfig
metadata:
  runtimeName: worker-a
  generation: "1"
spec:
  self:
    name: worker-a
    runtimeName: worker-a
    matrixUserId: "@worker-a:matrix.example.com"
    personalRoomId: "!worker-a:matrix.example.com"
  matrix:
    homeserver: https://matrix.example.com
    tokenEnv: AGENTTEAMS_WORKER_MATRIX_TOKEN
  defaultTeamName: team-alpha
  teams:
    - name: team-alpha
      teamRoomId: "!alpha:matrix.example.com"
      membership:
        role: {role}
      members:
        - type: agent
          name: leader-a
          runtimeName: leader-a
          role: leader
          matrixUserId: "@leader-a:matrix.example.com"
          personalRoomId: "!leader-a:matrix.example.com"
        - type: human
          name: operator-a
          role: human
          matrixUserId: "@operator-a:matrix.example.com"
""".lstrip()


def test_teams_provider_enables_collaboration_when_file_appears(tmp_path: Path) -> None:
    path = tmp_path / "teams.yaml"
    provider = TeamsProvider(path)

    assert provider.snapshot() is None

    path.write_text(_teams_yaml(), encoding="utf-8")
    snapshot = provider.snapshot()

    assert snapshot is not None
    assert snapshot.runtime_name == "worker-a"
    assert snapshot.matrix_token_env == "AGENTTEAMS_WORKER_MATRIX_TOKEN"
    assert snapshot.self_matrix_user_id == "@worker-a:matrix.example.com"
    assert snapshot.self_personal_room_id == "!worker-a:matrix.example.com"
    assert snapshot.teams["team-alpha"].role == "worker"
    assert snapshot.teams["team-alpha"].members[0].runtime_name == "leader-a"
    assert snapshot.teams["team-alpha"].members[1].type == "human"


def test_teams_provider_uses_last_good_and_disables_after_deletion(tmp_path: Path) -> None:
    path = tmp_path / "teams.yaml"
    path.write_text(_teams_yaml(), encoding="utf-8")
    provider = TeamsProvider(path)
    initial = provider.snapshot()

    path.write_text("kind: broken", encoding="utf-8")
    assert provider.snapshot() is initial

    path.unlink()
    assert provider.snapshot() is None


@pytest.mark.parametrize("matrix", ["", "  matrix: null\n"])
def test_teams_provider_accepts_identity_before_matrix_registration(
    tmp_path: Path, matrix: str
) -> None:
    path = tmp_path / "teams.yaml"
    path.write_text(
        "apiVersion: agentteams.io/v1alpha1\nkind: TeamsConfig\n"
        "metadata:\n  runtimeName: worker-a\nspec:\n"
        "  self:\n    name: worker-a\n    runtimeName: worker-a\n"
        + matrix + "  teams: []\n",
        encoding="utf-8",
    )
    provider = TeamsProvider(path)
    snapshot = provider.snapshot()
    assert snapshot is not None
    assert snapshot.matrix_token_env is None
    assert snapshot.self_matrix_user_id is None
    assert not snapshot.teams

    path.write_text(_teams_yaml(), encoding="utf-8")
    updated = provider.snapshot()
    assert updated is not None
    assert updated.matrix_token_env == "AGENTTEAMS_WORKER_MATRIX_TOKEN"
    assert "team-alpha" in updated.teams


def test_teams_provider_rejects_unknown_membership_role(tmp_path: Path) -> None:
    path = tmp_path / "teams.yaml"
    path.write_text(_teams_yaml(role="owner"), encoding="utf-8")

    with pytest.raises(CollaborationConfigError, match="membership.role"):
        TeamsProvider(path).snapshot()


def test_roster_without_matrix_ids_does_not_conflict(tmp_path: Path) -> None:
    path = tmp_path / "teams.yaml"
    content = _teams_yaml().replace(
        "  matrix:\n    homeserver: https://matrix.example.com\n"
        "    tokenEnv: AGENTTEAMS_WORKER_MATRIX_TOKEN\n", ""
    )
    path.write_text(
        "\n".join(line for line in content.splitlines() if "matrixUserId:" not in line),
        encoding="utf-8",
    )
    snapshot = TeamsProvider(path).snapshot()
    assert snapshot is not None
    assert len(snapshot.teams["team-alpha"].members) == 2


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        (
            '    matrixUserId: "@worker-a:matrix.example.com"\n',
            "",
            "spec.self.matrixUserId",
        ),
        ("    - name: team-alpha\n", "    - name: invalid/team\n", "mount-safe"),
        ("          name: operator-a\n", "          name: leader-a\n", "duplicate name"),
        (
            "        - type: human\n          name: operator-a\n          role: human\n",
            "        - type: agent\n          name: operator-a\n"
            "          runtimeName: leader-a\n          role: worker\n",
            "duplicate runtimeName",
        ),
        (
            '          matrixUserId: "@operator-a:matrix.example.com"\n',
            '          matrixUserId: "@leader-a:matrix.example.com"\n',
            "duplicate matrixUserId",
        ),
        (
            '          matrixUserId: "@operator-a:matrix.example.com"\n',
            '          matrixUserId: "@worker-a:matrix.example.com"\n',
            "conflicts with spec.self",
        ),
    ],
)
def test_teams_provider_matches_runtime_identity_and_roster_validation(
    tmp_path: Path,
    old: str,
    new: str,
    message: str,
) -> None:
    path = tmp_path / "teams.yaml"
    path.write_text(_teams_yaml().replace(old, new), encoding="utf-8")

    with pytest.raises(CollaborationConfigError, match=message):
        TeamsProvider(path).snapshot()


def test_teams_provider_reuses_snapshot_until_content_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "teams.yaml"
    path.write_text(_teams_yaml(), encoding="utf-8")
    parse_calls = 0
    parse = teams_module.parse_teams_config

    def counting_parse(data: bytes) -> Any:
        nonlocal parse_calls
        parse_calls += 1
        return parse(data)

    monkeypatch.setattr(teams_module, "parse_teams_config", counting_parse)
    provider = TeamsProvider(path)

    initial = provider.snapshot()
    assert provider.snapshot() is initial
    assert parse_calls == 1

    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert provider.snapshot() is initial
    assert parse_calls == 1

    path.write_text(_teams_yaml(role="member"), encoding="utf-8")
    updated = provider.snapshot()
    assert updated is not initial
    assert updated is not None
    assert updated.teams["team-alpha"].role == "member"
    assert parse_calls == 2


def test_teams_provider_caches_rejected_update(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = tmp_path / "teams.yaml"
    path.write_text(_teams_yaml(), encoding="utf-8")
    provider = TeamsProvider(path)
    initial = provider.snapshot()
    path.write_text("kind: broken", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        assert provider.snapshot() is initial
        assert provider.snapshot() is initial

    ignored = [
        record
        for record in caplog.records
        if record.message.startswith("agentcore.collaboration.teams.update_ignored")
    ]
    assert len(ignored) == 1
