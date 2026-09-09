from __future__ import annotations

import asyncio
from collections.abc import Iterable

import pytest

from agentcore.collaboration.debug import DebugCollaborationRuntime


def _teams_yaml(*, role: str = "worker", matrix_user_id: str = "@worker-a:example.com") -> bytes:
    return f"""apiVersion: agentteams.io/v1alpha1
kind: TeamsConfig
metadata:
  runtimeName: runtime-worker-a
spec:
  self:
    name: worker-a
    runtimeName: runtime-worker-a
    matrixUserId: "{matrix_user_id}"
  matrix:
    tokenEnv: AGENTTEAMS_WORKER_MATRIX_TOKEN
  defaultTeamName: team-a
  teams:
    - name: team-a
      membership:
        role: {role}
      members: []
""".encode()


class _DebugSource:
    matrix_url = "https://matrix.example.com/gateway"

    def __init__(self, configurations: Iterable[bytes | None]) -> None:
        self._configurations = iter(configurations)
        self.matrix_tokens = iter(["matrix-token-1", "matrix-token-2"])
        self.matrix_token_calls = 0
        self.config_calls = 0

    async def load_teams_config(self) -> bytes | None:
        self.config_calls += 1
        await asyncio.sleep(0)
        return next(self._configurations)

    async def exchange_matrix_token(self) -> str:
        self.matrix_token_calls += 1
        return next(self.matrix_tokens)


@pytest.mark.asyncio
async def test_debug_collaboration_refreshes_teams_and_reuses_token_for_same_identity() -> None:
    source = _DebugSource(
        [
            _teams_yaml(),
            _teams_yaml(role="member"),
            _teams_yaml(role="worker", matrix_user_id="@worker-b:example.com"),
        ]
    )
    runtime = DebugCollaborationRuntime(source, refresh_interval=0)

    first = await runtime.teams_snapshot()
    first_token = await runtime.token("AGENTTEAMS_WORKER_MATRIX_TOKEN")
    second = await runtime.teams_snapshot()
    reused_token = await runtime.token("AGENTTEAMS_WORKER_MATRIX_TOKEN")
    third = await runtime.teams_snapshot()
    changed_token = await runtime.token("AGENTTEAMS_WORKER_MATRIX_TOKEN")

    assert first is not None and first.teams["team-a"].role == "worker"
    assert second is not None and second.teams["team-a"].role == "member"
    assert third is not None and third.self_matrix_user_id == "@worker-b:example.com"
    assert (first_token, reused_token, changed_token) == (
        "matrix-token-1",
        "matrix-token-1",
        "matrix-token-2",
    )
    assert source.matrix_token_calls == 2


@pytest.mark.asyncio
async def test_debug_collaboration_retains_last_good_and_disables_on_confirmed_absence() -> None:
    source = _DebugSource([_teams_yaml(), b"kind: broken", None])
    runtime = DebugCollaborationRuntime(source, refresh_interval=0)

    first = await runtime.teams_snapshot()
    invalid_update = await runtime.teams_snapshot()
    deleted = await runtime.teams_snapshot()

    assert invalid_update is first
    assert deleted is None


@pytest.mark.asyncio
async def test_debug_collaboration_refresh_is_single_flight() -> None:
    source = _DebugSource([_teams_yaml()])
    runtime = DebugCollaborationRuntime(source)

    first, concurrent = await asyncio.gather(
        runtime.teams_snapshot(),
        runtime.teams_snapshot(),
    )

    assert concurrent is first
    assert source.config_calls == 1


@pytest.mark.asyncio
async def test_debug_collaboration_reuses_snapshot_for_unchanged_content() -> None:
    content = _teams_yaml()
    source = _DebugSource([content, content])
    runtime = DebugCollaborationRuntime(source, refresh_interval=0)

    first = await runtime.teams_snapshot()
    second = await runtime.teams_snapshot()

    assert second is first
    assert source.config_calls == 2


def test_debug_collaboration_derives_task_service_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = DebugCollaborationRuntime(_DebugSource([]))

    assert runtime.endpoint() == "https://matrix.example.com/gateway/agentteams-app"

    monkeypatch.setenv(
        "AGENTCORE_TASK_SERVICE_ENDPOINT",
        "https://task-service.example.com/custom/",
    )
    assert runtime.endpoint() == "https://task-service.example.com/custom"
