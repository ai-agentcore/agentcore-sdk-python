from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from agentcore.collaboration.errors import (
    CollaborationConfigError,
    CollaborationFileTooLargeError,
    CollaborationTaskConflictError,
    CollaborationTaskInvalidError,
    CollaborationTaskNotFoundError,
    CollaborationTaskUnauthorizedError,
)
from agentcore.collaboration.task_service import TaskServiceClient
from agentcore.collaboration.teams import parse_teams_config


def _snapshot() -> object:
    return parse_teams_config(
        b"""
apiVersion: agentteams.io/v1alpha1
kind: TeamsConfig
metadata:
  runtimeName: worker-a
spec:
  self:
    name: worker-a
    runtimeName: worker-a
    matrixUserId: "@worker-a:matrix.example.com"
  matrix:
    tokenEnv: AGENTTEAMS_WORKER_MATRIX_TOKEN
  teams: []
"""
    )


def _endpoint() -> str:
    return "https://agentteams.example.com/agentteams-app"


@pytest.mark.asyncio
async def test_task_service_requires_matrix_identity_before_request() -> None:
    def unexpected_call(*_args: object) -> str:
        pytest.fail("must not resolve credentials or endpoint without Matrix identity")

    client = TaskServiceClient(unexpected_call, endpoint_provider=unexpected_call)
    snapshot = replace(_snapshot(), matrix_token_env=None)  # type: ignore[type-var]
    try:
        with pytest.raises(CollaborationConfigError, match="Matrix identity is not configured"):
            await client.get_task(snapshot, "task-1")
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_task_service_refreshes_worker_token_once_after_401() -> None:
    tokens = iter(["old-token", "new-token"])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers["Authorization"] == "Bearer old-token":
            return httpx.Response(401, json={"error": "M_UNKNOWN_TOKEN"})
        return httpx.Response(200, json={"taskId": "task-1"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TaskServiceClient(
        lambda _name: next(tokens),
        endpoint_provider=_endpoint,
        http_client=http,
    )
    try:
        result = await client.get_task(_snapshot(), "task/1")  # type: ignore[arg-type]
    finally:
        await client.aclose()

    assert result == {"taskId": "task-1"}
    assert [request.url.raw_path for request in requests] == [
        b"/agentteams-app/v1/tasks/task%2F1",
        b"/agentteams-app/v1/tasks/task%2F1",
    ]
    assert [request.headers["Authorization"] for request in requests] == [
        "Bearer old-token",
        "Bearer new-token",
    ]


@pytest.mark.asyncio
async def test_task_service_supports_async_debug_providers_and_explicit_refresh() -> None:
    refreshes: list[str] = []
    requests: list[httpx.Request] = []

    async def token_provider(_name: str) -> str:
        return "old-token"

    async def refresh_token(_name: str, rejected: str) -> str:
        refreshes.append(rejected)
        return "new-token"

    async def endpoint_provider() -> str:
        return _endpoint()

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers["Authorization"] == "Bearer old-token":
            return httpx.Response(401)
        return httpx.Response(200, json={"taskId": "task-1"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TaskServiceClient(
        token_provider,
        endpoint_provider=endpoint_provider,
        token_refresher=refresh_token,
        http_client=http,
    )
    try:
        result = await client.get_task(_snapshot(), "task-1")  # type: ignore[arg-type]
    finally:
        await client.aclose()

    assert result == {"taskId": "task-1"}
    assert refreshes == ["old-token"]
    assert [request.headers["Authorization"] for request in requests] == [
        "Bearer old-token",
        "Bearer new-token",
    ]


@pytest.mark.asyncio
async def test_task_service_sends_related_event_and_idempotency_key() -> None:
    captured: httpx.Request | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured
        captured = request
        return httpx.Response(200, json={"status": "ACKED"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TaskServiceClient(
        lambda _name: "worker-token",
        endpoint_provider=_endpoint,
        http_client=http,
    )
    try:
        await client.ack_subtask(
            _snapshot(),  # type: ignore[arg-type]
            "subtask-1",
            event_id="$event-1",
        )
    finally:
        await client.aclose()

    assert captured is not None
    assert captured.url.path == "/agentteams-app/v1/sub-tasks/subtask-1/ack"
    assert captured.headers["Idempotency-Key"]
    assert captured.headers["Authorization"] == "Bearer worker-token"
    assert captured.content == b'{"relatedRoomMessageId":"$event-1"}'


@pytest.mark.asyncio
async def test_task_service_stops_after_one_401_retry_without_exposing_token() -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "raw downstream detail"})

    def token_provider(_name: str) -> str:
        nonlocal calls
        calls += 1
        return "secret-worker-token"

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TaskServiceClient(
        token_provider,
        endpoint_provider=_endpoint,
        http_client=http,
    )
    try:
        with pytest.raises(CollaborationTaskUnauthorizedError) as caught:
            await client.get_task(_snapshot(), "task-1")  # type: ignore[arg-type]
    finally:
        await client.aclose()

    assert calls == 2
    assert "secret-worker-token" not in str(caught.value)
    assert "raw downstream detail" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected_error"),
    [
        (400, CollaborationTaskInvalidError),
        (404, CollaborationTaskNotFoundError),
        (409, CollaborationTaskConflictError),
        (413, CollaborationFileTooLargeError),
        (422, CollaborationTaskInvalidError),
    ],
)
async def test_task_service_preserves_actionable_rejection_category(
    status: int,
    expected_error: type[Exception],
) -> None:
    http = httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _request: httpx.Response(status, json={"message": "downstream detail"})
        )
    )
    client = TaskServiceClient(
        lambda _name: "worker-token",
        endpoint_provider=_endpoint,
        http_client=http,
    )
    try:
        with pytest.raises(expected_error) as caught:
            await client.get_task(_snapshot(), "task-1")  # type: ignore[arg-type]
    finally:
        await client.aclose()

    assert "downstream detail" not in str(caught.value)


@pytest.mark.asyncio
async def test_task_service_downloads_large_file_without_returning_signed_url(
    tmp_path: Path,
) -> None:
    content = b"x" * (2 * 1024 * 1024)
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.host == "agentteams.example.com":
            return httpx.Response(
                200,
                json={
                    "fileRef": "shared/large.bin",
                    "downloadUrl": "https://download.example.com/large.bin?signature=secret",
                    "expiresAt": "2026-09-02T12:00:00Z",
                },
            )
        return httpx.Response(200, content=content)

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TaskServiceClient(
        lambda _name: "worker-token",
        endpoint_provider=_endpoint,
        http_client=http,
    )
    output = tmp_path / "large.bin"
    try:
        size = await client.download_task_file_to_path(
            _snapshot(),  # type: ignore[arg-type]
            "task-1",
            file_ref="shared/large.bin",
            destination=output,
        )
    finally:
        await client.aclose()

    assert size == len(content)
    assert output.read_bytes() == content
    assert [(request.method, request.url.host) for request in requests] == [
        ("GET", "agentteams.example.com"),
        ("GET", "download.example.com"),
    ]
    assert "Authorization" not in requests[1].headers


@pytest.mark.asyncio
async def test_task_service_worker_surface_matches_agentteams_paths() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/files/content"):
            return httpx.Response(200, content=b"file-content")
        return httpx.Response(200, json={"ok": True})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TaskServiceClient(
        lambda _name: "worker-token",
        endpoint_provider=_endpoint,
        http_client=http,
    )
    snapshot = _snapshot()
    try:
        await client.list_tasks(  # type: ignore[arg-type]
            snapshot,
            status="RUNNING",
            team_id="team-alpha",
            assigned_to="@worker:example.com",
            search="review",
            cursor="task-next",
            limit=25,
        )
        await client.get_subtask(snapshot, "sub-1")  # type: ignore[arg-type]
        await client.list_subtasks(  # type: ignore[arg-type]
            snapshot, task_id="task-1", status="RUNNING", assigned_to="@worker:example.com"
        )
        await client.report_subtask_progress(  # type: ignore[arg-type]
            snapshot, "sub-1", content={"percent": 50}, event_id="$event-1"
        )
        await client.heartbeat_subtask(snapshot, "sub-1")  # type: ignore[arg-type]
        await client.block_subtask(  # type: ignore[arg-type]
            snapshot,
            "sub-1",
            reason="waiting",
            evidence={"ticket": "A-1"},
            event_id="$event-1",
        )
        await client.list_task_files(  # type: ignore[arg-type]
            snapshot, "task-1", prefix="src/", cursor=None, limit=20
        )
        await client.list_subtask_files(  # type: ignore[arg-type]
            snapshot, "sub-1", prefix=None, cursor="next", limit=10
        )
        assert (
            await client.read_task_file(  # type: ignore[arg-type]
                snapshot, "task-1", file_ref="shared/a.txt"
            )
            == b"file-content"
        )
        assert (
            await client.read_subtask_file(  # type: ignore[arg-type]
                snapshot, "sub-1", file_ref="shared/b.txt"
            )
            == b"file-content"
        )
        await client.write_subtask_file(  # type: ignore[arg-type]
            snapshot,
            "sub-1",
            path="result.txt",
            content=b"done",
            content_type="text/plain",
        )
        await client.submit_subtask_result(  # type: ignore[arg-type]
            snapshot,
            "sub-1",
            summary="done",
            file_refs=["shared/result.txt"],
            event_id="$event-1",
        )
        await client.list_results(  # type: ignore[arg-type]
            snapshot, "task-1", subtask_id="sub-1"
        )
        await client.list_events(  # type: ignore[arg-type]
            snapshot,
            "task-1",
            subtask_id="sub-1",
            event_type="progress_reported",
            cursor="event-next",
            limit=30,
        )
        await client.get_task_file_download_ref(  # type: ignore[arg-type]
            snapshot, "task-1", file_ref="shared/a.txt"
        )
        await client.get_subtask_file_download_ref(  # type: ignore[arg-type]
            snapshot, "sub-1", file_ref="shared/b.txt"
        )
        await client.list_team_files(  # type: ignore[arg-type]
            snapshot,
            "team-alpha",
            path="shared/knowledge",
            cursor="team-next",
            limit=40,
        )
        await client.stat_team_file(  # type: ignore[arg-type]
            snapshot, "team-alpha", path="shared/knowledge/guide.md"
        )
        assert (
            await client.read_team_file(  # type: ignore[arg-type]
                snapshot, "team-alpha", path="shared/knowledge/guide.md"
            )
            == b"file-content"
        )
    finally:
        await client.aclose()

    assert [(request.method, request.url.path) for request in requests] == [
        ("GET", "/agentteams-app/v1/tasks"),
        ("GET", "/agentteams-app/v1/sub-tasks/sub-1"),
        ("GET", "/agentteams-app/v1/sub-tasks"),
        ("POST", "/agentteams-app/v1/sub-tasks/sub-1/progress"),
        ("POST", "/agentteams-app/v1/sub-tasks/sub-1/heartbeat"),
        ("POST", "/agentteams-app/v1/sub-tasks/sub-1/block"),
        ("GET", "/agentteams-app/v1/tasks/task-1/files"),
        ("GET", "/agentteams-app/v1/sub-tasks/sub-1/files"),
        ("GET", "/agentteams-app/v1/tasks/task-1/files/content"),
        ("GET", "/agentteams-app/v1/sub-tasks/sub-1/files/content"),
        ("PUT", "/agentteams-app/v1/sub-tasks/sub-1/files"),
        ("POST", "/agentteams-app/v1/sub-tasks/sub-1/results"),
        ("GET", "/agentteams-app/v1/tasks/task-1/results"),
        ("GET", "/agentteams-app/v1/tasks/task-1/events"),
        ("GET", "/agentteams-app/v1/tasks/task-1/files/download"),
        ("GET", "/agentteams-app/v1/sub-tasks/sub-1/files/download"),
        ("GET", "/agentteams-app/v1/teams/team-alpha/files"),
        ("GET", "/agentteams-app/v1/teams/team-alpha/files/stat"),
        ("GET", "/agentteams-app/v1/teams/team-alpha/files/content"),
    ]
    assert dict(requests[0].url.params) == {
        "status": "RUNNING",
        "teamId": "team-alpha",
        "assignedTo": "@worker:example.com",
        "search": "review",
        "cursor": "task-next",
        "limit": "25",
    }
    assert dict(requests[2].url.params) == {
        "taskId": "task-1",
        "status": "RUNNING",
        "assignedTo": "@worker:example.com",
    }
    assert requests[10].url.params["path"] == "result.txt"
    assert b'name="file"; filename="result.txt"' in requests[10].content
    assert dict(requests[13].url.params) == {
        "subTaskId": "sub-1",
        "type": "progress_reported",
        "cursor": "event-next",
        "limit": "30",
    }
    assert dict(requests[16].url.params) == {
        "path": "shared/knowledge",
        "cursor": "team-next",
        "limit": "40",
    }


@pytest.mark.asyncio
async def test_task_service_streams_local_uploads_as_multipart(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"path": request.url.params["path"]})

    source = tmp_path / "artifact.bin"
    source.write_bytes(b"\x00binary\xff")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TaskServiceClient(
        lambda _name: "worker-token",
        endpoint_provider=_endpoint,
        http_client=http,
    )
    snapshot = _snapshot()
    try:
        await client.write_subtask_file_from_path(  # type: ignore[arg-type]
            snapshot,
            "sub-1",
            path="result/artifact.bin",
            source=source,
            content_type="application/octet-stream",
        )
        await client.write_team_file_from_path(  # type: ignore[arg-type]
            snapshot,
            "team-alpha",
            path="shared/knowledge/artifact.bin",
            source=source,
            content_type="application/octet-stream",
        )
    finally:
        await client.aclose()

    assert [(request.method, request.url.path) for request in requests] == [
        ("PUT", "/agentteams-app/v1/sub-tasks/sub-1/files"),
        ("PUT", "/agentteams-app/v1/teams/team-alpha/files"),
    ]
    assert b'filename="artifact.bin"' in requests[0].content
    assert b"\x00binary\xff" in requests[0].content
    assert requests[1].url.params["path"] == "shared/knowledge/artifact.bin"


@pytest.mark.asyncio
async def test_task_service_rewinds_local_upload_before_401_retry(tmp_path: Path) -> None:
    tokens = iter(["old-token", "new-token"])
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.headers["Authorization"] == "Bearer old-token":
            return httpx.Response(401, json={"error": "M_UNKNOWN_TOKEN"})
        return httpx.Response(200, json={"path": "result/artifact.bin"})

    source = tmp_path / "artifact.bin"
    source.write_bytes(b"complete-binary-content")
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TaskServiceClient(
        lambda _name: next(tokens),
        endpoint_provider=_endpoint,
        http_client=http,
    )
    try:
        await client.write_subtask_file_from_path(  # type: ignore[arg-type]
            _snapshot(),
            "sub-1",
            path="result/artifact.bin",
            source=source,
            content_type="application/octet-stream",
        )
    finally:
        await client.aclose()

    assert len(requests) == 2
    assert b"complete-binary-content" in requests[0].content
    assert b"complete-binary-content" in requests[1].content


@pytest.mark.asyncio
async def test_task_service_rereads_endpoint_for_each_request() -> None:
    endpoints = iter(
        [
            "https://old.example.com/agentteams-app",
            "https://new.example.com/agentteams-app",
        ]
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"taskId": "task-1"})

    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = TaskServiceClient(
        lambda _name: "worker-token",
        endpoint_provider=lambda: next(endpoints),
        http_client=http,
    )
    try:
        await client.get_task(_snapshot(), "task-1")  # type: ignore[arg-type]
        await client.get_task(_snapshot(), "task-1")  # type: ignore[arg-type]
    finally:
        await client.aclose()

    assert [request.url.host for request in requests] == ["old.example.com", "new.example.com"]
