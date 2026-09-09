"""Async AgentTeams Task Service transport for AgentCore Worker operations."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import anyio
import httpx

from agentcore.collaboration.errors import (
    CollaborationConfigError,
    CollaborationFileTooLargeError,
    CollaborationTaskConflictError,
    CollaborationTaskInvalidError,
    CollaborationTaskNotFoundError,
    CollaborationTaskUnauthorizedError,
    CollaborationTaskUnavailableError,
)
from agentcore_collaboration.teams import TeamsSnapshot

ProviderValue = str | Awaitable[str]
TokenProvider = Callable[[str], ProviderValue]
TokenRefresher = Callable[[str, str], ProviderValue]
EndpointProvider = Callable[[], ProviderValue]


class TaskServiceClient:
    """Small async client covering only the Worker Task Service surface."""

    def __init__(
        self,
        token_provider: TokenProvider,
        *,
        endpoint_provider: EndpointProvider,
        token_refresher: TokenRefresher | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._token_provider = token_provider
        self._token_refresher = token_refresher
        self._endpoint_provider = endpoint_provider
        self._http = http_client or httpx.AsyncClient(timeout=timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def get_task(self, snapshot: TeamsSnapshot, task_id: str) -> Any:
        return await self._request(snapshot, "GET", f"/v1/tasks/{_id(task_id)}")

    async def list_tasks(
        self,
        snapshot: TeamsSnapshot,
        *,
        status: str | None,
        team_id: str | None,
        assigned_to: str | None,
        search: str | None,
        cursor: str | None,
        limit: int,
    ) -> Any:
        return await self._request(
            snapshot,
            "GET",
            "/v1/tasks",
            params={
                "status": status,
                "teamId": team_id,
                "assignedTo": assigned_to,
                "search": search,
                "cursor": cursor,
                "limit": limit,
            },
        )

    async def get_subtask(self, snapshot: TeamsSnapshot, subtask_id: str) -> Any:
        return await self._request(
            snapshot,
            "GET",
            f"/v1/sub-tasks/{_id(subtask_id)}",
        )

    async def list_subtasks(
        self,
        snapshot: TeamsSnapshot,
        *,
        task_id: str | None,
        status: str | None,
        assigned_to: str | None,
    ) -> Any:
        return await self._request(
            snapshot,
            "GET",
            "/v1/sub-tasks",
            params={
                "taskId": task_id,
                "status": status,
                "assignedTo": assigned_to,
            },
        )

    async def ack_subtask(
        self,
        snapshot: TeamsSnapshot,
        subtask_id: str,
        *,
        event_id: str,
    ) -> Any:
        body = {"relatedRoomMessageId": event_id}
        return await self._request(
            snapshot,
            "POST",
            f"/v1/sub-tasks/{_id(subtask_id)}/ack",
            json_body=body,
            idempotency_key=_idempotency_key("ack", subtask_id, event_id, body),
        )

    async def report_subtask_progress(
        self,
        snapshot: TeamsSnapshot,
        subtask_id: str,
        *,
        content: Mapping[str, Any],
        event_id: str,
    ) -> Any:
        return await self._request(
            snapshot,
            "POST",
            f"/v1/sub-tasks/{_id(subtask_id)}/progress",
            json_body={
                "content": dict(content),
                "relatedRoomMessageId": event_id,
            },
        )

    async def heartbeat_subtask(
        self,
        snapshot: TeamsSnapshot,
        subtask_id: str,
    ) -> Any:
        return await self._request(
            snapshot,
            "POST",
            f"/v1/sub-tasks/{_id(subtask_id)}/heartbeat",
        )

    async def block_subtask(
        self,
        snapshot: TeamsSnapshot,
        subtask_id: str,
        *,
        reason: str,
        evidence: Mapping[str, Any] | None,
        event_id: str,
    ) -> Any:
        body: dict[str, Any] = {
            "reason": reason,
            "relatedRoomMessageId": event_id,
        }
        if evidence is not None:
            body["evidence"] = dict(evidence)
        return await self._request(
            snapshot,
            "POST",
            f"/v1/sub-tasks/{_id(subtask_id)}/block",
            json_body=body,
            idempotency_key=_idempotency_key("block", subtask_id, event_id, body),
        )

    async def list_task_files(
        self,
        snapshot: TeamsSnapshot,
        task_id: str,
        *,
        prefix: str | None,
        cursor: str | None,
        limit: int,
    ) -> Any:
        return await self._list_files(
            snapshot,
            "tasks",
            task_id,
            prefix=prefix,
            cursor=cursor,
            limit=limit,
        )

    async def list_subtask_files(
        self,
        snapshot: TeamsSnapshot,
        subtask_id: str,
        *,
        prefix: str | None,
        cursor: str | None,
        limit: int,
    ) -> Any:
        return await self._list_files(
            snapshot,
            "sub-tasks",
            subtask_id,
            prefix=prefix,
            cursor=cursor,
            limit=limit,
        )

    async def read_task_file(
        self,
        snapshot: TeamsSnapshot,
        task_id: str,
        *,
        file_ref: str,
    ) -> bytes:
        return await self._read_file(snapshot, "tasks", task_id, file_ref=file_ref)

    async def read_subtask_file(
        self,
        snapshot: TeamsSnapshot,
        subtask_id: str,
        *,
        file_ref: str,
    ) -> bytes:
        return await self._read_file(
            snapshot,
            "sub-tasks",
            subtask_id,
            file_ref=file_ref,
        )

    async def write_subtask_file(
        self,
        snapshot: TeamsSnapshot,
        subtask_id: str,
        *,
        path: str,
        content: bytes,
        content_type: str,
    ) -> Any:
        return await self._request(
            snapshot,
            "PUT",
            f"/v1/sub-tasks/{_id(subtask_id)}/files",
            params={"path": path},
            files={"file": (_path_name(path), content, content_type)},
        )

    async def write_subtask_file_from_path(
        self,
        snapshot: TeamsSnapshot,
        subtask_id: str,
        *,
        path: str,
        source: Path,
        content_type: str,
    ) -> Any:
        return await self._write_file_from_path(
            snapshot,
            f"/v1/sub-tasks/{_id(subtask_id)}/files",
            path=path,
            source=source,
            content_type=content_type,
        )

    async def submit_subtask_result(
        self,
        snapshot: TeamsSnapshot,
        subtask_id: str,
        *,
        summary: str,
        file_refs: Sequence[str],
        event_id: str,
    ) -> Any:
        body = {
            "summary": summary,
            "fileRefs": list(file_refs),
            "relatedRoomMessageId": event_id,
        }
        return await self._request(
            snapshot,
            "POST",
            f"/v1/sub-tasks/{_id(subtask_id)}/results",
            json_body=body,
            idempotency_key=_idempotency_key("result", subtask_id, event_id, body),
        )

    async def list_results(
        self,
        snapshot: TeamsSnapshot,
        task_id: str,
        *,
        subtask_id: str | None,
    ) -> Any:
        return await self._request(
            snapshot,
            "GET",
            f"/v1/tasks/{_id(task_id)}/results",
            params={"subTaskId": subtask_id},
        )

    async def list_events(
        self,
        snapshot: TeamsSnapshot,
        task_id: str,
        *,
        subtask_id: str | None,
        event_type: str | None,
        cursor: str | None,
        limit: int,
    ) -> Any:
        return await self._request(
            snapshot,
            "GET",
            f"/v1/tasks/{_id(task_id)}/events",
            params={
                "subTaskId": subtask_id,
                "type": event_type,
                "cursor": cursor,
                "limit": limit,
            },
        )

    async def get_task_file_download_ref(
        self,
        snapshot: TeamsSnapshot,
        task_id: str,
        *,
        file_ref: str,
    ) -> Any:
        return await self._get_file_download_ref(
            snapshot,
            "tasks",
            task_id,
            file_ref=file_ref,
        )

    async def get_subtask_file_download_ref(
        self,
        snapshot: TeamsSnapshot,
        subtask_id: str,
        *,
        file_ref: str,
    ) -> Any:
        return await self._get_file_download_ref(
            snapshot,
            "sub-tasks",
            subtask_id,
            file_ref=file_ref,
        )

    async def download_task_file_to_path(
        self,
        snapshot: TeamsSnapshot,
        task_id: str,
        *,
        file_ref: str,
        destination: Path,
    ) -> int:
        reference = await self.get_task_file_download_ref(
            snapshot,
            task_id,
            file_ref=file_ref,
        )
        return await self._download_reference(reference, destination)

    async def download_subtask_file_to_path(
        self,
        snapshot: TeamsSnapshot,
        subtask_id: str,
        *,
        file_ref: str,
        destination: Path,
    ) -> int:
        reference = await self.get_subtask_file_download_ref(
            snapshot,
            subtask_id,
            file_ref=file_ref,
        )
        return await self._download_reference(reference, destination)

    async def list_team_files(
        self,
        snapshot: TeamsSnapshot,
        team_id: str,
        *,
        path: str,
        cursor: str | None,
        limit: int,
    ) -> Any:
        return await self._request(
            snapshot,
            "GET",
            f"/v1/teams/{_id(team_id)}/files",
            params={"path": path, "cursor": cursor, "limit": limit},
        )

    async def stat_team_file(
        self,
        snapshot: TeamsSnapshot,
        team_id: str,
        *,
        path: str,
    ) -> Any:
        return await self._request(
            snapshot,
            "GET",
            f"/v1/teams/{_id(team_id)}/files/stat",
            params={"path": path},
        )

    async def read_team_file(
        self,
        snapshot: TeamsSnapshot,
        team_id: str,
        *,
        path: str,
    ) -> bytes:
        result = await self._request(
            snapshot,
            "GET",
            f"/v1/teams/{_id(team_id)}/files/content",
            params={"path": path},
            binary=True,
        )
        assert isinstance(result, bytes)
        return result

    async def write_team_file_from_path(
        self,
        snapshot: TeamsSnapshot,
        team_id: str,
        *,
        path: str,
        source: Path,
        content_type: str,
    ) -> Any:
        return await self._write_file_from_path(
            snapshot,
            f"/v1/teams/{_id(team_id)}/files",
            path=path,
            source=source,
            content_type=content_type,
        )

    async def _get_file_download_ref(
        self,
        snapshot: TeamsSnapshot,
        resource: str,
        resource_id: str,
        *,
        file_ref: str,
    ) -> Any:
        return await self._request(
            snapshot,
            "GET",
            f"/v1/{resource}/{_id(resource_id)}/files/download",
            params={"fileRef": file_ref},
        )

    async def _download_reference(self, reference: Any, destination: Path) -> int:
        url = _download_url(reference)
        total = 0
        try:
            async with self._http.stream(
                "GET",
                url,
                headers={"Accept": "*/*"},
                follow_redirects=True,
            ) as response:
                if not 200 <= response.status_code < 300:
                    raise CollaborationTaskUnavailableError(
                        "The file download reference is temporarily unavailable."
                    )
                async with await anyio.open_file(destination, "wb") as target:
                    async for chunk in response.aiter_bytes():
                        await target.write(chunk)
                        total += len(chunk)
        except httpx.RequestError as exc:
            raise CollaborationTaskUnavailableError(
                "The file download reference is temporarily unavailable."
            ) from exc
        except OSError as exc:
            raise CollaborationConfigError("The local download path is unavailable.") from exc
        return total

    async def _write_file_from_path(
        self,
        snapshot: TeamsSnapshot,
        endpoint: str,
        *,
        path: str,
        source: Path,
        content_type: str,
    ) -> Any:
        with source.open("rb") as content:
            return await self._request(
                snapshot,
                "PUT",
                endpoint,
                params={"path": path},
                files={"file": (source.name, content, content_type)},
            )

    async def _list_files(
        self,
        snapshot: TeamsSnapshot,
        resource: str,
        resource_id: str,
        *,
        prefix: str | None,
        cursor: str | None,
        limit: int,
    ) -> Any:
        return await self._request(
            snapshot,
            "GET",
            f"/v1/{resource}/{_id(resource_id)}/files",
            params={"prefix": prefix, "cursor": cursor, "limit": limit},
        )

    async def _read_file(
        self,
        snapshot: TeamsSnapshot,
        resource: str,
        resource_id: str,
        *,
        file_ref: str,
    ) -> bytes:
        result = await self._request(
            snapshot,
            "GET",
            f"/v1/{resource}/{_id(resource_id)}/files/content",
            params={"fileRef": file_ref},
            binary=True,
        )
        assert isinstance(result, bytes)
        return result

    async def _request(
        self,
        snapshot: TeamsSnapshot,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        files: Any = None,
        binary: bool = False,
    ) -> Any:
        token_env = snapshot.matrix_token_env
        if token_env is None:
            raise CollaborationConfigError("Worker Task Service Matrix identity is not configured.")
        url = f"{await self._endpoint()}/{path.lstrip('/')}"
        response: httpx.Response | None = None
        retry_token: str | None = None
        for attempt in range(2):
            token = retry_token or await self._token(token_env)
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "*/*" if binary else "application/json",
            }
            if idempotency_key is not None:
                headers["Idempotency-Key"] = idempotency_key
            try:
                kwargs: dict[str, Any] = {
                    "headers": headers,
                    "params": {
                        key: value for key, value in (params or {}).items() if value is not None
                    },
                }
                if json_body is not None:
                    kwargs["json"] = dict(json_body)
                if files is not None:
                    kwargs["files"] = files
                response = await self._http.request(method, url, **kwargs)
            except httpx.RequestError as exc:
                raise CollaborationTaskUnavailableError(
                    "Task Service is temporarily unavailable."
                ) from exc
            if response.status_code != 401 or attempt == 1:
                break
            if self._token_refresher is not None:
                retry_token = await self._refresh_token(
                    token_env,
                    token,
                )
        assert response is not None
        _raise_for_status(response)
        if binary:
            return response.content
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise CollaborationTaskUnavailableError(
                "Task Service returned an invalid response."
            ) from exc

    async def _token(self, environment_name: str) -> str:
        try:
            value = self._token_provider(environment_name)
            if inspect.isawaitable(value):
                value = await value
        except Exception as exc:
            raise CollaborationConfigError("Worker Task Service token is unavailable.") from exc
        if not isinstance(value, str) or not value.strip():
            raise CollaborationConfigError("Worker Task Service token is unavailable.")
        return value.strip()

    async def _refresh_token(self, environment_name: str, rejected: str) -> str:
        assert self._token_refresher is not None
        try:
            value = self._token_refresher(environment_name, rejected)
            if inspect.isawaitable(value):
                value = await value
        except Exception as exc:
            raise CollaborationConfigError("Worker Task Service token is unavailable.") from exc
        if not isinstance(value, str) or not value.strip():
            raise CollaborationConfigError("Worker Task Service token is unavailable.")
        return value.strip()

    async def _endpoint(self) -> str:
        try:
            value = self._endpoint_provider()
            if inspect.isawaitable(value):
                value = await value
        except Exception as exc:
            raise CollaborationConfigError("Worker Task Service endpoint is unavailable.") from exc
        if not isinstance(value, str) or not value.strip():
            raise CollaborationConfigError("Worker Task Service endpoint is unavailable.")
        return value.strip().rstrip("/")


def _id(value: str) -> str:
    return quote(value, safe="")


def _path_name(path: str) -> str:
    return path.rsplit("/", 1)[-1] or "file"


def _download_url(reference: Any) -> str:
    if not isinstance(reference, Mapping):
        raise CollaborationTaskUnavailableError(
            "Task Service returned an invalid download reference."
        )
    value = reference.get("downloadUrl")
    if not isinstance(value, str) or not value.strip():
        raise CollaborationTaskUnavailableError(
            "Task Service returned an invalid download reference."
        )
    value = value.strip()
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise CollaborationTaskUnavailableError(
            "Task Service returned an invalid download reference."
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise CollaborationTaskUnavailableError(
            "Task Service returned an invalid download reference."
        )
    return value


def _idempotency_key(
    operation: str,
    target: str,
    event_id: str,
    body: Mapping[str, Any],
) -> str:
    payload = json.dumps(body, sort_keys=True, separators=(",", ":"))
    value = f"{operation}\0{target}\0{event_id}\0{payload}".encode()
    return f"agentcore-{hashlib.sha256(value).hexdigest()}"


def _raise_for_status(response: httpx.Response) -> None:
    if 200 <= response.status_code < 300:
        return
    if response.status_code in {401, 403}:
        raise CollaborationTaskUnauthorizedError("Task Service rejected this Worker operation.")
    if response.status_code in {400, 422}:
        raise CollaborationTaskInvalidError("Task Service rejected the operation input.")
    if response.status_code == 404:
        raise CollaborationTaskNotFoundError("The requested Task Service resource was not found.")
    if response.status_code == 409:
        raise CollaborationTaskConflictError(
            "Task Service rejected the current Task or Subtask state."
        )
    if response.status_code == 413:
        raise CollaborationFileTooLargeError("The requested file exceeds the supported size limit.")
    raise CollaborationTaskUnavailableError("Task Service is temporarily unavailable.")
