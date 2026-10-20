"""Room media transport. No polling, Task permissions or file understanding."""

from __future__ import annotations

import inspect
import logging
import mimetypes
import re
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlsplit

import anyio
import httpx

from agentcore.collaboration.context import CollaborationTurnContext
from agentcore.collaboration.errors import (
    CollaborationConfigError,
    CollaborationError,
    CollaborationFileTooLargeError,
)
from agentcore_collaboration._logging import safe_error_message
from agentcore_collaboration.task_service import EndpointProvider, TokenProvider, TokenRefresher
from agentcore_collaboration.teams import TeamsSnapshot

logger = logging.getLogger("agentcore.collaboration.room_files")


class RoomFileError(CollaborationError):
    def __init__(self, kind: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = "COLLABORATION_ROOM_FILE_" + kind
        self.retryable = retryable


@dataclass(frozen=True)
class Attachment:
    event_id: str
    filename: str
    content_type: str
    media_path: str
    limit: int


class MatrixFileClient:
    def __init__(
        self,
        token_provider: TokenProvider,
        *,
        endpoint_provider: EndpointProvider,
        token_refresher: TokenRefresher | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._token_provider = token_provider
        self._endpoint_provider = endpoint_provider
        self._token_refresher = token_refresher
        self._http = http_client or httpx.AsyncClient(timeout=30.0, follow_redirects=False)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def attachment(self, snapshot: TeamsSnapshot, room: str, event_id: str) -> Attachment:
        event = await self._json(snapshot, "GET", _event_path(room, event_id))
        content = _content(event, room)
        relation = content.get("m.relates_to", {})
        if isinstance(relation, dict) and relation.get("rel_type") == "m.replace":
            original_id = relation.get("event_id")
            if not isinstance(original_id, str) or not original_id:
                raise _unsupported()
            original = await self._json(snapshot, "GET", _event_path(room, original_id))
            original_content = _content(original, room)
            if (
                event.get("sender") != original.get("sender")
                or not event.get("sender")
                or _relation(original_content).get("rel_type") in {"m.replace", "m.thread"}
            ):
                raise _unsupported()
            content = content.get("m.new_content", {})
        if not isinstance(content, dict) or content.get("msgtype") not in {"m.file", "m.image"}:
            raise _unsupported()
        if "file" in content or _relation(content).get("rel_type") == "m.thread":
            raise _unsupported()
        media_path = _media_path(content.get("url"))
        filename = safe_filename(content.get("filename") or content.get("body"))
        info = content.get("info", {})
        mime = info.get("mimetype") if isinstance(info, dict) else None
        return Attachment(
            event_id,
            filename,
            mime if isinstance(mime, str) and mime else "application/octet-stream",
            media_path,
            await self._limit(snapshot),
        )

    async def download_to(
        self,
        snapshot: TeamsSnapshot,
        attachment: Attachment,
        destination: Path,
    ) -> int:
        response = await self._request(snapshot, "GET", attachment.media_path)
        size = 0
        try:
            async with await anyio.open_file(destination, "wb") as output:
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > attachment.limit:
                        raise CollaborationFileTooLargeError("File exceeds the Matrix media limit.")
                    await output.write(chunk)
        except httpx.TransportError as exc:
            raise RoomFileError(
                "UNAVAILABLE", "Media download was interrupted.", retryable=True
            ) from exc
        finally:
            with anyio.CancelScope(shield=True):
                await response.aclose()
        return size

    async def send(
        self,
        snapshot: TeamsSnapshot,
        context: CollaborationTurnContext,
        source: Path,
        caption: str | None,
    ) -> dict[str, Any]:
        limit = await self._limit(snapshot)
        size = (await anyio.Path(source).stat()).st_size
        if size > limit:
            raise CollaborationFileTooLargeError("File exceeds the Matrix media limit.")
        mime = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        uploaded = await self._json(
            snapshot,
            "POST",
            "/_matrix/media/v3/upload?filename=" + quote(source.name, safe=""),
            source=source,
            mime=mime,
            limit=limit,
        )
        uri = uploaded.get("content_uri")
        try:
            _media_path(uri)
        except RoomFileError as exc:
            raise RoomFileError(
                "SEND_UNKNOWN", "Media upload result could not be confirmed."
            ) from exc
        content = {
            "msgtype": "m.image"
            if mime in {"image/png", "image/jpeg", "image/gif", "image/webp"}
            else "m.file",
            "body": caption or source.name,
            "filename": source.name,
            "url": uri,
            "info": {"mimetype": mime, "size": size},
            "m.relates_to": {"m.in_reply_to": {"event_id": context.event_id}},
        }
        transaction = uuid.uuid4().hex
        sent = await self._json(
            snapshot,
            "PUT",
            f"/_matrix/client/v3/rooms/{quote(context.room_id, safe='')}"
            f"/send/m.room.message/{transaction}",
            body=content,
        )
        if not isinstance(sent.get("event_id"), str) or not sent["event_id"]:
            raise RoomFileError("SEND_UNKNOWN", "Room file delivery could not be confirmed.")
        logger.info(
            "agentcore.room_file.sent room_id=%s request_event_id=%s event_id=%s bytes=%s",
            context.room_id, context.event_id, sent["event_id"], size,
        )
        return {"event_id": sent["event_id"], "filename": source.name, "status": "sent"}

    async def _limit(self, snapshot: TeamsSnapshot) -> int:
        data = await self._json(snapshot, "GET", "/_matrix/client/v1/media/config")
        value = data.get("m.upload.size")
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RoomFileError("UNAVAILABLE", "Matrix media limit is unavailable.", retryable=True)
        return value

    async def _json(
        self,
        snapshot: TeamsSnapshot,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response = await self._request(snapshot, method, path, **kwargs)
        try:
            await response.aread()
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("object required")
            return data
        except (ValueError, httpx.TransportError) as exc:
            raise _unconfirmed(method) from exc
        finally:
            with anyio.CancelScope(shield=True):
                await response.aclose()

    async def _request(
        self,
        snapshot: TeamsSnapshot,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        source: Path | None = None,
        mime: str = "application/octet-stream",
        limit: int = 0,
    ) -> httpx.Response:
        name = snapshot.matrix_token_env
        if not name:
            raise CollaborationConfigError("Worker Matrix identity is not configured.")
        endpoint = self._endpoint_provider()
        endpoint = await endpoint if inspect.isawaitable(endpoint) else endpoint
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise CollaborationConfigError("Matrix endpoint is unavailable.")
        token_value = self._token_provider(name)
        token = await token_value if inspect.isawaitable(token_value) else token_value
        uncertain = False
        for attempt in range(2):
            if not token:
                raise CollaborationConfigError("Worker Matrix token is unavailable.")
            headers = {"Authorization": f"Bearer {token}"}
            if source is not None:
                headers["Content-Type"] = mime
            request = self._http.build_request(
                method,
                endpoint.rstrip("/") + path,
                headers=headers,
                json=body,
                content=_file_chunks(source, limit) if source is not None else None,
            )
            try:
                response = await self._http.send(request, stream=True, follow_redirects=False)
            except httpx.TransportError as exc:
                if method == "PUT" and attempt == 0:
                    uncertain = True
                    logger.info("agentcore.room_file.send_retry retries=1 same_transaction=true")
                    continue  # Same token, transaction and content; never upload again.
                raise _unconfirmed(method) from exc
            if response.status_code == 401 and attempt == 0 and not uncertain:
                await response.aclose()
                value = (
                    self._token_refresher(name, token)
                    if self._token_refresher
                    else self._token_provider(name)
                )
                token = await value if inspect.isawaitable(value) else value
                continue
            if not response.is_success:
                status = response.status_code
                logger.warning(
                    "agentcore.collaboration.room_files.http_failed method=%s status=%s "
                    "request_id=%s retries=%s",
                    method,
                    status,
                    safe_error_message(
                        response.headers.get("x-request-id")
                        or response.headers.get("x-acs-request-id")
                    ),
                    attempt,
                )
                await response.aclose()
                if uncertain:
                    raise _unconfirmed(method)
                if status in {401, 403}:
                    raise RoomFileError("UNAUTHORIZED", f"Matrix denied access (HTTP {status}).")
                if status == 404:
                    raise RoomFileError("NOT_FOUND", "Room event or media does not exist.")
                if status == 413:
                    raise CollaborationFileTooLargeError("File exceeds the Matrix media limit.")
                raise RoomFileError(
                    "UNAVAILABLE",
                    f"Matrix request failed (HTTP {status}).",
                    retryable=method == "GET" and status >= 500,
                )
            return response
        raise _unconfirmed(method)


async def _file_chunks(source: Path, limit: int) -> AsyncIterator[bytes]:
    size = 0
    async with await anyio.open_file(source, "rb") as stream:
        while chunk := await stream.read(65536):
            size += len(chunk)
            if size > limit:
                raise CollaborationFileTooLargeError("File exceeds the Matrix media limit.")
            yield chunk


def _content(event: dict[str, Any], room: str) -> dict[str, Any]:
    if event.get("room_id") != room or event.get("type") != "m.room.message":
        raise _unsupported()
    content = event.get("content")
    if not isinstance(content, dict):
        raise _unsupported()
    return content


def _relation(content: dict[str, Any]) -> dict[str, Any]:
    value = content.get("m.relates_to")
    return value if isinstance(value, dict) else {}


def _media_path(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"mxc://[^/\s]+/[^/?#\s]+", value):
        raise _unsupported()
    server, media = value[6:].split("/", 1)
    return (
        "/_matrix/client/v1/media/download/" + quote(server, safe="") + "/" + quote(media, safe="")
    )


def _event_path(room: str, event: str) -> str:
    return f"/_matrix/client/v3/rooms/{quote(room, safe='')}/event/{quote(event, safe='')}"


def safe_filename(value: object) -> str:
    name = str(value or "attachment").replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r'[\x00-\x1f\x7f<>:"|?*]', "_", name).strip(" .")[:120]
    if not name:
        return "attachment"
    if re.match(r"(?i)^(con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)", name):
        name = "_" + name
    return name


def _unsupported() -> RoomFileError:
    return RoomFileError(
        "UNSUPPORTED", "This event is not a supported unencrypted room attachment."
    )


def _unconfirmed(method: str) -> RoomFileError:
    return (
        RoomFileError("UNAVAILABLE", "Matrix media service is unavailable.", retryable=True)
        if method == "GET"
        else RoomFileError(
            "SEND_UNKNOWN", "File delivery could not be confirmed; do not resend blindly."
        )
    )
