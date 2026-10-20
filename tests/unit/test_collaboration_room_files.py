from __future__ import annotations

import asyncio
import base64
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import anyio
import httpx
import pytest
from agentcore_collaboration.matrix_files import MatrixFileClient, RoomFileError
from agentcore_collaboration.teams import parse_teams_config
from agentcore_collaboration.worker import WorkerCollaboration

from agentcore.collaboration.context import CollaborationTurnContext


def snapshot():
    return parse_teams_config(b"""
apiVersion: agentteams.io/v1alpha1
kind: TeamsConfig
metadata: {runtimeName: worker}
spec:
  self: {name: worker, runtimeName: worker, matrixUserId: "@worker:ws"}
  matrix: {tokenEnv: MATRIX_TOKEN}
  teams: []
""")


def client(handler):
    return MatrixFileClient(
        lambda _: "secret",
        endpoint_provider=lambda: "https://matrix.example",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


@pytest.mark.asyncio
async def test_download_uses_room_event_and_authenticated_media(tmp_path: Path):
    seen = []

    def handler(request):
        seen.append(request)
        if request.url.path.endswith("/config"):
            return httpx.Response(200, json={"m.upload.size": 100})
        if "/event/" in request.url.path:
            return httpx.Response(
                200,
                json={
                    "type": "m.room.message",
                    "room_id": "!room",
                    "sender": "@human:ws",
                    "content": {
                        "msgtype": "m.file",
                        "body": "../data.txt",
                        "url": "mxc://ws/media",
                    },
                },
            )
        return httpx.Response(200, content=b"hello")

    api = client(handler)
    try:
        attachment = await api.attachment(snapshot(), "!room", "$file")
        assert attachment.filename == "data.txt"
        size = await api.download_to(snapshot(), attachment, tmp_path / "file")
        assert size == 5
        assert (tmp_path / "file").read_bytes() == b"hello"
        assert all(r.headers["Authorization"] == "Bearer secret" for r in seen)
        assert seen[-1].url.path == "/_matrix/client/v1/media/download/ws/media"
    finally:
        await api.aclose()


@pytest.mark.asyncio
async def test_send_uploads_raw_bytes_then_replies_to_request(tmp_path: Path):
    seen = []
    source = tmp_path / "image.png"
    source.write_bytes(b"image")

    async def handler(request):
        seen.append((request, await request.aread()))
        if request.url.path.endswith("/config"):
            return httpx.Response(200, json={"m.upload.size": 100})
        if request.method == "POST":
            return httpx.Response(200, json={"content_uri": "mxc://ws/media"})
        return httpx.Response(200, json={"event_id": "$sent"})

    api = client(handler)
    try:
        context = CollaborationTurnContext(
            session_id=None, team_id=None, room_id="!room", event_id="$request", room_kind="group"
        )
        result = await api.send(snapshot(), context, source, "result")
        assert result == {"event_id": "$sent", "filename": "image.png", "status": "sent"}
        assert seen[1][1] == b"image"
        import json

        content = json.loads(seen[2][1])
        assert content["msgtype"] == "m.image"
        assert content["m.relates_to"] == {"m.in_reply_to": {"event_id": "$request"}}
        assert "m.mentions" not in content
        assert "mxc://" not in str(result)
    finally:
        await api.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,code", [(403, "UNAUTHORIZED"), (404, "NOT_FOUND"), (503, "UNAVAILABLE")]
)
async def test_safe_status_errors(status, code):
    api = client(lambda _: httpx.Response(status, text="secret echoed upstream body"))
    try:
        with pytest.raises(RoomFileError) as error:
            await api.attachment(snapshot(), "!room", "$file")
        assert error.value.code == "COLLABORATION_ROOM_FILE_" + code
        assert "secret" not in str(error.value)
    finally:
        await api.aclose()


def headers(room="!room", event="$request"):
    value = {"version": 1, "roomId": room, "eventId": event, "roomKind": "group"}
    return {
        "X-AgentCore-Collaboration-Context": base64.urlsafe_b64encode(json.dumps(value).encode())
        .decode()
        .rstrip("=")
    }


def attachment_event(content=None, room="!room", sender="@human:ws"):
    return {
        "type": "m.room.message",
        "room_id": room,
        "sender": sender,
        "content": content or {"msgtype": "m.file", "body": "file.txt", "url": "mxc://ws/media"},
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["oversize", "broken", "cancel"])
async def test_failed_download_cleans_partial_file_and_needs_no_team(tmp_path, mode):
    started = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"hello"
            started.set()
            if mode == "oversize":
                yield b"too large"
            elif mode == "broken":
                raise httpx.ReadError("private error")
            else:
                await asyncio.Event().wait()

    def handler(request):
        if "/event/" in request.url.path:
            return httpx.Response(200, json=attachment_event())
        if request.url.path.endswith("/config"):
            return httpx.Response(200, json={"m.upload.size": 5})
        return httpx.Response(200, stream=Stream())

    api = client(handler)
    worker = WorkerCollaboration(snapshot, object(), tmp_path, api)
    tool = next(t for t in worker.tools() if t.name == "agentteams_download_room_file")
    try:
        assert (await tool.ainvoke({"event_id": "$file"}))[
            "code"
        ] == "COLLABORATION_CONTEXT_REQUIRED"
        with worker.request_context(headers()):
            task = asyncio.create_task(tool.ainvoke({"event_id": "$file"}))
            if mode == "cancel":
                await started.wait()
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                result = await task
                assert result["code"] == (
                    "COLLABORATION_FILE_TOO_LARGE"
                    if mode == "oversize"
                    else "COLLABORATION_ROOM_FILE_UNAVAILABLE"
                )
        assert not [p for p in tmp_path.rglob("*") if p.is_file()]
    finally:
        await api.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["encrypted", "thread", "cross_room", "edit_other_sender"])
async def test_rejects_unsupported_event_without_downloading(kind):
    seen = []
    content = {"msgtype": "m.file", "body": "file", "url": "mxc://ws/media"}
    if kind == "encrypted":
        content["file"] = {"url": "mxc://ws/encrypted"}
    if kind == "thread":
        content["m.relates_to"] = {"rel_type": "m.thread", "event_id": "$root"}
    if kind == "edit_other_sender":
        content = {
            "m.relates_to": {"rel_type": "m.replace", "event_id": "$original"},
            "m.new_content": content,
        }

    def handler(request):
        seen.append(request.url.path)
        if request.url.path.endswith("/$original"):
            return httpx.Response(200, json=attachment_event(sender="@other:ws"))
        return httpx.Response(
            200, json=attachment_event(content, room="!other" if kind == "cross_room" else "!room")
        )

    api = client(handler)
    try:
        with pytest.raises(RoomFileError, match="supported"):
            await api.attachment(snapshot(), "!room", "$file")
        assert all("/event/" in path for path in seen)
    finally:
        await api.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("retry_status", [200, 401])
async def test_uncertain_send_reuses_token_transaction_and_never_reuploads(tmp_path, retry_status):
    source = tmp_path / "file.txt"
    source.write_text("result")
    sent = []
    uploads = []

    async def handler(request):
        if request.method == "GET":
            return httpx.Response(200, json={"m.upload.size": 100})
        if request.method == "POST":
            uploads.append(await request.aread())
            return httpx.Response(200, json={"content_uri": "mxc://ws/file"})
        sent.append((str(request.url), request.headers["Authorization"], await request.aread()))
        if len(sent) == 1:
            raise httpx.ReadError("lost response")
        return httpx.Response(retry_status, json={"event_id": "$sent"})

    api = client(handler)
    try:
        context = CollaborationTurnContext(None, None, "!room", "$request", "group")
        if retry_status == 200:
            assert (await api.send(snapshot(), context, source, None))["status"] == "sent"
        else:
            with pytest.raises(RoomFileError) as error:
                await api.send(snapshot(), context, source, None)
            assert error.value.code == "COLLABORATION_ROOM_FILE_SEND_UNKNOWN"
            assert not error.value.retryable
        assert len(uploads) == 1 and len(sent) == 2 and sent[0] == sent[1]
    finally:
        await api.aclose()


@pytest.mark.asyncio
async def test_real_http_download_via_native_tool_and_request_context(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            assert self.headers["Authorization"] == "Bearer secret"
            value = (
                json.dumps(attachment_event()).encode()
                if "/event/" in self.path
                else b'{"m.upload.size":100}'
                if self.path.endswith("/config")
                else b"hello"
            )
            self.send_response(200)
            self.send_header("Content-Length", str(len(value)))
            self.end_headers()
            self.wfile.write(value)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    api = MatrixFileClient(
        lambda _: "secret", endpoint_provider=lambda: f"http://127.0.0.1:{server.server_port}"
    )
    worker = WorkerCollaboration(snapshot, object(), tmp_path, api)
    try:
        tool = next(t for t in worker.tools() if t.name == "agentteams_download_room_file")
        with worker.request_context(headers()):
            result = await tool.ainvoke({"event_id": "$file"})
        assert result["ok"]
        assert await anyio.Path(result["data"]["local_path"]).read_text() == "hello"
        assert set(result["data"]) == {"event_id", "local_path", "filename", "content_type", "size"}
        assert "room-files" in [s.name for s in worker.skills()]
    finally:
        await api.aclose()
        await anyio.to_thread.run_sync(server.shutdown)
        server.server_close()
        thread.join()


@pytest.mark.asyncio
async def test_shared_tools_keep_concurrent_room_context_and_refresh_rejected_token(tmp_path):
    source = tmp_path / "result.txt"
    source.write_text("result")
    token = "old"
    sends = []

    async def handler(request):
        nonlocal token
        if request.headers["Authorization"] == "Bearer old":
            token = "new"
            return httpx.Response(401)
        if request.method == "GET":
            return httpx.Response(200, json={"m.upload.size": 100})
        if request.method == "POST":
            return httpx.Response(200, json={"content_uri": "mxc://ws/media"})
        await asyncio.sleep(0)
        sends.append((request.url.path, json.loads(await request.aread())))
        return httpx.Response(200, json={"event_id": "$sent"})

    api = MatrixFileClient(
        lambda _: token,
        endpoint_provider=lambda: "https://matrix.example",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    worker = WorkerCollaboration(snapshot, object(), tmp_path, api)
    tool = next(t for t in worker.tools() if t.name == "agentteams_send_room_file")

    async def call(identity):
        with worker.request_context(headers("!room-" + identity, "$request-" + identity)):
            assert (await tool.ainvoke({"local_path": "result.txt"}))["ok"]

    try:
        await asyncio.gather(call("one"), call("two"))
        assert len(sends) == 2
        for identity in ("one", "two"):
            path, content = next(item for item in sends if "!room-" + identity in item[0])
            assert content["m.relates_to"]["m.in_reply_to"]["event_id"] == "$request-" + identity
    finally:
        await api.aclose()
