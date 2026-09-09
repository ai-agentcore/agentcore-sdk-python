"""Exercise the real OpenAPI HTTP transport against a local contract server."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

from agentcore import AsyncAgentCore
from agentcore.auth import AccessKeyCredential
from agentcore.memory import MemoryScope


@pytest.mark.asyncio
async def test_common_requests_are_signed_and_use_memory_wire_contract() -> None:
    calls: list[tuple[str, str, dict[str, str], bytes]] = []

    class Handler(BaseHTTPRequestHandler):
        def handle_request(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            headers = {key.lower(): value for key, value in self.headers.items()}
            calls.append((self.command, self.path, headers, raw))
            action = headers["x-acs-action"]
            body: dict[str, Any] = {"success": True, "requestId": "local-request"}
            if action == "AddMemories":
                body["data"] = {"memories": [{"memoryId": "memory-1"}]}
            elif action == "SearchMemories":
                body["data"] = {"memories": []}
            elif action in {"GetMemory", "UpdateMemory"}:
                body["data"] = {
                    "memoryId": "memory-1",
                    "content": {"text": "test"},
                    "scope": {"userId": "user-1"},
                }
            else:
                body["items"] = []
            payload = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_PUT = do_DELETE = handle_request

        def log_message(self, format: str, *args: Any) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    core = AsyncAgentCore.auto(
        workspace_id="ws /中文",
        region_id="cn-hangzhou",
        control_plane_endpoint=f"http://127.0.0.1:{server.server_port}",
        access_key_credential=AccessKeyCredential(
            access_key_id="test-ak",
            access_key_secret="test-sk",
            security_token="test-sts",
        ),
    )
    try:
        store = core.memory_store("store /中文")
        result = await store.add_memories(text="记住", scope=MemoryScope(user_id="user-1"))
        assert result.memory_ids == ("memory-1",)
        await store.search_memories("query", enable_rerank=False, min_score=0.0)
        await store.list_memories(user_id="user-1", max_results=5, next_token="a +/=")
        await store.get_memory("id /中文")
        await store.update_memory("memory-1", metadata={})
        await store.delete_memory("memory-1")
        await store.list_memory_sessions(user_id="user-1")
        await store.list_memory_session_messages("session-1", user_id="user-1")
    finally:
        await core.aclose()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert len(calls) == 8
    for _, path, headers, _ in calls:
        assert path.startswith(
            "/workspaces/ws%20%2F%E4%B8%AD%E6%96%87/memorystores/store%20%2F%E4%B8%AD%E6%96%87/"
        )
        assert headers["authorization"]
        assert headers["x-acs-security-token"] == "test-sts"
        assert headers["x-acs-version"] == "2026-08-04"
    assert calls[0][2]["content-type"].startswith("application/x-www-form-urlencoded")
    assert json.loads(parse_qs(calls[0][3].decode())["body"][0]) == {
        "text": "记住",
        "scope": {"userId": "user-1"},
    }
    assert json.loads(parse_qs(calls[1][3].decode())["body"][0]) == {
        "query": "query",
        "enableRerank": False,
        "minScore": 0.0,
    }
    assert parse_qs(urlsplit(calls[2][1]).query) == {
        "userId": ["user-1"],
        "maxResults": ["5"],
        "nextToken": ["a +/="],
    }
    assert calls[3][1].endswith("/memories/id%20%2F%E4%B8%AD%E6%96%87")
    assert json.loads(parse_qs(calls[4][3].decode())["body"][0]) == {"metadata": {}}
    assert calls[5][3] == b""
