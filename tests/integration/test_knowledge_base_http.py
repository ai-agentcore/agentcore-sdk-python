from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agentcore import AgentCore, AsyncAgentCore
from agentcore.auth import AccessKeyCredential
from agentcore.errors import InvocationError


@pytest.fixture
def service():
    state = {
        "calls": [],
        "status": 200,
        "response": {
            "success": True,
            "data": {
                "chunks": [
                    {
                        "content": "产品说明",
                        "assetId": "a-1",
                        "assetName": "Guide",
                        "fileName": "guide.pdf",
                        "chunkSeq": 2,
                        "chunkToken": "ct-1",
                        "score": 0.032,
                        "scores": {"fusion": 0.032},
                    }
                ],
                "timeSpent": 12,
                "scoreType": "rrf",
            },
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            state["calls"].append(
                (
                    self.path,
                    self.headers,
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            payload = json.dumps(state["response"]).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    state["options"] = dict(
        workspace_id="ws-test",
        region_id="cn-hangzhou",
        control_plane_endpoint=f"http://127.0.0.1:{server.server_port}",
        access_key_credential=AccessKeyCredential("test-ak", "test-sk", "test-sts"),
    )
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


async def test_async_search_signed_json_and_result(service):
    async with AsyncAgentCore.auto(**service["options"]) as core:
        kb = core.knowledge_base("kb /中文")
        assert service["calls"] == []
        result = await kb.search(
            "产品",
            top_k=5,
            rerank=False,
            vector_weight=0,
            metadata_filter='{"Key":"dept","Operator":"Equals","Value":"support"}',
        )
        assert result.chunks[0].content == "产品说明"
        assert result.chunks[0].file_name == "guide.pdf"
        assert result.chunks[0].chunk_token == "ct-1"
        assert result.chunks[0].scores == {"fusion": 0.032}
        assert result.time_spent == 12 and result.score_type == "rrf"
        await kb.search("第二次")
    path, headers, body = service["calls"][0]
    assert path == "/workspaces/ws-test/knowledge-bases/kb%20%2F%E4%B8%AD%E6%96%87/search"
    assert headers["x-acs-action"] == "SearchKnowledgeBase"
    assert "Signature=" in headers["Authorization"]
    assert headers["x-acs-security-token"] == "test-sts"
    assert body["topK"] == 5 and body["rerank"] is False and body["vectorWeight"] == 0
    assert isinstance(body["metadataFilter"], str)
    assert service["calls"][1][2] == {"query": "第二次"}


def test_sync_search_empty_results_and_closed_core(service):
    service["response"]["data"]["chunks"] = []
    with AgentCore.auto(**service["options"]) as core:
        kb = core.knowledge_base("kb-1")
        assert kb.search("empty").chunks == ()
    with pytest.raises(RuntimeError):
        kb.search("closed")


@pytest.mark.parametrize("status", [200, 403])
async def test_errors_are_not_empty_results(service, status, caplog):
    service["status"] = status
    service["response"] = {
        "success": False,
        "code": "Forbidden",
        "Code": "Forbidden",
        "message": "access denied",
        "RequestId": "req-kb",
        "requestId": "req-kb",
    }
    async with AsyncAgentCore.auto(**service["options"]) as core:
        with pytest.raises(InvocationError):
            await core.knowledge_base("kb-1").search("private query")
    assert "req-kb" in caplog.text
    assert "private query" not in caplog.text


async def test_malformed_result_is_not_empty(service):
    service["response"] = {"success": True, "data": {}}
    async with AsyncAgentCore.auto(**service["options"]) as core:
        with pytest.raises(InvocationError):
            await core.knowledge_base("kb-1").search("query")
