"""Local protocol -> Worker -> real HTTP -> mock Task Service smoke tests.

No cloud credentials or model are required. The deterministic handler tests SDK
wiring, not LLM planning, Matrix delivery, or the production Task Service policy.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any

import httpx
import pytest

from agentcore import AsyncAgentCore
from agentcore.collaboration import current_collaboration_context
from agentcore.server import AgentCoreServer


@pytest.fixture
def task_service() -> Iterator[tuple[str, dict[str, Any]]]:
    state: dict[str, Any] = {"status": "assigned", "requests": [], "results": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def respond(self, status: int, payload: Any) -> None:
            content = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            self.dispatch()

        def do_POST(self) -> None:
            self.dispatch()

        def dispatch(self) -> None:
            content = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            body = json.loads(content) if content else None
            state["requests"].append((self.command, self.path, dict(self.headers), body))
            if self.headers.get("Authorization") != "Bearer local-worker-token":
                self.respond(401, {"error": "M_UNKNOWN_TOKEN"})
                return
            task = "/agentteams-app/v1/tasks/task-1"
            subtask = "/agentteams-app/v1/sub-tasks/subtask-1"
            if self.command == "GET" and self.path == task:
                self.respond(200, {"taskId": "task-1", "roomId": "!alpha:matrix.example.com"})
            elif self.command == "GET" and self.path == subtask:
                self.respond(
                    200, {"subTaskId": "subtask-1", "taskId": "task-1", "status": state["status"]}
                )
            elif self.command == "POST" and self.path == subtask + "/ack":
                if state["status"] != "assigned":
                    self.respond(409, {"message": "already acknowledged"})
                else:
                    state["status"] = "running"
                    self.respond(200, {"status": "running"})
            elif self.command == "POST" and self.path in (
                subtask + "/progress",
                subtask + "/heartbeat",
                subtask + "/results",
            ):
                if state["status"] != "running":
                    self.respond(409, {"message": "subtask is not running"})
                else:
                    if self.path.endswith("/results"):
                        state["results"].append(body)
                        state["status"] = "submitted"
                    self.respond(200, {"status": state["status"]})
            else:
                self.respond(404, {"message": "unknown mock route"})

    service = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=service.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{service.server_port}/agentteams-app", state
    finally:
        service.shutdown()
        service.server_close()
        thread.join()


@pytest.mark.asyncio
@pytest.mark.parametrize("protocol", ["agui", "openai", "openai-stream"])
@pytest.mark.parametrize("scenario", ["success", "no-context", "wrong-room", "conflict"])
async def test_collaboration_protocol_to_task_service(
    protocol: str,
    scenario: str,
    task_service: tuple[str, dict[str, Any]],
    agent_config_path: Path,
    teams_config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoint, state = task_service
    # Keep a developer's exported bootstrap credentials out of this local test.
    monkeypatch.delenv("AGENTCORE_DEBUG_TOKEN", raising=False)
    monkeypatch.delenv("AGENTCORE_DEBUG_TOKEN_FILE", raising=False)
    env_path = tmp_path / "env"
    env_path.write_text(
        f"export AGENTCORE_TASK_SERVICE_ENDPOINT={endpoint}\n"
        "export AGENTTEAMS_WORKER_MATRIX_TOKEN=local-worker-token\n"
    )
    if scenario == "conflict":
        state["status"] = "submitted"
    async with AsyncAgentCore(
        agent_config_path,
        env_path=env_path,
        teams_path=teams_config_path,
        collaboration_workspace_dir=tmp_path,
    ) as core:
        worker = core.collaboration.worker()
        tools = {tool.name: tool for tool in worker.tools()}
        server = AgentCoreServer()

        @server.invoke
        async def invoke(_request: Any, _context: Any) -> str:
            results = []
            actions: list[tuple[str, dict[str, Any]]] = [
                ("ack_subtask", {}),
                ("report_subtask_progress", {"content": {"percent": 50}}),
                ("heartbeat_subtask", {}),
                ("submit_subtask_result", {"summary": "local test done", "file_refs": []}),
            ]
            for name, arguments in actions:
                result = await tools[f"agentteams_{name}"].ainvoke(
                    {"subtask_id": "subtask-1", **arguments}
                )
                results.append(result)
                if not result["ok"]:
                    break
            return json.dumps(results)

        headers = {}
        if scenario != "no-context":
            headers["X-AgentCore-Collaboration-Context"] = (
                base64.urlsafe_b64encode(
                    json.dumps(
                        {
                            "version": 1,
                            "teamId": "team-alpha",
                            "eventId": "$local-event",
                            "roomKind": "task",
                            "roomId": "!other:matrix.example.com"
                            if scenario == "wrong-room"
                            else "!alpha:matrix.example.com",
                        }
                    ).encode()
                )
                .decode()
                .rstrip("=")
            )
        payload: dict[str, Any]
        if protocol == "agui":
            path = "/ag-ui/agent"
            payload = {
                "threadId": "local-session",
                "runId": "local-run",
                "messages": [],
                "state": {},
                "tools": [],
                "context": [],
                "forwardedProps": {},
            }
        else:
            path = "/openai/v1/chat/completions"
            payload = {
                "model": "local-worker",
                "messages": [],
                "stream": protocol == "openai-stream",
            }
        async with server.app.router.lifespan_context(server.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server),
                base_url="http://local-agent",
            ) as client:
                response = await client.post(path, json=payload, headers=headers)
        assert response.status_code == 200
        if protocol == "openai":
            output = response.json()["choices"][0]["message"]["content"]
        else:
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ") and line != "data: [DONE]"
            ]
            if protocol == "agui":
                assert events[-1]["type"] == "RUN_FINISHED"
                output = "".join(e["delta"] for e in events if e["type"] == "TEXT_MESSAGE_CONTENT")
            else:
                assert "data: [DONE]" in response.text
                output = "".join(e["choices"][0]["delta"].get("content", "") for e in events)
        results = json.loads(output)

    assert current_collaboration_context(required=False) is None
    writes = [r for r in state["requests"] if r[0] == "POST"]
    if scenario == "success":
        assert len(results) == 4 and all(result["ok"] for result in results)
        assert state["status"] == "submitted"
        assert [r[1].rsplit("/", 1)[1] for r in writes] == [
            "ack",
            "progress",
            "heartbeat",
            "results",
        ]
        assert state["results"] == [
            {"summary": "local test done", "fileRefs": [], "relatedRoomMessageId": "$local-event"}
        ]
        for _, _, headers, _ in state["requests"]:
            assert headers["Authorization"] == "Bearer local-worker-token"
        for request in (writes[0], writes[-1]):
            assert request[2]["Idempotency-Key"]
            assert request[3]["relatedRoomMessageId"] == "$local-event"
    else:
        expected = {
            "no-context": "COLLABORATION_CONTEXT_REQUIRED",
            "wrong-room": "COLLABORATION_TASK_UNAUTHORIZED",
            "conflict": "COLLABORATION_TASK_CONFLICT",
        }
        assert results[0]["code"] == expected[scenario]
        assert not results[0]["ok"] and not state["results"]
        assert len(writes) == (1 if scenario == "conflict" else 0)
