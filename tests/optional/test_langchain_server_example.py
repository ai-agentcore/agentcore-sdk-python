"""Exercise the public example with real LangChain and mocked cloud resources."""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("langchain")
pytest.importorskip("langchain_openai")
pytest.importorskip("ag_ui")
pytest.importorskip("fastapi")

from langchain_openai import ChatOpenAI

from agentcore import AsyncAgentCore
from agentcore.integrations import langchain
from agentcore.integrations.common import Tool
from agentcore.skill import AsyncSkills

ROOT = Path(__file__).parents[2]


@pytest.mark.asyncio
@pytest.mark.skipif(
    sys.version_info < (3, 11),
    reason="The documented create_agent event-stream example requires Python 3.11+",
)
@pytest.mark.parametrize("protocol", ["agui", "openai"])
async def test_example_preserves_tool_flow_and_protocol_boundaries(monkeypatch, protocol):
    calls = []
    model_inputs = []

    def respond(request):
        body = json.loads(request.content)
        model_inputs.append(body)
        tool_calls = (
            [
                {
                    "index": 0,
                    "id": "call-time",
                    "type": "function",
                    "function": {"name": "clock", "arguments": "{}"},
                },
                {
                    "index": 1,
                    "id": "call-skill",
                    "type": "function",
                    "function": {"name": "load_skills", "arguments": '{"name":"greeting"}'},
                },
            ]
            if len(model_inputs) == 1
            else []
        )
        delta = {"role": "assistant", "content": "先查询工具。" if tool_calls else "最终答案。"}
        if tool_calls:
            delta["tool_calls"] = tool_calls
        chunks = [
            {
                "id": f"answer-{len(model_inputs)}",
                "model": "example",
                "created": 1,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            },
            {
                "id": f"answer-{len(model_inputs)}",
                "model": "example",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if tool_calls else "stop",
                    }
                ],
            },
        ]
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n",
        )

    transport = httpx.MockTransport(respond)
    sync_http = httpx.Client(transport=transport)
    async_http = httpx.AsyncClient(transport=transport)
    chat = ChatOpenAI(
        model="example",
        api_key="not-a-real-key",
        streaming=True,
        http_client=sync_http,
        http_async_client=async_http,
    )

    async def clock(arguments):
        calls.append(arguments)
        return "Asia/Shanghai"

    skill = (await AsyncSkills(None, None).local(ROOT / "examples/skills"))[0]
    core = SimpleNamespace(
        mcp=AsyncMock(
            return_value=SimpleNamespace(
                list_tools=AsyncMock(
                    return_value=[
                        Tool("clock", "Get timezone", {"type": "object", "properties": {}}, clock),
                    ]
                )
            )
        ),
        skills=SimpleNamespace(managed=AsyncMock(return_value=skill)),
        aclose=AsyncMock(),
    )
    monkeypatch.setattr(AsyncAgentCore, "auto", lambda: core)
    monkeypatch.setattr(langchain, "model", lambda *args, **kwargs: chat)
    example = runpy.run_path(str(ROOT / "examples/langchain_server.py"))
    server = example["server"]
    try:
        async with server.app.router.lifespan_context(server.app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=server), base_url="http://example"
            ) as client:
                assert (await client.get("/readyz")).status_code == 200
                if protocol == "agui":
                    path = "/ag-ui/agent"
                    body = {
                        "threadId": "example-thread-1",
                        "runId": "example-run-1",
                        "messages": [{"id": "message-1", "role": "user", "content": "用工具回答"}],
                        "state": {},
                        "tools": [],
                        "context": [],
                        "forwardedProps": {},
                    }
                else:
                    path = "/openai/v1/chat/completions"
                    body = {
                        "model": "app",
                        "messages": [{"role": "user", "content": "用工具回答"}],
                        "stream": True,
                    }
                response = await client.post(path, json=body)
        assert response.status_code == 200
        frames = [line[6:] for line in response.text.splitlines() if line.startswith("data: ")]
        events = [json.loads(frame) for frame in frames if frame != "[DONE]"]
        assert calls == [{}]
        assert len(model_inputs) == 2
        assert {
            message["tool_call_id"]
            for message in model_inputs[1]["messages"]
            if message["role"] == "tool"
        } == {"call-time", "call-skill"}
        if protocol == "agui":
            assert events[0]["type"] == "RUN_STARTED"
            assert events[-1]["type"] == "RUN_FINISHED"
            starts = [e for e in events if e["type"] == "TEXT_MESSAGE_START"]
            ends = [e for e in events if e["type"] == "TEXT_MESSAGE_END"]
            assert len(starts) == len(ends) == 2, events
            assert starts[0]["messageId"] != starts[1]["messageId"]
            for call_id in ("call-time", "call-skill"):
                start = next(
                    i
                    for i, e in enumerate(events)
                    if e["type"] == "TOOL_CALL_START" and e["toolCallId"] == call_id
                )
                result = next(
                    i
                    for i, e in enumerate(events)
                    if e["type"] == "TOOL_CALL_RESULT" and e["toolCallId"] == call_id
                )
                assert events.index(ends[0]) < start < result < events.index(starts[1])
        else:
            assert frames[-1] == "[DONE]"
            deltas = [e["choices"][0]["delta"] for e in events]
            assert "".join(d.get("content", "") for d in deltas) == "先查询工具。最终答案。"
            assert {
                call["id"] for d in deltas for call in d.get("tool_calls", []) if call.get("id")
            } == {"call-time", "call-skill"}
            assert all("result" not in d and "tool_results" not in d for d in deltas)
        core.aclose.assert_awaited_once()
        assert sync_http.is_closed and async_http.is_closed
    finally:
        await async_http.aclose()
        sync_http.close()  # noqa: ASYNC212 -- in-memory MockTransport, no network I/O


@pytest.mark.asyncio
async def test_example_closes_core_when_resource_resolution_fails(monkeypatch):
    core = SimpleNamespace(
        mcp=AsyncMock(side_effect=RuntimeError("resolve failed")), aclose=AsyncMock()
    )
    monkeypatch.setattr(AsyncAgentCore, "auto", lambda: core)
    example = runpy.run_path(str(ROOT / "examples/langchain_server.py"))
    with pytest.raises(RuntimeError, match="resolve failed"):
        await example["startup"]()
    core.aclose.assert_awaited_once()
