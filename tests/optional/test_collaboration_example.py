from __future__ import annotations

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("langchain")
pytest.importorskip("fastapi")
pytest.importorskip("ag_ui")

from langchain_core.messages import AIMessageChunk

from agentcore import AsyncAgentCore, RequestContext
from agentcore.server import AgentRequest


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("hello", ["hello"]),
        ([{"type": "text", "text": "hello", "index": 0}], ["hello"]),
        ([{"type": "thinking", "thinking": "internal"}], []),
    ],
)
async def test_stream_extracts_text_from_native_chunks(monkeypatch, content, expected):
    monkeypatch.setattr(AsyncAgentCore, "auto", lambda: SimpleNamespace(aclose=AsyncMock()))
    example = runpy.run_path(str(Path(__file__).parents[2] / "examples" / "collaboration_agent.py"))

    class Agent:
        async def astream_events(self, *args, **kwargs):
            yield {
                "event": "on_chat_model_stream",
                "data": {"chunk": AIMessageChunk(content=content)},
            }

    invoke = example["invoke"]
    monkeypatch.setitem(invoke.__globals__, "agent", Agent())
    request = AgentRequest("ag-ui", [], True, None, None, {})
    events = [event async for event in invoke(request, RequestContext(headers={}))]
    assert [event.data["delta"] for event in events] == expected
