from __future__ import annotations

import asyncio
import json
import runpy
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pytest

pytest.importorskip("agentscope")

from agentscope.agent import Agent
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from pydantic import SecretStr

from agentcore.integrations.memory import MemoryScopes
from agentcore.integrations.memory.agentscope import AgentCoreMemoryMiddleware
from agentcore.memory import (
    AddMemoriesResult,
    Memory,
    MemoryContent,
    MemoryMessage,
    MemoryScope,
    MemorySearchHit,
    SearchMemoriesResult,
)


def make_store():
    store = AsyncMock()
    store.memory_store_name = "memory"

    async def search(query, *, scope, top_k):
        return SearchMemoriesResult(
            memories=[
                MemorySearchHit(
                    Memory("id", MemoryContent(f"memory-for-{scope.agent_id}"), scope),
                    1,
                    1,
                )
            ]
        )

    store.search_memories.side_effect = search
    store.add_memories.return_value = AddMemoriesResult()
    return store


def make_model(handler):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    model = OpenAIChatModel(
        credential=OpenAICredential(api_key=SecretStr("test"), base_url="https://model.invalid/v1"),
        model="gpt-4o",
        stream=False,
        max_retries=0,
        client_kwargs={"http_client": http},
    )
    return model, http


def response(finish="stop"):
    return httpx.Response(
        200,
        json={
            "id": "c1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": finish,
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def scopes(agent):
    return MemoryScopes(
        read=MemoryScope(agent_id=agent.name),
        write=MemoryScope(agent_id=agent.name, session_id=agent.state.session_id),
    )


@pytest.mark.asyncio
async def test_real_agentscope_reply_and_stream_do_not_pollute_history():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        return response()

    store = make_store()
    model, http = make_model(handler)
    agent = Agent(
        "alice",
        "original instructions",
        model,
        middlewares=[
            AgentCoreMemoryMiddleware(store, scope_resolver=scopes, write_back=True),
        ],
    )
    try:
        reply = await agent.reply(UserMsg("user", "question one"))
        assert reply.get_text_content() == "answer"
        async for _ in agent.reply_stream(UserMsg("user", "question two")):
            pass
        assert store.search_memories.await_count == 2
        assert store.add_memories.await_count == 2
        assert store.add_memories.await_args.kwargs["messages"] == [
            MemoryMessage("user", "question two"),
            MemoryMessage("assistant", "answer"),
        ]
        assert "memory-for-alice" not in str(agent.state.context)
        assert agent._system_prompt == "original instructions"
        for call in seen:
            assert "original instructions" in str(call["messages"][0])
            assert str(call["messages"]).count("memory-for-alice") == 1
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_shared_agentscope_middleware_keeps_per_task_reference():
    seen = []

    async def handler(request):
        seen.append(json.loads(request.content))
        await asyncio.sleep(0)
        return response()

    store = make_store()
    middleware = AgentCoreMemoryMiddleware(store, scope_resolver=scopes)
    model, http = make_model(handler)
    agents = [Agent(user, "original", model, middlewares=[middleware]) for user in ["alice", "bob"]]
    try:
        await asyncio.gather(*[a.reply(UserMsg("user", a.name)) for a in agents])
        assert len(seen) == 2
        for call in seen:
            text = str(call["messages"])
            assert ("memory-for-alice" in text) != ("memory-for-bob" in text)
        store.add_memories.assert_not_called()
        assert middleware._reference.get() == ""
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_agentscope_cancellation_does_not_write():
    started = asyncio.Event()

    async def handler(request):
        started.set()
        await asyncio.Event().wait()

    store = make_store()
    model, http = make_model(handler)
    agent = Agent(
        "a",
        "original",
        model,
        middlewares=[
            AgentCoreMemoryMiddleware(store, scope_resolver=scopes, write_back=True),
        ],
    )
    task = asyncio.create_task(agent.reply(UserMsg("user", "question")))
    try:
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass  # AgentScope may return an interrupted Msg instead.
        store.add_memories.assert_not_called()
    finally:
        await http.aclose()


@pytest.mark.asyncio
async def test_documented_agentscope_example():
    module = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/memory_frameworks/agentscope_agent.py")
    )
    store = make_store()
    model, http = make_model(lambda request: response())
    try:
        result = await module["run"](store, model, partition="a", session_id="s")
        assert result.get_text_content() == "answer"
        assert store.add_memories.await_count == 1
        assert store.add_memories.await_args.kwargs["scope"] == MemoryScope(
            agent_id="a", session_id="s"
        )
    finally:
        await http.aclose()
