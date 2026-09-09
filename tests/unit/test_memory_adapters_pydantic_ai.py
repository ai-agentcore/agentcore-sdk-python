from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("pydantic_ai.capabilities")

from pydantic_ai import Agent, DeferredToolRequests
from pydantic_ai.messages import ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from agentcore.integrations.memory import MemoryScopes
from agentcore.integrations.memory.pydantic_ai import AgentCoreMemoryCapability
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
                MemorySearchHit(Memory("id", MemoryContent(f"fact-{scope.agent_id}"), scope), 1, 1)
            ]
        )

    store.search_memories.side_effect = search
    store.add_memories.return_value = AddMemoriesResult()
    return store


def scopes(ctx):
    return MemoryScopes(
        MemoryScope(agent_id=ctx.deps), MemoryScope(agent_id=ctx.deps, session_id="s")
    )


@pytest.mark.asyncio
async def test_pydantic_real_tool_loop_and_new_turn_only():
    store = make_store()
    seen = []

    def model(messages, info):
        seen.append(info.instructions)
        if len(seen) % 2:
            return ModelResponse(parts=[ToolCallPart("lookup", {}, "call-1")])
        return ModelResponse(
            parts=[ThinkingPart("private reasoning"), TextPart("answer")], finish_reason="stop"
        )

    agent = Agent(
        FunctionModel(model),
        instructions="original",
        capabilities=[AgentCoreMemoryCapability(store, scope_resolver=scopes, write_back=True)],
    )

    @agent.tool_plain
    async def lookup() -> str:
        return "private tool output"

    first = await agent.run("question one", deps="alice")
    await agent.run("question two", deps="alice", message_history=first.all_messages())
    assert store.search_memories.await_count == 2
    assert all("original" in text and text.count("fact-alice") == 1 for text in seen)
    assert store.add_memories.await_count == 2
    assert store.add_memories.await_args.kwargs["messages"] == [
        MemoryMessage("user", "question two"),
        MemoryMessage("assistant", "answer"),
    ]
    # Instructions may be framework message metadata, but never become conversation parts.
    assert all("fact-alice" not in str(message.parts) for message in first.all_messages())


@pytest.mark.asyncio
async def test_pydantic_shared_capability_isolates_concurrent_runs():
    store = make_store()

    async def model(messages, info):
        await asyncio.sleep(0)
        return ModelResponse(parts=[TextPart(info.instructions)], finish_reason="stop")

    agent = Agent(
        FunctionModel(model), capabilities=[AgentCoreMemoryCapability(store, scope_resolver=scopes)]
    )
    a, b = await asyncio.gather(agent.run("q", deps="alice"), agent.run("q", deps="bob"))
    assert "fact-alice" in a.output and "fact-bob" not in a.output
    assert "fact-bob" in b.output and "fact-alice" not in b.output
    store.add_memories.assert_not_called()


@pytest.mark.asyncio
async def test_pydantic_cancel_and_model_error_never_write():
    store = make_store()
    entered = asyncio.Event()

    async def model(messages, info):
        entered.set()
        await asyncio.Future()

    capability = AgentCoreMemoryCapability(store, scope_resolver=scopes, write_back=True)
    agent = Agent(FunctionModel(model), capabilities=[capability])
    running = asyncio.create_task(agent.run("q", deps="alice"))
    await asyncio.wait_for(entered.wait(), 3)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running

    def broken(messages, info):
        raise ValueError("model failed")

    with pytest.raises(ValueError, match="model failed"):
        await Agent(FunctionModel(broken), capabilities=[capability]).run("q", deps="alice")
    store.add_memories.assert_not_called()


@pytest.mark.asyncio
async def test_pydantic_stream_writes_only_on_consumed_completion():
    store = make_store()

    async def stream(messages, info):
        assert "fact-alice" in info.instructions
        yield "answer"

    agent = Agent(
        FunctionModel(stream_function=stream),
        capabilities=[AgentCoreMemoryCapability(store, scope_resolver=scopes, write_back=True)],
    )
    async with agent.run_stream("q", deps="alice") as result:
        assert "".join([part async for part in result.stream_text(delta=True)]) == "answer"
    assert store.add_memories.await_count == 1


@pytest.mark.asyncio
async def test_pydantic_truncated_response_does_not_write():
    store = make_store()
    agent = Agent(
        FunctionModel(
            lambda messages, info: ModelResponse(
                parts=[TextPart("partial")], finish_reason="length"
            )
        ),
        capabilities=[AgentCoreMemoryCapability(store, scope_resolver=scopes, write_back=True)],
    )
    # Some PydanticAI versions raise on truncation; either way it must not become memory.
    try:
        await agent.run("q", deps="alice")
    except Exception as exc:
        assert type(exc).__name__ in {"UnexpectedModelBehavior", "IncompleteToolCall"}
    store.add_memories.assert_not_called()


@pytest.mark.asyncio
async def test_pydantic_deferred_approval_does_not_write():
    store = make_store()
    agent = Agent(
        FunctionModel(
            lambda messages, info: ModelResponse(parts=[ToolCallPart("approve", {}, "call-1")])
        ),
        output_type=[str, DeferredToolRequests],
        capabilities=[AgentCoreMemoryCapability(store, scope_resolver=scopes, write_back=True)],
    )

    @agent.tool_plain(requires_approval=True)
    def approve() -> str:
        return "approved"

    result = await agent.run("q", deps="alice")
    assert isinstance(result.output, DeferredToolRequests)
    store.add_memories.assert_not_called()


@pytest.mark.asyncio
async def test_pydantic_cancelled_stream_does_not_write():
    store = make_store()
    entered = asyncio.Event()

    async def stream(messages, info):
        yield "partial"
        entered.set()
        await asyncio.Future()

    agent = Agent(
        FunctionModel(stream_function=stream),
        capabilities=[AgentCoreMemoryCapability(store, scope_resolver=scopes, write_back=True)],
    )

    async def consume():
        async with agent.run_stream("q", deps="alice") as result:
            async for _ in result.stream_text(delta=True):
                pass

    running = asyncio.create_task(consume())
    await asyncio.wait_for(entered.wait(), 3)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    store.add_memories.assert_not_called()
