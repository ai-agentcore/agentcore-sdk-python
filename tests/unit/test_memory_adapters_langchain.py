from __future__ import annotations

import asyncio
import runpy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("langchain")

from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import Field

from agentcore.errors import MemoryAPIError, MemoryValidationError
from agentcore.integrations.memory import MemoryScopes
from agentcore.integrations.memory.langchain import AgentCoreMemoryMiddleware, message_text
from agentcore.memory import (
    AddMemoriesResult,
    Memory,
    MemoryContent,
    MemoryMessage,
    MemoryScope,
    MemorySearchHit,
    SearchMemoriesResult,
)


class TestModel(FakeMessagesListChatModel):
    __test__ = False
    seen: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def make_store():
    store = AsyncMock()
    store.memory_store_name = "test-memory"

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


def scopes(context):
    return MemoryScopes(
        read=MemoryScope(agent_id=context["user"]),
        write=MemoryScope(agent_id=context["user"], session_id=context["session"]),
    )


@pytest.mark.asyncio
async def test_user_only_scope_recall_and_write_back():
    store = make_store()
    scope = MemoryScope(user_id="alice")
    agent = create_agent(
        TestModel(responses=[AIMessage(content="answer")]),
        middleware=[AgentCoreMemoryMiddleware(
            store,
            scope_resolver=lambda context: MemoryScopes(read=scope, write=scope),
            write_back=True,
        )],
    )
    result = await agent.ainvoke({"messages": [HumanMessage("question")]})
    assert result["messages"][-1].content == "answer"
    store.search_memories.assert_awaited_once_with("question", scope=scope, top_k=5)
    store.add_memories.assert_awaited_once_with(
        scope=scope,
        messages=[MemoryMessage("user", "question"), MemoryMessage("assistant", "answer")],
    )


@pytest.mark.asyncio
async def test_real_agent_tool_loop_and_checkpoint_history():
    store = make_store()

    def lookup(query: str) -> str:
        """Look up a value."""
        return "private-tool-result"

    model = TestModel(
        responses=[
            AIMessage(
                content="", tool_calls=[{"name": "lookup", "args": {"query": "q"}, "id": "c1"}]
            ),
            AIMessage(content="answer one"),
            AIMessage(content="answer two"),
        ]
    )
    agent = create_agent(
        model=model,
        tools=[lookup],
        system_prompt="original instructions",
        middleware=[AgentCoreMemoryMiddleware(store, scope_resolver=scopes, write_back=True)],
        checkpointer=InMemorySaver(),
    )
    context = {"user": "alice", "session": "s1"}
    config = {"configurable": {"thread_id": "thread-1"}}
    first = await agent.ainvoke(
        {"messages": [HumanMessage("question one")]}, config, context=context
    )
    assert first["messages"][-1].content == "answer one"
    async for _ in agent.astream(
        {"messages": [HumanMessage("question two")]}, config, context=context
    ):
        pass
    assert store.search_memories.await_count == 2  # not once per model call
    assert store.add_memories.await_count == 2
    assert store.add_memories.await_args_list[0].kwargs["messages"] == [
        MemoryMessage("user", "question one"),
        MemoryMessage("assistant", "answer one"),
    ]
    assert store.add_memories.await_args_list[1].kwargs["messages"] == [
        MemoryMessage("user", "question two"),
        MemoryMessage("assistant", "answer two"),
    ]
    for call in model.seen:
        assert "original instructions" in call[0].content
        assert call[0].content.count("memory-for-alice") == 1
    snapshot = await agent.aget_state(config)
    assert all("memory-for-alice" not in str(m.content) for m in snapshot.values["messages"])


@pytest.mark.asyncio
async def test_shared_middleware_concurrent_scopes_and_no_write_by_default():
    store = make_store()
    model = TestModel(responses=[AIMessage(content="done")])
    middleware = AgentCoreMemoryMiddleware(store, scope_resolver=scopes)
    agent = create_agent(model, middleware=[middleware])
    await asyncio.gather(
        *[
            agent.ainvoke(
                {"messages": [HumanMessage(user)]}, context={"user": user, "session": user}
            )
            for user in ["alice", "bob"]
        ]
    )
    assert {c.kwargs["scope"].agent_id for c in store.search_memories.await_args_list} == {
        "alice",
        "bob",
    }
    for call in model.seen:
        user = call[-1].content
        assert f"memory-for-{user}" in call[0].content
        assert ("bob" if user == "alice" else "alice") not in call[0].content
    store.add_memories.assert_not_called()


@pytest.mark.asyncio
async def test_paused_tool_does_not_write():
    from langgraph.types import Command, interrupt

    def approve() -> str:
        """Ask for approval."""
        return interrupt("approve?")

    store = make_store()
    agent = create_agent(
        TestModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": "approve", "args": {}, "id": "c1"}]),
                AIMessage(content="approved"),
            ]
        ),
        tools=[approve],
        checkpointer=InMemorySaver(),
        middleware=[AgentCoreMemoryMiddleware(store, scope_resolver=scopes, write_back=True)],
    )
    context = {"user": "a", "session": "s"}
    config = {"configurable": {"thread_id": "paused"}}
    await agent.ainvoke({"messages": [HumanMessage("please approve")]}, config, context=context)
    store.add_memories.assert_not_called()
    await agent.ainvoke(Command(resume="yes"), config, context=context)
    assert store.add_memories.await_count == 1
    assert store.add_memories.await_args.kwargs["messages"][0].content == "please approve"


def test_only_visible_text_blocks_and_original_system_preserved():
    from agentcore.integrations.memory.langchain import memory_messages

    msg = AIMessage(
        content=[
            {"type": "text", "text": "answer"},
            {"type": "reasoning", "reasoning": "private"},
        ]
    )
    assert message_text(msg) == "answer"
    original = SystemMessage(content=[{"type": "text", "text": "original"}])
    combined = memory_messages(original, "reference")
    assert original.content == [{"type": "text", "text": "original"}]
    assert combined.content[0] == original.content[0]


@pytest.mark.asyncio
async def test_memory_service_failure_is_best_effort_but_bad_scope_is_not():
    store = make_store()
    store.search_memories.side_effect = MemoryAPIError(operation="search")
    store.add_memories.side_effect = MemoryAPIError(operation="add")
    model = TestModel(responses=[AIMessage(content="answer")])
    agent = create_agent(
        model,
        middleware=[AgentCoreMemoryMiddleware(store, scope_resolver=scopes, write_back=True)],
    )
    result = await agent.ainvoke(
        {"messages": [HumanMessage("question")]}, context={"user": "a", "session": "s"}
    )
    assert result["messages"][-1].content == "answer"
    assert store.add_memories.await_count == 1
    assert len(model.seen) == 1
    with pytest.raises(MemoryValidationError):
        await agent.ainvoke(
            {"messages": [HumanMessage("question")]}, context={"user": "a", "session": ""}
        )
    assert len(model.seen) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "final",
    [
        AIMessage(content="unfinished", response_metadata={"finish_reason": "length"}),
        AIMessage(content=""),
    ],
)
async def test_incomplete_or_empty_reply_does_not_write(final):
    store = make_store()
    agent = create_agent(
        TestModel(responses=[final]),
        middleware=[AgentCoreMemoryMiddleware(store, scope_resolver=scopes, write_back=True)],
    )
    await agent.ainvoke(
        {"messages": [HumanMessage("question")]}, context={"user": "a", "session": "s"}
    )
    store.add_memories.assert_not_called()


@pytest.mark.asyncio
async def test_cancelled_tool_does_not_write():
    started = asyncio.Event()

    async def blocked() -> str:
        """Wait for an external result."""
        started.set()
        await asyncio.Event().wait()
        return "unreachable"

    store = make_store()
    agent = create_agent(
        TestModel(
            responses=[
                AIMessage(content="", tool_calls=[{"name": "blocked", "args": {}, "id": "c1"}]),
            ]
        ),
        tools=[blocked],
        middleware=[AgentCoreMemoryMiddleware(store, scope_resolver=scopes, write_back=True)],
    )
    task = asyncio.create_task(
        agent.ainvoke(
            {"messages": [HumanMessage("question")]}, context={"user": "a", "session": "s"}
        )
    )
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    store.add_memories.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("example", ["langchain_agent", "langgraph_agent"])
async def test_documented_examples_use_real_frameworks(example):
    module = runpy.run_path(
        str(Path(__file__).parents[2] / "examples" / "memory_frameworks" / f"{example}.py")
    )
    store = make_store()
    responses = [AIMessage(content="answer")]
    if example == "langgraph_agent":
        responses.insert(
            0,
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "add",
                        "args": {"left": 2, "right": 3},
                        "id": "c1",
                    }
                ],
            ),
        )
    result = await module["run"](
        store,
        TestModel(responses=responses),
        partition="a",
        session_id="s",
    )
    assert result["messages"][-1].content == "answer"
    assert store.search_memories.await_count == 1
    assert store.add_memories.await_count == 1
    assert store.add_memories.await_args.kwargs["scope"] == MemoryScope(
        agent_id="a", session_id="s"
    )
