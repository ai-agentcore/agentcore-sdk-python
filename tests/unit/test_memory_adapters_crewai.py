from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("crewai")

from crewai import Agent, Crew, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput
from pydantic import Field

from agentcore.errors import MemoryAPIError, MemoryValidationError
from agentcore.integrations.memory import MemoryScopes
from agentcore.integrations.memory.crewai import AgentCoreTaskMemory
from agentcore.memory import (
    AddMemoriesResult,
    Memory,
    MemoryContent,
    MemoryMessage,
    MemoryScope,
    MemorySearchHit,
    SearchMemoriesResult,
)


class ReplyModel(BaseLLM):
    seen: list = Field(default_factory=list)

    def call(self, messages, **kwargs):
        self.seen.append(messages)
        return "Final Answer: answer"

    async def acall(self, messages, **kwargs):
        await asyncio.sleep(0)
        return self.call(messages, **kwargs)


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


def scopes(name):
    return MemoryScopes(MemoryScope(agent_id=name), MemoryScope(agent_id=name, session_id="s"))


def crew(binding, model):
    agent = Agent(role="assistant", goal="Answer", backstory="original", llm=model, verbose=False)
    task = Task(
        description="question\n" + binding.instructions,
        expected_output="text",
        agent=agent,
        callback=binding.callback,
    )
    return Crew(agents=[agent], tasks=[task], verbose=False, memory=False, tracing=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["akickoff", "kickoff_async"])
async def test_real_crewai_task_callback_uses_owner_loop_and_original_query(method):
    store = make_store()
    owner = asyncio.get_running_loop()

    async def add(**kwargs):
        assert asyncio.get_running_loop() is owner
        return AddMemoriesResult()

    store.add_memories.side_effect = add
    adapter = AgentCoreTaskMemory(store, write_back=True)
    binding = await adapter.prepare("question", scopes=scopes("alice"))
    model = ReplyModel(model="test")
    result = await getattr(crew(binding, model), method)()
    assert result.raw == "answer"
    assert "fact-alice" in str(model.seen)
    assert "original" in str(model.seen)
    store.search_memories.assert_awaited_once()
    store.add_memories.assert_awaited_once_with(
        scope=scopes("alice").write,
        messages=[MemoryMessage("user", "question"), MemoryMessage("assistant", "answer")],
    )


@pytest.mark.asyncio
async def test_crewai_read_only_and_concurrent_task_bindings():
    store = make_store()
    adapter = AgentCoreTaskMemory(store)
    bindings = await asyncio.gather(
        *[adapter.prepare("question", scopes=scopes(n)) for n in ["alice", "bob"]]
    )
    models = [ReplyModel(model="test") for _ in bindings]
    assert all(binding.callback is None for binding in bindings)
    await asyncio.gather(
        *[crew(binding, model).akickoff() for binding, model in zip(bindings, models, strict=True)]
    )
    assert "fact-alice" in str(models[0].seen) and "fact-bob" not in str(models[0].seen)
    assert "fact-bob" in str(models[1].seen) and "fact-alice" not in str(models[1].seen)
    store.add_memories.assert_not_called()


@pytest.mark.asyncio
async def test_crewai_callback_ignores_structured_empty_output_and_does_not_retry(caplog):
    store = make_store()
    adapter = AgentCoreTaskMemory(store, write_back=True)
    binding = await adapter.prepare("private query", scopes=scopes("a"))
    assert binding.callback is not None
    await binding.callback(TaskOutput(description="injected reference", raw="", agent="a"))
    await binding.callback(
        TaskOutput(
            description="injected reference",
            raw='{"x":1}',
            agent="a",
            json_dict={"x": 1},
            output_format=OutputFormat.JSON,
        )
    )
    store.add_memories.assert_not_called()
    store.add_memories.side_effect = MemoryAPIError("AddMemories", request_id="request-1")
    await binding.callback(TaskOutput(description="injected reference", raw="answer", agent="a"))
    assert store.add_memories.await_count == 1
    assert "request-1" in caplog.text and "private query" not in caplog.text


@pytest.mark.asyncio
async def test_crewai_invalid_write_scope_fails_before_search():
    store = make_store()
    with pytest.raises(MemoryValidationError):
        await AgentCoreTaskMemory(store, write_back=True).prepare(
            "q", scopes=MemoryScopes(MemoryScope(agent_id="a"))
        )
    store.search_memories.assert_not_called()


@pytest.mark.asyncio
async def test_crewai_failed_or_cancelled_task_never_calls_memory():
    store = make_store()
    binding = await AgentCoreTaskMemory(store, write_back=True).prepare("q", scopes=scopes("a"))

    class BrokenModel(ReplyModel):
        def call(self, messages, **kwargs):
            raise ValueError("model failed")

        async def acall(self, messages, **kwargs):
            raise ValueError("model failed")

    with pytest.raises(ValueError, match="model failed"):
        await crew(binding, BrokenModel(model="test")).akickoff()
    store.add_memories.assert_not_called()

    class CancelledModel(ReplyModel):
        def call(self, messages, **kwargs):
            raise asyncio.CancelledError()

        async def acall(self, messages, **kwargs):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await crew(binding, CancelledModel(model="test")).akickoff()
    store.add_memories.assert_not_called()
