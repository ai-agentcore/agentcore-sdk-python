from __future__ import annotations

import runpy
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("google.adk")

from google.adk import Runner
from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_response import LlmResponse
from google.adk.sessions import InMemorySessionService, Session
from google.adk.tools.preload_memory_tool import preload_memory_tool
from google.genai import types
from pydantic import Field

from agentcore.integrations.memory.google_adk import AgentCoreMemoryService
from agentcore.memory import (
    AddMemoriesResult,
    Memory,
    MemoryContent,
    MemoryMessage,
    MemoryScope,
    MemorySearchHit,
    MemoryValidationError,
    SearchMemoriesResult,
)


class TestModel(BaseLlm):
    __test__ = False
    seen: list = Field(default_factory=list)

    async def generate_content_async(self, llm_request, stream=False):
        self.seen.append(llm_request)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text="answer")]))


def make_store():
    store = AsyncMock()
    store.memory_store_name = "memory"
    store.search_memories.return_value = SearchMemoriesResult(
        memories=[
            MemorySearchHit(
                Memory("id", MemoryContent("historical preference"), MemoryScope(agent_id="a")),
                1,
                1,
            )
        ]
    )
    store.add_memories.return_value = AddMemoriesResult()
    return store


def user_event(text):
    return Event(author="user", content=types.Content(role="user", parts=[types.Part(text=text)]))


@pytest.mark.asyncio
async def test_real_adk_runner_preloads_and_explicitly_saves_turn_delta():
    store = make_store()
    service = AgentCoreMemoryService(store, partition_resolver=lambda app, user: f"{app}/{user}")
    model = TestModel(model="test-model")
    agent = Agent(
        name="assistant", model=model, instruction="original", tools=[preload_memory_tool]
    )
    sessions = InMemorySessionService()
    runner = Runner(app_name="app", agent=agent, session_service=sessions, memory_service=service)
    session_ids = {}
    try:
        for turn, user in enumerate(["alice", "bob", "alice"]):
            if user not in session_ids:
                session = await sessions.create_session(app_name="app", user_id=user)
                session_ids[user] = session.id
            session_id = session_ids[user]
            question = f"question-{turn}"
            event = user_event(question)
            emitted = [
                e
                async for e in runner.run_async(
                    user_id=user,
                    session_id=session_id,
                    new_message=event.content,
                )
            ]
            assert any(e.is_final_response() for e in emitted)
            assert "historical preference" in str(model.seen[-1].contents)
            assert "original" in str(model.seen[-1].config.system_instruction)
            await service.add_events_to_memory(
                app_name="app",
                user_id=user,
                session_id=session_id,
                events=[event, *emitted],
            )
            assert store.add_memories.await_args.kwargs["scope"] == MemoryScope(
                agent_id=f"app/{user}",
                session_id=session_id,
            )
            assert store.add_memories.await_args.kwargs["messages"] == [
                MemoryMessage("user", question),
                MemoryMessage("assistant", "answer"),
            ]
        assert {c.kwargs["scope"].agent_id for c in store.search_memories.await_args_list} == {
            "app/alice",
            "app/bob",
        }
        assert all(
            c.kwargs["scope"].session_id is None for c in store.search_memories.await_args_list
        )
        assert store.add_memories.await_count == 3
    finally:
        await runner.close()


@pytest.mark.asyncio
async def test_adk_session_import_validation_and_text_filtering():
    store = make_store()
    service = AgentCoreMemoryService(store, partition_resolver=lambda app, user: f"{app}/{user}")
    events = [
        user_event("input"),
        Event(
            author="assistant",
            partial=True,
            content=types.Content(role="model", parts=[types.Part(text="partial")]),
        ),
        Event(
            author="assistant",
            content=types.Content(
                role="model",
                parts=[types.Part(function_call=types.FunctionCall(name="tool", args={}))],
            ),
        ),
        Event(
            author="assistant",
            content=types.Content(
                role="user",
                parts=[
                    types.Part(
                        function_response=types.FunctionResponse(
                            name="tool", response={"text": "private"}
                        )
                    )
                ],
            ),
        ),
        Event(
            author="assistant",
            content=types.Content(
                role="model",
                parts=[
                    types.Part(text="thinking", thought=True),
                    types.Part(text="answer"),
                ],
            ),
        ),
    ]
    with pytest.raises(MemoryValidationError):
        await service.add_events_to_memory(
            app_name="a",
            user_id="u",
            events=events,
            session_id="s",
            custom_metadata={"key": 123},
        )
    store.add_memories.assert_not_called()
    await service.add_events_to_memory(app_name="a", user_id="u", events=events)
    assert store.add_memories.await_args.kwargs["scope"] == MemoryScope(agent_id="a/u")
    await service.add_session_to_memory(Session(id="s", app_name="a", user_id="u", events=events))
    assert store.add_memories.await_args.kwargs["messages"] == [
        MemoryMessage("user", "input"),
        MemoryMessage("assistant", "answer"),
    ]
    response = await service.search_memory(app_name="a", user_id="u", query="query")
    assert response.memories[0].id == "id"
    assert response.memories[0].author is None and response.memories[0].timestamp is None


@pytest.mark.asyncio
async def test_documented_adk_example():
    module = runpy.run_path(
        str(Path(__file__).parents[2] / "examples/memory_frameworks/google_adk_agent.py")
    )
    store = make_store()
    events = await module["run"](
        store,
        TestModel(model="test-model"),
        partition="a",
        session_id="s",
    )
    assert events[-1].is_final_response()
    assert store.search_memories.await_count == 1
    assert store.add_memories.await_count == 1
    assert store.add_memories.await_args.kwargs["scope"] == MemoryScope(
        agent_id="a", session_id="s"
    )
