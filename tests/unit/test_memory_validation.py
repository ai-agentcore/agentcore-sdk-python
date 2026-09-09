from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from agentcore.errors import MemoryValidationError
from agentcore.memory import AsyncMemoryStore, MemoryMessage, MemoryScope
from agentcore.memory._transport import _MemoryRuntime


class RuntimeProbe:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self) -> _MemoryRuntime:
        self.calls += 1
        raise AssertionError("runtime must not be resolved for invalid input")


def test_memory_store_name_is_validated_without_resolving_runtime() -> None:
    runtime = RuntimeProbe()

    with pytest.raises(MemoryValidationError, match="memory_store_name"):
        AsyncMemoryStore("   ", _runtime_provider=runtime)

    assert runtime.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invoke",
    [
        lambda store: store.add_memories(
            scope=MemoryScope(agent_id="agent", session_id="session")
        ),
        lambda store: store.add_memories(
            scope=MemoryScope(agent_id="agent", session_id="session"),
            text="text",
            messages=[MemoryMessage(role="user", content="hello")],
        ),
        lambda store: store.add_memories(
            scope=MemoryScope(agent_id="agent", session_id="session"), messages=[]
        ),
        lambda store: store.add_memories(
            scope=MemoryScope(agent_id="agent", session_id="session"),
            messages=[MemoryMessage(role="", content="hello")],
        ),
        lambda store: store.add_memories(
            scope=MemoryScope(agent_id="agent", session_id="session"),
            text="text",
            metadata={"invalid": 1},  # type: ignore[dict-item]
        ),
        lambda store: store.add_memories(
            scope=MemoryScope(user_id=""),
            text="text",
        ),
        lambda store: store.add_memories(
            scope=MemoryScope(agent_id="*"),
            text="text",
        ),
        lambda store: store.add_memories(
            scope=MemoryScope(session_id="__default__"),
            text="text",
        ),
        lambda store: store.search_memories("", top_k=1),
        lambda store: store.search_memories("query", scope=MemoryScope(agent_id="")),
        lambda store: store.search_memories("query", top_k=0),
        lambda store: store.search_memories("query", top_k=51),
        lambda store: store.search_memories("query", top_k=True),
        lambda store: store.search_memories("query", metadata={" ": "value"}),
        lambda store: store.search_memories(
            "query", metadata={"key": 1}  # type: ignore[dict-item]
        ),
        lambda store: store.search_memories("query", enable_rerank=1),
        lambda store: store.search_memories("query", min_score=True),
        lambda store: store.search_memories("query", min_score="0.5"),
        lambda store: store.list_memories(max_results=0),
        lambda store: store.list_memories(user_id="*"),
        lambda store: store.list_memories(agent_id="__default__"),
        lambda store: store.list_memories(max_results=True),
        lambda store: store.list_memories(next_token=1),  # type: ignore[arg-type]
        lambda store: store.get_memory(""),
        lambda store: store.update_memory("memory"),
        lambda store: store.update_memory("memory", text=""),
        lambda store: store.delete_memory(" "),
        lambda store: store.list_memory_sessions(agent_id=""),
        lambda store: store.list_memory_sessions(user_id="*"),
        lambda store: store.list_memory_sessions(max_results=101),
        lambda store: store.list_memory_session_messages("", agent_id="agent"),
        lambda store: store.list_memory_session_messages("session"),
        lambda store: store.list_memory_session_messages("session", user_id=""),
        lambda store: store.list_memory_session_messages("session", agent_id=""),
        lambda store: store.list_memory_session_messages("*", agent_id="agent"),
        lambda store: store.list_memory_session_messages(
            "session", user_id="__default__"
        ),
        lambda store: store.list_memory_session_messages(
            "session", agent_id="agent", max_results=False
        ),
    ],
)
async def test_invalid_public_inputs_fail_before_runtime(
    invoke: Callable[[AsyncMemoryStore], Awaitable[Any]],
) -> None:
    runtime = RuntimeProbe()
    store = AsyncMemoryStore("customer_memory", _runtime_provider=runtime)

    with pytest.raises(MemoryValidationError):
        await invoke(store)

    assert runtime.calls == 0
