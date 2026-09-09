"""Explicit memory nodes for application-owned LangGraph state and edges."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import BaseMessage, SystemMessage
from typing_extensions import NotRequired, TypedDict

from agentcore.integrations.memory._common import recall, record, reference_text, validate_top_k
from agentcore.memory import AsyncMemoryStore, MemoryMessage, MemoryScope


class MemoryNodeState(TypedDict):
    """Map these fields from your graph; SDK never infers a turn from history."""

    memory_query: NotRequired[str]
    memory_read_scope: NotRequired[MemoryScope]
    memory_write_scope: NotRequired[MemoryScope]
    memory_messages: NotRequired[list[MemoryMessage]]
    memory_text: NotRequired[str]


class AgentCoreMemoryNodes:
    def __init__(self, store: AsyncMemoryStore, *, top_k: int = 5) -> None:
        validate_top_k(top_k)
        self._store = store
        self._top_k = top_k

    async def recall(self, state: MemoryNodeState) -> dict[str, str]:
        text = await recall(
            self._store,
            state["memory_query"],
            state["memory_read_scope"],
            top_k=self._top_k,
        )
        return {"memory_text": text}

    async def record(self, state: MemoryNodeState) -> dict[str, list[MemoryMessage]]:
        await record(self._store, state["memory_messages"], state["memory_write_scope"])
        return {"memory_messages": []}


def with_memory(messages: Sequence[BaseMessage], memory_text: str) -> list[BaseMessage]:
    """Build model input without mutating or persisting the original messages."""
    result = list(messages)
    note = reference_text(memory_text)
    if not note:
        return result
    if result and isinstance(result[0], SystemMessage):
        system = result[0]
        content = system.content
        result[0] = system.model_copy(
            update={
                "content": content + "\n\n" + note
                if isinstance(content, str)
                else [*content, {"type": "text", "text": note}],
            }
        )
    else:
        result.insert(0, SystemMessage(content=note))
    return result
