"""ADK MemoryService; user partitions are mapped explicitly by the application."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from google.adk.events import Event
from google.adk.memory.base_memory_service import BaseMemoryService, SearchMemoryResponse
from google.adk.memory.memory_entry import MemoryEntry
from google.adk.sessions import Session
from google.genai import types

from agentcore.integrations.memory._common import validate_scope, validate_top_k
from agentcore.memory import AsyncMemoryStore, MemoryMessage, MemoryScope, MemoryValidationError


def _messages(events: Sequence[Event]) -> list[MemoryMessage]:
    result = []
    for event in events:
        content = event.content
        if content is None or event.partial or event.error_code:
            continue
        if event.author == "user":
            role = "user"
        elif content.role == "model" and event.is_final_response():
            role = "assistant"
        else:
            continue
        text = "\n".join(
            part.text for part in content.parts or [] if part.text and not part.thought
        )
        if text.strip():
            result.append(MemoryMessage(role, text))
    return result


class AgentCoreMemoryService(BaseMemoryService):  # type: ignore[misc,unused-ignore]
    """Async MemoryService for ADK 2.8+. It does not own the supplied store."""

    def __init__(
        self,
        store: AsyncMemoryStore,
        *,
        partition_resolver: Callable[[str, str], str],
        top_k: int = 5,
    ) -> None:
        validate_top_k(top_k)
        self._store = store
        self._partition = partition_resolver
        self._top_k = top_k

    async def search_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        query: str,
    ) -> SearchMemoryResponse:
        scope = MemoryScope(agent_id=self._partition(app_name, user_id))
        validate_scope(scope)
        result = await self._store.search_memories(query, scope=scope, top_k=self._top_k)
        return SearchMemoryResponse(
            memories=[
                MemoryEntry(
                    id=hit.memory.memory_id,
                    content=types.Content(parts=[types.Part(text=hit.memory.content.text)]),
                    timestamp=hit.memory.created_at,
                    custom_metadata=dict(hit.memory.metadata or {}),
                )
                for hit in result.memories
            ]
        )

    async def add_session_to_memory(self, session: Session) -> None:
        await self.add_events_to_memory(
            app_name=session.app_name,
            user_id=session.user_id,
            session_id=session.id,
            events=session.events,
        )

    async def add_events_to_memory(
        self,
        *,
        app_name: str,
        user_id: str,
        events: Sequence[Event],
        session_id: str | None = None,
        custom_metadata: Mapping[str, object] | None = None,
    ) -> None:
        scope = MemoryScope(
            agent_id=self._partition(app_name, user_id),
            session_id=session_id,
        )
        validate_scope(scope)
        metadata = None
        if custom_metadata is not None:
            metadata = {}
            for key, value in custom_metadata.items():
                if not isinstance(key, str) or not key.strip() or not isinstance(value, str):
                    raise MemoryValidationError(
                        "custom_metadata requires non-empty string keys and string values"
                    )
                metadata[key] = value
        messages = _messages(events)
        if messages:
            await self._store.add_memories(scope=scope, messages=messages, metadata=metadata)
