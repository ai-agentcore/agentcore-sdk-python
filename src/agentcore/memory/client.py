"""Async-first public client for one existing MemoryStore."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from agentcore.errors import MemoryValidationError
from agentcore.memory._transport import _ClientFactory, _MemoryTransport, _RuntimeProvider
from agentcore.memory.models import (
    AddMemoriesResult,
    Memory,
    MemoryMessage,
    MemoryScope,
    MemorySession,
    Page,
    SearchMemoriesResult,
)


class AsyncMemoryStore:
    """Data-plane client bound locally to one existing MemoryStore name."""

    def __init__(
        self,
        memory_store_name: str,
        *,
        _runtime_provider: _RuntimeProvider,
        _client_factory: _ClientFactory | None = None,
    ) -> None:
        self.memory_store_name = _required_string(memory_store_name, "memory_store_name")
        self._transport = _MemoryTransport(
            self.memory_store_name,
            _runtime_provider,
            client_factory=_client_factory,
        )

    async def add_memories(
        self,
        *,
        scope: MemoryScope | None = None,
        text: str | None = None,
        messages: Sequence[MemoryMessage] | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> AddMemoriesResult:
        validated_scope = _optional_scope(scope)
        has_text = text is not None
        has_messages = messages is not None
        if has_text == has_messages:
            raise MemoryValidationError("exactly one of text or messages must be provided")
        if text is not None:
            _required_string(text, "text")
        validated_messages = _messages(messages) if messages is not None else None
        validated_metadata = _metadata(metadata, "metadata")
        return await self._transport.add_memories(
            scope=validated_scope,
            text=text,
            messages=validated_messages,
            metadata=validated_metadata,
        )

    async def search_memories(
        self,
        query: str,
        *,
        scope: MemoryScope | None = None,
        top_k: int | None = None,
        metadata: Mapping[str, str] | None = None,
        enable_rerank: bool | None = None,
        min_similarity: float | None = None,
        min_score: float | None = None,
    ) -> SearchMemoriesResult:
        _required_string(query, "query")
        validated_scope = _optional_scope(scope)
        _bounded_integer(top_k, "top_k", maximum=50)
        validated_metadata = _metadata(metadata, "metadata")
        _optional_boolean(enable_rerank, "enable_rerank")
        _optional_number(min_similarity, "min_similarity")
        _optional_number(min_score, "min_score")
        return await self._transport.search_memories(
            query,
            scope=validated_scope,
            top_k=top_k,
            metadata=validated_metadata,
            enable_rerank=enable_rerank,
            min_similarity=min_similarity,
            min_score=min_score,
        )

    async def list_memories(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        session_id: str | None = None,
        max_results: int | None = None,
        next_token: str | None = None,
    ) -> Page[Memory]:
        _optional_scope_string(user_id, "user_id")
        _optional_scope_string(agent_id, "agent_id")
        _optional_scope_string(session_id, "session_id")
        _bounded_integer(max_results, "max_results", maximum=100)
        _next_token(next_token)
        return await self._transport.list_memories(
            user_id=user_id,
            agent_id=agent_id,
            session_id=session_id,
            max_results=max_results,
            next_token=next_token,
        )

    async def get_memory(self, memory_id: str) -> Memory:
        _required_string(memory_id, "memory_id")
        return await self._transport.get_memory(memory_id)

    async def update_memory(
        self,
        memory_id: str,
        *,
        text: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> Memory:
        _required_string(memory_id, "memory_id")
        if text is None and metadata is None:
            raise MemoryValidationError("at least one of text or metadata must be provided")
        if text is not None:
            _required_string(text, "text")
        validated_metadata = _metadata(metadata, "metadata")
        return await self._transport.update_memory(
            memory_id,
            text=text,
            metadata=validated_metadata,
        )

    async def delete_memory(self, memory_id: str) -> None:
        _required_string(memory_id, "memory_id")
        await self._transport.delete_memory(memory_id)

    async def list_memory_sessions(
        self,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        max_results: int | None = None,
        next_token: str | None = None,
    ) -> Page[MemorySession]:
        _optional_scope_string(user_id, "user_id")
        _optional_scope_string(agent_id, "agent_id")
        _bounded_integer(max_results, "max_results", maximum=100)
        _next_token(next_token)
        return await self._transport.list_memory_sessions(
            user_id=user_id,
            agent_id=agent_id,
            max_results=max_results,
            next_token=next_token,
        )

    async def list_memory_session_messages(
        self,
        session_id: str,
        *,
        user_id: str | None = None,
        agent_id: str | None = None,
        max_results: int | None = None,
        next_token: str | None = None,
    ) -> Page[MemoryMessage]:
        _required_string(session_id, "session_id")
        _optional_scope_string(session_id, "session_id")
        _optional_scope_string(user_id, "user_id")
        _optional_scope_string(agent_id, "agent_id")
        if user_id is None and agent_id is None:
            raise MemoryValidationError(
                "at least one of user_id or agent_id must be provided"
            )
        _bounded_integer(max_results, "max_results", maximum=100)
        _next_token(next_token)
        return await self._transport.list_memory_session_messages(
            session_id,
            user_id=user_id,
            agent_id=agent_id,
            max_results=max_results,
            next_token=next_token,
        )


def _required_string(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"{field_name} must be a non-empty string")
    return value


def _optional_non_empty_string(value: object, field_name: str) -> None:
    if value is not None:
        _required_string(value, field_name)


def _optional_scope_string(value: object, field_name: str) -> None:
    if value is None:
        return
    validated = _required_string(value, field_name)
    if validated.strip() in {"*", "__default__"}:
        raise MemoryValidationError(
            f"{field_name} must not use a reserved scope value"
        )


def _next_token(value: object) -> None:
    if value is not None and not isinstance(value, str):
        raise MemoryValidationError("next_token must be a string")


def _bounded_integer(value: object, field_name: str, *, maximum: int) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise MemoryValidationError(f"{field_name} must be an integer from 1 to {maximum}")


def _optional_boolean(value: object, field_name: str) -> None:
    if value is not None and not isinstance(value, bool):
        raise MemoryValidationError(f"{field_name} must be a boolean")


def _optional_number(value: object, field_name: str) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MemoryValidationError(f"{field_name} must be a number")


def _metadata(
    value: Mapping[str, str] | None,
    field_name: str,
) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str)
        and bool(key.strip())
        and isinstance(item, str)
        for key, item in value.items()
    ):
        raise MemoryValidationError(
            f"{field_name} must contain non-empty string keys and string values"
        )
    return dict(value)


def _optional_scope(value: MemoryScope | None) -> MemoryScope | None:
    if value is None:
        return None
    if not isinstance(value, MemoryScope):
        raise MemoryValidationError("scope must be a MemoryScope")
    _optional_scope_string(value.user_id, "scope.user_id")
    _optional_scope_string(value.agent_id, "scope.agent_id")
    _optional_scope_string(value.session_id, "scope.session_id")
    return value


def _messages(value: Sequence[MemoryMessage]) -> tuple[MemoryMessage, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)) or not value:
        raise MemoryValidationError("messages must be a non-empty sequence")
    copied = tuple(value)
    for message in copied:
        if not isinstance(message, MemoryMessage):
            raise MemoryValidationError("messages must contain MemoryMessage values")
        _required_string(message.role, "messages[].role")
        _required_string(message.content, "messages[].content")
    return copied
