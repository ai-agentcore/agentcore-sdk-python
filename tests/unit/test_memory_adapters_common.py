from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from agentcore.errors import MemoryAPIError, MemoryValidationError
from agentcore.integrations.memory import MemoryScopes
from agentcore.integrations.memory._common import recall, record
from agentcore.memory import MemoryMessage, MemoryScope, SearchMemoriesResult


def test_scopes_require_explicit_non_empty_ranges() -> None:
    with pytest.raises(MemoryValidationError):
        MemoryScopes(read=MemoryScope())
    with pytest.raises(MemoryValidationError):
        MemoryScopes(read=MemoryScope(agent_id="a"), write=MemoryScope())
    scopes = MemoryScopes(read=MemoryScope(agent_id="a"))
    with pytest.raises(MemoryValidationError):
        scopes.require_write()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [
    MemoryScope(user_id="u"),
    MemoryScope(agent_id="a"),
    MemoryScope(session_id="s"),
    MemoryScope(user_id="u", session_id="s"),
])
async def test_independent_scope_fields_reach_framework_read_and_write(scope) -> None:
    scopes = MemoryScopes(read=scope, write=scope)
    store = AsyncMock()
    store.memory_store_name = "memory"
    store.search_memories.return_value = SearchMemoriesResult()
    messages = [MemoryMessage("user", "question")]
    await recall(store, "query", scopes.read, top_k=5)
    await record(store, messages, scopes.require_write())
    store.search_memories.assert_awaited_once_with("query", scope=scope, top_k=5)
    store.add_memories.assert_awaited_once_with(scope=scope, messages=messages)


@pytest.mark.asyncio
async def test_only_automatic_remote_errors_are_best_effort(caplog) -> None:
    store = AsyncMock()
    store.memory_store_name = "memory"
    store.search_memories.side_effect = MemoryAPIError("SearchMemories", request_id="upstream-1")
    scope = MemoryScope(agent_id="a")
    assert await recall(store, "private-query", scope, top_k=5, best_effort=True) == ""
    assert "upstream-1" in caplog.text
    assert "private-query" not in caplog.text
    with pytest.raises(MemoryAPIError):
        await recall(store, "query", scope, top_k=5)
    store.search_memories.side_effect = ValueError("conversion bug")
    with pytest.raises(ValueError):
        await recall(store, "query", scope, top_k=5, best_effort=True)
    store.search_memories.side_effect = asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await recall(store, "query", scope, top_k=5, best_effort=True)


@pytest.mark.asyncio
async def test_no_empty_requests_or_write_retries() -> None:
    store = AsyncMock()
    store.memory_store_name = "memory"
    scope = MemoryScope(agent_id="a", session_id="s")
    assert await recall(store, " ", scope, top_k=5) == ""
    await record(store, [], scope)
    store.search_memories.assert_not_called()
    store.add_memories.assert_not_called()
    store.add_memories.side_effect = MemoryAPIError("AddMemories")
    await record(store, [MemoryMessage("user", "private")], scope, best_effort=True)
    assert store.add_memories.await_count == 1
    store.search_memories.side_effect = None
    store.search_memories.return_value = SearchMemoriesResult()
    assert await recall(store, "query", scope, top_k=5) == ""
