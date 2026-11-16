from unittest.mock import AsyncMock

import pytest

from agentcore.errors import MemoryAPIError
from agentcore.integrations.memory import MemoryScopes
from agentcore.integrations.memory._common import recall, record
from agentcore.memory import (
    AsyncContextMemoryClient,
    ContextMemorySearchResult,
    CreateEventsResult,
    EventReceipt,
    MemoryMessage,
    MemoryScope,
)


@pytest.mark.asyncio
async def test_context_adapter_uses_events_and_team_scope_without_polling():
    client = AsyncMock(spec=AsyncContextMemoryClient)
    client.memory_store_name = "test-memory"
    client.search_memories.return_value = ContextMemorySearchResult(())
    client.create_events.return_value = CreateEventsResult((EventReceipt(0, "STORED", "e1"),))
    scope = MemoryScope(team_id="team")
    scopes = MemoryScopes(read=scope, write=scope)
    assert await recall(client, "question", scopes.read, top_k=100) == ""
    await record(client, [MemoryMessage("user", "question")], scopes.require_write())
    client.search_memories.assert_awaited_once_with("question", scope=scope, top_k=100)
    call = client.create_events.await_args
    assert call.kwargs["scope"] == scope
    event = call.args[0][0]
    assert event.type == "message" and event.timestamp > 0
    assert event.to_dict()["data"] == {
        "role": "user",
        "content": [{"type": "text", "text": "question"}],
    }


@pytest.mark.asyncio
async def test_context_write_checks_receipts_and_does_not_retry(caplog):
    client = AsyncMock(spec=AsyncContextMemoryClient)
    client.memory_store_name = "test-memory"
    client.create_events.return_value = CreateEventsResult(
        (
            EventReceipt(0, "STORED", "e1"),
            EventReceipt(1, "REJECTED", None, "PolicyDenied", "Write rejected"),
        ),
        "request-partial",
    )
    messages = [MemoryMessage("user", "question"), MemoryMessage("assistant", "answer")]
    with pytest.raises(MemoryAPIError) as caught:
        await record(client, messages, MemoryScope(user_id="u"))
    assert caught.value.request_id == "request-partial"
    assert caught.value.service_code == "PolicyDenied"
    client.create_events.assert_awaited_once()
    await record(client, messages, MemoryScope(user_id="u"), best_effort=True)
    assert client.create_events.await_count == 2
    assert "request-partial" in caplog.text
    assert "write.succeeded" not in caplog.text


@pytest.mark.asyncio
async def test_context_write_batches_session_import_without_repeating_messages():
    client = AsyncMock(spec=AsyncContextMemoryClient)
    client.memory_store_name = "test-memory"

    async def create(events, **kwargs):
        return CreateEventsResult(
            tuple(EventReceipt(i, "STORED", f"e{i}") for i in range(len(events)))
        )

    client.create_events.side_effect = create
    await record(
        client, [MemoryMessage("user", str(i)) for i in range(101)], MemoryScope(user_id="u")
    )
    batches = [call.args[0] for call in client.create_events.await_args_list]
    assert [len(batch) for batch in batches] == [100, 1]
    assert [
        event.to_dict()["data"]["content"][0]["text"] for batch in batches for event in batch
    ] == [str(i) for i in range(101)]
