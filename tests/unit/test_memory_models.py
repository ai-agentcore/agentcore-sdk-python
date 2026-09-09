from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from agentcore.errors import (
    AddMemoriesOutcomeUnknownError,
    MemoryAPIError,
    MemoryContractError,
    MemoryValidationError,
)
from agentcore.memory import (
    AddMemoriesResult,
    Memory,
    MemoryContent,
    MemoryMessage,
    MemoryScope,
    MemorySearchHit,
    MemorySession,
    Page,
    SearchMemoriesResult,
)


def test_public_models_are_frozen_and_defensively_copy_collections() -> None:
    memory_metadata = {"memory": "before"}
    item_list = [
        MemorySession(agent_id="agent-1", session_id="session-1", user_id="user-1")
    ]
    id_list = ["memory-1"]

    message = MemoryMessage(role="user", content="hello")
    memory = Memory(
        memory_id="memory-1",
        content=MemoryContent(text="remember this"),
        scope=MemoryScope(agent_id="agent-1", session_id="session-1", user_id="user-1"),
        metadata=memory_metadata,
    )
    page = Page(items=item_list)
    added = AddMemoriesResult(memory_ids=id_list)  # type: ignore[arg-type]
    search = SearchMemoriesResult(
        memories=[MemorySearchHit(memory=memory, score=0.9, similarity=0.8)]
    )

    memory_metadata["memory"] = "after"
    item_list.clear()
    id_list.clear()

    assert message == MemoryMessage(role="user", content="hello")
    assert dict(memory.metadata or {}) == {"memory": "before"}
    assert page.items == (
        MemorySession(agent_id="agent-1", session_id="session-1", user_id="user-1"),
    )
    assert added.memory_ids == ("memory-1",)
    assert search.memories == (MemorySearchHit(memory=memory, score=0.9, similarity=0.8),)
    with pytest.raises(FrozenInstanceError):
        memory.memory_id = "other"  # type: ignore[misc]


def test_public_models_preserve_optional_fields_and_empty_results() -> None:
    scope = MemoryScope()
    message = MemoryMessage(role="assistant", content="answer")
    memory = Memory(
        memory_id="memory-1",
        content=MemoryContent(text="answer"),
        scope=scope,
    )

    assert scope.agent_id is None
    assert scope.session_id is None
    assert scope.user_id is None
    assert message.role == "assistant"
    assert message.content == "answer"
    assert memory.created_at is None
    assert memory.updated_at is None
    assert memory.metadata is None
    assert AddMemoriesResult().memory_ids == ()
    assert SearchMemoriesResult().memories == ()
    assert Page[Memory]().total_count is None


@pytest.mark.parametrize("metadata", [{"invalid": 1}, {" ": "value"}])
def test_model_metadata_rejects_invalid_entries_without_echoing_values(
    metadata: dict[str, object],
) -> None:
    marker = "SECRET_METADATA_VALUE"
    with pytest.raises(MemoryValidationError) as raised:
        Memory(
            memory_id="memory-1",
            content=MemoryContent(text="hello"),
            scope=MemoryScope(),
            metadata={"key": marker, **metadata},  # type: ignore[arg-type]
        )

    assert raised.value.code == "MEMORY_VALIDATION_FAILED"
    assert marker not in str(raised.value)
    assert marker not in repr(raised.value)


def test_memory_errors_have_stable_codes_and_safe_structured_fields() -> None:
    api_error = MemoryAPIError(
        "GetMemory",
        service_code="Resource.NotFound",
        http_status_code=404,
        request_id="request-1",
    )
    contract_error = MemoryContractError("GetMemory", "data is missing")
    unknown_error = AddMemoriesOutcomeUnknownError(
        "AddMemories",
        service_code="InternalError",
        http_status_code=500,
        request_id="request-2",
    )

    assert api_error.code == "MEMORY_API_FAILED"
    assert api_error.service_code == "Resource.NotFound"
    assert api_error.http_status_code == 404
    assert api_error.request_id == "request-1"
    assert contract_error.code == "MEMORY_RESPONSE_INVALID"
    assert unknown_error.code == "ADD_MEMORIES_OUTCOME_UNKNOWN"
    assert "may have succeeded" in str(unknown_error)
    assert api_error.__cause__ is None
    assert unknown_error.__cause__ is None


def test_memory_api_error_never_accepts_or_exposes_service_message() -> None:
    marker = "SECRET_SERVICE_MESSAGE"
    error = MemoryAPIError(
        "DeleteMemory",
        service_code="Forbidden",
        http_status_code=403,
        request_id="request-3",
    )

    assert marker not in str(error)
    assert marker not in repr(error)
    assert "message" not in vars(error) or vars(error)["message"] != marker
