from __future__ import annotations

import inspect

from alibabacloud_agentcore20260804 import models
from alibabacloud_agentcore20260804.client import Client


def test_generated_memory_data_plane_contract_is_available() -> None:
    expected_parameters = {
        "add_memories_with_options_async": (
            "self",
            "workspace_id",
            "memory_store_name",
            "tmp_req",
            "headers",
            "runtime",
        ),
        "search_memories_with_options_async": (
            "self",
            "workspace_id",
            "memory_store_name",
            "tmp_req",
            "headers",
            "runtime",
        ),
        "list_memories_with_options_async": (
            "self",
            "workspace_id",
            "memory_store_name",
            "request",
            "headers",
            "runtime",
        ),
        "get_memory_with_options_async": (
            "self",
            "workspace_id",
            "memory_store_name",
            "memory_id",
            "request",
            "headers",
            "runtime",
        ),
        "update_memory_with_options_async": (
            "self",
            "workspace_id",
            "memory_store_name",
            "memory_id",
            "tmp_req",
            "headers",
            "runtime",
        ),
        "delete_memory_with_options_async": (
            "self",
            "workspace_id",
            "memory_store_name",
            "memory_id",
            "request",
            "headers",
            "runtime",
        ),
        "list_memory_sessions_with_options_async": (
            "self",
            "workspace_id",
            "memory_store_name",
            "request",
            "headers",
            "runtime",
        ),
        "list_memory_session_messages_with_options_async": (
            "self",
            "workspace_id",
            "memory_store_name",
            "request",
            "headers",
            "runtime",
        ),
    }

    for method_name, parameters in expected_parameters.items():
        method = getattr(Client, method_name)
        assert tuple(inspect.signature(method).parameters) == parameters

    required_models = (
        "AddMemoriesRequest",
        "SearchMemoriesRequest",
        "ListMemoriesRequest",
        "GetMemoryRequest",
        "UpdateMemoryRequest",
        "DeleteMemoryRequest",
        "ListMemorySessionsRequest",
        "ListMemorySessionMessagesRequest",
    )
    assert all(getattr(models, name, None) is not None for name in required_models)
    assert tuple(inspect.signature(models.AddMemoriesRequestBodyMessages).parameters) == (
        "content",
        "role",
    )
    assert tuple(inspect.signature(models.AddMemoriesRequestBodyScope).parameters) == (
        "agent_id",
        "session_id",
        "user_id",
    )
    assert tuple(inspect.signature(models.SearchMemoriesRequestBodyScope).parameters) == (
        "agent_id",
        "session_id",
        "user_id",
    )
    assert tuple(inspect.signature(models.ListMemoriesRequest).parameters) == (
        "agent_id",
        "max_results",
        "next_token",
        "session_id",
        "user_id",
    )
    assert tuple(inspect.signature(models.ListMemorySessionsRequest).parameters) == (
        "agent_id",
        "max_results",
        "next_token",
        "user_id",
    )
    assert tuple(inspect.signature(models.ListMemorySessionMessagesRequest).parameters) == (
        "agent_id",
        "max_results",
        "next_token",
        "session_id",
        "user_id",
    )
    assert tuple(inspect.signature(models.SearchMemoriesRequestBody).parameters) == (
        "enable_rerank",
        "metadata",
        "min_score",
        "min_similarity",
        "query",
        "scope",
        "top_k",
    )
    assert tuple(
        inspect.signature(models.SearchMemoriesResponseBodyDataMemories).parameters
    ) == ("memory", "score", "similarity")
    assert tuple(
        inspect.signature(models.ListMemorySessionMessagesResponseBodyItems).parameters
    ) == ("content", "role")
