from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from alibabacloud_agentcore20260804 import models
from darabonba.exceptions import TeaException

from agentcore.auth import AccessKeyCredential
from agentcore.auth.resource_sts import ResourceCredential
from agentcore.errors import (
    AddMemoriesOutcomeUnknownError,
    MemoryAPIError,
    MemoryContractError,
)
from agentcore.memory import (
    AsyncMemoryStore,
    MemoryMessage,
    MemoryScope,
    MemorySession,
)
from agentcore.memory._transport import _MemoryRuntime, _MemoryTransport


@dataclass(frozen=True)
class GeneratedCall:
    operation: str
    args: tuple[Any, ...]


class FakeSTS:
    def __init__(self) -> None:
        self.purposes: list[str | None] = []
        self.credential = ResourceCredential(
            access_key_id="fake-ak",
            access_key_secret="fake-sk",
            security_token="fake-token",
            expiration=datetime.now(timezone.utc) + timedelta(hours=1),
        )

    async def get(self, purpose: str | None = None) -> ResourceCredential:
        self.purposes.append(purpose)
        return self.credential


class RecordingClient:
    def __init__(self, responses: dict[str, Any] | None = None) -> None:
        self.responses = _success_responses()
        if responses:
            self.responses.update(responses)
        self.calls: list[GeneratedCall] = []

    async def _call(self, operation: str, *args: Any) -> Any:
        self.calls.append(GeneratedCall(operation, args))
        result = self.responses[operation]
        if isinstance(result, Exception):
            raise result
        return result

    async def add_memories_with_options_async(self, *args: Any) -> Any:
        return await self._call("AddMemories", *args)

    async def search_memories_with_options_async(self, *args: Any) -> Any:
        return await self._call("SearchMemories", *args)

    async def list_memories_with_options_async(self, *args: Any) -> Any:
        return await self._call("ListMemories", *args)

    async def get_memory_with_options_async(self, *args: Any) -> Any:
        return await self._call("GetMemory", *args)

    async def update_memory_with_options_async(self, *args: Any) -> Any:
        return await self._call("UpdateMemory", *args)

    async def delete_memory_with_options_async(self, *args: Any) -> Any:
        return await self._call("DeleteMemory", *args)

    async def list_memory_sessions_with_options_async(self, *args: Any) -> Any:
        return await self._call("ListMemorySessions", *args)

    async def list_memory_session_messages_with_options_async(self, *args: Any) -> Any:
        return await self._call("ListMemorySessionMessages", *args)


@dataclass
class StoreHarness:
    store: AsyncMemoryStore
    client: RecordingClient
    sts: FakeSTS
    runtime_calls: list[_MemoryRuntime]
    factory_calls: list[tuple[_MemoryRuntime, AccessKeyCredential | ResourceCredential]]


def _harness(responses: dict[str, Any] | None = None) -> StoreHarness:
    client = RecordingClient(responses)
    sts = FakeSTS()
    runtime = _MemoryRuntime(
        workspace_id="workspace-runtime",
        region_id="cn-hangzhou",
        endpoint="https://agentcore-pre.example.com",
        resource_sts=sts,
    )
    runtime_calls: list[_MemoryRuntime] = []
    factory_calls: list[
        tuple[_MemoryRuntime, AccessKeyCredential | ResourceCredential]
    ] = []

    async def runtime_provider() -> _MemoryRuntime:
        runtime_calls.append(runtime)
        return runtime

    def client_factory(
        resolved_runtime: _MemoryRuntime,
        credential: AccessKeyCredential | ResourceCredential,
    ) -> RecordingClient:
        factory_calls.append((resolved_runtime, credential))
        return client

    store = AsyncMemoryStore(
        "customer_memory",
        _runtime_provider=runtime_provider,
        _client_factory=client_factory,
    )
    return StoreHarness(store, client, sts, runtime_calls, factory_calls)


def _memory_scope(scope_type: type[Any]) -> Any:
    return scope_type(agent_id="agent-1", session_id="session-1", user_id="user-1")


def _success_responses() -> dict[str, Any]:
    get_memory = models.GetMemoryResponseBodyData(
        memory_id="memory-get",
        content=models.GetMemoryResponseBodyDataContent(text="get text"),
        scope=_memory_scope(models.GetMemoryResponseBodyDataScope),
        metadata={"source": "get"},
        created_at="2026-09-02T01:00:00Z",
        updated_at="2026-09-02T02:00:00Z",
    )
    updated_memory = models.UpdateMemoryResponseBodyData(
        memory_id="memory-update",
        content=models.UpdateMemoryResponseBodyDataContent(text="updated text"),
        scope=_memory_scope(models.UpdateMemoryResponseBodyDataScope),
        metadata={"source": "update"},
        created_at="2026-09-02T01:00:00Z",
        updated_at="2026-09-02T03:00:00Z",
    )
    listed_memory = models.ListMemoriesResponseBodyItems(
        memory_id="memory-list",
        content=models.ListMemoriesResponseBodyItemsContent(text="listed text"),
        scope=_memory_scope(models.ListMemoriesResponseBodyItemsScope),
        metadata={"source": "list"},
        created_at="2026-09-02T01:00:00Z",
        updated_at=None,
    )
    searched_memory = models.SearchMemoriesResponseBodyDataMemoriesMemory(
        memory_id="memory-search",
        content=models.SearchMemoriesResponseBodyDataMemoriesMemoryContent(
            text="searched text"
        ),
        scope=_memory_scope(models.SearchMemoriesResponseBodyDataMemoriesMemoryScope),
        metadata={"source": "search"},
    )
    return {
        "AddMemories": models.AddMemoriesResponse(
            status_code=200,
            body=models.AddMemoriesResponseBody(
                success=True,
                http_status_code=200,
                request_id="request-add",
                data=models.AddMemoriesResponseBodyData(
                    memories=[
                        models.AddMemoriesResponseBodyDataMemories(memory_id="memory-add")
                    ]
                ),
            ),
        ),
        "SearchMemories": models.SearchMemoriesResponse(
            status_code=200,
            body=models.SearchMemoriesResponseBody(
                success=True,
                http_status_code=200,
                data=models.SearchMemoriesResponseBodyData(
                    memories=[
                        models.SearchMemoriesResponseBodyDataMemories(
                            memory=searched_memory,
                            score=0.95,
                            similarity=0.85,
                        )
                    ]
                ),
            ),
        ),
        "ListMemories": models.ListMemoriesResponse(
            status_code=200,
            body=models.ListMemoriesResponseBody(
                success=True,
                http_status_code=200,
                items=[listed_memory],
                max_results=10,
                next_token="next-memory-page",
                total_count=3,
            ),
        ),
        "GetMemory": models.GetMemoryResponse(
            status_code=200,
            body=models.GetMemoryResponseBody(
                success=True,
                http_status_code=200,
                data=get_memory,
            ),
        ),
        "UpdateMemory": models.UpdateMemoryResponse(
            status_code=200,
            body=models.UpdateMemoryResponseBody(
                success=True,
                http_status_code=200,
                data=updated_memory,
            ),
        ),
        "DeleteMemory": models.DeleteMemoryResponse(
            status_code=200,
            body=models.DeleteMemoryResponseBody(success=True, http_status_code=200),
        ),
        "ListMemorySessions": models.ListMemorySessionsResponse(
            status_code=200,
            body=models.ListMemorySessionsResponseBody(
                success=True,
                http_status_code=200,
                items=[
                    models.ListMemorySessionsResponseBodyItems(
                        agent_id="agent-1",
                        session_id="session-1",
                        user_id="user-1",
                    )
                ],
                max_results=20,
                next_token="",
                total_count=None,
            ),
        ),
        "ListMemorySessionMessages": models.ListMemorySessionMessagesResponse(
            status_code=200,
            body=models.ListMemorySessionMessagesResponseBody(
                success=True,
                http_status_code=200,
                items=[
                    models.ListMemorySessionMessagesResponseBodyItems(
                        role="user",
                        content="message content",
                    )
                ],
                max_results=30,
                next_token=None,
                total_count=1,
            ),
        ),
    }


@pytest.mark.asyncio
async def test_eight_actions_use_runtime_workspace_memory_store_sts_and_no_retry() -> None:
    harness = _harness()
    scope = MemoryScope(agent_id="agent-1", session_id="session-1", user_id="user-1")

    added = await harness.store.add_memories(
        scope=scope,
        text="remember",
        metadata={"kind": "fact"},
    )
    searched = await harness.store.search_memories(
        "find",
        scope=scope,
        top_k=5,
        metadata={"kind": "fact"},
        enable_rerank=False,
        min_similarity=0.2,
        min_score=0.3,
    )
    listed = await harness.store.list_memories(
        user_id="user-1",
        agent_id="agent-1",
        session_id="session-1",
        max_results=10,
        next_token="page-1",
    )
    fetched = await harness.store.get_memory("memory-get")
    updated = await harness.store.update_memory(
        "memory-update",
        text="new text",
        metadata={},
    )
    deleted = await harness.store.delete_memory("memory-delete")
    sessions = await harness.store.list_memory_sessions(
        user_id="user-1",
        agent_id="agent-1",
        max_results=20,
        next_token="session-page",
    )
    messages = await harness.store.list_memory_session_messages(
        "session-1",
        user_id="user-1",
        agent_id="agent-1",
        max_results=30,
        next_token="message-page",
    )

    assert added.memory_ids == ("memory-add",)
    assert searched.memories[0].memory.memory_id == "memory-search"
    assert searched.memories[0].score == 0.95
    assert searched.memories[0].similarity == 0.85
    assert listed.items[0].memory_id == "memory-list"
    assert fetched.content.text == "get text"
    assert dict(fetched.metadata or {}) == {"source": "get"}
    assert updated.memory_id == "memory-update"
    assert deleted is None
    assert sessions.items[0].session_id == "session-1"
    assert sessions.items[0].user_id == "user-1"
    assert sessions.next_token is None
    assert sessions.total_count is None
    assert messages.items[0] == MemoryMessage(role="user", content="message content")

    assert [call.operation for call in harness.client.calls] == [
        "AddMemories",
        "SearchMemories",
        "ListMemories",
        "GetMemory",
        "UpdateMemory",
        "DeleteMemory",
        "ListMemorySessions",
        "ListMemorySessionMessages",
    ]
    assert harness.sts.purposes == ["highcode_sdk"] * 8
    assert len(harness.runtime_calls) == 8
    assert len(harness.factory_calls) == 8
    assert all(call.args[0] == "workspace-runtime" for call in harness.client.calls)
    assert all(call.args[1] == "customer_memory" for call in harness.client.calls)
    for index, call in enumerate(harness.client.calls):
        runtime_options = call.args[-1]
        assert runtime_options.autoretry is False
        assert runtime_options.max_attempts == 1
        assert runtime_options.read_timeout == (120_000 if index == 0 else None)
        assert call.args[-2] == {}


@pytest.mark.asyncio
async def test_add_text_and_messages_map_to_generated_body_without_cross_mode_fields() -> None:
    scope = MemoryScope(agent_id="agent-1", session_id="session-1", user_id="user-1")
    text_harness = _harness()
    await text_harness.store.add_memories(
        scope=scope,
        text="remember",
        metadata={"kind": "text"},
    )
    text_body = text_harness.client.calls[0].args[2].body

    assert isinstance(text_harness.client.calls[0].args[2], models.AddMemoriesRequest)
    assert text_body.text == "remember"
    assert text_body.messages is None
    assert text_body.metadata == {"kind": "text"}
    assert text_body.scope.agent_id == "agent-1"
    assert text_body.scope.session_id == "session-1"
    assert text_body.scope.user_id == "user-1"

    message_harness = _harness()
    await message_harness.store.add_memories(
        scope=scope,
        messages=[
            MemoryMessage(
                role="user",
                content="hello",
            )
        ],
    )
    message_body = message_harness.client.calls[0].args[2].body

    assert message_body.text is None
    assert len(message_body.messages) == 1
    assert message_body.messages[0].to_map() == {
        "content": "hello",
        "role": "user",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (None, None),
        (MemoryScope(), {}),
        (MemoryScope(user_id="user-1"), {"userId": "user-1"}),
        (MemoryScope(agent_id="agent-1"), {"agentId": "agent-1"}),
        (MemoryScope(session_id="session-1"), {"sessionId": "session-1"}),
    ],
)
async def test_add_allows_omitted_and_independent_scope_fields(
    scope: MemoryScope | None,
    expected: dict[str, str] | None,
) -> None:
    harness = _harness()

    await harness.store.add_memories(scope=scope, text="remember")

    generated_scope = harness.client.calls[0].args[2].body.scope
    if expected is None:
        assert generated_scope is None
    else:
        assert generated_scope.to_map() == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (None, None),
        (MemoryScope(), (None, None, None)),
        (MemoryScope(user_id="user-1"), ("user-1", None, None)),
        (MemoryScope(agent_id="agent-1"), (None, "agent-1", None)),
        (MemoryScope(session_id="session-1"), (None, None, "session-1")),
        (
            MemoryScope(
                user_id="user-1",
                agent_id="agent-1",
                session_id="session-1",
            ),
            ("user-1", "agent-1", "session-1"),
        ),
    ],
)
async def test_search_preserves_independent_scope_fields(
    scope: MemoryScope | None,
    expected: tuple[str | None, str | None, str | None] | None,
) -> None:
    harness = _harness()

    await harness.store.search_memories("query", scope=scope, top_k=50)

    request = harness.client.calls[0].args[2]
    assert isinstance(request, models.SearchMemoriesRequest)
    assert request.body.query == "query"
    assert request.body.top_k == 50
    generated_scope = request.body.scope
    if expected is None:
        assert generated_scope is None
    else:
        assert (
            generated_scope.user_id,
            generated_scope.agent_id,
            generated_scope.session_id,
        ) == expected
    assert not hasattr(request, "next_token")


@pytest.mark.asyncio
async def test_search_maps_all_optional_body_fields() -> None:
    harness = _harness()

    await harness.store.search_memories(
        "query",
        metadata={"source": "test"},
        enable_rerank=False,
        min_similarity=-0.1,
        min_score=1.1,
    )

    body = harness.client.calls[0].args[2].body
    assert body.to_map() == {
        "enableRerank": False,
        "metadata": {"source": "test"},
        "minScore": 1.1,
        "minSimilarity": -0.1,
        "query": "query",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "agent_id", "session_id"),
    [
        (None, None, None),
        ("user-1", None, None),
        (None, "agent-1", None),
        (None, None, "session-1"),
        ("user-1", "agent-1", "session-1"),
    ],
)
async def test_list_memories_maps_query_fields_and_omissions(
    user_id: str | None,
    agent_id: str | None,
    session_id: str | None,
) -> None:
    harness = _harness()

    await harness.store.list_memories(
        user_id=user_id,
        agent_id=agent_id,
        session_id=session_id,
        max_results=100,
        next_token="opaque-token",
    )

    request = harness.client.calls[0].args[2]
    assert isinstance(request, models.ListMemoriesRequest)
    assert request.user_id == user_id
    assert request.agent_id == agent_id
    assert request.session_id == session_id
    assert request.max_results == 100
    assert request.next_token == "opaque-token"


@pytest.mark.asyncio
async def test_path_and_update_requests_never_include_scope() -> None:
    harness = _harness()

    await harness.store.get_memory("memory-get")
    await harness.store.update_memory("memory-update", metadata={})
    await harness.store.delete_memory("memory-delete")

    get_call, update_call, delete_call = harness.client.calls
    assert get_call.args[2] == "memory-get"
    assert isinstance(get_call.args[3], models.GetMemoryRequest)
    assert not hasattr(get_call.args[3], "scope")
    assert update_call.args[2] == "memory-update"
    assert update_call.args[3].body.text is None
    assert update_call.args[3].body.metadata == {}
    assert not hasattr(update_call.args[3].body, "scope")
    assert delete_call.args[2] == "memory-delete"
    assert isinstance(delete_call.args[3], models.DeleteMemoryRequest)
    assert not hasattr(delete_call.args[3], "scope")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "metadata"),
    [
        ("text only", None),
        (None, {"mode": "metadata-only"}),
        ("both", {"mode": "both"}),
        (None, {}),
    ],
)
async def test_update_sends_only_fields_explicitly_provided(
    text: str | None,
    metadata: dict[str, str] | None,
) -> None:
    harness = _harness()

    await harness.store.update_memory("memory-update", text=text, metadata=metadata)

    body = harness.client.calls[0].args[3].body
    assert body.text == text
    assert body.metadata == metadata
    mapped = body.to_map()
    assert ("text" in mapped) is (text is not None)
    assert ("metadata" in mapped) is (metadata is not None)


@pytest.mark.asyncio
async def test_session_and_message_list_query_and_path_parameters_are_exact() -> None:
    harness = _harness()

    await harness.store.list_memory_sessions(
        user_id="user-1",
        agent_id="agent-1",
        max_results=12,
        next_token="sessions-token",
    )
    await harness.store.list_memory_session_messages(
        "session-1",
        user_id="user-1",
        agent_id="agent-1",
        max_results=13,
        next_token="messages-token",
    )

    sessions_request = harness.client.calls[0].args[2]
    assert sessions_request.to_map() == {
        "agentId": "agent-1",
        "maxResults": 12,
        "nextToken": "sessions-token",
        "userId": "user-1",
    }
    messages_call = harness.client.calls[1]
    assert messages_call.args[2].to_map() == {
        "agentId": "agent-1",
        "maxResults": 13,
        "nextToken": "messages-token",
        "sessionId": "session-1",
        "userId": "user-1",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_id", "agent_id", "expected"),
    [
        ("user-1", None, {"sessionId": "session-1", "userId": "user-1"}),
        (None, "agent-1", {"agentId": "agent-1", "sessionId": "session-1"}),
        (
            "user-1",
            "agent-1",
            {
                "agentId": "agent-1",
                "sessionId": "session-1",
                "userId": "user-1",
            },
        ),
    ],
)
async def test_message_list_requires_one_identity_and_omits_the_other(
    user_id: str | None,
    agent_id: str | None,
    expected: dict[str, str],
) -> None:
    harness = _harness()

    await harness.store.list_memory_session_messages(
        "session-1",
        user_id=user_id,
        agent_id=agent_id,
    )

    assert harness.client.calls[0].args[2].to_map() == expected


@pytest.mark.asyncio
async def test_response_projection_preserves_omitted_default_scope_fields() -> None:
    responses = _success_responses()
    responses["GetMemory"].body.data.scope = models.GetMemoryResponseBodyDataScope(
        user_id="user-1"
    )
    responses["ListMemorySessions"].body.items = [
        models.ListMemorySessionsResponseBodyItems(user_id="user-1"),
        models.ListMemorySessionsResponseBodyItems(agent_id="agent-1"),
        models.ListMemorySessionsResponseBodyItems(session_id="session-1"),
    ]
    harness = _harness(responses)

    memory = await harness.store.get_memory("memory-get")
    sessions = await harness.store.list_memory_sessions()

    assert memory.scope == MemoryScope(user_id="user-1")
    assert sessions.items == (
        MemorySession(user_id="user-1"),
        MemorySession(agent_id="agent-1"),
        MemorySession(session_id="session-1"),
    )


@pytest.mark.asyncio
async def test_empty_add_search_and_pages_preserve_contractual_empty_results() -> None:
    responses = {
        "AddMemories": models.AddMemoriesResponse(
            status_code=200,
            body=models.AddMemoriesResponseBody(
                success=True,
                data=models.AddMemoriesResponseBodyData(memories=[]),
            ),
        ),
        "SearchMemories": models.SearchMemoriesResponse(
            status_code=200,
            body=models.SearchMemoriesResponseBody(
                success=True,
                data=models.SearchMemoriesResponseBodyData(memories=[]),
            ),
        ),
        "ListMemories": models.ListMemoriesResponse(
            status_code=200,
            body=models.ListMemoriesResponseBody(
                success=True,
                items=[],
                next_token="continue-after-empty-page",
                total_count=None,
            ),
        ),
    }
    harness = _harness(responses)
    scope = MemoryScope(agent_id="agent", session_id="session")

    added = await harness.store.add_memories(scope=scope, text="text")
    searched = await harness.store.search_memories("query")
    page = await harness.store.list_memories()

    assert added.memory_ids == ()
    assert searched.memories == ()
    assert page.items == ()
    assert page.next_token == "continue-after-empty-page"
    assert page.total_count is None


@pytest.mark.asyncio
async def test_unknown_generated_response_fields_are_ignored() -> None:
    responses = _success_responses()
    get_response = responses["GetMemory"]
    get_response.body.future_envelope_field = "ignored"
    get_response.body.data.future_memory_field = {"nested": "ignored"}
    list_response = responses["ListMemories"]
    list_response.body.future_page_field = "ignored"
    list_response.body.items[0].future_item_field = "ignored"
    harness = _harness(responses)

    memory = await harness.store.get_memory("memory-get")
    page = await harness.store.list_memories()

    assert memory.memory_id == "memory-get"
    assert page.items[0].memory_id == "memory-list"


@pytest.mark.asyncio
@pytest.mark.parametrize("next_token", [None, ""])
async def test_page_terminal_tokens_normalize_to_none(next_token: str | None) -> None:
    harness = _harness(
        {
            "ListMemories": models.ListMemoriesResponse(
                status_code=200,
                body=models.ListMemoriesResponseBody(
                    success=True,
                    items=[],
                    next_token=next_token,
                ),
            )
        }
    )

    page = await harness.store.list_memories()

    assert page.next_token is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        None,
        SimpleNamespace(status_code=200, body=None),
        SimpleNamespace(status_code=200, body=SimpleNamespace(success=None)),
        models.GetMemoryResponse(
            status_code=200,
            body=models.GetMemoryResponseBody(success=True, data=None),
        ),
        models.GetMemoryResponse(
            status_code=200,
            body=models.GetMemoryResponseBody(
                success=True,
                data=models.GetMemoryResponseBodyData(
                    memory_id="memory-1",
                    content=None,
                    scope=models.GetMemoryResponseBodyDataScope(
                        agent_id="agent",
                        session_id="session",
                    ),
                ),
            ),
        ),
    ],
)
async def test_malformed_get_success_response_is_contract_error(response: Any) -> None:
    harness = _harness({"GetMemory": response})

    with pytest.raises(MemoryContractError):
        await harness.store.get_memory("memory-1")


@pytest.mark.asyncio
async def test_malformed_add_success_response_is_outcome_unknown() -> None:
    response = models.AddMemoriesResponse(
        status_code=200,
        body=models.AddMemoriesResponseBody(
            success=True,
            request_id="request-malformed-add",
            data=models.AddMemoriesResponseBodyData(
                memories=[models.AddMemoriesResponseBodyDataMemories(memory_id=None)]
            ),
        ),
    )
    harness = _harness({"AddMemories": response})

    with pytest.raises(AddMemoriesOutcomeUnknownError) as raised:
        await harness.store.add_memories(
            scope=MemoryScope(agent_id="agent", session_id="session"),
            text="text",
        )

    assert raised.value.request_id == "request-malformed-add"


def _failure_response(status: int) -> Any:
    return SimpleNamespace(
        status_code=200,
        body=SimpleNamespace(
            success=False,
            code="BusinessFailed",
            message="SECRET_SERVICE_MESSAGE",
            http_status_code=status,
            request_id="request-business",
        ),
    )


def _tea_error(status: int) -> TeaException:
    return TeaException(
        {
            "code": "ServiceError",
            "message": "SECRET_SERVICE_MESSAGE",
            "data": {"statusCode": status, "requestId": f"request-{status}"},
        }
    )


OperationCall = Callable[[AsyncMemoryStore], Awaitable[Any]]


_OPERATIONS: list[tuple[str, OperationCall]] = [
    (
        "AddMemories",
        lambda store: store.add_memories(
            scope=MemoryScope(agent_id="agent", session_id="session"), text="text"
        ),
    ),
    ("SearchMemories", lambda store: store.search_memories("query")),
    ("ListMemories", lambda store: store.list_memories()),
    ("GetMemory", lambda store: store.get_memory("memory")),
    ("UpdateMemory", lambda store: store.update_memory("memory", text="text")),
    ("DeleteMemory", lambda store: store.delete_memory("memory")),
    ("ListMemorySessions", lambda store: store.list_memory_sessions()),
    (
        "ListMemorySessionMessages",
        lambda store: store.list_memory_session_messages("session", agent_id="agent"),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("operation", "invoke"), _OPERATIONS)
@pytest.mark.parametrize("failure_kind", ["transport", "4xx", "5xx", "business"])
async def test_each_action_sends_once_for_every_failure_class(
    operation: str,
    invoke: OperationCall,
    failure_kind: str,
) -> None:
    if failure_kind == "transport":
        response: Any = TimeoutError("SECRET_TIMEOUT_DETAIL")
    elif failure_kind == "4xx":
        response = _tea_error(400)
    elif failure_kind == "5xx":
        response = _tea_error(503)
    else:
        response = _failure_response(409)
    harness = _harness({operation: response})
    expected_error = (
        AddMemoriesOutcomeUnknownError
        if operation == "AddMemories" and failure_kind in {"transport", "5xx"}
        else MemoryAPIError
    )

    with pytest.raises(expected_error) as raised:
        await invoke(harness.store)

    assert [call.operation for call in harness.client.calls] == [operation]
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert "SECRET" not in str(raised.value)
    assert "SECRET" not in repr(raised.value)


@pytest.mark.asyncio
async def test_api_error_keeps_only_safe_service_fields_and_logs_no_payload(
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker_values = [
        "SECRET_TEXT",
        "SECRET_QUERY",
        "SECRET_METADATA",
        "SECRET_SERVICE_MESSAGE",
        "fake-ak",
        "fake-sk",
        "fake-token",
    ]
    harness = _harness({"SearchMemories": _tea_error(403)})

    with caplog.at_level("WARNING"):
        with pytest.raises(MemoryAPIError) as raised:
            await harness.store.search_memories(
                "SECRET_QUERY",
                scope=MemoryScope(agent_id="agent", session_id="session"),
            )

    assert raised.value.service_code == "ServiceError"
    assert raised.value.http_status_code == 403
    assert raised.value.request_id == "request-403"
    combined = "\n".join(
        [str(raised.value), repr(raised.value), caplog.text, repr(vars(raised.value))]
    )
    assert all(marker not in combined for marker in marker_values)


@pytest.mark.asyncio
async def test_get_not_found_preserves_public_error_identity() -> None:
    error = TeaException(
        {
            "code": "Resource.NotFound",
            "message": "memory details must stay private",
            "data": {"statusCode": 404, "requestId": "request-not-found"},
        }
    )
    harness = _harness({"GetMemory": error})

    with pytest.raises(MemoryAPIError) as raised:
        await harness.store.get_memory("missing-memory")

    assert raised.value.service_code == "Resource.NotFound"
    assert raised.value.http_status_code == 404
    assert raised.value.request_id == "request-not-found"
    assert "memory details" not in str(raised.value)


@pytest.mark.asyncio
async def test_nested_generated_error_preserves_safe_direct_fields() -> None:
    inner = RuntimeError("SECRET_INNER_DETAIL")
    inner.code = "DependencyServiceError"  # type: ignore[attr-defined]
    inner.status_code = 502  # type: ignore[attr-defined]
    inner.request_id = "request-direct-fields"  # type: ignore[attr-defined]
    outer = RuntimeError("SECRET_OUTER_DETAIL")
    outer.inner_exception = inner  # type: ignore[attr-defined]
    harness = _harness({"AddMemories": outer})

    with pytest.raises(AddMemoriesOutcomeUnknownError) as raised:
        await harness.store.add_memories(
            scope=MemoryScope(agent_id="agent", session_id="session"),
            text="text",
        )

    assert raised.value.service_code == "DependencyServiceError"
    assert raised.value.http_status_code == 502
    assert raised.value.request_id == "request-direct-fields"
    assert "SECRET" not in str(raised.value)
    assert "SECRET" not in repr(raised.value)


@pytest.mark.asyncio
async def test_failure_logging_and_exception_objects_do_not_retain_request_payloads(
    caplog: pytest.LogCaptureFixture,
) -> None:
    markers = {
        "SECRET_TEXT",
        "SECRET_MESSAGE_CONTENT",
        "SECRET_QUERY",
        "SECRET_METADATA_VALUE",
        "SECRET_SERVICE_MESSAGE",
        "fake-ak",
        "fake-sk",
        "fake-token",
    }
    failures: list[MemoryAPIError] = []
    cases: list[tuple[str, OperationCall]] = [
        (
            "AddMemories",
            lambda store: store.add_memories(
                scope=MemoryScope(agent_id="agent", session_id="session"),
                messages=[
                    MemoryMessage(
                        role="user",
                        content="SECRET_MESSAGE_CONTENT",
                    )
                ],
                metadata={"key": "SECRET_METADATA_VALUE"},
            ),
        ),
        (
            "SearchMemories",
            lambda store: store.search_memories("SECRET_QUERY"),
        ),
        (
            "UpdateMemory",
            lambda store: store.update_memory(
                "memory",
                text="SECRET_TEXT",
                metadata={"key": "SECRET_METADATA_VALUE"},
            ),
        ),
    ]

    with caplog.at_level("WARNING"):
        for operation, invoke in cases:
            harness = _harness({operation: _tea_error(400)})
            with pytest.raises(MemoryAPIError) as raised:
                await invoke(harness.store)
            failures.append(raised.value)

    exposed = caplog.text + "\n".join(
        str(error) + repr(error) + repr(vars(error)) for error in failures
    )
    assert all(marker not in exposed for marker in markers)
    assert all(error.__cause__ is None and error.__context__ is None for error in failures)


def test_default_generated_client_uses_runtime_region_endpoint_and_credentials() -> None:
    credential = FakeSTS().credential
    runtime = _MemoryRuntime(
        workspace_id="workspace-runtime",
        region_id="cn-shanghai",
        endpoint="https://agentcore-memory.example.com/",
        resource_sts=None,
    )

    client = _MemoryTransport._new_client(runtime, credential)

    assert client._region_id == "cn-shanghai"
    assert client._endpoint == "agentcore-memory.example.com"
    assert client._protocol == "https"


@pytest.mark.asyncio
async def test_explicit_access_key_takes_precedence_over_automatic_sts() -> None:
    client = RecordingClient()
    sts = FakeSTS()
    credential = AccessKeyCredential("explicit-ak", "explicit-sk", "explicit-token")
    runtime = _MemoryRuntime(
        workspace_id="workspace-runtime",
        region_id="cn-hangzhou",
        endpoint=None,
        resource_sts=sts,
        access_key_credential=credential,
    )
    received: list[AccessKeyCredential | ResourceCredential] = []

    async def runtime_provider() -> _MemoryRuntime:
        return runtime

    store = AsyncMemoryStore(
        "customer_memory",
        _runtime_provider=runtime_provider,
        _client_factory=lambda resolved, value: (received.append(value), client)[1],
    )

    await store.list_memories()

    assert received == [credential]
    assert sts.purposes == []


@pytest.mark.parametrize(
    "credential",
    [
        AccessKeyCredential("explicit-ak", "explicit-sk"),
        AccessKeyCredential("explicit-ak", "explicit-sk", "explicit-token"),
    ],
)
def test_default_generated_client_supports_long_term_access_key_and_static_sts(
    credential: AccessKeyCredential,
) -> None:
    runtime = _MemoryRuntime(
        workspace_id="workspace-runtime",
        region_id="cn-hangzhou",
        endpoint=None,
        resource_sts=None,
        access_key_credential=credential,
    )

    client = _MemoryTransport._new_client(runtime, credential)
    resolved = client._credential.get_credential()

    assert resolved.access_key_id == "explicit-ak"
    assert resolved.access_key_secret == "explicit-sk"
    assert resolved.security_token == credential.security_token
