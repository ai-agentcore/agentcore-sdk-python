from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any

import pytest
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


def _wire(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {
            key.split("_")[0] + "".join(p.title() for p in key.split("_")[1:]): _wire(item)
            for key, item in vars(value).items()
        }
    if isinstance(value, list):
        return [_wire(item) for item in value]
    return value


@dataclass(frozen=True)
class OpenAPICall:
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
        self.calls: list[OpenAPICall] = []

    async def _call(self, operation: str, *args: Any) -> Any:
        self.calls.append(OpenAPICall(operation, args))
        result = self.responses[operation]
        if isinstance(result, Exception):
            raise result
        return _wire(result)

    async def call_api_async(self, params: Any, request: Any, runtime: Any) -> Any:
        return await self._call(params.action, params, request, runtime)


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
    factory_calls: list[tuple[_MemoryRuntime, AccessKeyCredential | ResourceCredential]] = []

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
    get_memory = SimpleNamespace(
        memory_id="memory-get",
        content=SimpleNamespace(text="get text"),
        scope=_memory_scope(SimpleNamespace),
        metadata={"source": "get"},
        created_at="2026-09-02T01:00:00Z",
        updated_at="2026-09-02T02:00:00Z",
    )
    updated_memory = SimpleNamespace(
        memory_id="memory-update",
        content=SimpleNamespace(text="updated text"),
        scope=_memory_scope(SimpleNamespace),
        metadata={"source": "update"},
        created_at="2026-09-02T01:00:00Z",
        updated_at="2026-09-02T03:00:00Z",
    )
    listed_memory = SimpleNamespace(
        memory_id="memory-list",
        content=SimpleNamespace(text="listed text"),
        scope=_memory_scope(SimpleNamespace),
        metadata={"source": "list"},
        created_at="2026-09-02T01:00:00Z",
        updated_at=None,
    )
    searched_memory = SimpleNamespace(
        memory_id="memory-search",
        content=SimpleNamespace(text="searched text"),
        scope=_memory_scope(SimpleNamespace),
        metadata={"source": "search"},
    )
    return {
        "AddMemories": SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(
                success=True,
                http_status_code=200,
                request_id="request-add",
                data=SimpleNamespace(memories=[SimpleNamespace(memory_id="memory-add")]),
            ),
        ),
        "SearchMemories": SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(
                success=True,
                http_status_code=200,
                data=SimpleNamespace(
                    memories=[
                        SimpleNamespace(
                            memory=searched_memory,
                            score=0.95,
                            similarity=0.85,
                        )
                    ]
                ),
            ),
        ),
        "ListMemories": SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(
                success=True,
                http_status_code=200,
                items=[listed_memory],
                max_results=10,
                next_token="next-memory-page",
                total_count=3,
            ),
        ),
        "GetMemory": SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(
                success=True,
                http_status_code=200,
                data=get_memory,
            ),
        ),
        "UpdateMemory": SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(
                success=True,
                http_status_code=200,
                data=updated_memory,
            ),
        ),
        "DeleteMemory": SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(success=True, http_status_code=200),
        ),
        "ListMemorySessions": SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(
                success=True,
                http_status_code=200,
                items=[
                    SimpleNamespace(
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
        "ListMemorySessionMessages": SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(
                success=True,
                http_status_code=200,
                items=[
                    SimpleNamespace(
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
    assert all(
        call.args[0].pathname.startswith(
            "/workspaces/workspace-runtime/memorystores/customer_memory/"
        )
        for call in harness.client.calls
    )
    for index, call in enumerate(harness.client.calls):
        runtime_options = call.args[-1]
        assert runtime_options.autoretry is False
        assert runtime_options.max_attempts == 1
        assert runtime_options.read_timeout == (120_000 if index == 0 else None)
        assert call.args[1].headers == {}


@pytest.mark.asyncio
async def test_add_text_and_messages_map_to_wire_body_without_cross_mode_fields() -> None:
    scope = MemoryScope(agent_id="agent-1", session_id="session-1", user_id="user-1")
    text_harness = _harness()
    await text_harness.store.add_memories(
        scope=scope,
        text="remember",
        metadata={"kind": "text"},
    )
    text_body = json.loads(text_harness.client.calls[0].args[1].body["body"])

    assert text_body.get("text") == "remember"
    assert text_body.get("messages") is None
    assert text_body["metadata"] == {"kind": "text"}
    assert text_body["scope"]["agentId"] == "agent-1"
    assert text_body["scope"]["sessionId"] == "session-1"
    assert text_body["scope"]["userId"] == "user-1"

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
    message_body = json.loads(message_harness.client.calls[0].args[1].body["body"])

    assert message_body.get("text") is None
    assert len(message_body["messages"]) == 1
    assert message_body["messages"][0] == {
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

    wire_scope = json.loads(harness.client.calls[0].args[1].body["body"]).get("scope")
    if expected is None:
        assert wire_scope is None
    else:
        assert wire_scope == expected


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

    request = harness.client.calls[0].args[1]
    body = json.loads(request.body["body"])
    assert body["query"] == "query"
    assert body["topK"] == 50
    wire_scope = body.get("scope")
    if expected is None:
        assert wire_scope is None
    else:
        assert (
            wire_scope.get("userId"),
            wire_scope.get("agentId"),
            wire_scope.get("sessionId"),
        ) == expected
    assert "nextToken" not in body


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

    body = json.loads(harness.client.calls[0].args[1].body["body"])
    assert body == {
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

    request = harness.client.calls[0].args[1]
    assert request.query.get("userId") == user_id
    assert request.query.get("agentId") == agent_id
    assert request.query.get("sessionId") == session_id
    assert request.query.get("maxResults") == "100"
    assert request.query.get("nextToken") == "opaque-token"


@pytest.mark.asyncio
async def test_path_and_update_requests_never_include_scope() -> None:
    harness = _harness()

    await harness.store.get_memory("memory-get")
    await harness.store.update_memory("memory-update", metadata={})
    await harness.store.delete_memory("memory-delete")

    get_call, update_call, delete_call = harness.client.calls
    assert get_call.args[0].pathname.endswith("/memories/memory-get")
    assert get_call.args[1].to_map() == {"headers": {}}
    assert update_call.args[0].pathname.endswith("/memories/memory-update")
    assert json.loads(update_call.args[1].body["body"]) == {"metadata": {}}
    assert delete_call.args[0].pathname.endswith("/memories/memory-delete")
    assert delete_call.args[1].to_map() == {"headers": {}}


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

    body = json.loads(harness.client.calls[0].args[1].body["body"])
    assert body.get("text") == text
    assert body.get("metadata") == metadata
    assert ("text" in body) is (text is not None)
    assert ("metadata" in body) is (metadata is not None)


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

    sessions_request = harness.client.calls[0].args[1]
    assert sessions_request.query == {
        "agentId": "agent-1",
        "maxResults": "12",
        "nextToken": "sessions-token",
        "userId": "user-1",
    }
    messages_call = harness.client.calls[1]
    assert messages_call.args[1].query == {
        "agentId": "agent-1",
        "maxResults": "13",
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

    assert harness.client.calls[0].args[1].query == expected


@pytest.mark.asyncio
async def test_response_projection_preserves_omitted_default_scope_fields() -> None:
    responses = _success_responses()
    responses["GetMemory"].body.data.scope = SimpleNamespace(user_id="user-1")
    responses["ListMemorySessions"].body.items = [
        SimpleNamespace(user_id="user-1"),
        SimpleNamespace(agent_id="agent-1"),
        SimpleNamespace(session_id="session-1"),
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
        "AddMemories": SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(
                success=True,
                data=SimpleNamespace(memories=[]),
            ),
        ),
        "SearchMemories": SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(
                success=True,
                data=SimpleNamespace(memories=[]),
            ),
        ),
        "ListMemories": SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(
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
async def test_unknown_response_fields_are_ignored() -> None:
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
            "ListMemories": SimpleNamespace(
                status_code=200,
                body=SimpleNamespace(
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
        SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(success=True, data=None),
        ),
        SimpleNamespace(
            status_code=200,
            body=SimpleNamespace(
                success=True,
                data=SimpleNamespace(
                    memory_id="memory-1",
                    content=None,
                    scope=SimpleNamespace(
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
    response = SimpleNamespace(
        status_code=200,
        body=SimpleNamespace(
            success=True,
            request_id="request-malformed-add",
            data=SimpleNamespace(memories=[SimpleNamespace(memory_id=None)]),
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
