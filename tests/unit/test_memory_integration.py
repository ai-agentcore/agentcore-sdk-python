from __future__ import annotations

import inspect
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import anyio
import pytest

import agentcore.client as client_module
import agentcore.memory as memory_api
from agentcore.auth import AccessKeyCredential
from agentcore.auth.resource_sts import ResourceCredential
from agentcore.client import AgentCore, AsyncAgentCore
from agentcore.memory import (
    AddMemoriesResult,
    AsyncMemoryStore,
    Memory,
    MemoryMessage,
    MemoryScope,
    Page,
    SearchMemoriesResult,
)
from agentcore.memory._transport import _MemoryRuntime


class IntegrationSTS:
    def __init__(self) -> None:
        self.purposes: list[str | None] = []

    async def get(self, purpose: str | None = None) -> ResourceCredential:
        self.purposes.append(purpose)
        return ResourceCredential(
            access_key_id="integration-ak",
            access_key_secret="integration-sk",
            security_token="integration-token",
            expiration=datetime.now(timezone.utc) + timedelta(hours=1),
        )


class IntegrationClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []

    async def call_api_async(self, params: Any, request: Any, runtime: Any) -> Any:
        self.calls.append((params.action, (params, request, runtime)))
        body: dict[str, Any] = {"success": True}
        if params.action in {"AddMemories", "SearchMemories"}:
            body["data"] = {"memories": []}
        elif params.action in {"ListMemories", "ListMemorySessions", "ListMemorySessionMessages"}:
            body["items"] = []
        elif params.action in {"GetMemory", "UpdateMemory"}:
            body["data"] = {
                "memoryId": "memory-get" if params.action == "GetMemory" else "memory-update",
                "content": {"text": "stored text"},
                "scope": {"agentId": "agent-1", "sessionId": "session-1", "userId": "user-1"},
            }
        return {"statusCode": 200, "body": body}


@pytest.mark.asyncio
async def test_async_core_explicit_credential_covers_eight_actions_without_runtime_files() -> None:
    credential = AccessKeyCredential(
        access_key_id="explicit-ak",
        access_key_secret="explicit-sk",
        security_token="explicit-token",
    )
    core = AsyncAgentCore.auto(
        workspace_id="workspace-explicit",
        region_id="cn-hangzhou",
        control_plane_endpoint="https://runtime-agentcore.example.com",
        access_key_credential=credential,
    )
    client = IntegrationClient()
    credentials: list[AccessKeyCredential | ResourceCredential] = []

    store = core.memory_store("customer_memory")
    store._transport._client_factory = lambda runtime, value: (
        credentials.append(value),
        client,
    )[1]

    assert core._runtime is None
    assert core.config is None
    assert core._resource_sts is None
    assert client.calls == []

    scope = MemoryScope(agent_id="agent-1", session_id="session-1", user_id="user-1")
    results = [
        await store.add_memories(scope=scope, text="text"),
        await store.search_memories(
            "query",
            scope=MemoryScope(session_id="session-1"),
            metadata={"source": "integration"},
            enable_rerank=False,
            min_similarity=0.0,
            min_score=0.0,
        ),
        await store.list_memories(user_id="user-1", agent_id="agent-1"),
        await store.get_memory("memory-get"),
        await store.update_memory("memory-update", metadata={}),
        await store.delete_memory("memory-delete"),
        await store.list_memory_sessions(user_id="user-1"),
        await store.list_memory_session_messages("session-1", user_id="user-1"),
    ]

    assert core._runtime is None
    assert credentials == [credential] * 8
    assert [operation for operation, _ in client.calls] == [
        "AddMemories",
        "SearchMemories",
        "ListMemories",
        "GetMemory",
        "UpdateMemory",
        "DeleteMemory",
        "ListMemorySessions",
        "ListMemorySessionMessages",
    ]
    assert all(
        args[0].pathname.startswith("/workspaces/workspace-explicit/") for _, args in client.calls
    )
    assert isinstance(results[0], AddMemoriesResult)
    assert isinstance(results[1], SearchMemoriesResult)
    assert isinstance(results[2], Page)
    assert isinstance(results[3], Memory)
    assert isinstance(results[4], Memory)
    assert results[5] is None
    assert isinstance(results[6], Page)
    assert isinstance(results[7], Page)

    await core.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await store.list_memories()
    assert credentials == [credential] * 8


class SyncAsyncCore:
    def __init__(self, client: IntegrationClient | None = None) -> None:
        self.client = client or IntegrationClient()
        self.sts = IntegrationSTS()
        self.closed = False
        self.skills = None
        self.credentials = None

    async def _runtime(self) -> _MemoryRuntime:
        return _MemoryRuntime(
            workspace_id="workspace-sync",
            region_id="cn-hangzhou",
            endpoint=None,
            resource_sts=self.sts,
        )

    def memory_store(self, memory_store_name: str) -> AsyncMemoryStore:
        return AsyncMemoryStore(
            memory_store_name,
            _runtime_provider=self._runtime,
            _client_factory=lambda runtime, credential: self.client,
        )

    async def model(self, resource_name: str, *, model: str | None = None) -> Any:
        raise AssertionError

    def direct_model(self, **kwargs: Any) -> Any:
        raise AssertionError

    async def mcp(self, name: str) -> Any:
        raise AssertionError

    def direct_mcp(self, **kwargs: Any) -> Any:
        raise AssertionError

    async def aclose(self) -> None:
        self.closed = True


def _sync_core(async_core: SyncAsyncCore) -> AgentCore:
    core = AgentCore()
    core._factory = lambda: async_core
    return core


def test_sync_memory_store_reuses_portal_and_returns_async_domain_types(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_start = client_module.start_blocking_portal
    portal_starts = 0

    def counting_start(*args: Any, **kwargs: Any) -> Any:
        nonlocal portal_starts
        portal_starts += 1
        return original_start(*args, **kwargs)

    monkeypatch.setattr(client_module, "start_blocking_portal", counting_start)
    async_core = SyncAsyncCore()
    core = _sync_core(async_core)
    store = core.memory_store("customer_memory")
    portal = core._portal
    scope = MemoryScope(agent_id="agent-1", session_id="session-1", user_id="user-1")

    added = store.add_memories(scope=scope, text="text")
    searched = store.search_memories("query", scope=MemoryScope(session_id="session-1"))
    memories = store.list_memories(agent_id="agent-1")
    fetched = store.get_memory("memory-get")
    updated = store.update_memory("memory-update", metadata={})
    deleted = store.delete_memory("memory-delete")
    sessions = store.list_memory_sessions()
    messages = store.list_memory_session_messages("session-1", user_id="user-1")

    assert core._portal is portal
    assert portal_starts == 1
    assert isinstance(added, AddMemoriesResult)
    assert isinstance(searched, SearchMemoriesResult)
    assert isinstance(memories, Page)
    assert isinstance(fetched, Memory)
    assert isinstance(updated, Memory)
    assert deleted is None
    assert isinstance(sessions, Page)
    assert isinstance(messages, Page)
    assert async_core.sts.purposes == ["highcode_sdk"] * 8
    assert [operation for operation, _ in async_core.client.calls] == [
        "AddMemories",
        "SearchMemories",
        "ListMemories",
        "GetMemory",
        "UpdateMemory",
        "DeleteMemory",
        "ListMemorySessions",
        "ListMemorySessionMessages",
    ]
    core.close()
    assert async_core.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        store.get_memory("memory-get")


def test_sync_close_waits_for_active_memory_call() -> None:
    started = threading.Event()
    release = threading.Event()
    call_done = threading.Event()
    close_done = threading.Event()

    class BlockingClient(IntegrationClient):
        async def call_api_async(self, *args: Any) -> Any:
            started.set()
            await anyio.to_thread.run_sync(release.wait)
            return await super().call_api_async(*args)

    core = _sync_core(SyncAsyncCore(BlockingClient()))
    store = core.memory_store("customer_memory")

    def invoke() -> None:
        store.get_memory("memory-get")
        call_done.set()

    call_thread = threading.Thread(target=invoke)
    close_thread = threading.Thread(target=lambda: (core.close(), close_done.set()))
    call_thread.start()
    assert started.wait(timeout=1)
    close_thread.start()
    time.sleep(0.03)
    assert close_done.is_set() is False
    release.set()
    call_thread.join(timeout=1)
    close_thread.join(timeout=1)

    assert call_done.is_set() is True
    assert close_done.is_set() is True


def test_memory_public_surface_has_only_data_plane_and_no_runtime_overrides() -> None:
    expected_methods = {
        "add_memories",
        "search_memories",
        "list_memories",
        "get_memory",
        "update_memory",
        "delete_memory",
        "list_memory_sessions",
        "list_memory_session_messages",
    }
    control_plane_methods = {
        "create_memory_store",
        "get_memory_store",
        "list_memory_stores",
        "update_memory_store",
        "delete_memory_store",
    }
    forbidden_parameters = {
        "workspace_id",
        "region_id",
        "endpoint",
        "access_key_id",
        "access_key_secret",
        "security_token",
        "client_token",
        "sync",
        "runtime_options",
    }

    assert expected_methods <= set(dir(AsyncMemoryStore))
    assert control_plane_methods.isdisjoint(dir(AsyncMemoryStore))
    for method_name in expected_methods:
        parameters = inspect.signature(getattr(AsyncMemoryStore, method_name)).parameters
        assert forbidden_parameters.isdisjoint(parameters)
    assert set(inspect.signature(AsyncAgentCore.memory_store).parameters) == {
        "self",
        "memory_store_name",
    }
    assert set(inspect.signature(AgentCore.memory_store).parameters) == {
        "self",
        "memory_store_name",
    }
    assert "AsyncMemoryStore" in memory_api.__all__
    assert memory_api.MemoryMessage is MemoryMessage


@pytest.mark.asyncio
async def test_async_binding_after_close_fails_without_resolving_runtime(
    agent_config_path: Path,
) -> None:
    core = AsyncAgentCore(agent_config_path)
    await core.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        core.memory_store("customer_memory")
