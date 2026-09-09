from __future__ import annotations

import asyncio
import base64
import inspect
import json
import threading
import time
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from typing import Any

import anyio
import anyio.lowlevel
import httpx
import pytest

from agentcore.auth import AccessKeyCredential
from agentcore.client import AgentCore, AsyncAgentCore
from agentcore.controlplane import MCPDescriptor, ModelDescriptor
from agentcore.errors import ConfigError
from agentcore.integrations.common import Tool
from agentcore.runtime.startup import DebugRuntimeSource


class FakeModel:
    async def invoke(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        return {"messages": messages, **kwargs}

    async def stream(self, _messages, **_kwargs) -> AsyncIterator[dict[str, str]]:
        yield {"content": "a"}
        yield {"content": "b"}

    async def completion(self, messages, **kwargs):  # type: ignore[no-untyped-def]
        return {"operation": "completion", "messages": messages, **kwargs}

    async def responses(self, input, **kwargs):  # type: ignore[no-untyped-def,redefined-builtin]
        return {"operation": "responses", "input": input, **kwargs}

    async def responses_stream(self, _input: Any, **_kwargs: Any) -> AsyncIterator[dict[str, str]]:
        yield {"type": "response.output_text.delta", "delta": "a"}
        yield {"type": "response.output_text.delta", "delta": "b"}

    async def embedding(self, input, **kwargs):  # type: ignore[no-untyped-def,redefined-builtin]
        return {"operation": "embedding", "input": input, **kwargs}


class FakeMCP:
    async def tools(self) -> list[Tool]:
        return [
            Tool(
                name="echo",
                description="Echo",
                parameters={"type": "object", "properties": {"value": {"type": "string"}}},
                _call=lambda arguments: self.call_tool("echo", arguments),
            )
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return {"name": name, "arguments": arguments}


class FakeSkills:
    async def local(self, root: str) -> list[str]:
        return [root]

    async def managed(self, name: str, *, version: str | None = None) -> str:
        return f"{name}@{version}"


class FakeAsyncCore:
    def __init__(self) -> None:
        self.skills = FakeSkills()
        self.credentials = None
        self.closed = False

    async def model(self, resource_name: str, *, model: str | None = None) -> FakeModel:
        return FakeModel()

    async def mcp(self, _name: str, *, headers: Any = None, credential_name: Any = None) -> FakeMCP:
        return FakeMCP()

    def direct_mcp(self, **_kwargs: Any) -> FakeMCP:
        return FakeMCP()

    def direct_model(self, **_kwargs: Any) -> FakeModel:
        return FakeModel()

    async def aclose(self) -> None:
        self.closed = True


def _sync_core(factory):  # type: ignore[no-untyped-def]
    core = AgentCore()
    core._factory = factory
    return core


def _runtime_env(tmp_path: Any, *, control_endpoint: str | None = None) -> Any:
    token_file = tmp_path / "token"
    token_file.write_text("sa-token", encoding="utf-8")
    env_path = tmp_path / "env"
    lines = [
        "export AGENTTEAMS_CONTROLLER_URL='https://controller.example.com'",
        f"export AGENTTEAMS_AUTH_TOKEN_FILE='{token_file}'",
    ]
    if control_endpoint is not None:
        lines.append(f"export AGENTCORE_CONTROL_ENDPOINT='{control_endpoint}'")
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return env_path


def test_sync_client_uses_one_portal_for_calls_and_streaming() -> None:
    core = _sync_core(FakeAsyncCore)
    with core:
        response = core.model("service").invoke([{"role": "user"}], temperature=0)
        chunks = list(core.model("service").stream([]))
        direct = core.direct_model(
            model="qwen-plus",
            base_url="https://provider.example.com/v1",
            api_key="secret",
        )
        completion = direct.completion([])
        responses = direct.responses("hello")
        response_events = list(direct.responses_stream("hello"))
        embedding = direct.embedding(["hello"])
        tools = core.mcp("search").tools()
        tool_result = tools[0].invoke({"value": "tool"})
        async_tool_result = anyio.run(tools[0].ainvoke, {"value": "async-tool"})
        managed = core.skills.managed("review", version="1.0.0")

    assert response["temperature"] == 0
    assert chunks == [{"content": "a"}, {"content": "b"}]
    assert completion == {"operation": "completion", "messages": []}
    assert responses == {"operation": "responses", "input": "hello"}
    assert response_events == [
        {"type": "response.output_text.delta", "delta": "a"},
        {"type": "response.output_text.delta", "delta": "b"},
    ]
    assert embedding == {"operation": "embedding", "input": ["hello"]}
    assert tool_result == {
        "name": "echo",
        "arguments": {"value": "tool"},
    }
    assert async_tool_result == {
        "name": "echo",
        "arguments": {"value": "async-tool"},
    }
    assert managed == "review@1.0.0"


def test_langchain_sync_tool_reuses_agentcore_portal() -> None:
    pytest.importorskip("langchain_core")
    from agentcore.integrations.langchain import tools as langchain_tools

    core = _sync_core(FakeAsyncCore)
    with core:
        adapted = langchain_tools(core.mcp("search").tools())[0]
        result = adapted.invoke({"value": "tool"})

    assert result == {"name": "echo", "arguments": {"value": "tool"}}


def test_sync_client_initializes_one_core_under_concurrent_first_use() -> None:
    created = 0
    created_lock = threading.Lock()
    barrier = threading.Barrier(12)

    def factory() -> FakeAsyncCore:
        nonlocal created
        with created_lock:
            created += 1
        time.sleep(0.02)
        return FakeAsyncCore()

    core = _sync_core(factory)
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            barrier.wait()
            core.model("service").invoke([])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=invoke) for _ in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    core.close()

    assert errors == []
    assert created == 1


def test_sync_client_allows_concurrent_calls_after_single_initialization() -> None:
    class ConcurrentModel:
        def __init__(self) -> None:
            self.active = 0
            self.both_entered = anyio.Event()

        async def invoke(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.active += 1
            if self.active == 2:
                self.both_entered.set()
            with anyio.fail_after(1):
                await self.both_entered.wait()
            return {"messages": messages, **kwargs}

    class ConcurrentCore(FakeAsyncCore):
        def __init__(self) -> None:
            super().__init__()
            self.concurrent_model = ConcurrentModel()

        async def model(self, resource_name: str, *, model: str | None = None) -> ConcurrentModel:
            return self.concurrent_model

    core = _sync_core(ConcurrentCore)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def invoke() -> None:
        try:
            model = core.model("service")
            barrier.wait()
            model.invoke([])
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=invoke) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    core.close()

    assert errors == []


def test_sync_client_can_retry_after_factory_initialization_failure() -> None:
    attempts = 0

    def factory() -> FakeAsyncCore:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("initialization failed")
        return FakeAsyncCore()

    core = _sync_core(factory)

    with pytest.raises(RuntimeError, match="initialization failed"):
        core.model("service")

    assert core.model("service").invoke([]) == {"messages": []}
    core.close()
    assert attempts == 2


def test_sync_client_closes_once_under_concurrent_close() -> None:
    class CountingCore(FakeAsyncCore):
        def __init__(self) -> None:
            super().__init__()
            self.close_calls = 0

        async def aclose(self) -> None:
            self.close_calls += 1
            await anyio.sleep(0.02)
            await super().aclose()

    async_core = CountingCore()
    core = _sync_core(lambda: async_core)
    core.model("service")
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def close() -> None:
        try:
            barrier.wait()
            core.close()
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=close) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert async_core.close_calls == 1


def test_sync_client_close_is_terminal() -> None:
    created = 0

    def factory() -> FakeAsyncCore:
        nonlocal created
        created += 1
        return FakeAsyncCore()

    core = _sync_core(factory)
    core.model("service")
    core.close()

    with pytest.raises(RuntimeError, match="closed"):
        core.model("service")

    assert created == 1


@pytest.mark.asyncio
async def test_async_core_close_is_terminal(agent_config_path) -> None:  # type: ignore[no-untyped-def]
    core = AsyncAgentCore(agent_config_path)
    await core.aclose()
    with pytest.raises(RuntimeError, match="closed"):
        await core.model("service")
    with pytest.raises(RuntimeError, match="closed"):
        await core.mcp("search")
    with pytest.raises(RuntimeError, match="closed"):
        core.direct_mcp(url="https://mcp.example.com/mcp")
    with pytest.raises(RuntimeError, match="closed"):
        core.direct_model(
            model="qwen-plus",
            base_url="https://provider.example.com/v1",
        )
    with pytest.raises(RuntimeError, match="closed"):
        _ = core.skills
    with pytest.raises(RuntimeError, match="closed"):
        _ = core.credentials


@pytest.mark.asyncio
async def test_async_core_defers_collaboration_workspace_validation(
    agent_config_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENT_WORKSPACE", "/missing/agentcore-collaboration-workspace")

    core = AsyncAgentCore(agent_config_path)

    with pytest.raises(ConfigError, match="collaboration workspace is unavailable"):
        core.collaboration.worker()
    await core.aclose()


def test_async_core_public_constructor_does_not_expose_test_dependencies() -> None:
    parameters = inspect.signature(AsyncAgentCore).parameters

    assert "config_path" in parameters
    assert "config_provider" not in parameters
    assert {"workspace_id", "region_id", "access_key_credential"}.issubset(parameters)
    assert {"controller_endpoint", "agent_sa_token_file"}.isdisjoint(parameters)
    assert {
        "workload_tokens",
        "resource_sts",
        "model_http_client",
        "mcp_session_factory",
        "registry_transport",
        "bound_credential_transport",
    }.isdisjoint(parameters)
    assert "async_factory" not in inspect.signature(AgentCore).parameters
    assert "_async_factory" not in inspect.signature(AgentCore).parameters


@pytest.mark.asyncio
async def test_async_core_direct_clients_do_not_require_runtime_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("AGENTCORE_DEBUG_TOKEN", raising=False)
    monkeypatch.delenv("AGENTCORE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("AGENTCORE_ENV_PATH", raising=False)

    core = AsyncAgentCore.auto()
    try:
        model = core.direct_model(
            model="local-model",
            base_url="http://127.0.0.1:8001/v1",
        )
        mcp = core.direct_mcp(url="http://127.0.0.1:8002/mcp")

        assert model is not None
        assert mcp is not None
        assert core.config is None
    finally:
        await core.aclose()


@pytest.mark.asyncio
async def test_async_core_defers_explicit_agent_config_validation(tmp_path: Any) -> None:
    config_path = tmp_path / "empty-agent.yaml"
    config_path.write_text("", encoding="utf-8")
    core = AsyncAgentCore.auto(config_path=config_path)
    try:
        core.direct_model(
            model="local-model",
            base_url="http://127.0.0.1:8001/v1",
        )
        with pytest.raises(ConfigError, match="agent.yaml"):
            await core.model("production-model", model="qwen-plus")
    finally:
        await core.aclose()


def test_async_core_explicit_context_uses_access_key_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class ControlPlane:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    monkeypatch.setattr("agentcore.client.AgentCoreControlPlane", ControlPlane)
    credential = AccessKeyCredential("ak", "sk", "sts")
    core = AsyncAgentCore.auto(
        workspace_id="workspace-a",
        region_id="cn-hangzhou",
        access_key_credential=credential,
    )

    try:
        assert captured["workspace_id"] == "workspace-a"
        assert captured["region_id"] == "cn-hangzhou"
        assert captured["access_key_credential"] is credential
        assert captured["resource_sts"] is None
    finally:
        anyio.run(core.aclose)


@pytest.mark.asyncio
async def test_async_core_local_config_uses_access_key_without_controller_sts(
    agent_config_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class ControlPlane:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

    def unexpected_sts(*_args: object, **_kwargs: object) -> None:
        pytest.fail("explicit AccessKey must not allocate automatic Resource STS")

    monkeypatch.setattr("agentcore.client.AgentCoreControlPlane", ControlPlane)
    monkeypatch.setattr("agentcore.client.ResourceSTSProvider", unexpected_sts)
    core = AsyncAgentCore.auto(
        config_path=agent_config_path,
        access_key_credential=AccessKeyCredential("ak", "sk"),
    )
    try:
        await core._ensure_runtime()

        assert captured["workspace_id"] == "workspace-a"
        assert captured["access_key_credential"].security_token is None
        assert captured["resource_sts"] is None
    finally:
        await core.aclose()


def test_async_core_rejects_mixed_explicit_runtime_sources(agent_config_path: Any) -> None:
    with pytest.raises(ValueError, match="cannot be combined"):
        AsyncAgentCore.auto(
            config_path=agent_config_path,
            workspace_id="workspace-a",
            region_id="cn-hangzhou",
        )


@pytest.mark.asyncio
async def test_async_core_single_flights_concurrent_resource_creation(
    agent_config_path: Any,
    model_descriptor: ModelDescriptor,
    mcp_descriptor: MCPDescriptor,
) -> None:
    class ControlPlane:
        model_calls = 0
        mcp_calls = 0

        async def resolve_model(self, resource_name: str, model: str | None):  # type: ignore[no-untyped-def]
            self.model_calls += 1
            await anyio.lowlevel.checkpoint()
            return model_descriptor

        async def resolve_mcp(self, name: str):  # type: ignore[no-untyped-def]
            self.mcp_calls += 1
            await anyio.lowlevel.checkpoint()
            return mcp_descriptor

    core = AsyncAgentCore(agent_config_path)
    await core._ensure_runtime()
    control_plane = ControlPlane()
    core._control_plane = control_plane  # type: ignore[assignment]
    models: list[Any] = []
    mcps: list[Any] = []

    async def create_model() -> None:
        models.append(await core.model("model-service-a", model="qwen-plus"))

    async def create_mcp() -> None:
        mcps.append(await core.mcp("search"))

    async with anyio.create_task_group() as tasks:
        for _ in range(10):
            tasks.start_soon(create_model)
            tasks.start_soon(create_mcp)

    await core.aclose()

    assert len({id(value) for value in models}) == 1
    assert len({id(value) for value in mcps}) == 1
    assert control_plane.model_calls == 1
    assert control_plane.mcp_calls == 1


@pytest.mark.asyncio
async def test_async_core_rejects_explicit_missing_runtime_env_on_managed_use(
    agent_config_path: Any,
) -> None:
    core = AsyncAgentCore(
        agent_config_path,
        env_path=agent_config_path.parent / "missing-env",
    )
    try:
        with pytest.raises(ConfigError, match="runtime env"):
            await core.model("service")
    finally:
        await core.aclose()


@pytest.mark.asyncio
async def test_async_core_validates_control_endpoint_before_allocating_auth_clients(
    monkeypatch: pytest.MonkeyPatch,
    agent_config_path: Any,
    tmp_path: Any,
) -> None:
    env_path = _runtime_env(tmp_path)

    def unexpected_provider(*_args: object, **_kwargs: object) -> None:
        pytest.fail("auth clients must not be allocated before endpoint validation")

    monkeypatch.setattr("agentcore.client.WorkloadAccessTokenProvider", unexpected_provider)
    monkeypatch.setattr("agentcore.client.ResourceSTSProvider", unexpected_provider)

    core = AsyncAgentCore(
        agent_config_path,
        env_path=env_path,
        control_plane_endpoint="https://agentcore.example.com/invalid-path",
    )
    try:
        with pytest.raises(ConfigError, match="control endpoint"):
            await core.model("service")
        assert core.config is None
        assert core._resource_context is None
        with pytest.raises(ConfigError, match="control endpoint"):
            await core.skills.managed("skill")
    finally:
        await core.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("explicit", "expected"),
    [
        (
            "https://explicit-agentcore.example.com",
            "https://explicit-agentcore.example.com",
        ),
        (None, "https://env-agentcore.example.com"),
    ],
)
async def test_async_core_control_endpoint_precedence(
    monkeypatch: pytest.MonkeyPatch,
    agent_config_path: Any,
    tmp_path: Any,
    explicit: str | None,
    expected: str,
) -> None:
    captured: list[str | None] = []
    env_path = _runtime_env(tmp_path)
    monkeypatch.setenv(
        "AGENTCORE_CONTROL_ENDPOINT",
        "https://env-agentcore.example.com",
    )

    def control_plane(**kwargs: Any) -> object:
        captured.append(kwargs["endpoint"])
        return object()

    monkeypatch.setattr("agentcore.client.AgentCoreControlPlane", control_plane)
    core = AsyncAgentCore(
        agent_config_path,
        env_path=env_path,
        control_plane_endpoint=explicit,
    )
    try:
        await core._ensure_runtime()
        assert captured == [expected]
    finally:
        await core.aclose()


@pytest.mark.asyncio
async def test_async_core_debug_runtime_is_lazy_and_single_flight(
    monkeypatch: pytest.MonkeyPatch,
    model_descriptor: ModelDescriptor,
    mcp_descriptor: MCPDescriptor,
) -> None:
    envelope = base64.b64encode(
        json.dumps(
            {
                "product": "agentcore",
                "jwtToken": "bootstrap-jwt",
                "controllerUrl": "https://controller.example.com",
                "modelGatewayUrl": "https://public-model.example.com/v1",
                "matrixUrl": "https://public-matrix.example.com",
            }
        ).encode()
    ).decode()
    monkeypatch.setenv("AGENTCORE_DEBUG_TOKEN", envelope)
    exchanges = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal exchanges
        if request.url.path == "/api/v1/edge/token":
            exchanges += 1
            return httpx.Response(
                200,
                json={
                    "token": "agent-sa-token",
                    "expiresAt": (
                        datetime.now(timezone.utc) + timedelta(hours=1)
                    ).isoformat(),
                    "jwtToken": "refreshed-bootstrap-jwt",
                    "jwtExpiresAt": (
                        datetime.now(timezone.utc) + timedelta(hours=24)
                    ).isoformat(),
                },
            )
        if request.url.path == "/api/v1/credentials/sts":
            return httpx.Response(
                200,
                json={
                    "access_key_id": "sts-ak",
                    "access_key_secret": "sts-sk",
                    "security_token": "sts-token",
                    "expiration": (
                        datetime.now(timezone.utc) + timedelta(hours=1)
                    ).isoformat(),
                    "oss_endpoint": "https://oss-cn-hangzhou.aliyuncs.com",
                    "oss_bucket": "control-bucket",
                    "agent_config_path": "agents/runtime-debug/agent.yaml",
                },
            )
        if request.url.path == "/agents/runtime-debug/agent.yaml":
            return httpx.Response(
                200,
                content=b"""apiVersion: agentteams.io/v1alpha1
kind: AgentConfig
metadata:
  runtimeName: runtime-debug
  workspaceId: workspace-a
  regionId: cn-hangzhou
spec:
  model:
    gatewayUrl: http://private-model.internal/v1
  mcp:
    gatewayUrl: http://private-mcp.internal/mcp
  credentials:
    header:
      - key: Authorization
        value: Bearer gateway-secret
""",
            )
        return httpx.Response(404)

    class ControlPlane:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        async def resolve_model(
            self,
            _resource_name: str,
            _model: str | None,
        ) -> ModelDescriptor:
            return model_descriptor

        async def resolve_mcp(self, _name: str) -> MCPDescriptor:
            return mcp_descriptor

    monkeypatch.setattr("agentcore.client.AgentCoreControlPlane", ControlPlane)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        core = AsyncAgentCore.auto()
        source = core._runtime_source
        assert isinstance(source, DebugRuntimeSource)
        source._http = http
        source._owns_http = False

        core.direct_model(
            model="qwen-plus",
            base_url="https://provider.example.com/v1",
        )
        assert exchanges == 0

        await asyncio.gather(
            core.model("model-service-a", model="qwen-plus"),
            core.mcp("search"),
        )
        assert exchanges == 1
        assert core.config is not None
        assert core.config.spec.model.gateway_url == "https://public-model.example.com/v1"
        await core.aclose()


@pytest.mark.asyncio
async def test_async_core_debug_collaboration_reads_remote_teams_config(
    monkeypatch: pytest.MonkeyPatch,
    teams_config_text: str,
) -> None:
    envelope = base64.b64encode(
        json.dumps(
            {
                "product": "agentcore",
                "jwtToken": "bootstrap-jwt",
                "controllerUrl": "https://controller.example.com",
                "modelGatewayUrl": "https://public-model.example.com/v1",
                "matrixUrl": "https://public-matrix.example.com",
            }
        ).encode()
    ).decode()
    monkeypatch.setenv("AGENTCORE_DEBUG_TOKEN", envelope)
    core = AsyncAgentCore.auto()
    source = core._runtime_source
    assert isinstance(source, DebugRuntimeSource)

    async def load_teams_config() -> bytes:
        return teams_config_text.encode()

    monkeypatch.setattr(source, "load_teams_config", load_teams_config)
    get_context = next(
        tool
        for tool in core.collaboration.worker().tools()
        if tool.name == "agentteams_get_team_context"
    )
    try:
        result = await get_context.ainvoke({})
    finally:
        await core.aclose()

    assert result["ok"] is True
    assert result["data"]["member"]["name"] == "worker-a"
    assert result["data"]["teams"][0]["name"] == "team-alpha"
