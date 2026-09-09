from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from agentcore.auth import AccessKeyCredential
from agentcore.controlplane import AgentCoreControlPlane, ModelDescriptor
from agentcore.errors import ConfigError
from agentcore.integrations import _shared
from agentcore.integrations._shared import (
    framework_completion_parameters,
    framework_model_settings,
    tool_functions,
)
from agentcore.integrations.common import Tool
from agentcore.runtime.config import load_agent_config
from agentcore.runtime.startup import RuntimeBindings


@pytest.fixture
def framework_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_descriptor: ModelDescriptor,
) -> Path:
    token = tmp_path / "token"
    token.write_text("sa-token", encoding="utf-8")
    env = tmp_path / "env"
    env.write_text(
        "export AGENTTEAMS_CONTROLLER_URL='https://controller.example.com'\n"
        f"export AGENTTEAMS_AUTH_TOKEN_FILE='{token}'\n"
        "export AGENTCORE_CONTROL_ENDPOINT="
        "'https://agentcore-vpc.cn-hangzhou.aliyuncs.com'\n",
        encoding="utf-8",
    )

    async def resolve_model(
        self: AgentCoreControlPlane,
        resource_name: str,
        model_name: str | None = None,
    ) -> ModelDescriptor:
        assert resource_name == "model-service-a"
        assert model_name in {None, "qwen-plus"}
        assert self._endpoint == "agentcore-vpc.cn-hangzhou.aliyuncs.com"
        return model_descriptor

    monkeypatch.setattr(AgentCoreControlPlane, "resolve_model", resolve_model)
    return env


def test_framework_model_settings_project_gateway_without_repr_secret(
    agent_config_path: Path,
    framework_env: Path,
) -> None:
    settings = framework_model_settings(
        "model-service-a",
        model="qwen-plus",
        config_path=agent_config_path,
        env_path=framework_env,
    )

    assert settings.model == "qwen-plus"
    assert settings.base_url == ("https://gateway.example.com/model-connection/model-1/v1")
    assert settings.api_key == "gateway-secret"
    assert "gateway-secret" not in repr(settings)


def test_framework_model_settings_uses_explicit_access_key_without_resource_sts(
    agent_config_path: Path,
    framework_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_sts(*_args: object, **_kwargs: object) -> None:
        pytest.fail("explicit AccessKey must not allocate automatic Resource STS")

    monkeypatch.setattr(_shared, "ResourceSTSProvider", unexpected_sts)

    settings = framework_model_settings(
        "model-service-a",
        config_path=agent_config_path,
        env_path=framework_env,
        access_key_credential=AccessKeyCredential("ak", "sk", "sts"),
    )

    assert settings.model == "qwen-plus"


def test_framework_model_settings_add_openai_v1_to_gateway_route_prefix(
    agent_config_path: Path,
    framework_env: Path,
) -> None:
    agent_config_path.write_text(
        agent_config_path.read_text(encoding="utf-8").replace(
            "https://gateway.example.com/v1",
            "http://forwarder.internal:10000/v1",
        ),
        encoding="utf-8",
    )

    settings = framework_model_settings(
        "model-service-a",
        model="qwen-plus",
        config_path=agent_config_path,
        env_path=framework_env,
    )

    assert settings.base_url == ("http://forwarder.internal:10000/model-connection/model-1/v1")


def test_framework_model_fails_when_protocol_adapter_is_missing(
    agent_config_path: Path,
    framework_env: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def resolve_model(
        self: AgentCoreControlPlane,
        resource_name: str,
        model_name: str | None = None,
    ) -> ModelDescriptor:
        return replace(model_descriptor, protocol="Future/v1")

    monkeypatch.setattr(AgentCoreControlPlane, "resolve_model", resolve_model)

    with pytest.raises(ConfigError, match="protocol"):
        framework_model_settings(
            "model-service-a",
            config_path=agent_config_path,
            env_path=framework_env,
        )


def test_framework_model_settings_support_anthropic_without_openai_route(
    agent_config_path: Path,
    framework_env: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_anthropic_descriptor(monkeypatch, model_descriptor)

    settings = framework_model_settings(
        "model-service-a",
        config_path=agent_config_path,
        env_path=framework_env,
    )
    parameters = framework_completion_parameters(settings)

    assert settings.provider == "anthropic"
    assert settings.base_url == "https://gateway.example.com/model-connection/model-1"
    assert parameters["model"] == "anthropic/claude-sonnet"
    assert parameters["extra_headers"]["Authorization"] == "Bearer gateway-secret"
    assert parameters["use_bearer_for_custom_base"] is True


def test_google_adk_preserves_anthropic_litellm_parameters(
    agent_config_path: Path,
    framework_env: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_anthropic_descriptor(monkeypatch, model_descriptor)
    from agentcore.integrations.google_adk import _google_adk_parameters

    settings = framework_model_settings(
        "model-service-a",
        config_path=agent_config_path,
        env_path=framework_env,
    )

    parameters = _google_adk_parameters(settings, {})

    assert parameters["use_bearer_for_custom_base"] is True


def test_langchain_selects_anthropic_adapter_from_control_plane_protocol(
    agent_config_path: Path,
    framework_env: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("langchain_anthropic")
    _use_anthropic_descriptor(monkeypatch, model_descriptor)
    from agentcore.integrations.langchain import model

    llm = model(
        "model-service-a",
        config_path=agent_config_path,
        env_path=framework_env,
    )

    assert type(llm).__name__ == "ChatAnthropic"
    assert llm.anthropic_api_url == "https://gateway.example.com/model-connection/model-1"
    assert llm.anthropic_api_key.get_secret_value() == ""
    assert llm.default_headers["Authorization"] == "Bearer gateway-secret"


def test_agentscope_selects_anthropic_adapter_from_control_plane_protocol(
    agent_config_path: Path,
    framework_env: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("agentscope")
    _use_anthropic_descriptor(monkeypatch, model_descriptor)
    from agentcore.integrations.agentscope import model

    llm = model(
        "model-service-a",
        config_path=agent_config_path,
        env_path=framework_env,
    )

    assert type(llm).__name__ == "AnthropicChatModel"
    assert llm.credential.base_url == "https://gateway.example.com/model-connection/model-1"
    assert llm.credential.api_key.get_secret_value() == ""
    assert llm.client.auth_token == "gateway-secret"


def test_pydantic_ai_selects_anthropic_adapter_from_control_plane_protocol(
    agent_config_path: Path,
    framework_env: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pydantic_ai")
    _use_anthropic_descriptor(monkeypatch, model_descriptor)
    from agentcore.integrations.pydantic_ai import model

    llm = model(
        "model-service-a",
        config_path=agent_config_path,
        env_path=framework_env,
    )

    assert type(llm).__name__ == "AnthropicModel"
    assert llm.model_name == "claude-sonnet"


def test_framework_model_settings_honors_agentcore_env_path(
    agent_config_path: Path,
    framework_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTCORE_ENV_PATH", str(framework_env))

    settings = framework_model_settings(
        "model-service-a",
        model="qwen-plus",
        config_path=agent_config_path,
    )

    assert settings.model == "qwen-plus"


def test_framework_debug_runtime_is_reused_until_process_cleanup(
    agent_config_path: Path,
    framework_env: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TokenSource:
        async def get(self) -> str:
            return "sa-token"

        async def refresh(self, current: str) -> str:
            return current

    class RuntimeSource:
        initial = None

        def __init__(self) -> None:
            self.resolve_calls = 0
            self.close_calls = 0

        async def resolve(self) -> RuntimeBindings:
            self.resolve_calls += 1
            return RuntimeBindings(
                config=load_agent_config(agent_config_path),
                controller_endpoint="https://controller.example.com",
                control_plane_endpoint="https://agentcore-vpc.cn-hangzhou.aliyuncs.com",
                agent_sa_tokens=TokenSource(),
            )

        async def aclose(self) -> None:
            self.close_calls += 1

    source = RuntimeSource()
    created = 0

    def create_source(*_args: object, **_kwargs: object) -> RuntimeSource:
        nonlocal created
        created += 1
        return source

    _shared._close_framework_debug_runtime()
    monkeypatch.setenv("AGENTCORE_DEBUG_TOKEN", "debug-token")
    monkeypatch.setattr(_shared, "create_runtime_source", create_source)

    try:
        first = framework_model_settings("model-service-a")
        second = framework_model_settings("model-service-a", model="qwen-plus")

        assert first.model == second.model == "qwen-plus"
        assert created == 1
        assert source.resolve_calls == 2
        assert source.close_calls == 0
    finally:
        _shared._close_framework_debug_runtime()

    assert source.close_calls == 1


@pytest.mark.asyncio
async def test_langchain_model_keeps_startup_gateway_credentials(
    agent_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework_env: Path,
) -> None:
    pytest.importorskip("langchain_openai")
    requests: list[httpx.Request] = []

    async def handle_request(
        transport: httpx.AsyncHTTPTransport,
        request: httpx.Request,
    ) -> httpx.Response:
        requests.append(request)
        return _completion_response()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_request)
    from agentcore.integrations.langchain import model

    llm = model(
        "model-service-a",
        model_name="qwen-plus",
        config_path=agent_config_path,
        env_path=framework_env,
    )
    await llm.async_client.create(
        messages=[{"role": "user", "content": "first"}],
        model=llm.model_name,
        stream=False,
    )

    _rotate_gateway_credential(agent_config_path)

    await llm.async_client.create(
        messages=[{"role": "user", "content": "second"}],
        model=llm.model_name,
        stream=False,
    )
    await llm.root_async_client.close()

    assert len(requests) == 2
    expected = "https://gateway.example.com/model-connection/model-1/v1/chat/completions"
    assert requests[0].url == expected
    assert requests[0].headers["authorization"] == "Bearer gateway-secret"
    assert json.loads(requests[0].content)["model"] == "qwen-plus"
    assert requests[1].url == expected
    assert requests[1].headers["authorization"] == "Bearer gateway-secret"
    assert json.loads(requests[1].content)["model"] == "qwen-plus"


def test_langchain_sync_model_keeps_startup_gateway_credentials(
    agent_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework_env: Path,
) -> None:
    pytest.importorskip("langchain_openai")
    requests: list[httpx.Request] = []

    def handle_request(
        transport: httpx.HTTPTransport,
        request: httpx.Request,
    ) -> httpx.Response:
        requests.append(request)
        return _completion_response()

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", handle_request)
    from agentcore.integrations.langchain import model

    llm = model(
        "model-service-a",
        model_name="qwen-plus",
        config_path=agent_config_path,
        env_path=framework_env,
    )
    llm.client.create(
        messages=[{"role": "user", "content": "first"}],
        model=llm.model_name,
        stream=False,
    )
    _rotate_gateway_credential(agent_config_path)
    llm.client.create(
        messages=[{"role": "user", "content": "second"}],
        model=llm.model_name,
        stream=False,
    )
    llm.root_client.close()

    assert requests[1].url == (
        "https://gateway.example.com/model-connection/model-1/v1/chat/completions"
    )
    assert requests[1].headers["authorization"] == "Bearer gateway-secret"
    assert json.loads(requests[1].content)["model"] == "qwen-plus"


@pytest.mark.asyncio
async def test_framework_model_does_not_reload_agent_yaml(
    agent_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework_env: Path,
) -> None:
    pytest.importorskip("langchain_openai")
    requests: list[httpx.Request] = []

    async def handle_request(
        transport: httpx.AsyncHTTPTransport,
        request: httpx.Request,
    ) -> httpx.Response:
        requests.append(request)
        return _completion_response()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_request)
    from agentcore.integrations.langchain import model

    llm = model(
        "model-service-a",
        model_name="qwen-plus",
        config_path=agent_config_path,
        env_path=framework_env,
    )
    _replace_agent_config(agent_config_path)
    await llm.async_client.create(
        messages=[{"role": "user", "content": "hello"}],
        model=llm.model_name,
        stream=False,
    )
    await llm.root_async_client.close()

    assert requests[0].url == (
        "https://gateway.example.com/model-connection/model-1/v1/chat/completions"
    )
    assert requests[0].headers["authorization"] == "Bearer gateway-secret"
    assert json.loads(requests[0].content)["model"] == "qwen-plus"


@pytest.mark.asyncio
async def test_langchain_model_credential_transport_preserves_streaming(
    agent_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework_env: Path,
) -> None:
    pytest.importorskip("langchain_openai")

    class CompletionStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield (
                b'data: {"id":"completion-1","object":"chat.completion.chunk",'
                b'"created":0,"model":"qwen-plus","choices":[{"index":0,'
                b'"delta":{"role":"assistant","content":"ok"},'
                b'"finish_reason":null}]}\n\n'
            )
            yield b"data: [DONE]\n\n"

    async def handle_request(
        transport: httpx.AsyncHTTPTransport,
        request: httpx.Request,
    ) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=CompletionStream(),
        )

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_request)

    from agentcore.integrations.langchain import model

    llm = model(
        "model-service-a",
        model_name="qwen-plus",
        config_path=agent_config_path,
        env_path=framework_env,
    )
    chunks = [chunk async for chunk in llm.astream("hello")]
    await llm.root_async_client.close()

    assert "".join(str(chunk.content) for chunk in chunks) == "ok"


@pytest.mark.asyncio
async def test_agentscope_model_keeps_startup_gateway_credentials(
    agent_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework_env: Path,
) -> None:
    pytest.importorskip("agentscope")
    requests: list[httpx.Request] = []

    async def handle_request(
        transport: httpx.AsyncHTTPTransport,
        request: httpx.Request,
    ) -> httpx.Response:
        requests.append(request)
        return _completion_response()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_request)
    from agentcore.integrations.agentscope import model

    llm = model("model-service-a", config_path=agent_config_path, env_path=framework_env)
    await llm.client.chat.completions.create(
        messages=[{"role": "user", "content": "first"}],
        model=llm.model,
        stream=False,
    )
    _rotate_gateway_credential(agent_config_path)
    await llm.client.chat.completions.create(
        messages=[{"role": "user", "content": "second"}],
        model=llm.model,
        stream=False,
    )
    await llm.client.close()

    assert requests[1].url == (
        "https://gateway.example.com/model-connection/model-1/v1/chat/completions"
    )
    assert requests[1].headers["authorization"] == "Bearer gateway-secret"
    assert json.loads(requests[1].content)["model"] == "qwen-plus"


@pytest.mark.asyncio
async def test_pydantic_ai_model_keeps_startup_gateway_credentials(
    agent_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework_env: Path,
) -> None:
    pytest.importorskip("pydantic_ai")
    requests: list[httpx.Request] = []

    async def handle_request(
        transport: httpx.AsyncHTTPTransport,
        request: httpx.Request,
    ) -> httpx.Response:
        requests.append(request)
        return _completion_response()

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_request)
    from agentcore.integrations.pydantic_ai import model

    llm = model("model-service-a", config_path=agent_config_path, env_path=framework_env)
    await llm.client.chat.completions.create(
        messages=[{"role": "user", "content": "first"}],
        model=llm.model_name,
        stream=False,
    )
    _rotate_gateway_credential(agent_config_path)
    await llm.client.chat.completions.create(
        messages=[{"role": "user", "content": "second"}],
        model=llm.model_name,
        stream=False,
    )
    await llm.client.close()

    assert requests[1].url == (
        "https://gateway.example.com/model-connection/model-1/v1/chat/completions"
    )
    assert requests[1].headers["authorization"] == "Bearer gateway-secret"
    assert json.loads(requests[1].content)["model"] == "qwen-plus"


@pytest.mark.asyncio
async def test_google_adk_model_keeps_startup_gateway_credentials(
    agent_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework_env: Path,
) -> None:
    pytest.importorskip("google.adk.models.lite_llm")
    from google.adk.models.lite_llm import LiteLLMClient

    calls: list[dict[str, object]] = []

    async def acompletion(
        client: LiteLLMClient,
        model: object,
        messages: object,
        tools: object,
        **kwargs: object,
    ) -> None:
        calls.append({"model": model, **kwargs})

    monkeypatch.setattr(LiteLLMClient, "acompletion", acompletion)
    from agentcore.integrations.google_adk import model

    llm = model("model-service-a", config_path=agent_config_path, env_path=framework_env)
    await llm.llm_client.acompletion(model=llm.model, messages=[], tools=[])
    _rotate_gateway_credential(agent_config_path)
    await llm.llm_client.acompletion(model=llm.model, messages=[], tools=[])

    assert calls[1]["model"] == "openai/qwen-plus"
    assert calls[1]["api_base"] == ("https://gateway.example.com/model-connection/model-1/v1")
    assert calls[1]["api_key"] == "gateway-secret"


@pytest.mark.asyncio
async def test_crewai_model_keeps_startup_gateway_credentials(
    agent_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    framework_env: Path,
) -> None:
    pytest.importorskip("crewai")
    from crewai import LLM

    calls: list[dict[str, object]] = []

    async def handle_response(
        llm: LLM,
        params: dict[str, object],
        **kwargs: object,
    ) -> str:
        calls.append(params)
        return "ok"

    monkeypatch.setattr(LLM, "_ahandle_non_streaming_response", handle_response)
    from agentcore.integrations.crewai import model

    llm = model("model-service-a", config_path=agent_config_path, env_path=framework_env)
    await llm.acall("first")
    _rotate_gateway_credential(agent_config_path)
    await llm.acall("second")

    assert calls[1]["model"] == "openai/qwen-plus"
    assert calls[1]["base_url"] == ("https://gateway.example.com/model-connection/model-1/v1")
    assert calls[1]["api_key"] == "gateway-secret"


def _completion_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "completion-1",
            "object": "chat.completion",
            "created": 0,
            "model": "ignored-by-test",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


def _replace_agent_config(agent_config_path: Path) -> None:
    rotated = agent_config_path.read_text(encoding="utf-8")
    rotated = rotated.replace(
        "https://gateway.example.com/v1",
        "https://rotated-gateway.example.com/v2",
    )
    rotated = rotated.replace("gateway-secret", "rotated-gateway-secret")
    agent_config_path.write_text(rotated, encoding="utf-8")


def _rotate_gateway_credential(agent_config_path: Path) -> None:
    rotated = agent_config_path.read_text(encoding="utf-8")
    rotated = rotated.replace("gateway-secret", "rotated-gateway-secret")
    agent_config_path.write_text(rotated, encoding="utf-8")


def _use_anthropic_descriptor(
    monkeypatch: pytest.MonkeyPatch,
    model_descriptor: ModelDescriptor,
) -> None:
    async def resolve_model(
        self: AgentCoreControlPlane,
        resource_name: str,
        model_name: str | None = None,
    ) -> ModelDescriptor:
        assert resource_name == "model-service-a"
        return replace(
            model_descriptor,
            protocol="Anthropic",
            provider_type="ANTHROPIC",
            model_name="claude-sonnet",
        )

    monkeypatch.setattr(AgentCoreControlPlane, "resolve_model", resolve_model)


@pytest.mark.asyncio
async def test_tool_functions_keep_schema_signature_and_async_call() -> None:
    async def invoke(arguments):  # type: ignore[no-untyped-def]
        return arguments["query"]

    tool = Tool(
        name="search",
        description="Search",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        _call=invoke,
    )
    function = tool_functions([tool])[0]

    assert list(inspect.signature(function).parameters) == ["query"]
    assert await function(query="agentcore") == "agentcore"


@pytest.mark.asyncio
async def test_tool_functions_allow_business_session_and_user_arguments() -> None:
    async def invoke(arguments):  # type: ignore[no-untyped-def]
        return arguments

    tool = Tool(
        name="store",
        description="Store",
        parameters={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "user_id": {"type": "string"},
            },
        },
        _call=invoke,
    )

    function = tool_functions([tool])[0]

    assert await function(session_id="session-a", user_id="user-a") == {
        "session_id": "session-a",
        "user_id": "user-a",
    }


@pytest.mark.asyncio
async def test_tool_functions_preserve_non_python_json_schema_property_names() -> None:
    async def invoke(arguments):  # type: ignore[no-untyped-def]
        return arguments

    tool = Tool(
        name="copy",
        description="Copy",
        parameters={
            "type": "object",
            "properties": {
                "file-path": {"type": "string"},
                "from": {"type": "string"},
            },
            "required": ["file-path"],
        },
        _call=invoke,
    )

    function = tool_functions([tool])[0]

    assert await function(**{"file-path": "a.txt", "from": "workspace"}) == {
        "file-path": "a.txt",
        "from": "workspace",
    }
    assert function.__agentcore_json_schema__ == tool.parameters  # type: ignore[attr-defined]


def test_langchain_adapter_preserves_non_python_json_schema_properties() -> None:
    pytest.importorskip("langchain_core")
    from agentcore.integrations.langchain import tools

    canonical = _non_python_property_tool()
    adapted = tools([canonical])[0]

    assert adapted.args_schema == canonical.parameters


@pytest.mark.asyncio
async def test_langchain_adapter_supports_sync_and_async_tool_invocation() -> None:
    pytest.importorskip("langchain_core")
    from agentcore.integrations.langchain import tools

    async def invoke(arguments):  # type: ignore[no-untyped-def]
        return arguments

    canonical = Tool(
        name="copy",
        description="Copy",
        parameters={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        },
        _call=invoke,
        _sync_call=lambda arguments: arguments,
    )
    adapted = tools([canonical])[0]

    assert adapted.invoke({"value": "sync"}) == {"value": "sync"}
    assert await adapted.ainvoke({"value": "async"}) == {"value": "async"}


def test_agentscope_adapter_preserves_non_python_json_schema_properties() -> None:
    pytest.importorskip("agentscope")
    from agentcore.integrations.agentscope import tools

    canonical = _non_python_property_tool()
    adapted = tools([canonical])[0]

    assert adapted.input_schema == canonical.parameters


def test_google_adk_adapter_preserves_non_python_json_schema_properties() -> None:
    pytest.importorskip("google.adk")
    from agentcore.integrations.google_adk import tools

    canonical = _non_python_property_tool()
    adapted = tools([canonical])[0]
    declaration = adapted._get_declaration()

    assert declaration.parameters_json_schema == canonical.parameters


def test_pydantic_ai_adapter_preserves_non_python_json_schema_properties() -> None:
    pytest.importorskip("pydantic_ai")
    from agentcore.integrations.pydantic_ai import tools

    canonical = _non_python_property_tool()
    adapted = tools([canonical])[0]
    prepared = adapted.prepare(None, adapted.tool_def)

    assert prepared.parameters_json_schema == canonical.parameters


@pytest.mark.asyncio
async def test_crewai_adapter_preserves_and_invokes_non_python_schema_properties() -> None:
    pytest.importorskip("crewai")
    from agentcore.integrations.crewai import tools

    canonical = _non_python_property_tool()
    adapted = tools([canonical])[0]
    schema = adapted.args_schema.model_json_schema()

    assert schema["properties"] == canonical.parameters["properties"]
    assert schema["required"] == canonical.parameters["required"]
    assert await adapted.arun(**{"file-path": "a.txt", "from": "workspace"}) == {
        "file-path": "a.txt",
        "from": "workspace",
    }


def _non_python_property_tool() -> Tool:
    async def invoke(arguments):  # type: ignore[no-untyped-def]
        return arguments

    return Tool(
        name="copy",
        description="Copy",
        parameters={
            "type": "object",
            "properties": {
                "file-path": {"type": "string"},
                "from": {"type": "string"},
            },
            "required": ["file-path", "from"],
        },
        _call=invoke,
    )
