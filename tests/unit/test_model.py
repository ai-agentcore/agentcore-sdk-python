from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import anyio
import httpx
import httpx2
import pytest

from agentcore.controlplane import ModelDescriptor
from agentcore.errors import ConfigError, InvocationError
from agentcore.model import AsyncModelClient
from agentcore.runtime.config import load_agent_config
from agentcore.runtime.context import RequestContext, use_context


def _use_openai_transport(
    monkeypatch: pytest.MonkeyPatch,
    handler: Callable[[httpx.Request], httpx.Response],
) -> None:
    async def handle_request(
        _transport: httpx.AsyncHTTPTransport,
        request: httpx.Request,
    ) -> httpx.Response:
        return handler(request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", handle_request)


@pytest.mark.asyncio
async def test_managed_model_uses_bound_route_credentials_without_context_headers(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _chat_completion_response()

    _use_openai_transport(monkeypatch, handler)
    client = AsyncModelClient(load_agent_config(agent_config_path), model_descriptor)
    with use_context(
        RequestContext(
            {
                "x-agentcore-session-id": "session-a",
                "x-agentcore-user-id": "user-a",
            }
        )
    ):
        response = await client.invoke([{"role": "user", "content": "hello"}])
    await client.aclose()

    assert client.descriptor is model_descriptor
    assert response["choices"][0]["message"]["content"] == "ok"
    assert seen[0].url == (
        "https://gateway.example.com/model-connection/model-1/v1/chat/completions"
    )
    assert seen[0].headers["Authorization"] == "Bearer gateway-secret"
    assert "X-AgentCore-Session-ID" not in seen[0].headers
    assert "X-AgentCore-User-ID" not in seen[0].headers
    assert json.loads(seen[0].content)["model"] == "qwen-plus"


@pytest.mark.asyncio
async def test_model_adds_openai_v1_to_gateway_route_prefix(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def replace_gateway_url() -> None:
        agent_config_path.write_text(
            agent_config_path.read_text(encoding="utf-8").replace(
                "https://gateway.example.com/v1",
                "http://forwarder.internal:10000/v1",
            ),
            encoding="utf-8",
        )

    await anyio.to_thread.run_sync(replace_gateway_url)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return _chat_completion_response()

    _use_openai_transport(monkeypatch, handler)
    client = AsyncModelClient(load_agent_config(agent_config_path), model_descriptor)
    await client.invoke([])
    await client.aclose()

    assert seen[0].url == (
        "http://forwarder.internal:10000/model-connection/model-1/v1/chat/completions"
    )


@pytest.mark.asyncio
async def test_model_does_not_allow_rebinding_after_creation(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
) -> None:
    client = AsyncModelClient(load_agent_config(agent_config_path), model_descriptor)

    with pytest.raises(ValueError, match="resource_name"):
        await client.invoke([], resource_name="other-service")
    with pytest.raises(ValueError, match="model"):
        await client.invoke([], model="other-model")

    await client.aclose()


def test_platform_model_fails_when_protocol_adapter_is_missing(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
) -> None:
    descriptor = replace(model_descriptor, protocol="Future/v1")

    with pytest.raises(ConfigError, match="protocol"):
        AsyncModelClient(load_agent_config(agent_config_path), descriptor)


@pytest.mark.asyncio
async def test_managed_anthropic_uses_messages_route_and_bearer_credential(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = replace(
        model_descriptor,
        protocol="Anthropic",
        provider_type="ANTHROPIC",
        model_name="claude-sonnet",
    )
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return _anthropic_message_response()

    anthropic_http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        follow_redirects=False,
    )
    import anthropic

    monkeypatch.setattr(
        anthropic,
        "DefaultAsyncHttpxClient",
        lambda **kwargs: anthropic_http,
    )
    client = AsyncModelClient(load_agent_config(agent_config_path), descriptor)
    response = await client.completion([{"role": "user", "content": "hello"}])
    await client.aclose()

    assert response["content"][0]["text"] == "ok"
    assert seen[0].url == (
        "https://gateway.example.com/model-connection/model-1/v1/messages"
    )
    assert seen[0].headers["Authorization"] == "Bearer gateway-secret"
    assert "x-api-key" not in seen[0].headers
    payload = json.loads(seen[0].content)
    assert payload == {
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": "hello"}],
        "model": "claude-sonnet",
        "stream": False,
    }


@pytest.mark.asyncio
async def test_managed_anthropic_stream_preserves_native_events(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = replace(
        model_descriptor,
        protocol="Anthropic",
        provider_type="ANTHROPIC",
        model_name="claude-sonnet",
    )
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return httpx2.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'event: message_start\ndata: {"type":"message_start","message":'
                '{"id":"msg_1","type":"message","role":"assistant",'
                '"model":"claude-sonnet","content":[],"stop_reason":null,'
                '"stop_sequence":null,"usage":{"input_tokens":1,'
                '"output_tokens":0}}}\n\n'
                'event: content_block_start\ndata: {"type":"content_block_start",'
                '"index":0,"content_block":{"type":"text","text":""}}\n\n'
                'event: content_block_delta\ndata: {"type":"content_block_delta",'
                '"index":0,"delta":{"type":"text_delta","text":"ok"}}\n\n'
                'event: content_block_stop\ndata: {"type":"content_block_stop",'
                '"index":0}\n\n'
                'event: message_delta\ndata: {"type":"message_delta","delta":'
                '{"stop_reason":"end_turn","stop_sequence":null},"usage":'
                '{"output_tokens":1}}\n\n'
                'event: message_stop\ndata: {"type":"message_stop"}\n\n'
            ),
        )

    anthropic_http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(handler),
        follow_redirects=False,
    )
    import anthropic

    monkeypatch.setattr(
        anthropic,
        "DefaultAsyncHttpxClient",
        lambda **kwargs: anthropic_http,
    )
    client = AsyncModelClient(load_agent_config(agent_config_path), descriptor)
    events = [event async for event in client.stream([])]
    await client.aclose()

    assert json.loads(seen[0].content)["stream"] is True
    assert [event["type"] for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]


@pytest.mark.asyncio
async def test_managed_anthropic_rejects_protocol_specific_unsupported_apis(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = replace(model_descriptor, protocol="Anthropic")
    import anthropic

    anthropic_http = httpx2.AsyncClient(
        transport=httpx2.MockTransport(lambda request: _anthropic_message_response()),
        follow_redirects=False,
    )
    monkeypatch.setattr(
        anthropic,
        "DefaultAsyncHttpxClient",
        lambda **kwargs: anthropic_http,
    )
    client = AsyncModelClient(load_agent_config(agent_config_path), descriptor)

    with pytest.raises(InvocationError, match="Responses API"):
        await client.responses("hello")
    with pytest.raises(InvocationError, match="Responses API"):
        _ = [event async for event in client.responses_stream("hello")]
    with pytest.raises(InvocationError, match="Embedding API"):
        await client.embedding(["hello"])

    await client.aclose()


@pytest.mark.asyncio
async def test_managed_model_stream_uses_openai_sse(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"id":"chunk-a","object":"chat.completion.chunk",'
                '"created":1,"model":"qwen-plus","choices":[{"index":0,'
                '"delta":{"content":"a"},"finish_reason":null}]}\n\n'
                'data: {"id":"chunk-a","object":"chat.completion.chunk",'
                '"created":1,"model":"qwen-plus","choices":[{"index":0,'
                '"delta":{"content":"b"},"finish_reason":"stop"}]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    _use_openai_transport(monkeypatch, handler)
    client = AsyncModelClient(load_agent_config(agent_config_path), model_descriptor)
    chunks = [chunk async for chunk in client.stream([])]
    await client.aclose()

    assert json.loads(seen[0].content)["stream"] is True
    assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks] == ["a", "b"]


@pytest.mark.asyncio
async def test_managed_model_wraps_openai_errors_without_retry(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise httpx.ConnectError("rejected", request=request)

    _use_openai_transport(monkeypatch, handler)
    client = AsyncModelClient(load_agent_config(agent_config_path), model_descriptor)
    with pytest.raises(InvocationError, match="model request failed") as error:
        await client.completion([])
    await client.aclose()

    assert calls == 1
    assert error.value.__cause__ is not None


@pytest.mark.asyncio
async def test_managed_model_logs_openai_http_status(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            request=request,
            json={"error": {"message": "rate limited", "type": "rate_limit_error"}},
        )

    _use_openai_transport(monkeypatch, handler)
    client = AsyncModelClient(load_agent_config(agent_config_path), model_descriptor)
    with pytest.raises(InvocationError):
        await client.completion([])
    await client.aclose()

    assert "status=429" in caplog.text


@pytest.mark.asyncio
async def test_managed_openai_v1_supports_chat_and_responses_routes(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/chat/completions"):
            return _chat_completion_response()
        return httpx.Response(
            200,
            json={
                "id": "response-a",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "qwen-plus",
                "output": [],
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    _use_openai_transport(monkeypatch, handler)
    client = AsyncModelClient(load_agent_config(agent_config_path), model_descriptor)
    await client.completion([{"role": "user", "content": "hello"}])
    response = await client.responses("hello")
    with pytest.raises(InvocationError, match="Embedding API"):
        await client.embedding(["hello"])
    await client.aclose()

    assert response["id"] == "response-a"
    assert requests[0].url == (
        "https://gateway.example.com/model-connection/model-1/v1/chat/completions"
    )
    assert requests[1].url == (
        "https://gateway.example.com/model-connection/model-1/v1/responses"
    )
    assert json.loads(requests[1].content)["input"] == "hello"


@pytest.mark.asyncio
async def test_managed_responses_stream_is_forwarded_as_native_stream(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"type":"response.created","sequence_number":0,'
                '"response":{"id":"response-a","object":"response",'
                '"created_at":1,"status":"in_progress","model":"qwen-plus",'
                '"output":[]}}\n\n'
                'data: {"type":"response.output_text.delta","sequence_number":1,'
                '"item_id":"message-a","output_index":0,"content_index":0,'
                '"delta":"hello","logprobs":[]}\n\n'
                "data: [DONE]\n\n"
            ),
        )

    _use_openai_transport(monkeypatch, handler)
    client = AsyncModelClient(load_agent_config(agent_config_path), model_descriptor)
    events = [event async for event in client.responses_stream("hello")]
    await client.aclose()

    assert requests[0].url == (
        "https://gateway.example.com/model-connection/model-1/v1/responses"
    )
    assert json.loads(requests[0].content)["stream"] is True
    assert [event["type"] for event in events] == [
        "response.created",
        "response.output_text.delta",
    ]


def _chat_completion_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "completion-a",
            "object": "chat.completion",
            "created": 1,
            "model": "qwen-plus",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        },
    )


def _anthropic_message_response() -> httpx2.Response:
    return httpx2.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet",
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        },
    )
