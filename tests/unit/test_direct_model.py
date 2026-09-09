from __future__ import annotations

import inspect
from collections.abc import AsyncIterator
from typing import Any

import litellm
import pytest

from agentcore.errors import ConfigError, InvocationError
from agentcore.model import AsyncModelClient


@pytest.mark.asyncio
async def test_direct_model_completion_uses_litellm_and_latest_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    tokens = iter(["token-a", "token-b"])

    async def completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    async def headers() -> dict[str, str]:
        return {"X-Tenant": "tenant-a"}

    monkeypatch.setattr(litellm, "acompletion", completion)
    client = AsyncModelClient.direct(
        model="qwen-plus",
        provider="dashscope",
        base_url="https://provider.example.com/v1",
        api_key_provider=lambda: next(tokens),
        headers_provider=headers,
    )

    first = await client.completion([{"role": "user", "content": "hello"}])
    second = await client.invoke([{"role": "user", "content": "again"}])

    assert first["choices"][0]["message"]["content"] == "ok"
    assert second["choices"][0]["message"]["content"] == "ok"
    assert [call["api_key"] for call in calls] == ["token-a", "token-b"]
    assert calls[0]["base_url"] == "https://provider.example.com/v1"
    assert calls[0]["model"] == "qwen-plus"
    assert calls[0]["custom_llm_provider"] == "dashscope"
    assert calls[0]["caching"] is False
    assert calls[0]["max_retries"] == 0
    assert calls[0]["extra_headers"] == {"X-Tenant": "tenant-a"}
    assert calls[0]["messages"] == [{"role": "user", "content": "hello"}]
    assert "token-a" not in repr(client)


@pytest.mark.asyncio
async def test_direct_model_supports_responses_and_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response_calls: list[dict[str, Any]] = []
    embedding_calls: list[dict[str, Any]] = []

    async def responses(**kwargs: Any) -> dict[str, Any]:
        response_calls.append(kwargs)
        return {"id": "response-a"}

    async def embedding(**kwargs: Any) -> dict[str, Any]:
        embedding_calls.append(kwargs)
        return {"data": [{"embedding": [0.1, 0.2]}]}

    monkeypatch.setattr(litellm, "aresponses", responses)
    monkeypatch.setattr(litellm, "aembedding", embedding)
    client = AsyncModelClient.direct(
        model="custom-model",
        provider="openai",
        base_url="https://provider.example.com/openai",
        api_key="secret",
    )

    response = await client.responses("hello", temperature=0)
    embeddings = await client.embedding(["hello"])

    assert response == {"id": "response-a"}
    assert embeddings == {"data": [{"embedding": [0.1, 0.2]}]}
    assert response_calls == [
        {
            "api_key": "secret",
            "api_base": "https://provider.example.com/openai",
            "model": "custom-model",
            "custom_llm_provider": "openai",
            "caching": False,
            "max_retries": 0,
            "input": "hello",
            "temperature": 0,
        }
    ]
    assert embedding_calls == [
        {
            "api_key": "secret",
            "api_base": "https://provider.example.com/openai",
            "model": "custom-model",
            "custom_llm_provider": "openai",
            "caching": False,
            "max_retries": 0,
            "input": ["hello"],
        }
    ]


@pytest.mark.asyncio
async def test_direct_model_supports_responses_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []

    async def responses(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        calls.append(kwargs)

        async def events() -> AsyncIterator[dict[str, Any]]:
            yield {"type": "response.output_text.delta", "delta": "a"}
            yield {"type": "response.output_text.delta", "delta": "b"}

        return events()

    monkeypatch.setattr(litellm, "aresponses", responses)
    client = AsyncModelClient.direct(
        model="custom-model",
        provider="openai",
        base_url="https://provider.example.com/openai",
        api_key="secret",
    )

    events = [event async for event in client.responses_stream("hello")]

    assert calls[0]["api_base"] == "https://provider.example.com/openai"
    assert calls[0]["stream"] is True
    assert [event["delta"] for event in events] == ["a", "b"]


@pytest.mark.asyncio
async def test_direct_model_stream_uses_one_credential_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    credential_calls = 0

    def api_key() -> str:
        nonlocal credential_calls
        credential_calls += 1
        return "stream-token"

    async def completion(**kwargs: Any) -> AsyncIterator[dict[str, Any]]:
        calls.append(kwargs)

        async def chunks() -> AsyncIterator[dict[str, Any]]:
            yield {"choices": [{"delta": {"content": "a"}}]}
            yield {"choices": [{"delta": {"content": "b"}}]}

        return chunks()

    monkeypatch.setattr(litellm, "acompletion", completion)
    client = AsyncModelClient.direct(
        model="qwen-plus",
        base_url="https://provider.example.com/v1",
        api_key_provider=api_key,
    )

    chunks = [chunk async for chunk in client.stream([])]

    assert credential_calls == 1
    assert calls[0]["stream"] is True
    assert [chunk["choices"][0]["delta"]["content"] for chunk in chunks] == ["a", "b"]


def test_direct_model_rejects_ambiguous_credentials_and_invalid_endpoint() -> None:
    with pytest.raises(ConfigError, match="api_key"):
        AsyncModelClient.direct(
            model="qwen-plus",
            base_url="https://provider.example.com/v1",
            api_key="secret",
            api_key_provider=lambda: "latest",
        )

    with pytest.raises(ConfigError, match="HTTP"):
        AsyncModelClient.direct(
            model="qwen-plus",
            base_url="file:///tmp/model",
        )


@pytest.mark.asyncio
async def test_direct_model_does_not_retry_failed_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    async def completion(**_kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        raise RuntimeError("rejected")

    monkeypatch.setattr(litellm, "acompletion", completion)
    client = AsyncModelClient.direct(
        model="qwen-plus",
        base_url="https://provider.example.com/v1",
        api_key="secret",
    )

    with pytest.raises(InvocationError, match="completion failed"):
        await client.completion([])

    assert calls == 1


def test_model_client_constructor_does_not_expose_backend_injection() -> None:
    assert "_backend" not in inspect.signature(AsyncModelClient).parameters
