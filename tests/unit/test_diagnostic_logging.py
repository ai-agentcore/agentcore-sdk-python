from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

import httpx
import httpx2
import pytest
from alibabacloud_tea_openapi.exceptions import ClientException
from Tea.exceptions import TeaException

from agentcore.controlplane import ModelDescriptor
from agentcore.controlplane.client import _invoke_control_plane
from agentcore.errors import InvocationError
from agentcore.integrations._shared import FrameworkModelSettings, _gateway_credential_request
from agentcore.mcp import AsyncMCPClient, MCPConnection
from agentcore.mcp import client as mcp_module
from agentcore.model import AsyncModelClient
from agentcore.runtime.config import load_agent_config


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy", [False, True])
async def test_control_plane_failure_logs_upstream_request_id(
    caplog: pytest.LogCaptureFixture,
    legacy: bool,
) -> None:
    data = {"RequestId": "upstream-123", "statusCode": 403}
    error = (
        TeaException({"code": "Forbidden", "message": "denied", "data": data})
        if legacy
        else ClientException(
            code="Forbidden",
            message="denied",
            status_code=403,
            data=data,
            request_id="upstream-123",
        )
    )

    async def fail() -> None:
        raise error

    with pytest.raises(InvocationError) as raised:
        await _invoke_control_plane("list_mcps", fail())
    assert raised.value.__cause__ is error
    assert "upstream_request_id=upstream-123" in caplog.text
    assert "status=403" in caplog.text and "code=Forbidden" in caplog.text


@pytest.mark.asyncio
async def test_mcp_timeout_logs_path_and_port_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    @asynccontextmanager
    async def fail(_connection: MCPConnection) -> AsyncIterator[None]:
        raise TimeoutError("initialize timed out")
        yield  # pragma: no cover

    monkeypatch.setattr(mcp_module, "_official_session", fail)
    caplog.set_level(logging.INFO, logger="agentcore")
    client = AsyncMCPClient.direct(
        url="http://user:password@gateway.example:19011/mcp-servers/mcp-servers/id"
        "?token=query-secret#fragment-secret",
    )
    try:
        with pytest.raises(TimeoutError):
            await client.list_tools()
    finally:
        await client.aclose()
    failures = [r.message for r in caplog.records if "tools.list.failed" in r.message]
    assert "url=http://gateway.example:19011/mcp-servers/mcp-servers/id" in failures[0]
    assert "session.open.started" in caplog.text
    for secret in ("password", "query-secret", "fragment-secret"):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_model_logs_resolved_route_and_actual_http_path(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fail(_transport: httpx.AsyncHTTPTransport, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("connect timeout", request=request)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fail)
    caplog.set_level(logging.INFO, logger="agentcore")
    config = load_agent_config(agent_config_path)
    config = replace(
        config,
        spec=replace(
            config.spec,
            model=replace(
                config.spec.model,
                gateway_url="http://gateway.example:19011/model-connection",
            ),
        ),
    )
    client = AsyncModelClient(config, model_descriptor)
    try:
        with pytest.raises(InvocationError):
            await client.completion([])
    finally:
        await client.aclose()
    expected = "http://gateway.example:19011/model-connection/model-connection/model-1/v1"
    assert f"base_url={expected}" in caplog.text
    assert f"url={expected}/chat/completions" in caplog.text
    failures = [r.message for r in caplog.records if "model.request.failed" in r.message]
    assert f"base_url={expected}" in failures[0]
    assert "gateway-secret" not in caplog.text


def test_framework_request_logs_actual_url_without_secrets(
    model_descriptor: ModelDescriptor,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="agentcore")
    settings = FrameworkModelSettings(
        model="qwen-plus",
        provider="openai",
        base_url="http://gateway.example:19011/v1",
        api_key="consumer-secret",
        headers={},
        descriptor=model_descriptor,
    )
    request = httpx.Request(
        "POST",
        "http://gateway.example:19011/v1/chat/completions?token=query-secret",
        content=b"private-prompt",
    )
    projected = _gateway_credential_request(settings, request)
    assert projected.url == request.url
    assert projected.headers["authorization"] == "Bearer consumer-secret"
    assert "url=http://gateway.example:19011/v1/chat/completions" in caplog.text
    for secret in ("consumer-secret", "query-secret", "private-prompt"):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_anthropic_request_hook_logs_messages_route(
    agent_config_path: Path,
    model_descriptor: ModelDescriptor,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fail(
        _transport: httpx2.AsyncHTTPTransport, request: httpx2.Request
    ) -> httpx2.Response:
        raise httpx2.ConnectTimeout("connect timeout", request=request)

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", fail)
    caplog.set_level(logging.INFO, logger="agentcore")
    descriptor = replace(model_descriptor, protocol="Anthropic", model_name="claude-sonnet")
    client = AsyncModelClient(load_agent_config(agent_config_path), descriptor)
    try:
        with pytest.raises(InvocationError):
            await client.completion([{"role": "user", "content": "private-prompt"}])
    finally:
        await client.aclose()
    assert "url=https://gateway.example.com/model-connection/model-1/v1/messages" in caplog.text
    assert "gateway-secret" not in caplog.text
    assert "private-prompt" not in caplog.text
