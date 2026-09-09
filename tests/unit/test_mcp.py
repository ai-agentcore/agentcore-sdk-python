from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anyio
import httpx
import pytest

from agentcore import AsyncAgentCore
from agentcore.controlplane import MCPDescriptor
from agentcore.errors import ConfigError
from agentcore.mcp import AsyncMCPClient, MCPConnection
from agentcore.mcp import client as mcp_client_module
from agentcore.runtime.config import load_agent_config
from agentcore.runtime.context import (
    RequestContext,
    use_context,
)


class FakeSession:
    def __init__(self) -> None:
        self.calls = 0

    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search",
                "description": "Search",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "session_id": {"type": "string"},
                    },
                },
            }
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls += 1
        return {"name": name, "arguments": arguments}


@pytest.mark.asyncio
async def test_platform_mcp_uses_bound_url_and_gateway_headers(
    agent_config_path: Path,
    mcp_descriptor: MCPDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[MCPConnection] = []
    session = FakeSession()

    @asynccontextmanager
    async def factory(connection: MCPConnection) -> AsyncIterator[FakeSession]:
        opened.append(connection)
        yield session

    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.platform(load_agent_config(agent_config_path), mcp_descriptor)
    with use_context(RequestContext({"X-AgentCore-Session-ID": "gateway-session"})):
        tools = await client.tools()
    result = await tools[0].ainvoke({"query": "agentcore", "session_id": "tool-session"})
    await client.aclose()

    assert opened[0].url == "https://gateway.example.com/mcp-servers/search-id"
    assert opened[0].transport == "streamable-http"
    assert await opened[0].resolve_headers() == {"Authorization": "Bearer gateway-secret"}
    assert "session_id" in tools[0].parameters["properties"]
    assert result["arguments"] == {
        "query": "agentcore",
        "session_id": "tool-session",
    }


def test_platform_mcp_requires_gateway(
    agent_config_path: Path,
    mcp_descriptor: MCPDescriptor,
) -> None:
    agent_config_path.write_text(
        agent_config_path.read_text(encoding="utf-8").replace(
            "  mcp:\n    gatewayUrl: https://gateway.example.com/mcp\n", ""
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="spec.mcp"):
        AsyncMCPClient.platform(load_agent_config(agent_config_path), mcp_descriptor)


def test_platform_mcp_uses_streamable_http_for_sse_upstream(
    agent_config_path: Path,
    mcp_descriptor: MCPDescriptor,
) -> None:
    descriptor = MCPDescriptor(
        mcp_server_id=mcp_descriptor.mcp_server_id,
        name=mcp_descriptor.name,
        protocol="SSE",
        type=mcp_descriptor.type,
        status=mcp_descriptor.status,
    )

    client = AsyncMCPClient.platform(load_agent_config(agent_config_path), descriptor)

    assert client._connection.url == "https://gateway.example.com/mcp-servers/search-id"
    assert client._connection.transport == "streamable-http"


def test_direct_mcp_rejects_invalid_ipv6_url() -> None:
    with pytest.raises(ConfigError, match=r"valid HTTP\(S\) URL"):
        AsyncMCPClient.direct(url="http://[invalid")


@pytest.mark.asyncio
async def test_direct_mcp_does_not_inject_gateway_headers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[MCPConnection] = []

    @asynccontextmanager
    async def factory(connection: MCPConnection) -> AsyncIterator[FakeSession]:
        opened.append(connection)
        yield FakeSession()

    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(
        url="https://direct.example.com/mcp",
        headers_provider=lambda: {"Authorization": "Bearer direct"},
    )
    await client.list_tools()
    await client.aclose()

    assert await opened[0].resolve_headers() == {"Authorization": "Bearer direct"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [42, {"Authorization": 42}],
)
async def test_direct_mcp_rejects_invalid_headers_provider_result(headers: object) -> None:
    connection = MCPConnection(
        "https://direct.example.com/mcp",
        "streamable-http",
        lambda: headers,  # type: ignore[return-value]
    )

    with pytest.raises(ConfigError, match="MCP headers"):
        await connection.resolve_headers()


@pytest.mark.asyncio
async def test_direct_mcp_header_provider_is_connection_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def headers() -> dict[str, str]:
        nonlocal calls
        calls += 1
        return {"Authorization": f"Bearer direct-{calls}"}

    @asynccontextmanager
    async def factory(connection: MCPConnection) -> AsyncIterator[FakeSession]:
        await connection.resolve_headers()
        yield FakeSession()

    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(
        url="https://direct.example.com/mcp",
        headers_provider=headers,
    )

    await client.list_tools()
    await client.call_tool("search", {"query": "one"})
    await client.call_tool("search", {"query": "two"})
    await client.aclose()

    assert calls == 1


@pytest.mark.asyncio
async def test_persistent_session_owns_transport_cancel_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def factory(_connection: MCPConnection) -> AsyncIterator[FakeSession]:
        async with anyio.create_task_group():
            yield FakeSession()

    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(url="https://direct.example.com/mcp")

    tools = await client.list_tools()
    await client.aclose()

    assert [tool.name for tool in tools] == ["search"]


@pytest.mark.asyncio
async def test_persistent_session_propagates_single_background_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DisconnectedSession(FakeSession):
        async def list_tools(self) -> list[dict[str, Any]]:
            raise anyio.ClosedResourceError

    async def fail_request() -> None:
        raise httpx.ReadError("connection lost")

    try:
        async with anyio.create_task_group() as task_group:
            task_group.start_soon(fail_request)
    except BaseException as exc:
        background_failure = exc
    else:  # pragma: no cover
        raise AssertionError("background failure was not raised")

    @asynccontextmanager
    async def factory(_connection: MCPConnection) -> AsyncIterator[DisconnectedSession]:
        try:
            yield DisconnectedSession()
        finally:
            raise background_failure

    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(url="https://direct.example.com/mcp")

    with pytest.raises(httpx.ReadError, match="connection lost"):
        await client.list_tools()
    await client.aclose()


@pytest.mark.asyncio
async def test_official_mcp_session_lists_all_tool_pages() -> None:
    class PagedSession:
        def __init__(self) -> None:
            self.cursors: list[str | None] = []

        async def list_tools(self, cursor: str | None = None) -> Any:
            self.cursors.append(cursor)
            if cursor is None:
                return SimpleNamespace(tools=["first"], nextCursor="page-2")
            return SimpleNamespace(tools=["second"], nextCursor=None)

    session = PagedSession()

    tools = await mcp_client_module._OfficialMCPSession(session).list_tools()

    assert tools == ["first", "second"]
    assert session.cursors == [None, "page-2"]


@pytest.mark.asyncio
async def test_mcp_call_tool_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    class FailingSession(FakeSession):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            self.calls += 1
            raise RuntimeError("side effect may have happened")

    session = FailingSession()

    @asynccontextmanager
    async def factory(_connection: MCPConnection) -> AsyncIterator[FailingSession]:
        yield session

    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(url="https://direct.example.com/mcp")
    with pytest.raises(RuntimeError):
        await client.call_tool("write", {"value": 1})
    await client.aclose()

    assert session.calls == 1


@pytest.mark.asyncio
async def test_mcp_transport_failure_reconnects_before_next_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenSession(FakeSession):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            self.calls += 1
            raise httpx.ReadError("connection lost")

    first = BrokenSession()
    second = FakeSession()
    sessions = iter((first, second))
    opened: list[FakeSession] = []
    closed: list[FakeSession] = []

    @asynccontextmanager
    async def factory(_connection: MCPConnection) -> AsyncIterator[FakeSession]:
        session = next(sessions)
        opened.append(session)
        try:
            yield session
        finally:
            closed.append(session)

    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(url="https://direct.example.com/mcp")

    with pytest.raises(httpx.ReadError):
        await client.call_tool("write", {"value": 1})
    result = await client.call_tool("write", {"value": 2})
    await client.aclose()

    assert result == {"name": "write", "arguments": {"value": 2}}
    assert opened == [first, second]
    assert closed == [first, second]
    assert first.calls == 1
    assert second.calls == 1


@pytest.mark.asyncio
async def test_mcp_connection_closed_error_reconnects_before_next_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.shared.exceptions import McpError
    from mcp.types import CONNECTION_CLOSED, ErrorData

    class ClosedSession(FakeSession):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            self.calls += 1
            raise McpError(ErrorData(code=CONNECTION_CLOSED, message="Connection closed"))

    sessions = iter((ClosedSession(), FakeSession()))
    opened: list[FakeSession] = []

    @asynccontextmanager
    async def factory(_connection: MCPConnection) -> AsyncIterator[FakeSession]:
        session = next(sessions)
        opened.append(session)
        yield session

    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(url="https://direct.example.com/mcp")

    with pytest.raises(McpError):
        await client.call_tool("write", {"value": 1})
    result = await client.call_tool("write", {"value": 2})
    await client.aclose()

    assert result == {"name": "write", "arguments": {"value": 2}}
    assert len(opened) == 2


@pytest.mark.asyncio
async def test_mcp_tool_error_keeps_the_current_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mcp.shared.exceptions import McpError
    from mcp.types import INTERNAL_ERROR, ErrorData

    class ToolErrorSession(FakeSession):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            self.calls += 1
            if self.calls == 1:
                raise McpError(ErrorData(code=INTERNAL_ERROR, message="tool failed"))
            return {"name": name, "arguments": arguments}

    session = ToolErrorSession()
    opened = 0

    @asynccontextmanager
    async def factory(_connection: MCPConnection) -> AsyncIterator[ToolErrorSession]:
        nonlocal opened
        opened += 1
        yield session

    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(url="https://direct.example.com/mcp")

    with pytest.raises(McpError):
        await client.call_tool("write", {"value": 1})
    result = await client.call_tool("write", {"value": 2})
    await client.aclose()

    assert result == {"name": "write", "arguments": {"value": 2}}
    assert opened == 1


@pytest.mark.asyncio
async def test_mcp_metadata_call_has_a_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    class BlockingSession(FakeSession):
        async def list_tools(self) -> list[dict[str, Any]]:
            await anyio.sleep_forever()

    @asynccontextmanager
    async def factory(_connection: MCPConnection) -> AsyncIterator[BlockingSession]:
        yield BlockingSession()

    monkeypatch.setattr(mcp_client_module, "_MCP_METADATA_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(url="https://direct.example.com/mcp")

    with pytest.raises(TimeoutError):
        await client.list_tools()
    await client.aclose()


@pytest.mark.asyncio
async def test_mcp_session_establishment_has_a_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def factory(_connection: MCPConnection) -> AsyncIterator[FakeSession]:
        await anyio.sleep_forever()
        yield FakeSession()

    monkeypatch.setattr(mcp_client_module, "_MCP_SESSION_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(url="https://direct.example.com/mcp")

    with pytest.raises(TimeoutError):
        await client.list_tools()
    await client.aclose()


@pytest.mark.asyncio
async def test_mcp_tool_call_has_a_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    class BlockingSession(FakeSession):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            await anyio.sleep_forever()

    sessions = iter((BlockingSession(), FakeSession()))
    opened: list[FakeSession] = []

    @asynccontextmanager
    async def factory(_connection: MCPConnection) -> AsyncIterator[FakeSession]:
        session = next(sessions)
        opened.append(session)
        yield session

    monkeypatch.setattr(mcp_client_module, "_MCP_TOOL_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(url="https://direct.example.com/mcp")

    with pytest.raises(TimeoutError):
        await client.call_tool("write", {"value": 1})
    result = await client.call_tool("write", {"value": 2})
    await client.aclose()

    assert result == {"name": "write", "arguments": {"value": 2}}
    assert len(opened) == 2


@pytest.mark.asyncio
async def test_mcp_close_wait_is_bounded_by_active_call_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = anyio.Event()
    timed_out = anyio.Event()

    class BlockingSession(FakeSession):
        async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
            entered.set()
            await anyio.sleep_forever()

    @asynccontextmanager
    async def factory(_connection: MCPConnection) -> AsyncIterator[BlockingSession]:
        yield BlockingSession()

    monkeypatch.setattr(mcp_client_module, "_MCP_TOOL_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(url="https://direct.example.com/mcp")

    async def call() -> None:
        with pytest.raises(TimeoutError):
            await client.call_tool("write", {"value": 1})
        timed_out.set()

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(call)
        await entered.wait()
        with anyio.fail_after(0.5):
            await client.aclose()

    assert timed_out.is_set()


@pytest.mark.asyncio
async def test_async_core_reuses_platform_mcp_client_by_bound_name(
    agent_config_path: Path,
    mcp_descriptor: MCPDescriptor,
) -> None:
    core = AsyncAgentCore(agent_config_path)

    class ControlPlane:
        async def resolve_mcp(self, name: str) -> MCPDescriptor:
            assert name == "search"
            return mcp_descriptor

    await core._ensure_runtime()
    core._control_plane = ControlPlane()  # type: ignore[assignment]

    first = await core.mcp("search")
    second = await core.mcp("search")

    assert first is second
    await core.aclose()


@pytest.mark.asyncio
async def test_closed_mcp_client_cannot_open_a_new_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened = 0

    @asynccontextmanager
    async def factory(_connection: MCPConnection) -> AsyncIterator[FakeSession]:
        nonlocal opened
        opened += 1
        yield FakeSession()

    monkeypatch.setattr(mcp_client_module, "_official_session", factory)
    client = AsyncMCPClient.direct(url="https://direct.example.com/mcp")
    await client.list_tools()
    await client.aclose()

    with pytest.raises(RuntimeError, match="closed"):
        await client.list_tools()

    assert opened == 1


def test_direct_mcp_repr_redacts_url_credentials() -> None:
    connection = MCPConnection(
        "https://user:password@mcp.example.com/mcp?token=secret",
        "streamable-http",
        lambda: {"Authorization": "Bearer secret"},
    )

    rendered = repr(connection)
    assert "password" not in rendered
    assert "token" not in rendered
    assert "Bearer" not in rendered
