"""Persistent async MCP sessions with explicit platform/direct routing."""

from __future__ import annotations

import asyncio
import inspect
import logging
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import quote, urlsplit

import anyio
import httpx

if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

from agentcore._logging import safe_url
from agentcore.controlplane import MCPDescriptor
from agentcore.errors import ConfigError
from agentcore.integrations.common import Tool
from agentcore.mcp._headers import merge_mcp_headers
from agentcore.runtime.config import AgentConfig

HeadersProvider = Callable[[], Mapping[str, str] | Awaitable[Mapping[str, str]]]
_MCP_SESSION_TIMEOUT_SECONDS = 30.0
_MCP_METADATA_TIMEOUT_SECONDS = 60.0
_MCP_TOOL_TIMEOUT_SECONDS = 600.0
logger = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False)
class MCPConnection:
    url: str
    transport: str
    headers_provider: HeadersProvider | None = None

    async def resolve_headers(self) -> dict[str, str]:
        if self.headers_provider is None:
            return {}
        result = self.headers_provider()
        if inspect.isawaitable(result):
            result = await result
        if not isinstance(result, Mapping):
            raise ConfigError("MCP headers_provider must return a mapping")
        headers = dict(result)
        if not all(
            isinstance(key, str) and isinstance(value, str) for key, value in headers.items()
        ):
            raise ConfigError("MCP headers must contain string keys and values")
        return headers

    def __repr__(self) -> str:
        parsed = urlsplit(self.url)
        safe_url = f"{parsed.scheme}://{parsed.hostname or ''}{parsed.path}"
        return f"MCPConnection(url={safe_url!r}, transport={self.transport!r}, headers=<redacted>)"


class MCPSession(Protocol):
    async def list_tools(self) -> Sequence[Any]: ...

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


@dataclass
class _SessionState:
    context: AbstractAsyncContextManager[MCPSession]
    session: MCPSession
    active: int = 0
    idle: anyio.Event | None = None
    invalidated: bool = False


class AsyncMCPClient:
    def __init__(
        self,
        connection: MCPConnection,
        *,
        descriptor: MCPDescriptor | None = None,
    ) -> None:
        self._connection = connection
        self.descriptor = descriptor
        self._state: _SessionState | None = None
        self._lock = anyio.Lock()
        self._closed = False
        logger.info(
            "agentcore.mcp.client.created mode=%s name=%s transport=%s host=%s url=%s",
            "managed" if descriptor is not None else "direct",
            descriptor.name if descriptor is not None else "direct",
            connection.transport,
            urlsplit(connection.url).hostname or "",
            safe_url(connection.url),
        )

    async def __aenter__(self) -> AsyncMCPClient:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.aclose()

    @classmethod
    def platform(
        cls,
        config: AgentConfig,
        descriptor: MCPDescriptor,
        *,
        headers: Mapping[str, str] | None = None,
        credential_headers_provider: HeadersProvider | None = None,
    ) -> AsyncMCPClient:
        return cls(
            _platform_connection(descriptor, config, headers, credential_headers_provider),
            descriptor=descriptor,
        )

    @classmethod
    def direct(
        cls,
        *,
        url: str,
        transport: str = "streamable-http",
        headers_provider: HeadersProvider | None = None,
    ) -> AsyncMCPClient:
        try:
            parsed = urlsplit(url)
        except ValueError as exc:
            raise ConfigError("Direct MCP URL must be a valid HTTP(S) URL") from exc
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ConfigError("Direct MCP URL must be an absolute HTTP(S) URL")
        if transport not in {"streamable-http", "sse"}:
            raise ConfigError("Direct MCP transport must be streamable-http or sse")
        return cls(
            MCPConnection(url=url, transport=transport, headers_provider=headers_provider),
        )

    async def list_tools(self) -> list[Tool]:
        logger.info(
            "agentcore.mcp.tools.list.started name=%s url=%s timeout_seconds=%s",
            self._log_name,
            safe_url(self._connection.url),
            _MCP_METADATA_TIMEOUT_SECONDS,
        )
        try:
            state = await self._acquire_session_with_deadline()
            with anyio.fail_after(_MCP_METADATA_TIMEOUT_SECONDS):
                try:
                    definitions = await state.session.list_tools()
                except BaseException as exc:
                    if _session_is_unusable(exc):
                        with anyio.CancelScope(shield=True):
                            await self._invalidate_session(state, exc)
                    raise
                finally:
                    with anyio.CancelScope(shield=True):
                        await self._release_session(state)
            tools = [self._to_tool(definition) for definition in definitions]
        except Exception as exc:
            logger.warning(
                "agentcore.mcp.tools.list.failed name=%s error_type=%s url=%s",
                self._log_name,
                type(exc).__name__,
                safe_url(self._connection.url),
            )
            raise
        logger.info(
            "agentcore.mcp.tools.list.succeeded name=%s tool_count=%s",
            self._log_name,
            len(tools),
        )
        return tools

    async def tools(self) -> list[Tool]:
        return await self.list_tools()

    async def call_tool(self, name: str, arguments: Mapping[str, Any]) -> Any:
        logger.info(
            "agentcore.mcp.tool.call.started name=%s tool=%s url=%s timeout_seconds=%s",
            self._log_name,
            name,
            safe_url(self._connection.url),
            _MCP_TOOL_TIMEOUT_SECONDS,
        )
        try:
            state = await self._acquire_session_with_deadline()
            with anyio.fail_after(_MCP_TOOL_TIMEOUT_SECONDS):
                try:
                    result = await state.session.call_tool(name, dict(arguments))
                except BaseException as exc:
                    if _session_is_unusable(exc):
                        with anyio.CancelScope(shield=True):
                            await self._invalidate_session(state, exc)
                    raise
                finally:
                    with anyio.CancelScope(shield=True):
                        await self._release_session(state)
        except Exception as exc:
            logger.warning(
                "agentcore.mcp.tool.call.failed name=%s tool=%s error_type=%s url=%s",
                self._log_name,
                name,
                type(exc).__name__,
                safe_url(self._connection.url),
            )
            raise
        logger.info(
            "agentcore.mcp.tool.call.succeeded name=%s tool=%s",
            self._log_name,
            name,
        )
        return result

    async def aclose(self) -> None:
        while True:
            async with self._lock:
                self._closed = True
                state = self._state
                if state is None:
                    logger.info("agentcore.mcp.client.closed name=%s", self._log_name)
                    return
                await self._close_if_idle_unlocked(state)
                if self._state is None:
                    return
                assert state.idle is not None
                waiter = state.idle
            await waiter.wait()

    async def _acquire_session_with_deadline(self) -> _SessionState:
        with anyio.fail_after(_MCP_SESSION_TIMEOUT_SECONDS):
            return await self._acquire_session()

    async def _acquire_session(self) -> _SessionState:
        while True:
            async with self._lock:
                if self._closed:
                    raise RuntimeError("MCP client is closed")
                state = self._state
                if state is not None and state.invalidated:
                    await self._close_if_idle_unlocked(state)
                    if self._state is state:
                        assert state.idle is not None
                        waiter = state.idle
                    else:
                        continue
                else:
                    if state is None:
                        logger.info(
                            "agentcore.mcp.session.open.started name=%s transport=%s "
                            "url=%s timeout_seconds=%s",
                            self._log_name,
                            self._connection.transport,
                            safe_url(self._connection.url),
                            _MCP_SESSION_TIMEOUT_SECONDS,
                        )
                        context = _owned_session(self._connection)
                        session = await context.__aenter__()
                        state = _SessionState(context, session)
                        self._state = state
                        logger.info("agentcore.mcp.session.opened name=%s", self._log_name)
                    if state.active == 0:
                        state.idle = anyio.Event()
                    state.active += 1
                    return state
            await waiter.wait()

    async def _invalidate_session(self, state: _SessionState, exc: BaseException) -> None:
        async with self._lock:
            if state.invalidated:
                return
            state.invalidated = True
            logger.warning(
                "agentcore.mcp.session.invalidated name=%s error_type=%s",
                self._log_name,
                type(exc).__name__,
            )
            await self._close_if_idle_unlocked(state)

    async def _release_session(self, state: _SessionState) -> None:
        async with self._lock:
            state.active -= 1
            if state.active < 0:
                raise RuntimeError("MCP Session lease underflow")
            await self._close_if_idle_unlocked(state)

    async def _close_if_idle_unlocked(self, state: _SessionState) -> None:
        if state.active != 0:
            return
        if state.idle is not None:
            state.idle.set()
        if (self._closed or state.invalidated) and self._state is state:
            self._state = None
            await state.context.__aexit__(None, None, None)
            logger.info("agentcore.mcp.session.closed name=%s", self._log_name)

    @property
    def _log_name(self) -> str:
        return self.descriptor.name if self.descriptor is not None else "direct"

    def _to_tool(self, definition: Any) -> Tool:
        name = _field(definition, "name")
        description = _field(definition, "description", default="")
        schema = _field(definition, "inputSchema", default={})
        if not isinstance(name, str) or not name:
            raise ConfigError("MCP returned a tool without a valid name")
        if not isinstance(description, str) or not isinstance(schema, Mapping):
            raise ConfigError(f"MCP returned invalid metadata for tool {name}")

        async def invoke(arguments: dict[str, Any]) -> Any:
            return await self.call_tool(name, arguments)

        return Tool(
            name=name,
            description=description,
            parameters=dict(schema),
            _call=invoke,
        )


def _platform_connection(
    descriptor: MCPDescriptor,
    config: AgentConfig,
    headers: Mapping[str, str] | None = None,
    credential_headers_provider: HeadersProvider | None = None,
) -> MCPConnection:
    mcp = config.spec.mcp
    gateway = mcp.gateway_url.rstrip("/")
    if gateway.endswith("/mcp"):
        gateway = gateway[:-4]
    server_id = quote(descriptor.mcp_server_id, safe="")
    captured_headers = merge_mcp_headers(config.spec.gateway_headers, headers)
    custom_headers = dict(headers or {})

    async def resolve_headers() -> Mapping[str, str]:
        if credential_headers_provider is None:
            return captured_headers
        credential_headers = credential_headers_provider()
        if inspect.isawaitable(credential_headers):
            credential_headers = await credential_headers
        bound = merge_mcp_headers(config.spec.gateway_headers, credential_headers)
        return merge_mcp_headers(bound, custom_headers)

    return MCPConnection(
        url=f"{gateway}/mcp-servers/{server_id}",
        transport="streamable-http",
        headers_provider=resolve_headers,
    )


def _field(value: Any, name: str, *, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _session_is_unusable(exc: BaseException) -> bool:
    if isinstance(
        exc,
        (
            TimeoutError,
            anyio.get_cancelled_exc_class(),
            anyio.BrokenResourceError,
            anyio.ClosedResourceError,
            anyio.EndOfStream,
            httpx.TransportError,
        ),
    ):
        return True
    try:
        from mcp.shared.exceptions import McpError
        from mcp.types import CONNECTION_CLOSED
    except ImportError:
        return False
    if not isinstance(exc, McpError):
        return False
    # Official Streamable HTTP transport maps HTTP 404 to this exact error.
    return exc.error.code == CONNECTION_CLOSED or (
        exc.error.code == 32600 and exc.error.message == "Session terminated"
    )


class _OfficialMCPSession:
    def __init__(self, session: Any) -> None:
        self._session = session

    async def list_tools(self) -> Sequence[Any]:
        tools: list[Any] = []
        cursor: str | None = None
        while True:
            result = await self._session.list_tools(cursor=cursor)
            tools.extend(result.tools)
            cursor = result.nextCursor
            if not cursor:
                return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return await self._session.call_tool(name, arguments)


@asynccontextmanager
async def _owned_session(connection: MCPConnection) -> AsyncIterator[MCPSession]:
    ready: asyncio.Future[MCPSession] = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()

    async def run() -> None:
        try:
            async with _official_session(connection) as session:
                ready.set_result(session)
                await stop.wait()
        except BaseException as exc:
            if not ready.done():
                ready.set_exception(exc)
            raise

    task = asyncio.create_task(run())
    try:
        yield await asyncio.shield(ready)
    finally:
        stop.set()
        if not ready.done():
            ready.cancel()
            task.cancel()
        try:
            await task
        except BaseException as exc:
            cause = _unwrap_single_exception_group(exc)
            if cause is exc:
                raise
            raise cause from exc


def _unwrap_single_exception_group(exc: BaseException) -> BaseException:
    while isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
        exc = exc.exceptions[0]
    return exc


@asynccontextmanager
async def _official_session(connection: MCPConnection) -> AsyncIterator[MCPSession]:
    try:
        from mcp import ClientSession
    except ImportError as exc:
        raise ImportError(
            "MCP support requires the 'mcp' extra: "
            "pip install alibabacloud-agentcore-sdk[mcp]"
        ) from exc

    headers = await connection.resolve_headers()
    if connection.transport == "streamable-http":
        from mcp.client.streamable_http import streamablehttp_client

        async with streamablehttp_client(connection.url, headers=headers) as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield _OfficialMCPSession(session)
    else:
        from mcp.client.sse import sse_client

        async with sse_client(connection.url, headers=headers) as streams:
            read_stream, write_stream = streams[0], streams[1]
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield _OfficialMCPSession(session)
