"""Public synchronous MCP APIs and request cleanup across the SDK portal."""

import asyncio
import json
import socket
from collections.abc import AsyncIterator
from pathlib import Path

import anyio
import pytest

from agentcore import AgentCore
from agentcore.auth import AccessKeyCredential
from agentcore.runtime.context import RequestContext
from agentcore.server import AgentRequest
from agentcore.server.invoker import AgentInvoker


@pytest.fixture
async def sync_mcp_endpoint(
    agent_config_path: Path,
) -> AsyncIterator[tuple[str, list[tuple[str, dict[str, str]]]]]:
    pytest.importorskip("mcp")
    uvicorn = pytest.importorskip("uvicorn")
    from mcp.server.fastmcp import FastMCP

    service = FastMCP("sync-mcp", streamable_http_path="/mcp-servers/search-id", json_response=True)

    @service.tool()
    def echo(value: str) -> str:
        return value

    app = service.streamable_http_app()
    received: list[tuple[str, dict[str, str]]] = []

    async def routes(scope, receive, send):  # type: ignore[no-untyped-def]
        if scope["type"] == "http":
            if scope["path"] == "/workspaces/workspace-a/mcp-servers":
                body = json.dumps(
                    {
                        "items": [
                            {
                                "mcpServerId": "search-id",
                                "name": "search",
                                "protocol": "HTTP",
                                "type": "CUSTOM",
                                "status": "READY",
                            }
                        ]
                    }
                ).encode()
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send({"type": "http.response.body", "body": body})
                return
            received.append(
                (scope["method"], {key.decode(): value.decode() for key, value in scope["headers"]})
            )
        await app(scope, receive, send)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        config_path = anyio.Path(agent_config_path)
        await config_path.write_text(
            (await config_path.read_text()).replace("https://gateway.example.com/mcp", url)
        )
        server = uvicorn.Server(uvicorn.Config(routes, log_config=None))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        try:

            async def wait_for_start() -> None:
                while not server.started:
                    if serving.done():
                        await serving
                        raise AssertionError("MCP server exited before startup")
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_start(), 5)
            yield url, received
        finally:
            server.should_exit = True
            await asyncio.wait_for(serving, 5)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["completed", "failed", "cancelled"])
@pytest.mark.parametrize("close_in_handler", [False, True])
async def test_sync_request_closes_mcp_on_its_own_loop(
    agent_config_path: Path,
    sync_mcp_endpoint: tuple[str, list[tuple[str, dict[str, str]]]],
    finish: str,
    close_in_handler: bool,
) -> None:
    url, received = sync_mcp_endpoint
    core = AgentCore.auto(
        config_path=agent_config_path,
        control_plane_endpoint=url,
        access_key_credential=AccessKeyCredential("test-key", "test-secret"),
    )

    def handler(_request: AgentRequest, _context: RequestContext) -> list[str]:
        try:
            mcp = core.mcp("search")
            tools = mcp.list_tools()
            assert tools[0].invoke({"value": "first"}).content[0].text == "first"
            assert mcp.call_tool("echo", {"value": "second"}).content[0].text == "second"
        finally:
            if close_in_handler:
                core.close()
        if finish == "failed":
            raise ValueError("handler failed")
        return ["first", "second"]

    invoker = AgentInvoker()
    invoker.set_handler(handler)
    request = AgentRequest("openai", [], True, None, None, {})
    stream = invoker.invoke_stream(request, {"X-AgentCore-Accessor-WAT": "sync-wat"})

    async def consume() -> None:
        if finish == "failed":
            with pytest.raises(ValueError, match="handler failed"):
                await anext(stream)
        else:
            await anext(stream)
            if finish == "cancelled":
                await stream.aclose()
            else:
                assert len([event async for event in stream]) == 1

    try:
        await asyncio.wait_for(consume(), 5)
        deletes = [headers for method, headers in received if method == "DELETE"]
        assert len(deletes) == 1, "The request session must close before invocation cleanup returns"
        assert all(headers.get("x-agentcore-accessor-wat") == "sync-wat" for _, headers in received)
        assert all(
            headers.get("authorization") == "Bearer gateway-secret" for _, headers in received
        )
        assert (
            len(
                {
                    headers["mcp-session-id"]
                    for _, headers in received
                    if "mcp-session-id" in headers
                }
            )
            == 1
        )
    finally:
        await anyio.to_thread.run_sync(core.close)


@pytest.mark.asyncio
@pytest.mark.parametrize("managed", [False, True])
async def test_sync_mcp_per_call_headers_reuse_session(
    agent_config_path: Path,
    sync_mcp_endpoint: tuple[str, list[tuple[str, dict[str, str]]]],
    managed: bool,
) -> None:
    url, received = sync_mcp_endpoint

    def call() -> None:
        with AgentCore.auto(
            config_path=agent_config_path,
            control_plane_endpoint=url,
            access_key_credential=AccessKeyCredential("test-key", "test-secret"),
        ) as core:
            mcp = (
                core.mcp("search")
                if managed
                else core.direct_mcp(
                    url=f"{url}/mcp-servers/search-id",
                    headers_provider=lambda: {"Authorization": "Bearer direct"},
                )
            )
            assert mcp.tools(headers={"X-Call": "tools"})[0].name == "echo"
            assert mcp.list_tools(headers={"X-Call": "list"})[0].name == "echo"
            assert (
                mcp.call_tool("echo", {"value": "call"}, headers={"X-Call": "call"}).content[0].text
                == "call"
            )
            assert mcp.call_tool("echo", {"value": "plain"}).content[0].text == "plain"
            with pytest.raises(Exception, match="protected"):
                mcp.call_tool("echo", {}, headers={"X-AgentCore-Accessor-WAT": "forged"})

    await asyncio.wait_for(anyio.to_thread.run_sync(call), 5)
    assert {headers.get("x-call") for _, headers in received} == {"tools", "list", "call", None}
    assert all("x-agentcore-accessor-wat" not in headers for _, headers in received)
    auth = "Bearer gateway-secret" if managed else "Bearer direct"
    assert all(headers.get("authorization") == auth for _, headers in received)
    posts = [headers for method, headers in received if method == "POST"]
    assert "x-call" not in posts[-1]
    assert (
        len({headers["mcp-session-id"] for _, headers in received if "mcp-session-id" in headers})
        == 1
    )
    assert received[-1][0] == "DELETE" and "x-call" not in received[-1][1]
