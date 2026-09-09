"""Exercise managed fixed headers through the official MCP HTTP transport."""

import asyncio
import socket
from dataclasses import replace
from pathlib import Path

import pytest

from agentcore.controlplane import MCPDescriptor
from agentcore.mcp import AsyncMCPClient
from agentcore.runtime.config import load_agent_config


@pytest.mark.asyncio
@pytest.mark.parametrize("use_credential", [False, True])
async def test_managed_mcp_headers_over_http(
    agent_config_path: Path,
    mcp_descriptor: MCPDescriptor,
    use_credential: bool,
) -> None:
    pytest.importorskip("mcp")
    uvicorn = pytest.importorskip("uvicorn")
    from mcp.server.fastmcp import FastMCP
    from mcp.shared.exceptions import McpError

    server_mcp = FastMCP(
        "headers-test",
        streamable_http_path="/mcp-servers/search-id",
        json_response=True,
    )

    @server_mcp.tool()
    async def echo(value: str) -> str:
        return value

    app = server_mcp.streamable_http_app()
    received: list[dict[str, str]] = []
    reject_sessions = False

    async def capture(scope, receive, send):  # type: ignore[no-untyped-def]
        if scope["type"] == "http":
            headers = {name.decode(): value.decode() for name, value in scope["headers"]}
            received.append(headers)
            if reject_sessions and "mcp-session-id" in headers:
                await send({"type": "http.response.start", "status": 404, "headers": []})
                await send({"type": "http.response.body", "body": b"session missing"})
                return
        await app(scope, receive, send)

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(capture, log_config=None))
        serving = asyncio.create_task(server.serve(sockets=[listener]))
        config = load_agent_config(agent_config_path)
        config = replace(
            config,
            spec=replace(config.spec, mcp=replace(config.spec.mcp, gateway_url=url)),
        )
        fetches = {"a": 0, "b": 0}

        def provider(key):
            async def get():
                fetches[key] += 1
                return {"X-API-Key": key}

            return get

        clients = [
            AsyncMCPClient.platform(
                config,
                mcp_descriptor,
                headers=None if use_credential or key is None else {"X-API-Key": key},
                credential_headers_provider=provider(key) if use_credential and key else None,
            )
            for key in ("a", "b", None)
        ]
        try:

            async def wait_for_start() -> None:
                while not server.started:
                    if serving.done():
                        await serving
                        raise AssertionError("MCP server exited before startup")
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_start(), 5)
            await asyncio.gather(*(client.list_tools() for client in clients))
            results = await asyncio.gather(
                *(
                    client.call_tool("echo", {"value": str(index)})
                    for index, client in enumerate(clients)
                )
            )
            assert [result.content[0].text for result in results] == ["0", "1", "2"]
            assert {headers.get("x-api-key") for headers in received} == {"a", "b", None}
            sessions: dict[str, set[str | None]] = {}
            for headers in received:
                assert headers["authorization"] == "Bearer gateway-secret"
                if session_id := headers.get("mcp-session-id"):
                    sessions.setdefault(session_id, set()).add(headers.get("x-api-key"))
            assert len(sessions) == 3
            assert all(len(values) == 1 for values in sessions.values())
            if use_credential:
                assert fetches == {"a": 1, "b": 1}
                reject_sessions = True
                with pytest.raises(McpError, match="Session terminated"):
                    await clients[0].call_tool("echo", {"value": "lost-session"})
                reject_sessions = False
                await clients[0].call_tool("echo", {"value": "reconnected"})
                assert fetches == {"a": 2, "b": 1}
        finally:
            await asyncio.gather(*(client.aclose() for client in clients))
            server.should_exit = True
            await asyncio.wait_for(serving, 5)
