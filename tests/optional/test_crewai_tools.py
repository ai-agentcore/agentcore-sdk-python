from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any

import pytest

from agentcore.integrations.common import Tool

pytest.importorskip("crewai")

from agentcore.integrations.crewai import tools  # noqa: E402


def make_tool(call: Any) -> Tool:
    return Tool(
        "echo",
        "Echo a value",
        {"type": "object", "properties": {"value": {"type": "string"}}},
        call,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["run", "structured_invoke", "arun"])
async def test_crewai_tool_runs_on_its_original_loop(entry: str) -> None:
    owner = asyncio.get_running_loop()
    calls = []

    async def invoke(arguments: dict[str, Any]) -> Any:
        calls.append(asyncio.get_running_loop())
        # A real loop-owned Future reproduces the affinity of a persistent MCP Session.
        response = owner.create_future()
        owner.call_soon(response.set_result, arguments)
        return await response

    adapted = tools([make_tool(invoke)])[0]
    if entry == "run":
        result = await asyncio.to_thread(adapted.run, value="test")
    elif entry == "structured_invoke":
        result = await asyncio.to_thread(adapted.to_structured_tool().invoke, {"value": "test"})
    else:
        result = await adapted.arun(value="test")

    assert result == {"value": "test"}
    assert calls == [owner]


@pytest.mark.asyncio
async def test_crewai_tool_propagates_upstream_exception() -> None:
    error = ValueError("tool failed")

    async def invoke(arguments: dict[str, Any]) -> Any:
        raise error

    adapted = tools([make_tool(invoke)])[0]
    with pytest.raises(ValueError, match="tool failed") as caught:
        await asyncio.to_thread(adapted.to_structured_tool().invoke, {})
    assert caught.value is error


@pytest.mark.asyncio
async def test_crewai_foreign_loop_cancellation_reaches_original_tool() -> None:
    started = threading.Event()
    cancelled = asyncio.Event()

    async def invoke(arguments: dict[str, Any]) -> Any:
        started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.set()

    adapted = tools([make_tool(invoke)])[0]

    async def foreign_call() -> None:
        task = asyncio.create_task(adapted.arun())
        assert await asyncio.to_thread(started.wait, 3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    await asyncio.wait_for(asyncio.to_thread(asyncio.run, foreign_call()), 5)
    await asyncio.wait_for(cancelled.wait(), 1)


def test_crewai_sync_capable_tool_is_not_bound_to_conversion_loop() -> None:
    async def invoke(arguments: dict[str, Any]) -> Any:
        pytest.fail("sync-capable tool should use its existing bridge")

    canonical = make_tool(invoke).with_sync_call(lambda arguments: arguments)

    async def convert() -> Any:
        return tools([canonical])[0]

    adapted = asyncio.run(convert())
    assert adapted.run(value="test") == {"value": "test"}


def test_crewai_stateless_tool_can_be_converted_without_running_loop() -> None:
    async def invoke(arguments: dict[str, Any]) -> Any:
        return arguments

    adapted = tools([make_tool(invoke)])[0]
    assert adapted.run(value="test") == {"value": "test"}


def test_crewai_closed_owner_loop_reports_lifecycle_error() -> None:
    async def invoke(arguments: dict[str, Any]) -> Any:
        pytest.fail("closed owner loop must not execute the tool on a different loop")

    async def convert() -> Any:
        return tools([make_tool(invoke)])[0]

    adapted = asyncio.run(convert())
    with pytest.raises(RuntimeError, match="original tool event loop"):
        adapted.run(value="test")


@pytest.mark.asyncio
async def test_crewai_reuses_real_streamable_http_mcp_session() -> None:
    pytest.importorskip("mcp")
    uvicorn = pytest.importorskip("uvicorn")
    from mcp.server.fastmcp import FastMCP

    from agentcore.mcp import AsyncMCPClient

    server_calls = []
    mcp_server = FastMCP("crewai-test")

    @mcp_server.tool()
    async def echo(value: str) -> str:
        server_calls.append(value)
        return value

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(mcp_server.streamable_http_app(), log_config=None))
        serving = asyncio.create_task(server.serve(sockets=[listener]))

        async def wait_for_start() -> None:
            while not server.started:
                if serving.done():
                    await serving
                    raise AssertionError("MCP server exited before startup")
                await asyncio.sleep(0.01)

        client = AsyncMCPClient.direct(url=f"http://127.0.0.1:{port}/mcp")
        try:
            await asyncio.wait_for(wait_for_start(), 5)
            adapted = tools(await client.list_tools())[0]
            structured = adapted.to_structured_tool()
            for value in ["first", "second"]:
                result = await asyncio.wait_for(
                    asyncio.to_thread(structured.invoke, {"value": value}), 5
                )
                assert not result.isError
                assert result.content[0].text == value
            assert server_calls == ["first", "second"]
        finally:
            await client.aclose()
            server.should_exit = True
            await asyncio.wait_for(serving, 5)
