from __future__ import annotations

import base64
import inspect
import json
import logging
import threading
from typing import Any

import anyio
import httpx
import pytest
from fastapi import APIRouter, Request

from agentcore.collaboration import current_collaboration_context
from agentcore.runtime.context import current_context
from agentcore.server import (
    AgentCoreServer,
    AgentEvent,
    AgentRequest,
    EventType,
    OpenAIProtocolHandler,
    ProtocolHandler,
)
from agentcore.server.sse import with_heartbeat


def test_agent_event_type_cannot_be_overridden_by_data() -> None:
    event = AgentEvent("CUSTOM", {"type": "OVERRIDE", "value": 1})

    assert event.to_dict() == {"type": "CUSTOM", "value": 1}


async def request(
    app: AgentCoreServer,
    path: str,
    *,
    method: str | None = None,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    request_body = body
    if path == "/ag-ui/agent" and body is not None:
        request_body = {
            "threadId": "test-thread",
            "runId": "test-run",
            "messages": [],
            "tools": [],
            "context": [],
            "forwardedProps": {},
            **body,
        }
        request_body["messages"] = [
            {"id": f"message-{index}", **message}
            if isinstance(message, dict) and "id" not in message
            else message
            for index, message in enumerate(request_body["messages"])
        ]
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.request(
            method or ("POST" if path == "/ag-ui/agent" else "GET"),
            path,
            json=request_body,
            headers=headers,
        )
    return response.status_code, dict(response.headers), response.content


@pytest.mark.asyncio
async def test_server_injects_request_context() -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(payload, context):  # type: ignore[no-untyped-def]
        assert isinstance(payload, AgentRequest)
        assert payload.raw_payload["input"] == "hello"
        assert payload.protocol == "agui"
        return AgentEvent(
            "CUSTOM",
            {
                "sessionHeader": context.headers.get("x-agentcore-session-id"),
                "userHeader": context.headers.get("x-agentcore-user-id"),
                "requestId": context.headers.get("x-request-id"),
            },
        )

    status, _, body = await request(
        server,
        "/ag-ui/agent",
        body={"input": "hello", "threadId": "protocol-session", "runId": "run-a"},
        headers={
            "X-AgentCore-Session-ID": "gateway-session",
            "X-AgentCore-User-ID": "user-a",
            "X-Request-ID": "request-a",
        },
    )

    assert status == 200
    events = [
        json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")
    ]
    assert events[0]["type"] == "RUN_STARTED"
    assert events[0]["threadId"] == "protocol-session"
    assert events[1] == {
        "type": "CUSTOM",
        "sessionHeader": "gateway-session",
        "userHeader": "user-a",
        "requestId": "request-a",
    }
    assert events[2]["type"] == "RUN_FINISHED"


@pytest.mark.asyncio
async def test_server_agui_stream_sends_heartbeat_while_handler_is_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agentcore.server.sse._HEARTBEAT_INTERVAL_SECONDS", 0.01)
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        await anyio.sleep(0.03)
        return "done"

    status, _, body = await request(server, "/ag-ui/agent", body={})

    assert status == 200
    assert ": ping\n\n" in body.decode()


@pytest.mark.asyncio
async def test_sse_heartbeat_closes_source_when_client_disconnects() -> None:
    closed = anyio.Event()

    async def source():  # type: ignore[no-untyped-def]
        try:
            yield "data: first\n\n"
            await anyio.sleep_forever()
        finally:
            closed.set()

    stream = with_heartbeat(source())
    assert await anext(stream) == "data: first\n\n"

    await stream.aclose()

    assert closed.is_set()


@pytest.mark.asyncio
async def test_server_accepts_request_without_agentcore_session_header() -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        return AgentEvent("CUSTOM", {})

    status, _, body = await request(
        server,
        "/ag-ui/agent",
        body={"threadId": "protocol-session"},
    )

    assert status == 200
    events = [
        json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")
    ]
    assert events[0]["threadId"] == "protocol-session"


@pytest.mark.asyncio
async def test_server_parses_agui_tools_using_the_official_flat_shape() -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(payload, _context):  # type: ignore[no-untyped-def]
        assert payload.tools is not None
        assert payload.tools[0].name == "search"
        assert payload.tools[0].description == "Search documents"
        assert payload.tools[0].parameters == {
            "type": "object",
            "properties": {"query": {"type": "string"}},
        }
        return "ok"

    status, _, body = await request(
        server,
        "/ag-ui/agent",
        body={
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [
                {
                    "name": "search",
                    "description": "Search documents",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                }
            ],
        },
    )

    assert status == 200
    assert b'TEXT_MESSAGE_CONTENT' in body


@pytest.mark.asyncio
async def test_server_rejects_invalid_agui_message_role() -> None:
    server = AgentCoreServer()
    invoked = False

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        nonlocal invoked
        invoked = True
        return "unexpected"

    status, _, body = await request(
        server,
        "/ag-ui/agent",
        body={"messages": [{"role": "invalid", "content": "hello"}]},
    )

    assert status == 200
    assert not invoked
    assert b'"type":"RUN_ERROR"' in body
    assert b'"code":"INVALID_REQUEST"' in body


@pytest.mark.asyncio
async def test_server_health_and_readiness() -> None:
    server = AgentCoreServer(readiness=lambda: True)
    assert (await request(server, "/healthz"))[0] == 200
    assert (await request(server, "/readyz"))[0] == 200


@pytest.mark.asyncio
async def test_server_default_readiness_only_requires_handler() -> None:
    server = AgentCoreServer()
    assert {"config_provider", "config_path"}.isdisjoint(
        inspect.signature(AgentCoreServer).parameters
    )
    assert (await request(server, "/readyz"))[0] == 503

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        return AgentEvent("CUSTOM", {})

    assert (await request(server, "/readyz"))[0] == 200


@pytest.mark.asyncio
async def test_server_lifespan_runs_callbacks() -> None:
    calls: list[str] = []

    async def startup() -> None:
        calls.append("startup")

    async def shutdown() -> None:
        calls.append("shutdown")

    server = AgentCoreServer(startup=startup, shutdown=shutdown)
    incoming = iter(
        [
            {"type": "lifespan.startup"},
            {"type": "lifespan.shutdown"},
        ]
    )
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return next(incoming)

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await server({"type": "lifespan"}, receive, send)

    assert calls == ["startup", "shutdown"]
    assert [message["type"] for message in messages] == [
        "lifespan.startup.complete",
        "lifespan.shutdown.complete",
    ]


@pytest.mark.asyncio
async def test_server_runs_sync_handler_off_loop_and_propagates_context() -> None:
    server = AgentCoreServer()
    event_loop_thread = threading.get_ident()

    @server.invoke
    def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        context = current_context()
        assert context is not None
        return AgentEvent(
            "CUSTOM",
            {
                "requestId": context.headers.get("x-request-id"),
                "offLoop": threading.get_ident() != event_loop_thread,
            },
        )

    status, _, body = await request(
        server,
        "/ag-ui/agent",
        body={"threadId": "session-a"},
        headers={"X-Request-ID": "request-a"},
    )
    events = [
        json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")
    ]

    assert status == 200
    assert events[1] == {"type": "CUSTOM", "requestId": "request-a", "offLoop": True}


@pytest.mark.asyncio
async def test_server_converts_stream_failure_to_agui_run_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        yield AgentEvent("CUSTOM", {"value": 1})
        raise RuntimeError("secret internal detail")

    with caplog.at_level(logging.WARNING):
        status, _, body = await request(
            server,
            "/ag-ui/agent",
            body={"threadId": "session-a", "runId": "run-a"},
            headers={"X-Request-ID": "request-a"},
        )
    events = [
        json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")
    ]

    assert status == 200
    assert [event["type"] for event in events] == [
        "RUN_STARTED",
        "CUSTOM",
        "RUN_ERROR",
    ]
    assert "secret internal detail" not in body.decode()
    assert caplog.text.count("agentcore.server.invoke.failed") == 1
    assert "agentcore.server.stream.failed" not in caplog.text
    assert "request-a" in caplog.text
    assert "session-a" in caplog.text
    assert "run-a" in caplog.text
    assert "secret internal detail" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_server_logs_openai_handler_failure_once(
    caplog: pytest.LogCaptureFixture,
    stream: bool,
) -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        raise RuntimeError("openai internal detail")

    with caplog.at_level(logging.WARNING):
        status, _, body = await request(
            server,
            "/openai/v1/chat/completions",
            method="POST",
            body={"model": "test-agent", "messages": [], "stream": stream},
            headers={"X-Request-ID": "request-openai"},
        )

    assert status == (200 if stream else 500)
    assert "openai internal detail" not in body.decode()
    assert caplog.text.count("agentcore.server.invoke.failed") == 1
    assert "agentcore.server.openai.stream.failed" not in caplog.text
    assert "agentcore.server.openai.failed" not in caplog.text
    assert "openai internal detail" in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "body"),
    [
        ("/ag-ui/agent", {"threadId": "session-a", "runId": "run-a"}),
        (
            "/openai/v1/chat/completions",
            {"model": "test-agent", "messages": [], "stream": True},
        ),
    ],
)
async def test_server_closes_handler_stream_after_error_event(
    caplog: pytest.LogCaptureFixture,
    path: str,
    body: dict[str, Any],
) -> None:
    server = AgentCoreServer()
    finalized = False

    async def events():  # type: ignore[no-untyped-def]
        nonlocal finalized
        try:
            yield AgentEvent(EventType.ERROR, {"message": "expected failure"})
            yield "not consumed"
        finally:
            finalized = True

    event_stream = events()

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        return event_stream

    with caplog.at_level(logging.ERROR):
        status, _, _ = await request(server, path, method="POST", body=body)

    assert status == 200
    assert finalized
    assert "created in a different Context" not in caplog.text


@pytest.mark.asyncio
async def test_server_logs_agui_encoding_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        return AgentEvent("CUSTOM", {"value": object()})

    with caplog.at_level(logging.ERROR):
        status, _, body = await request(
            server,
            "/ag-ui/agent",
            body={"threadId": "session-a", "runId": "run-a"},
            headers={"X-Request-ID": "request-a"},
        )

    assert status == 200
    assert b'"type":"RUN_ERROR"' in body
    assert "agentcore.server.agui.encode.failed" in caplog.text
    assert "Object of type object is not JSON serializable" in caplog.text


@pytest.mark.asyncio
async def test_server_awaits_awaitable_returned_by_sync_wrapper() -> None:
    server = AgentCoreServer()

    async def implementation() -> AgentEvent:
        return AgentEvent("CUSTOM", {"value": "ok"})

    @server.invoke
    def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        return implementation()

    _, _, body = await request(server, "/ag-ui/agent", body={"threadId": "session-a"})
    events = [
        json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")
    ]

    assert events[1] == {"type": "CUSTOM", "value": "ok"}


@pytest.mark.asyncio
async def test_server_openai_chat_completions_non_stream() -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(payload, context):  # type: ignore[no-untyped-def]
        assert isinstance(payload, AgentRequest)
        assert payload.protocol == "openai"
        assert payload.messages[0].role.value == "user"
        assert payload.messages[0].content == "hello"
        assert payload.raw_payload["messages"] == [{"role": "user", "content": "hello"}]
        assert context.headers["x-request-id"] == "request-openai"
        return "hello from agent"

    status, response_headers, body = await request(
        server,
        "/openai/v1/chat/completions",
        method="POST",
        body={
            "model": "test-agent",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        },
        headers={"X-Request-ID": "request-openai"},
    )
    response = json.loads(body)

    assert status == 200
    assert response_headers["content-type"] == "application/json"
    assert response["object"] == "chat.completion"
    assert response["model"] == "test-agent"
    assert response["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "hello from agent"},
            "finish_reason": "stop",
        }
    ]


@pytest.mark.asyncio
async def test_server_binds_collaboration_context_for_openai_requests() -> None:
    server = AgentCoreServer()
    encoded_context = base64.urlsafe_b64encode(
        json.dumps(
            {
                "version": 1,
                "teamId": "team-alpha",
                "roomId": "!room:matrix.example.com",
                "eventId": "$event",
                "roomKind": "task",
            },
            separators=(",", ":"),
        ).encode()
    ).decode().rstrip("=")

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        collaboration = current_collaboration_context()
        assert collaboration is not None
        return collaboration.team_id

    status, _, body = await request(
        server,
        "/openai/v1/chat/completions",
        method="POST",
        body={"model": "test-agent", "messages": [], "stream": False},
        headers={
            "X-AgentCore-Session-ID": "session-1",
            "X-AgentCore-Collaboration-Context": encoded_context,
        },
    )

    assert status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "team-alpha"
    assert current_collaboration_context(required=False) is None


@pytest.mark.asyncio
async def test_server_preserves_invalid_collaboration_context_code_in_agui_stream() -> None:
    server = AgentCoreServer()
    called = False

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        return "unexpected"

    status, _, body = await request(
        server,
        "/ag-ui/agent",
        body={},
        headers={"X-AgentCore-Collaboration-Context": "not-base64"},
    )
    events = [
        json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")
    ]

    assert status == 200
    assert [event["type"] for event in events] == ["RUN_STARTED", "RUN_ERROR"]
    assert events[1]["code"] == "COLLABORATION_CONTEXT_INVALID"
    assert called is False


@pytest.mark.asyncio
async def test_server_preserves_invalid_collaboration_context_code_in_openai_stream() -> None:
    server = AgentCoreServer()
    called = False

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        nonlocal called
        called = True
        return "unexpected"

    status, response_headers, body = await request(
        server,
        "/openai/v1/chat/completions",
        method="POST",
        body={"model": "test-agent", "messages": [], "stream": True},
        headers={"X-AgentCore-Collaboration-Context": "not-base64"},
    )

    data = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")]

    assert status == 200
    assert response_headers["content-type"] == "text/event-stream; charset=utf-8"
    assert json.loads(data[-2])["error"]["code"] == "COLLABORATION_CONTEXT_INVALID"
    assert data[-1] == "[DONE]"
    assert called is False


@pytest.mark.asyncio
async def test_server_openai_chat_completions_stream() -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        yield AgentEvent(EventType.TEXT, {"delta": "hello "})
        yield AgentEvent(EventType.TEXT, {"delta": "world"})

    status, response_headers, body = await request(
        server,
        "/openai/v1/chat/completions",
        method="POST",
        body={"model": "test-agent", "messages": [], "stream": True},
    )
    data = [line[6:] for line in body.decode().splitlines() if line.startswith("data: ")]
    chunks = [json.loads(item) for item in data[:-1]]

    assert status == 200
    assert response_headers["content-type"] == "text/event-stream; charset=utf-8"
    assert data[-1] == "[DONE]"
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert [chunk["choices"][0]["delta"].get("content") for chunk in chunks[1:3]] == [
        "hello ",
        "world",
    ]
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_server_openai_stream_sends_heartbeat_while_handler_is_idle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agentcore.server.sse._HEARTBEAT_INTERVAL_SECONDS", 0.01)
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        await anyio.sleep(0.03)
        return "done"

    status, _, body = await request(
        server,
        "/openai/v1/chat/completions",
        method="POST",
        body={"model": "test-agent", "messages": [], "stream": True},
    )

    assert status == 200
    assert ": ping\n\n" in body.decode()


@pytest.mark.asyncio
async def test_server_openai_chat_completions_maps_tool_calls() -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        yield AgentEvent(
            EventType.TOOL_CALL,
            {"id": "call-1", "name": "search", "args": {"query": "agentcore"}},
        )

    status, _, body = await request(
        server,
        "/openai/v1/chat/completions",
        method="POST",
        body={"model": "test-agent", "messages": [], "stream": False},
    )
    response = json.loads(body)

    assert status == 200
    assert response["choices"][0] == {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"query":"agentcore"}'},
                }
            ],
        },
        "finish_reason": "tool_calls",
    }


@pytest.mark.asyncio
async def test_server_rejects_openai_message_without_role() -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        return "unexpected"

    status, _, body = await request(
        server,
        "/openai/v1/chat/completions",
        method="POST",
        body={"messages": [{"content": "hello"}]},
    )

    assert status == 400
    assert json.loads(body)["error"]["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_server_does_not_accept_mapping_events() -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_payload, _context):  # type: ignore[no-untyped-def]
        return {"type": "CUSTOM", "value": "old shape"}

    status, _, body = await request(
        server,
        "/ag-ui/agent",
        body={},
    )
    events = [
        json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")
    ]

    assert status == 200
    assert [event["type"] for event in events] == ["RUN_STARTED", "RUN_ERROR"]


@pytest.mark.asyncio
async def test_server_openai_models() -> None:
    server = AgentCoreServer(protocols=[OpenAIProtocolHandler(model_name="test-agent")])

    status, _, body = await request(server, "/openai/v1/models", method="GET")

    assert status == 200
    assert json.loads(body)["data"][0]["id"] == "test-agent"


@pytest.mark.asyncio
async def test_server_accepts_custom_protocol_handler() -> None:
    class EchoProtocolHandler(ProtocolHandler):
        name = "echo"

        def get_prefix(self) -> str:
            return "/echo"

        def as_fastapi_router(self, invoker):  # type: ignore[no-untyped-def]
            router = APIRouter()

            @router.post("")
            async def echo(request: Request):
                payload = await request.json()
                agent_request = AgentRequest(
                    protocol=self.name,
                    messages=[],
                    stream=False,
                    tools=None,
                    raw_request=request,
                    raw_payload=payload,
                )
                events = await invoker.invoke(agent_request, request.headers)
                return {"result": "".join(event.data["delta"] for event in events)}

            return router

    server = AgentCoreServer(protocols=[EchoProtocolHandler()])

    @server.invoke
    async def invoke(payload, _context):  # type: ignore[no-untyped-def]
        return payload.raw_payload["value"]

    status, _, body = await request(
        server,
        "/echo",
        method="POST",
        body={"value": "custom protocol"},
    )

    assert status == 200
    assert json.loads(body) == {"result": "custom protocol"}


@pytest.mark.asyncio
async def test_server_normalizes_sync_generator_without_blocking_loop() -> None:
    server = AgentCoreServer()
    event_loop_thread = threading.get_ident()

    @server.invoke
    def invoke(_request, _context):  # type: ignore[no-untyped-def]
        def generate():  # type: ignore[no-untyped-def]
            assert threading.get_ident() != event_loop_thread
            yield "hello "
            yield AgentEvent(EventType.TEXT, {"delta": "world"})

        return generate()

    status, _, body = await request(
        server,
        "/openai/v1/chat/completions",
        method="POST",
        body={"messages": [], "stream": False},
    )

    assert status == 200
    assert json.loads(body)["choices"][0]["message"]["content"] == "hello world"


@pytest.mark.asyncio
async def test_server_maps_core_events_to_agui_boundaries() -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_request, _context):  # type: ignore[no-untyped-def]
        yield AgentEvent(EventType.TEXT, {"delta": "working"})
        yield AgentEvent(
            EventType.TOOL_CALL_CHUNK,
            {"id": "call-1", "name": "search", "args_delta": '{"q":"sdk"}'},
        )
        yield AgentEvent(EventType.TOOL_RESULT, {"id": "call-1", "result": "found"})

    status, _, body = await request(
        server,
        "/ag-ui/agent",
        body={"threadId": "thread-1", "runId": "run-1", "messages": []},
    )
    events = [
        json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")
    ]

    assert status == 200
    assert [event["type"] for event in events] == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "RUN_FINISHED",
    ]


@pytest.mark.asyncio
async def test_server_emits_complete_agui_reasoning_lifecycle() -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_request, _context):  # type: ignore[no-untyped-def]
        yield AgentEvent(EventType.REASONING, {"delta": "thinking"})
        yield AgentEvent(EventType.TEXT, {"delta": "answer"})

    _, _, body = await request(server, "/ag-ui/agent", body={})
    events = [
        json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")
    ]

    assert [event["type"] for event in events] == [
        "RUN_STARTED",
        "REASONING_START",
        "REASONING_MESSAGE_START",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_END",
        "REASONING_END",
        "TEXT_MESSAGE_START",
        "TEXT_MESSAGE_CONTENT",
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]


@pytest.mark.asyncio
async def test_server_wraps_direct_tool_result_with_agui_boundaries() -> None:
    server = AgentCoreServer()

    @server.invoke
    async def invoke(_request, _context):  # type: ignore[no-untyped-def]
        return AgentEvent(
            EventType.TOOL_RESULT,
            {"id": "call-1", "name": "search", "result": "found"},
        )

    _, _, body = await request(
        server,
        "/ag-ui/agent",
        body={"threadId": "thread-1", "runId": "run-1"},
    )
    events = [
        json.loads(line[6:]) for line in body.decode().splitlines() if line.startswith("data: ")
    ]

    assert [event["type"] for event in events] == [
        "RUN_STARTED",
        "TOOL_CALL_START",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "RUN_FINISHED",
    ]


def test_server_exposes_fastapi_app() -> None:
    server = AgentCoreServer()

    assert server.as_fastapi_app() is server.app
