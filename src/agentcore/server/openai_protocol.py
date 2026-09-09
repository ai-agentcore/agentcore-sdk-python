"""OpenAI Chat Completions protocol for AgentCore Server."""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from agentcore.errors import AgentCoreError, ConfigError, ContextError
from agentcore.server.events import AgentEvent, EventType
from agentcore.server.invoker import AgentInvoker
from agentcore.server.model import AgentRequest, Message, MessageRole, Tool, ToolCall
from agentcore.server.protocol import ProtocolHandler
from agentcore.server.sse import with_heartbeat

logger = logging.getLogger(__name__)
_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@dataclass
class _ToolCallState:
    id: str
    index: int
    name: str = ""
    arguments: str = ""


@dataclass
class _CompletionState:
    content: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    tool_calls: dict[str, _ToolCallState] = field(default_factory=dict)

    def consume(self, event: AgentEvent) -> list[dict[str, Any]]:
        if event.type == EventType.TEXT.value:
            delta = event.data.get("delta")
            if isinstance(delta, str) and delta:
                self.content.append(delta)
                return [{"content": delta}]
            return []
        if event.type == EventType.REASONING.value:
            delta = event.data.get("delta")
            if isinstance(delta, str) and delta:
                self.reasoning.append(delta)
                return [{"reasoning_content": delta}]
            return []
        if event.type != EventType.TOOL_CALL_CHUNK.value:
            return []
        tool_id = event.data.get("id")
        if not isinstance(tool_id, str) or not tool_id:
            return []
        call = self.tool_calls.get(tool_id)
        started = call is None
        if call is None:
            call = _ToolCallState(id=tool_id, index=len(self.tool_calls))
            self.tool_calls[tool_id] = call
        name = event.data.get("name")
        if isinstance(name, str) and name:
            call.name = name
        arguments = event.data.get("args_delta")
        if isinstance(arguments, str):
            call.arguments += arguments
        deltas: list[dict[str, Any]] = []
        if started:
            deltas.append(
                {
                    "tool_calls": [
                        {
                            "index": call.index,
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.name, "arguments": ""},
                        }
                    ]
                }
            )
        if isinstance(arguments, str) and arguments:
            deltas.append(
                {
                    "tool_calls": [
                        {
                            "index": call.index,
                            "function": {"arguments": arguments},
                        }
                    ]
                }
            )
        return deltas


class OpenAIProtocolHandler(ProtocolHandler):
    """Expose OpenAI-compatible Chat Completions and Models endpoints."""

    name = "openai_chat_completions"

    def __init__(self, prefix: str = "/openai/v1", *, model_name: str = "agentcore") -> None:
        self.prefix = prefix.rstrip("/")
        self.model_name = model_name

    def get_prefix(self) -> str:
        return self.prefix

    def as_fastapi_router(self, agent_invoker: AgentInvoker) -> APIRouter:
        router = APIRouter()

        @router.post("/chat/completions")
        async def chat_completions(request: Request) -> Response:
            if not agent_invoker.configured:
                return _error(503, "handler is not configured", "server_error")
            try:
                payload = await request.json()
                agent_request = _parse_request(request, payload)
                model = _text(payload.get("model")) or self.model_name
                session_id = _first_text(payload, "sessionId", "threadId", "session_id")
                run_id = _first_text(payload, "runId", "run_id")
                if agent_request.stream:
                    event_stream = agent_invoker.invoke_stream(
                        agent_request,
                        request.headers,
                        session_id=session_id,
                        run_id=run_id,
                    )
                    return StreamingResponse(
                        with_heartbeat(
                            _format_stream(
                                event_stream,
                                model=model,
                                request_id=request.headers.get("x-request-id"),
                                session_id=session_id,
                                run_id=run_id,
                            )
                        ),
                        media_type="text/event-stream",
                        headers=_SSE_HEADERS,
                    )
                event_list = await agent_invoker.invoke(
                    agent_request,
                    request.headers,
                    session_id=session_id,
                    run_id=run_id,
                )
                return JSONResponse(_format_non_stream(event_list, model=model))
            except (ValueError, json.JSONDecodeError, ContextError, ConfigError) as exc:
                message = exc.message if isinstance(exc, AgentCoreError) else "invalid request"
                code = exc.code if isinstance(exc, AgentCoreError) else None
                return _error(400, message, "invalid_request_error", code=code)
            except AgentCoreError as exc:
                return _error(502, exc.message, "server_error", code=exc.code)
            except Exception:
                return _error(500, "handler failed", "server_error")

        @router.get("/models")
        async def models() -> dict[str, Any]:
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.model_name,
                        "object": "model",
                        "created": int(time.time()),
                        "owned_by": "agentcore",
                    }
                ],
            }

        return router


def _parse_request(request: Request, payload: Any) -> AgentRequest:
    if not isinstance(payload, dict):
        raise ValueError("request body must be an object")
    raw_messages = payload.get("messages")
    if not isinstance(raw_messages, list):
        raise ValueError("messages must be an array")
    return AgentRequest(
        protocol="openai",
        messages=_parse_messages(raw_messages),
        stream=payload.get("stream") is True,
        tools=_parse_tools(payload.get("tools")),
        raw_request=request,
        raw_payload=payload,
    )


def _parse_messages(values: Sequence[Any]) -> list[Message]:
    messages: list[Message] = []
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("message must be an object")
        role_value = value.get("role")
        if role_value not in {"developer", "system", "user", "assistant", "tool"}:
            raise ValueError("invalid message role")
        try:
            role = MessageRole(role_value)
        except ValueError:
            raise ValueError("invalid message role") from None
        messages.append(
            Message(
                id=_text(value.get("id")),
                role=role,
                content=value.get("content"),
                name=_text(value.get("name")),
                tool_calls=_parse_tool_calls(value.get("tool_calls")),
                tool_call_id=_text(value.get("tool_call_id")),
            )
        )
    return messages


def _parse_tool_calls(value: Any) -> list[ToolCall] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("tool_calls must be an array")
    calls: list[ToolCall] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("tool call must be an object")
        function = item.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("tool call function must be an object")
        tool_call_id = item.get("id")
        if not isinstance(tool_call_id, str) or not tool_call_id:
            raise ValueError("tool call id is required")
        calls.append(
            ToolCall(
                id=tool_call_id,
                type=str(item.get("type") or "function"),
                function=function,
            )
        )
    return calls or None


def _parse_tools(value: Any) -> list[Tool] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("tools must be an array")
    tools: list[Tool] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("tool must be an object")
        function = item.get("function")
        if item.get("type", "function") != "function" or not isinstance(function, Mapping):
            raise ValueError("only function tools are supported")
        name = function.get("name")
        description = function.get("description", "")
        parameters = function.get("parameters", {})
        if not isinstance(name, str) or not name:
            raise ValueError("tool function name is required")
        if not isinstance(description, str):
            raise ValueError("tool function description must be a string")
        if not isinstance(parameters, Mapping):
            raise ValueError("tool function parameters must be an object")
        tools.append(
            Tool(
                name=name,
                description=description,
                parameters=parameters,
            )
        )
    return tools or None


def _format_non_stream(events: Sequence[AgentEvent], *, model: str) -> dict[str, Any]:
    state = _CompletionState()
    for event in events:
        if event.type == EventType.ERROR.value:
            raise RuntimeError("agent reported an error")
        state.consume(event)
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "".join(state.content) if state.content else None,
    }
    if state.reasoning:
        message["reasoning_content"] = "".join(state.reasoning)
    if state.tool_calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in state.tool_calls.values()
        ]
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": "tool_calls" if state.tool_calls else "stop",
            }
        ],
    }


async def _format_stream(
    events: AsyncGenerator[AgentEvent, None],
    *,
    model: str,
    request_id: str | None,
    session_id: str | None,
    run_id: str | None,
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    state = _CompletionState()
    yield _sse(_chunk(completion_id, created, model, {"role": "assistant"}))
    failed = False
    async with aclosing(events):
        try:
            async for event in events:
                if event.type == EventType.ERROR.value:
                    yield _sse({"error": {"message": "handler failed", "type": "server_error"}})
                    failed = True
                    break
                try:
                    chunks = [
                        _sse(_chunk(completion_id, created, model, delta))
                        for delta in state.consume(event)
                    ]
                except Exception:
                    logger.exception(
                        "agentcore.server.openai.encode.failed request_id=%s "
                        "session_id=%s run_id=%s event_type=%s",
                        request_id or "-",
                        session_id or "-",
                        run_id or "-",
                        event.type,
                    )
                    yield _sse(
                        {"error": {"message": "handler failed", "type": "server_error"}}
                    )
                    failed = True
                    break
                for chunk in chunks:
                    yield chunk
        except Exception as exc:
            error: dict[str, Any] = {"message": "handler failed", "type": "server_error"}
            if isinstance(exc, AgentCoreError):
                error["code"] = exc.code
            yield _sse({"error": error})
            failed = True
        if not failed:
            yield _sse(
                _chunk(
                    completion_id,
                    created,
                    model,
                    {},
                    finish_reason="tool_calls" if state.tool_calls else "stop",
                )
            )
    yield "data: [DONE]\n\n"


def _chunk(
    completion_id: str,
    created: int,
    model: str,
    delta: Mapping[str, Any],
    *,
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
    }


def _sse(value: Mapping[str, Any]) -> str:
    return f"data: {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _error(
    status: int,
    message: str,
    error_type: str,
    *,
    code: str | None = None,
) -> JSONResponse:
    error: dict[str, Any] = {"message": message, "type": error_type}
    if code is not None:
        error["code"] = code
    return JSONResponse({"error": error}, status_code=status)


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _first_text(payload: Mapping[str, Any], *names: str) -> str | None:
    for name in names:
        value = _text(payload.get(name))
        if value is not None:
            return value
    return None
