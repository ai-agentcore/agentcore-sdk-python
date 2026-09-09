"""AG-UI protocol for AgentCore Server."""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncGenerator, AsyncIterator, Mapping
from contextlib import aclosing
from dataclasses import dataclass, field
from typing import Any

from ag_ui.core import (
    ReasoningEndEvent,
    ReasoningMessageContentEvent,
    ReasoningMessageEndEvent,
    ReasoningMessageStartEvent,
    ReasoningStartEvent,
    RunAgentInput,
    RunErrorEvent,
    RunFinishedEvent,
    RunStartedEvent,
    TextMessageContentEvent,
    TextMessageEndEvent,
    TextMessageStartEvent,
    ToolCallArgsEvent,
    ToolCallEndEvent,
    ToolCallResultEvent,
    ToolCallStartEvent,
)
from ag_ui.encoder import EventEncoder
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError
from starlette.responses import Response

from agentcore.errors import AgentCoreError
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
class _StreamState:
    text_message_id: str | None = None
    reasoning_message_id: str | None = None
    tools: dict[str, str] = field(default_factory=dict)
    tool_ids: set[str] = field(default_factory=set)
    text_ids: dict[str, None] = field(default_factory=dict)
    reasoning_ids: dict[str, None] = field(default_factory=dict)


class AGUIProtocolHandler(ProtocolHandler):
    """Expose the AgentCore AG-UI SSE endpoint."""

    name = "ag-ui"

    def __init__(self, prefix: str = "/ag-ui/agent") -> None:
        self.prefix = prefix.rstrip("/")
        self._encoder = EventEncoder()

    def get_prefix(self) -> str:
        return self.prefix

    def as_fastapi_router(self, agent_invoker: AgentInvoker) -> APIRouter:
        router = APIRouter()

        @router.post("")
        async def invoke(request: Request) -> Response:
            if not agent_invoker.configured:
                return JSONResponse(
                    {"error": {"code": "HANDLER_NOT_CONFIGURED"}},
                    status_code=503,
                )
            try:
                payload = await request.json()
                agent_request, thread_id, run_id = _parse_request(request, payload)
            except (json.JSONDecodeError, ValidationError, ValueError):
                return StreamingResponse(
                    _single_error_stream(self._encoder, "invalid request", "INVALID_REQUEST"),
                    media_type=self._encoder.get_content_type(),
                    headers=_SSE_HEADERS,
                )
            events = agent_invoker.invoke_stream(
                agent_request,
                request.headers,
                session_id=thread_id,
                run_id=run_id,
            )
            return StreamingResponse(
                with_heartbeat(
                    _format_stream(
                        events,
                        encoder=self._encoder,
                        thread_id=thread_id,
                        run_id=run_id,
                        request_id=request.headers.get("x-request-id"),
                    )
                ),
                media_type=self._encoder.get_content_type(),
                headers=_SSE_HEADERS,
            )

        return router


def _parse_request(request: Request, payload: Any) -> tuple[AgentRequest, str, str]:
    if not isinstance(payload, dict):
        raise ValueError("request body must be an object")
    validated = RunAgentInput.model_validate(payload)
    normalized = validated.model_dump(by_alias=True, mode="json")
    messages = _parse_messages(normalized["messages"])
    tools = _parse_tools(normalized["tools"])
    return (
        AgentRequest(
            protocol="agui",
            messages=messages,
            stream=True,
            tools=tools,
            raw_request=request,
            raw_payload=payload,
        ),
        validated.thread_id,
        validated.run_id,
    )


def _parse_messages(value: Any) -> list[Message]:
    if not isinstance(value, list):
        raise ValueError("messages must be an array")
    messages: list[Message] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("message must be an object")
        try:
            role = MessageRole(item["role"])
        except ValueError:
            raise ValueError("invalid message role") from None
        except KeyError:
            raise ValueError("message role is required") from None
        messages.append(
            Message(
                id=_text(item.get("id")),
                role=role,
                content=item.get("content"),
                name=_text(item.get("name")),
                tool_calls=_parse_tool_calls(item.get("toolCalls")),
                tool_call_id=_text(item.get("toolCallId")),
            )
        )
    return messages


def _parse_tool_calls(value: Any) -> list[ToolCall] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("toolCalls must be an array")
    calls: list[ToolCall] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("tool call must be an object")
        function = item.get("function")
        if not isinstance(function, Mapping):
            raise ValueError("tool call function must be an object")
        calls.append(
            ToolCall(
                id=str(item["id"]),
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
        name = item.get("name")
        description = item.get("description")
        parameters = item.get("parameters")
        if not isinstance(name, str) or not name:
            raise ValueError("tool name is required")
        if not isinstance(description, str):
            raise ValueError("tool description must be a string")
        if not isinstance(parameters, Mapping):
            raise ValueError("tool parameters must be an object")
        tools.append(
            Tool(
                name=name,
                description=description,
                parameters=parameters,
            )
        )
    return tools or None


async def _format_stream(
    events: AsyncGenerator[AgentEvent, None],
    *,
    encoder: EventEncoder,
    thread_id: str,
    run_id: str,
    request_id: str | None,
) -> AsyncIterator[str]:
    state = _StreamState()
    yield encoder.encode(RunStartedEvent(thread_id=thread_id, run_id=run_id))
    async with aclosing(events):
        try:
            async for event in events:
                if event.type == EventType.ERROR.value:
                    yield encoder.encode(
                        RunErrorEvent(
                            message=str(event.data.get("message") or "agent failed"),
                            code=_text(event.data.get("code")),
                        )
                    )
                    return
                try:
                    encoded_events = _encode_event(event, state=state, encoder=encoder)
                except Exception:
                    logger.exception(
                        "agentcore.server.agui.encode.failed request_id=%s "
                        "session_id=%s run_id=%s event_type=%s",
                        request_id or "-",
                        thread_id,
                        run_id,
                        event.type,
                    )
                    yield encoder.encode(
                        RunErrorEvent(message="handler failed", code="INTERNAL_ERROR")
                    )
                    return
                for encoded in encoded_events:
                    yield encoded
        except Exception as exc:
            code = exc.code if isinstance(exc, AgentCoreError) else "INTERNAL_ERROR"
            yield encoder.encode(RunErrorEvent(message="handler failed", code=code))
            return
    for encoded in _finish_reasoning(state, encoder):
        yield encoded
    for encoded in _finish_tools(state, encoder):
        yield encoded
    for encoded in _finish_text(state, encoder):
        yield encoded
    for encoded in _finish_identified(state, encoder, reasoning=False):
        yield encoded
    for encoded in _finish_identified(state, encoder, reasoning=True):
        yield encoded
    yield encoder.encode(RunFinishedEvent(thread_id=thread_id, run_id=run_id))


def _encode_event(
    event: AgentEvent,
    *,
    state: _StreamState,
    encoder: EventEncoder,
) -> list[str]:
    if event.type == EventType.TEXT_END.value:
        message_id = _text(event.data.get("message_id"))
        return [
            *(_finish_text(state, encoder) if message_id is None else []),
            *_finish_identified(state, encoder, reasoning=False, message_id=message_id),
        ]
    if event.type == EventType.REASONING_END.value:
        message_id = _text(event.data.get("message_id"))
        return [
            *(_finish_reasoning(state, encoder) if message_id is None else []),
            *_finish_identified(state, encoder, reasoning=True, message_id=message_id),
        ]
    if event.type == EventType.TEXT.value:
        encoded = [*_finish_reasoning(state, encoder), *_finish_tools(state, encoder)]
        message_id = _text(event.data.get("message_id"))
        if message_id:
            encoded.extend(_finish_text(state, encoder))
            started = message_id in state.text_ids
            state.text_ids[message_id] = None
        else:
            started = state.text_message_id is not None
            message_id = state.text_message_id or f"message-{uuid.uuid4().hex}"
            state.text_message_id = message_id
        if not started:
            encoded.append(
                encoder.encode(TextMessageStartEvent(message_id=message_id, role="assistant"))
            )
        delta = event.data.get("delta")
        if isinstance(delta, str) and delta:
            encoded.append(
                encoder.encode(TextMessageContentEvent(message_id=message_id, delta=delta))
            )
        return encoded
    if event.type == EventType.REASONING.value:
        encoded = [*_finish_text(state, encoder), *_finish_tools(state, encoder)]
        message_id = _text(event.data.get("message_id"))
        if message_id:
            encoded.extend(_finish_reasoning(state, encoder))
            started = message_id in state.reasoning_ids
            state.reasoning_ids[message_id] = None
        else:
            started = state.reasoning_message_id is not None
            message_id = state.reasoning_message_id or f"reasoning-{uuid.uuid4().hex}"
            state.reasoning_message_id = message_id
        if not started:
            encoded.append(encoder.encode(ReasoningStartEvent(message_id=message_id)))
            encoded.append(
                encoder.encode(
                    ReasoningMessageStartEvent(
                        message_id=message_id,
                        role="reasoning",
                    )
                )
            )
        delta = event.data.get("delta")
        if isinstance(delta, str) and delta:
            encoded.append(
                encoder.encode(
                    ReasoningMessageContentEvent(
                        message_id=message_id,
                        delta=delta,
                    )
                )
            )
        return encoded
    if event.type == EventType.TOOL_CALL_CHUNK.value:
        encoded = [*_finish_text(state, encoder), *_finish_reasoning(state, encoder)]
        tool_id = _text(event.data.get("id"))
        if tool_id is None:
            return encoded
        if tool_id not in state.tool_ids:
            name = _text(event.data.get("name")) or ""
            state.tools[tool_id] = name
            state.tool_ids.add(tool_id)
            encoded.append(
                encoder.encode(ToolCallStartEvent(tool_call_id=tool_id, tool_call_name=name))
            )
        arguments = event.data.get("args_delta")
        if isinstance(arguments, str) and arguments:
            encoded.append(encoder.encode(ToolCallArgsEvent(tool_call_id=tool_id, delta=arguments)))
        return encoded
    if event.type == EventType.TOOL_RESULT.value:
        tool_id = _text(event.data.get("id"))
        if tool_id is None:
            return []
        encoded = [*_finish_text(state, encoder), *_finish_reasoning(state, encoder)]
        if tool_id not in state.tool_ids:
            name = _text(event.data.get("name")) or ""
            state.tools[tool_id] = name
            state.tool_ids.add(tool_id)
            encoded.append(
                encoder.encode(ToolCallStartEvent(tool_call_id=tool_id, tool_call_name=name))
            )
        encoded.extend(_finish_tool(tool_id, state, encoder))
        result = event.data.get("result", "")
        content = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        encoded.append(
            encoder.encode(
                ToolCallResultEvent(
                    message_id=f"tool-result-{tool_id}",
                    tool_call_id=tool_id,
                    content=content,
                    role="tool",
                )
            )
        )
        return encoded
    return [_encode_raw(event.to_dict())]


def _finish_identified(
    state: _StreamState,
    encoder: EventEncoder,
    *,
    reasoning: bool,
    message_id: str | None = None,
) -> list[str]:
    # Explicit identities may be interleaved. Only their own END closes them.
    active = state.reasoning_ids if reasoning else state.text_ids
    ids = list(active) if message_id is None else [message_id]
    encoded = []
    for identity in ids:
        if identity not in active:
            continue
        del active[identity]
        if reasoning:
            encoded.extend(
                [
                    encoder.encode(ReasoningMessageEndEvent(message_id=identity)),
                    encoder.encode(ReasoningEndEvent(message_id=identity)),
                ]
            )
        else:
            encoded.append(encoder.encode(TextMessageEndEvent(message_id=identity)))
    return encoded


def _finish_text(state: _StreamState, encoder: EventEncoder) -> list[str]:
    if state.text_message_id is None:
        return []
    message_id = state.text_message_id
    state.text_message_id = None
    return [encoder.encode(TextMessageEndEvent(message_id=message_id))]


def _finish_reasoning(state: _StreamState, encoder: EventEncoder) -> list[str]:
    if state.reasoning_message_id is None:
        return []
    message_id = state.reasoning_message_id
    state.reasoning_message_id = None
    return [
        encoder.encode(ReasoningMessageEndEvent(message_id=message_id)),
        encoder.encode(ReasoningEndEvent(message_id=message_id)),
    ]


def _finish_tool(tool_id: str, state: _StreamState, encoder: EventEncoder) -> list[str]:
    if state.tools.pop(tool_id, None) is None:
        return []
    return [encoder.encode(ToolCallEndEvent(tool_call_id=tool_id))]


def _finish_tools(state: _StreamState, encoder: EventEncoder) -> list[str]:
    encoded: list[str] = []
    for tool_id in list(state.tools):
        encoded.extend(_finish_tool(tool_id, state, encoder))
    return encoded


async def _single_error_stream(
    encoder: EventEncoder,
    message: str,
    code: str,
) -> AsyncIterator[str]:
    yield encoder.encode(RunErrorEvent(message=message, code=code))


def _encode_raw(value: Mapping[str, Any]) -> str:
    return f"data: {json.dumps(value, ensure_ascii=False, separators=(',', ':'))}\n\n"


def _text(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
