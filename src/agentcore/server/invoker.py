"""Protocol-neutral invocation and result normalization."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import (
    AsyncGenerator,
    AsyncIterable,
    AsyncIterator,
    Callable,
    Iterable,
    Mapping,
)
from functools import partial
from typing import Any

import anyio

from agentcore.collaboration.context import bind_collaboration_context
from agentcore.errors import AgentCoreError
from agentcore.runtime.context import RequestContext, request_context_from_headers, use_context
from agentcore.server.events import AgentEvent, EventType
from agentcore.server.model import AgentRequest

InvokeHandler = Callable[[AgentRequest, RequestContext], Any]
logger = logging.getLogger(__name__)
_STOP = object()


class AgentInvoker:
    """Call one user handler and normalize its values into Agent events."""

    def __init__(self) -> None:
        self._handler: InvokeHandler | None = None

    @property
    def configured(self) -> bool:
        return self._handler is not None

    def set_handler(self, handler: InvokeHandler) -> None:
        self._handler = handler

    async def invoke(
        self,
        request: AgentRequest,
        headers: Mapping[str, str],
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> list[AgentEvent]:
        return [
            event
            async for event in self.invoke_stream(
                request,
                headers,
                session_id=session_id,
                run_id=run_id,
            )
        ]

    async def invoke_stream(
        self,
        request: AgentRequest,
        headers: Mapping[str, str],
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> AsyncGenerator[AgentEvent, None]:
        handler = self._handler
        if handler is None:
            raise RuntimeError("AgentCore handler is not configured")
        context = request_context_from_headers(headers)
        request_id = context.headers.get("x-request-id")
        logger.info(
            "agentcore.server.invoke.started protocol=%s request_id=%s session_id=%s run_id=%s",
            request.protocol,
            request_id or "-",
            session_id or "-",
            run_id or "-",
        )
        try:
            with use_context(context), bind_collaboration_context(headers):
                result = await _call_handler(handler, request, context)
                async for item in _iterate_result(result):
                    for event in _normalize_item(item):
                        yield event
        except Exception as exc:
            code = exc.code if isinstance(exc, AgentCoreError) else "INTERNAL_ERROR"
            logger.exception(
                "agentcore.server.invoke.failed protocol=%s request_id=%s session_id=%s "
                "run_id=%s code=%s error_type=%s",
                request.protocol,
                request_id or "-",
                session_id or "-",
                run_id or "-",
                code,
                type(exc).__name__,
            )
            raise
        logger.info(
            "agentcore.server.invoke.succeeded protocol=%s request_id=%s session_id=%s run_id=%s",
            request.protocol,
            request_id or "-",
            session_id or "-",
            run_id or "-",
        )


async def _call_handler(
    handler: InvokeHandler,
    request: AgentRequest,
    context: RequestContext,
) -> Any:
    if inspect.iscoroutinefunction(handler) or inspect.isasyncgenfunction(handler):
        result = handler(request, context)
    else:
        result = await anyio.to_thread.run_sync(partial(handler, request, context))
    if inspect.isawaitable(result):
        return await result
    return result


async def _iterate_result(result: Any) -> AsyncIterator[Any]:
    if isinstance(result, AsyncIterable):
        iterator = aiter(result)
        try:
            async for item in iterator:
                yield item
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                await close()
        return
    if _is_sync_iterable(result):
        iterator = iter(result)

        def safe_next() -> Any:
            try:
                return next(iterator)
            except StopIteration:
                return _STOP

        while True:
            item = await anyio.to_thread.run_sync(safe_next)
            if item is _STOP:
                return
            yield item
    else:
        yield result


def _is_sync_iterable(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(
        value,
        (str, bytes, Mapping, AgentEvent),
    )


def _normalize_item(item: Any) -> list[AgentEvent]:
    if item is None:
        return []
    if isinstance(item, str):
        return [AgentEvent(EventType.TEXT, {"delta": item})] if item else []
    if isinstance(item, AgentEvent):
        if item.type != EventType.TOOL_CALL.value:
            return [item]
        arguments = item.data.get("args", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False, separators=(",", ":"))
        return [
            AgentEvent(
                EventType.TOOL_CALL_CHUNK,
                {
                    "id": str(item.data.get("id") or ""),
                    "name": str(item.data.get("name") or ""),
                    "args_delta": arguments,
                },
            )
        ]
    raise TypeError("handler must return text, AgentEvent, or an event iterable")
