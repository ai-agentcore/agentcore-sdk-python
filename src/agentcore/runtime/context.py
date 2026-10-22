"""Request-scoped HTTP context."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType

from agentcore.errors import ContextError


@dataclass(frozen=True, repr=False, eq=False)
class RequestContext:
    headers: Mapping[str, str]
    _cleanups: list[Callable[[], Awaitable[None]]] = field(default_factory=list, init=False)

    def on_close(self, cleanup: Callable[[], Awaitable[None]]) -> None:
        self._cleanups.append(cleanup)

    async def aclose(self) -> None:
        cleanups = list(self._cleanups)
        self._cleanups.clear()

        async def run(cleanup: Callable[[], Awaitable[None]]) -> None:
            await cleanup()

        results = await asyncio.gather(
            *(run(cleanup) for cleanup in cleanups), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                raise result

    def __post_init__(self) -> None:
        normalized = {key.lower(): value for key, value in self.headers.items()}
        object.__setattr__(self, "headers", MappingProxyType(normalized))

    def __repr__(self) -> str:
        return "RequestContext(headers=<redacted>)"


_current_context: ContextVar[RequestContext | None] = ContextVar(
    "agentcore_request_context", default=None
)


def request_context_from_headers(
    headers: Mapping[str, str],
) -> RequestContext:
    return RequestContext(headers=headers)


def current_context(*, required: bool = True) -> RequestContext | None:
    context = _current_context.get()
    if context is None and required:
        raise ContextError("no AgentCore request context is active")
    return context


@contextmanager
def use_context(context: RequestContext) -> Iterator[None]:
    token = _current_context.set(context)
    try:
        yield
    finally:
        _current_context.reset(token)
