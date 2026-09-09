"""Request-scoped HTTP context."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType

from agentcore.errors import ContextError


@dataclass(frozen=True, repr=False)
class RequestContext:
    headers: Mapping[str, str]

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
