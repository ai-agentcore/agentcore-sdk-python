"""Small shared utilities for per-invocation framework event converters."""

from collections.abc import AsyncIterable, AsyncIterator
from typing import Any

from agentcore.events import AgentEvent, EventType


def field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


class EventConverter:
    """Create one converter per invocation; pass the full native event stream."""

    def convert(self, event: Any) -> list[AgentEvent]:
        raise NotImplementedError

    async def stream(self, events: AsyncIterable[Any]) -> AsyncIterator[AgentEvent]:
        async for event in events:
            for converted in self.convert(event):
                yield converted


def end_text() -> AgentEvent:
    return AgentEvent(EventType.TEXT_END, {})
