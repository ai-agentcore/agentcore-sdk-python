"""Protocol-neutral streaming events emitted by AgentCore handlers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any


class EventType(str, Enum):
    TEXT = "TEXT"
    TEXT_END = "TEXT_END"
    REASONING = "REASONING"
    REASONING_END = "REASONING_END"
    TOOL_CALL = "TOOL_CALL"
    TOOL_CALL_CHUNK = "TOOL_CALL_CHUNK"
    TOOL_RESULT = "TOOL_RESULT"
    ERROR = "ERROR"


@dataclass(frozen=True)
class AgentEvent:
    event: EventType | str
    data: Mapping[str, Any]

    @property
    def type(self) -> str:
        return self.event.value if isinstance(self.event, EventType) else self.event

    def to_dict(self) -> dict[str, Any]:
        return {**self.data, "type": self.type}
