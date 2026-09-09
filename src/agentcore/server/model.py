"""Protocol-neutral request models shared by AgentCore server handlers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    DEVELOPER = "developer"
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    ACTIVITY = "activity"
    REASONING = "reasoning"


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: Mapping[str, Any]
    type: str = "function"


@dataclass(frozen=True)
class Message:
    role: MessageRole
    content: str | Sequence[Mapping[str, Any]] | None = None
    id: str | None = None
    name: str | None = None
    tool_calls: Sequence[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Mapping[str, Any]


@dataclass(frozen=True)
class AgentRequest:
    """Protocol-neutral Agent request."""

    protocol: str
    messages: Sequence[Message]
    stream: bool
    tools: Sequence[Tool] | None
    raw_request: Any
    raw_payload: Mapping[str, Any]
