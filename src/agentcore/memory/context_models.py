"""Immutable native Context Memory models and source-event conversion."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal

from agentcore.errors import MemoryValidationError
from agentcore.memory.models import MemoryContent, MemoryScope, _frozen_metadata

SourceEventType = Literal["message", "tool_call", "tool_result", "model_request", "model_response"]


class MemoryType(str, Enum):
    SEMANTIC = "semantic"
    SKILL = "skill"
    USER_PROFILE = "user_profile"
    SESSION_SUMMARY = "session_summary"


class MemorySource(str, Enum):
    SESSION_EXTRACTION = "session_extraction"
    CONTENT_EXTRACTION = "content_extraction"
    DIRECT = "direct"
    IMPORTED = "imported"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise MemoryValidationError("JSON object keys must be strings")
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            _thaw(value), ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
        )
    except (TypeError, ValueError) as exc:
        raise MemoryValidationError("event must contain finite JSON values") from exc


def _nonempty(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MemoryValidationError(f"{name} must be a non-empty string")


def _blocks(value: Any) -> None:
    if not isinstance(value, (list, tuple)):
        raise MemoryValidationError("event content must be a list")
    for block in value:
        if not isinstance(block, Mapping):
            raise MemoryValidationError("event content blocks must be objects")
        _nonempty(block.get("type"), "content[].type")
        if block["type"] == "text" and not isinstance(block.get("text"), str):
            raise MemoryValidationError("text content must contain text")
        if block["type"] == "opaque":
            _nonempty(block.get("format"), "opaque.format")
            if "value" not in block:
                raise MemoryValidationError("opaque content must contain value")


@dataclass(frozen=True)
class ContextEvent:
    type: SourceEventType
    timestamp: int
    data: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.timestamp) is not int or not 0 <= self.timestamp <= 2**53 - 1:
            raise MemoryValidationError("timestamp must be a non-negative safe integer")
        if not isinstance(self.data, Mapping):
            raise MemoryValidationError("event data must be an object")
        data = self.data
        if self.type == "message":
            if data.get("role") not in ("user", "assistant", "system", "developer", "tool"):
                raise MemoryValidationError("invalid message role")
            _blocks(data.get("content"))
            if data.get("role") == "tool" or "callId" in data:
                _nonempty(data.get("callId"), "callId")
        elif self.type == "tool_call":
            _nonempty(data.get("callId"), "callId")
            _nonempty(data.get("name"), "name")
            if "arguments" not in data:
                raise MemoryValidationError("tool_call requires arguments")
        elif self.type == "tool_result":
            _nonempty(data.get("callId"), "callId")
            if data.get("status") not in ("succeeded", "failed", "cancelled"):
                raise MemoryValidationError("invalid tool_result status")
            _blocks(data.get("content"))
        elif self.type == "model_request":
            if not isinstance(data.get("input"), (list, tuple)):
                raise MemoryValidationError("model_request requires input list")
        elif self.type == "model_response":
            if data.get("status") not in ("succeeded", "failed", "cancelled"):
                raise MemoryValidationError("invalid model_response status")
        else:
            raise MemoryValidationError("unsupported event type")
        frozen = _freeze(data)
        if (
            len(
                _canonical_json(
                    {"type": self.type, "timestamp": self.timestamp, "data": frozen}
                ).encode("utf-8")
            )
            > 900000
        ):
            raise MemoryValidationError("event exceeds 900000 UTF-8 bytes")
        object.__setattr__(self, "data", frozen)

    @classmethod
    def message(
        cls,
        timestamp: int,
        role: str,
        content: Sequence[Mapping[str, object]],
        *,
        call_id: str | None = None,
    ) -> ContextEvent:
        data: dict[str, object] = {"role": role, "content": list(content)}
        if call_id is not None:
            data["callId"] = call_id
        return cls.from_source("message", timestamp, data)

    @classmethod
    def typed(
        cls,
        event_type: SourceEventType,
        timestamp: int,
        data: Mapping[str, object],
    ) -> ContextEvent:
        """Opt in to typed events only when the deployment supports them."""
        return cls(event_type, timestamp, data)

    @classmethod
    def from_source(
        cls,
        event_type: SourceEventType,
        timestamp: int,
        data: Mapping[str, object],
    ) -> ContextEvent:
        """Convert a source event into the deployed message envelope."""
        event = cls.typed(event_type, timestamp, data)
        wire = event.to_dict()["data"]
        if event_type == "message":
            content = []
            for block in wire["content"]:
                if block["type"] == "opaque":
                    block = {
                        **block,
                        "format": block["format"] + ";encoding=json",
                        "value": _canonical_json(block["value"]),
                    }
                content.append(block)
            message_data: dict[str, object] = {**wire, "content": content}
        else:
            message_data = {
                "role": "tool" if event_type == "tool_result" else "assistant",
                "content": [
                    {
                        "type": "opaque",
                        "format": f"harness.{event_type};encoding=json",
                        "value": _canonical_json(wire),
                    }
                ],
            }
            if event_type == "tool_result":
                message_data["callId"] = wire["callId"]
        return cls("message", timestamp, message_data)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "timestamp": self.timestamp, "data": _thaw(self.data)}


@dataclass(frozen=True)
class ContextMemory:
    memory_id: str
    content: MemoryContent
    scope: MemoryScope
    metadata: Mapping[str, str] | None = None
    type: MemoryType | None = None
    source: MemorySource | None = None
    valid: bool | None = None
    source_event_ids: tuple[str, ...] | None = None
    version: str | int | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata, field_name="metadata"))
        if self.source_event_ids is not None:
            object.__setattr__(self, "source_event_ids", tuple(self.source_event_ids))


@dataclass(frozen=True)
class ContextMemorySearchHit:
    memory: ContextMemory
    score: float
    similarity: float


@dataclass(frozen=True)
class ContextMemorySearchResult:
    memories: tuple[ContextMemorySearchHit, ...] = ()
    request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "memories", tuple(self.memories))


@dataclass(frozen=True)
class EventReceipt:
    input_index: int
    status: str
    event_id: str | None = None
    code: str | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if type(self.input_index) is not int or self.input_index < 0:
            raise MemoryValidationError("input_index must be a non-negative integer")
        _nonempty(self.status, "status")
        if self.status == "STORED" or self.event_id is not None:
            _nonempty(self.event_id, "event_id")


@dataclass(frozen=True)
class CreateEventsResult:
    results: tuple[EventReceipt, ...]
    request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "results", tuple(self.results))


@dataclass(frozen=True)
class EventPage:
    """Service event envelopes, preserved without imposing a guessed row schema."""

    items: tuple[Mapping[str, object], ...]
    next_token: str | None = None
    max_results: int | None = None
    total_count: int | None = None
    request_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(_freeze(item) for item in self.items))
