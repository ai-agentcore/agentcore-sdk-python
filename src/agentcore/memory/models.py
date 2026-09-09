"""Stable public models for the AgentCore Memory data plane."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Generic, TypeVar

from agentcore.errors import MemoryValidationError

T = TypeVar("T")


def _frozen_metadata(
    value: Mapping[str, str] | None,
    *,
    field_name: str,
) -> Mapping[str, str] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str)
        and bool(key.strip())
        and isinstance(item, str)
        for key, item in value.items()
    ):
        raise MemoryValidationError(
            f"{field_name} must contain non-empty string keys and string values"
        )
    return MappingProxyType(dict(value))


@dataclass(frozen=True)
class MemoryScope:
    agent_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None


@dataclass(frozen=True)
class MemoryMessage:
    role: str
    content: str


@dataclass(frozen=True)
class MemoryContent:
    text: str


@dataclass(frozen=True)
class Memory:
    memory_id: str
    content: MemoryContent
    scope: MemoryScope
    metadata: Mapping[str, str] | None = None
    created_at: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metadata",
            _frozen_metadata(self.metadata, field_name="memory.metadata"),
        )


@dataclass(frozen=True)
class MemorySearchHit:
    memory: Memory
    score: float
    similarity: float


@dataclass(frozen=True)
class AddMemoriesResult:
    memory_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "memory_ids", tuple(self.memory_ids))


@dataclass(frozen=True)
class SearchMemoriesResult:
    memories: tuple[MemorySearchHit, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "memories", tuple(self.memories))


@dataclass(frozen=True)
class MemorySession:
    agent_id: str | None = None
    session_id: str | None = None
    user_id: str | None = None


@dataclass(frozen=True)
class Page(Generic[T]):
    items: tuple[T, ...] = ()
    max_results: int | None = None
    next_token: str | None = None
    total_count: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "items", tuple(self.items))
