"""Small framework-neutral model and tool types."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import anyio

ToolCallable = Callable[[dict[str, Any]], Any | Awaitable[Any]]
SyncToolCallable = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class Message:
    role: str
    content: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "role": self.role,
                "content": self.content,
                "name": self.name,
            }.items()
            if value is not None
        }


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    _call: ToolCallable
    _sync_call: SyncToolCallable | None = field(default=None, repr=False, compare=False)

    async def ainvoke(self, arguments: Mapping[str, Any]) -> Any:
        result = self._call(dict(arguments))
        if inspect.isawaitable(result):
            return await result
        return result

    @property
    def supports_sync(self) -> bool:
        return self._sync_call is not None

    def invoke(self, arguments: Mapping[str, Any]) -> Any:
        if self._sync_call is None:
            raise RuntimeError("Tool is async-only; use ainvoke() or the AgentCore sync facade")
        return self._sync_call(dict(arguments))

    def with_sync_call(self, call: SyncToolCallable) -> Tool:
        async def call_from_async(arguments: dict[str, Any]) -> Any:
            return await anyio.to_thread.run_sync(call, arguments)

        return replace(self, _call=call_from_async, _sync_call=call)

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": dict(self.parameters),
            },
        }
