"""PydanticAI Capability for per-run reference memory, not conversation history."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from typing import Any

from pydantic_ai import RunContext
from pydantic_ai.capabilities import AbstractCapability, WrapRunHandler
from pydantic_ai.run import AgentRunResult

from agentcore.integrations.memory._common import (
    MemoryScopes,
    logger,
    recall,
    record,
    reference_text,
    validate_top_k,
)
from agentcore.memory import AsyncMemoryStore, MemoryMessage


class AgentCoreMemoryCapability(AbstractCapability[Any]):
    """Async text runs; verified with PydanticAI 2.31.1's Capability lifecycle."""

    def __init__(
        self,
        store: AsyncMemoryStore,
        *,
        scope_resolver: Callable[[RunContext[Any]], MemoryScopes],
        write_back: bool = False,
        top_k: int = 5,
    ) -> None:
        validate_top_k(top_k)
        self._store = store
        self._scopes = scope_resolver
        self._write_back = write_back
        self._top_k = top_k
        self._reference = ""

    async def for_run(self, ctx: RunContext[Any]) -> AgentCoreMemoryCapability:
        # PydanticAI rebinds the instruction callback to this per-run instance.
        return copy(self)

    def get_instructions(self) -> Callable[[RunContext[Any]], str]:
        return self._instructions

    def _instructions(self, ctx: RunContext[Any]) -> str:
        if self._reference:
            logger.info("agentcore.memory.adapter.injected framework=pydantic_ai")
        return self._reference

    async def wrap_run(
        self, ctx: RunContext[Any], *, handler: WrapRunHandler
    ) -> AgentRunResult[Any]:
        scopes = self._scopes(ctx)
        write_scope = scopes.require_write() if self._write_back else None
        prompt = ctx.prompt
        text = (
            prompt
            if isinstance(prompt, str)
            else "\n".join(part for part in prompt or [] if isinstance(part, str))
        )
        self._reference = reference_text(
            await recall(self._store, text, scopes.read, top_k=self._top_k, best_effort=True)
        )
        try:
            result = await handler()
            if (
                write_scope is not None
                and text.strip()
                and isinstance(result.output, str)
                and result.output.strip()
                and result.response.finish_reason in {None, "stop"}
            ):
                await record(
                    self._store,
                    [MemoryMessage("user", text), MemoryMessage("assistant", result.output)],
                    write_scope,
                    best_effort=True,
                )
            return result
        finally:
            self._reference = ""
