"""AgentScope 2.x reply middleware for AgentCore Memory."""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import aclosing
from contextvars import ContextVar
from typing import Any

from agentscope.agent import Agent
from agentscope.event import ReplyEndEvent
from agentscope.message import Msg
from agentscope.middleware import MiddlewareBase
from agentscope.types import ReplyFinishedReason

from agentcore.integrations.memory._common import (
    MemoryScopes,
    logger,
    recall,
    record,
    reference_text,
    validate_top_k,
)
from agentcore.memory import AsyncMemoryStore, MemoryMessage


class AgentCoreMemoryMiddleware(MiddlewareBase):  # type: ignore[misc,unused-ignore]
    def __init__(
        self,
        store: AsyncMemoryStore,
        *,
        scope_resolver: Callable[[Agent], MemoryScopes],
        write_back: bool = False,
        top_k: int = 5,
    ) -> None:
        validate_top_k(top_k)
        self._store = store
        self._scopes = scope_resolver
        self._write_back = write_back
        self._top_k = top_k
        self._reference: ContextVar[str] = ContextVar("agentcore_memory_reference", default="")

    async def on_reply(
        self,
        agent: Agent,
        input_kwargs: dict[str, Any],
        next_handler: Callable[..., AsyncGenerator[Any, None]],
    ) -> AsyncGenerator[Any, None]:
        scopes = self._scopes(agent)
        write_scope = scopes.require_write() if self._write_back else None
        inputs = input_kwargs.get("inputs")
        candidates = (
            [inputs] if isinstance(inputs, Msg) else inputs if isinstance(inputs, list) else []
        )
        incoming = [
            MemoryMessage("user", text)
            for msg in candidates
            if isinstance(msg, Msg) and msg.role == "user"
            if (text := msg.get_text_content()) and text.strip()
        ]
        text = await recall(
            self._store,
            "\n".join(msg.content for msg in incoming),
            scopes.read,
            top_k=self._top_k,
            best_effort=True,
        )
        token = self._reference.set(reference_text(text))
        completed = False
        final: Msg | None = None
        try:
            async with aclosing(next_handler(**input_kwargs)) as events:
                async for item in events:
                    if isinstance(item, ReplyEndEvent):
                        completed = item.finished_reason == ReplyFinishedReason.COMPLETED
                    elif isinstance(item, Msg) and item.role == "assistant":
                        final = item
                    yield item
            # Not in finally: cancellation or a failed/parked reply never writes.
            if completed and incoming and final is not None and write_scope is not None:
                answer = final.get_text_content()
                if answer and answer.strip():
                    await record(
                        self._store,
                        [*incoming, MemoryMessage("assistant", answer)],
                        write_scope,
                        best_effort=True,
                    )
        finally:
            self._reference.reset(token)

    async def on_system_prompt(self, agent: Agent, current_prompt: str) -> str:
        note = self._reference.get()
        if not note:
            return current_prompt
        logger.info("agentcore.memory.adapter.injected framework=agentscope")
        return current_prompt + "\n\n" + note
