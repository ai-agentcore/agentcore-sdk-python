"""Per-Task memory context and native completion callback for CrewAI."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput

from agentcore.integrations.crewai import _run_on_loop
from agentcore.integrations.memory._common import (
    MemoryScopes,
    logger,
    recall,
    record,
    reference_text,
    validate_top_k,
)
from agentcore.memory import AsyncMemoryStore, MemoryMessage


@dataclass(frozen=True)
class TaskMemory:
    """Application adds instructions to Task.description and assigns its callback."""

    instructions: str
    callback: Callable[[TaskOutput], Awaitable[None]] | None


class AgentCoreTaskMemory:
    """Prepare a fresh binding per Task; never mutate a Task, Crew or native memory backend."""

    def __init__(
        self, store: AsyncMemoryStore, *, write_back: bool = False, top_k: int = 5
    ) -> None:
        validate_top_k(top_k)
        self._store = store
        self._write_back = write_back
        self._top_k = top_k

    async def prepare(self, query: str, *, scopes: MemoryScopes) -> TaskMemory:
        write_scope = scopes.require_write() if self._write_back else None
        instructions = reference_text(
            await recall(self._store, query, scopes.read, top_k=self._top_k, best_effort=True)
        )
        if instructions:
            logger.info("agentcore.memory.adapter.prepared framework=crewai")
        if write_scope is None or not query.strip():
            return TaskMemory(instructions, None)

        owner_loop = asyncio.get_running_loop()

        async def callback(output: TaskOutput) -> None:
            # Crew's final TaskOutput, not its transcript or injected description.
            if output.output_format != OutputFormat.RAW or not output.raw.strip():
                return
            await _run_on_loop(
                lambda: record(
                    self._store,
                    [MemoryMessage("user", query), MemoryMessage("assistant", output.raw)],
                    write_scope,
                    best_effort=True,
                ),
                owner_loop,
            )

        return TaskMemory(instructions, callback)
