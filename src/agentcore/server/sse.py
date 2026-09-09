"""Shared SSE transport helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress
from dataclasses import dataclass
from typing import cast

_HEARTBEAT_INTERVAL_SECONDS = 15.0
_HEARTBEAT_FRAME = ": ping\n\n"


@dataclass(frozen=True)
class _StreamFailure:
    error: Exception


async def with_heartbeat(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    """Emit an SSE comment while the application stream is idle."""
    end = object()
    queue: asyncio.Queue[str | _StreamFailure | object] = asyncio.Queue(maxsize=1)

    async def forward() -> None:
        try:
            async for chunk in stream:
                await queue.put(chunk)
        except Exception as exc:
            await queue.put(_StreamFailure(exc))
        else:
            await queue.put(end)

    task = asyncio.create_task(forward())
    try:
        while True:
            try:
                item = await asyncio.wait_for(
                    queue.get(),
                    timeout=_HEARTBEAT_INTERVAL_SECONDS,
                )
            except asyncio.TimeoutError:
                yield _HEARTBEAT_FRAME
                continue
            if item is end:
                return
            if isinstance(item, _StreamFailure):
                raise item.error
            yield cast(str, item)
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
