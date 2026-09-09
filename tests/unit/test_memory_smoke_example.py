from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest

from agentcore.memory import (
    AddMemoriesResult,
    Memory,
    MemoryContent,
    MemoryScope,
    MemorySearchHit,
    MemorySession,
    Page,
    SearchMemoriesResult,
)

SMOKE_PATH = Path(__file__).parents[2] / "examples" / "memory_data_plane_smoke.py"
SMOKE_SPEC = importlib.util.spec_from_file_location("memory_data_plane_smoke", SMOKE_PATH)
assert SMOKE_SPEC is not None and SMOKE_SPEC.loader is not None
SMOKE_MODULE = importlib.util.module_from_spec(SMOKE_SPEC)
SMOKE_SPEC.loader.exec_module(SMOKE_MODULE)
run_smoke = SMOKE_MODULE.run_smoke


class FakeStore:
    def __init__(self) -> None:
        self.operations: list[str] = []
        self.memory = Memory(
            memory_id="memory-smoke",
            content=MemoryContent(text="smoke"),
            scope=MemoryScope(agent_id="agent", session_id="session"),
        )

    async def add_memories(self, **kwargs: Any) -> AddMemoriesResult:
        self.operations.append("AddMemories")
        return AddMemoriesResult((self.memory.memory_id,))

    async def search_memories(self, *args: Any, **kwargs: Any) -> SearchMemoriesResult:
        self.operations.append("SearchMemories")
        return SearchMemoriesResult((MemorySearchHit(self.memory, 1.0, 0.9),))

    async def list_memories(self, **kwargs: Any) -> Page[Memory]:
        self.operations.append("ListMemories")
        return Page((self.memory,))

    async def get_memory(self, memory_id: str) -> Memory:
        self.operations.append("GetMemory")
        return self.memory

    async def update_memory(self, memory_id: str, **kwargs: Any) -> Memory:
        self.operations.append("UpdateMemory")
        return self.memory

    async def delete_memory(self, memory_id: str) -> None:
        self.operations.append("DeleteMemory")

    async def list_memory_sessions(self, **kwargs: Any) -> Page[MemorySession]:
        self.operations.append("ListMemorySessions")
        return Page((MemorySession("agent", "session"),))

    async def list_memory_session_messages(self, *args: Any, **kwargs: Any) -> Page[Any]:
        self.operations.append("ListMemorySessionMessages")
        return Page()


class FakeCore:
    def __init__(self) -> None:
        self.store = FakeStore()

    def memory_store(self, name: str) -> FakeStore:
        assert name == "ready_store"
        return self.store


class MultiMemoryStore(FakeStore):
    async def add_memories(self, **kwargs: Any) -> AddMemoriesResult:
        self.operations.append("AddMemories")
        return AddMemoriesResult((self.memory.memory_id, "memory-extra"))


class MultiMemoryCore(FakeCore):
    def __init__(self) -> None:
        self.store = MultiMemoryStore()


@pytest.mark.asyncio
async def test_smoke_uses_only_eight_data_plane_operations_and_cleans_memories() -> None:
    core = FakeCore()

    report = await run_smoke(core, "ready_store", search_attempts=1, search_delay=0)

    assert report["success"] is True
    assert report["remainingMemoryIds"] == 0
    assert report["cleanupErrors"] == []
    assert core.store.operations == [
        "AddMemories",
        "SearchMemories",
        "ListMemories",
        "GetMemory",
        "UpdateMemory",
        "ListMemorySessions",
        "ListMemorySessionMessages",
        "DeleteMemory",
    ]
    assert not hasattr(core, "create_memory_store")
    assert not hasattr(core, "delete_memory_store")


@pytest.mark.asyncio
async def test_smoke_cleans_every_memory_returned_by_message_add() -> None:
    core = MultiMemoryCore()

    report = await run_smoke(core, "ready_store", search_attempts=1, search_delay=0)

    assert report["success"] is True
    assert report["remainingMemoryIds"] == 0
    assert report["cleanupErrors"] == []
    assert core.store.operations.count("DeleteMemory") == 2
