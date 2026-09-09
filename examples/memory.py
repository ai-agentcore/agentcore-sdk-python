"""Write and search memories in an existing AgentCore MemoryStore."""

import asyncio

from agentcore import AsyncAgentCore
from agentcore.memory import MemoryScope


async def main():
    async with AsyncAgentCore.auto() as core:
        store = core.memory_store("test-memory")
        await store.add_memories(
            scope=MemoryScope(user_id="example-user", session_id="session-1"),
            text="用户喜欢简短的中文回答。",
        )

        # Newly added memories may take time to become searchable.
        result = await store.search_memories(
            "用户有哪些回答偏好？",
            scope=MemoryScope(user_id="example-user"),
            top_k=5,
        )
        for hit in result.memories:
            print(hit.memory.content.text)


if __name__ == "__main__":
    asyncio.run(main())
