"""Search an existing AgentCore knowledge base."""

import asyncio

from agentcore import AsyncAgentCore


async def main():
    async with AsyncAgentCore.auto() as core:
        kb = core.knowledge_base("你的知识库 ID")
        result = await kb.search("产品如何配置？", top_k=5)
        for chunk in result.chunks:
            print(chunk.file_name or chunk.asset_name, chunk.content)


if __name__ == "__main__":
    asyncio.run(main())
