"""PydanticAI Agent with managed model, MCP and Skill."""

import asyncio
import logging

from pydantic_ai import Agent

from agentcore import AsyncAgentCore
from agentcore.integrations.pydantic_ai import model, skill_tools, tools


async def main():
    async with AsyncAgentCore.auto() as core:
        mcp = await core.mcp("test-mcp")
        skill = await core.skills.managed("test-skill")
        agent = Agent(
            model("test-mc", model_name="qwen3.8-max"),
            instructions="使用工具完成请求，不要编造工具执行结果。",
            tools=[*tools(await mcp.list_tools()), *skill_tools([skill])],
        )
        result = await agent.run("加载可用 Skill，按其说明完成一个示范。")
        print(result.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
