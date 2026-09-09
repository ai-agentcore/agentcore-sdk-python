"""LangChain Agent using a managed model, MCP and Skill."""

import asyncio
import logging

from langchain.agents import create_agent

from agentcore import AsyncAgentCore
from agentcore.integrations.langchain import model, skill_tools, tools


async def main():
    async with AsyncAgentCore.auto() as core:
        chat = model("test-mc", model_name="qwen3.8-max")
        mcp = await core.mcp("test-mcp")
        skill = await core.skills.managed("test-skill")
        agent = create_agent(
            model=chat,
            tools=[*tools(await mcp.list_tools()), *skill_tools([skill])],
            system_prompt="使用工具完成请求，不要编造工具执行结果。",
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": "加载可用 Skill，按其说明完成一个示范。"}]}
        )
        print(result["messages"][-1].content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
