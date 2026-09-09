"""CrewAI: construct tools and run the Crew on the same live event loop."""

import asyncio
import logging

from crewai import Agent, Crew, Task

from agentcore import AsyncAgentCore
from agentcore.integrations.crewai import model, skill_tools, tools


async def main():
    async with AsyncAgentCore.auto() as core:
        mcp = await core.mcp("test-mcp")
        skill = await core.skills.managed("test-skill")
        agent = Agent(
            role="资源助手",
            goal="使用 Skill 和 MCP 完成用户请求",
            backstory="根据工具返回的信息回答，不编造结果。",
            llm=model("test-mc", model_name="qwen3.8-max"),
            tools=[*tools(await mcp.list_tools()), *skill_tools([skill])],
        )
        task = Task(
            description="加载可用 Skill，按其说明完成一个示范。",
            expected_output="简短的执行结果说明",
            agent=agent,
        )
        # Keep the owning loop alive while asynchronous MCP tools run.
        result = await Crew(agents=[agent], tasks=[task], memory=False).akickoff()
        print(result.raw)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
