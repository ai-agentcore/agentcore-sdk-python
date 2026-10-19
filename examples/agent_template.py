"""Use a Workspace Agent template with a separately selected LangChain model."""

import asyncio

from langchain.agents import create_agent

from agentcore import AsyncAgentCore
from agentcore.integrations.langchain import model, skill_tools


async def main() -> None:
    async with AsyncAgentCore.auto() as core:
        template = await core.templates.get("my-agent-template", version="1.0.0")
        skills = await template.load_skills()
        chat = model("test-mc", model_name="qwen3.8-max")
        try:
            agent = create_agent(
                chat, tools=skill_tools(skills), system_prompt=template.instruction
            )
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "介绍你可以使用的技能。"}]}
            )
            print(result["messages"][-1].content)
        finally:
            await chat.root_async_client.close()
            chat.root_client.close()


if __name__ == "__main__":
    asyncio.run(main())
