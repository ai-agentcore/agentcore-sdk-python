"""Connect to your own services and load a Skill shipped with the application."""

import asyncio
import logging
import os
from pathlib import Path

from agentcore import AsyncAgentCore
from agentcore.skill import skill_tools


async def main():
    async with AsyncAgentCore() as core:
        model = core.direct_model(
            provider="openai",
            model=os.environ["CUSTOM_MODEL_NAME"],
            base_url=os.environ["CUSTOM_MODEL_BASE_URL"],
            api_key=os.environ["CUSTOM_MODEL_API_KEY"],
        )
        result = await model.invoke([{"role": "user", "content": "你好"}])
        print(result["choices"][0]["message"]["content"])
        mcp = core.direct_mcp(
            url=os.environ["CUSTOM_MCP_URL"],
            # For an authenticated service, provide your own header resolver:
            # headers_provider=lambda: {"Authorization": os.environ["CUSTOM_MCP_AUTH"]},
        )
        print("MCP tools:", [tool.name for tool in await mcp.list_tools()])
        skills = await core.skills.local(Path(__file__).parent / "skills")
        print("Local Skills:", [skill.name for skill in skills])
        print("Skill tools:", [tool.name for tool in skill_tools(skills)])
        # Pass MCP / Skill tools to your chosen framework to run the tool loop.


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
