"""Use named resources in an AgentCore cloud container."""

import asyncio
import logging

from agentcore import AsyncAgentCore


async def main():
    async with AsyncAgentCore.auto() as core:
        model = await core.model("test-mc", model="qwen3.8-max")
        result = await model.invoke([{"role": "user", "content": "请简短介绍你自己。"}])
        print(result["choices"][0]["message"]["content"])
        # Optional fixed headers apply to this MCP client's connections and calls.
        mcp = await core.mcp("test-mcp", headers={"x-business-id": "example-app"})
        print("MCP tools:", [tool.name for tool in await mcp.list_tools()])
        # Set version="..." to select an existing published version.
        skill = await core.skills.managed("test-skill")
        print("Skill:", skill.name, skill.version)
        # To execute MCP tools, use a framework example or mcp.call_tool(name, arguments).


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
