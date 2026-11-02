"""Select managed resources using the environment variables documented in README."""

import asyncio

from agentcore import AsyncAgentCore


async def main():
    async with AsyncAgentCore.auto() as core:
        model = await core.model()
        mcps = await core.mcps_from_env()
        skills = await core.skills.from_env()
        template = await core.templates.get()

        print("Model:", model.descriptor.model_name)
        for mcp in mcps:
            print("MCP tools:", [tool.name for tool in await mcp.list_tools()])
        print("Skills:", [skill.name for skill in skills])
        print("AgentSpec:", template.name, template.version)
        # Load AgentSpec Skills explicitly when needed:
        # template_skills = await template.load_skills()
        # Pass the model, tools, instruction and Skills to your framework adapter.


if __name__ == "__main__":
    asyncio.run(main())
