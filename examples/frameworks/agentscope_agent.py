"""AgentScope 2.x (Python 3.11+) with explicitly selected trusted tools."""

import asyncio
import logging

from agentscope.agent import Agent
from agentscope.message import UserMsg
from agentscope.permission import PermissionBehavior, PermissionContext, PermissionRule
from agentscope.state import AgentState
from agentscope.tool import Toolkit

from agentcore import AsyncAgentCore
from agentcore.integrations.agentscope import model, skill_tools, tools


async def main():
    async with AsyncAgentCore.auto() as core:
        mcp = await core.mcp("test-mcp")
        skill = await core.skills.managed("test-skill")
        selected = [*tools(await mcp.list_tools()), *skill_tools([skill])]
        toolkit = Toolkit()
        await toolkit.add_tool(selected)
        # This example authorizes only the tools deliberately selected above.
        permissions = PermissionContext(
            allow_rules={
                tool.name: [
                    PermissionRule(
                        tool_name=tool.name,
                        rule_content=None,
                        behavior=PermissionBehavior.ALLOW,
                        source="application",
                    )
                ]
                for tool in selected
            }
        )
        agent = Agent(
            name="resource_assistant",
            system_prompt="使用工具完成请求，不要编造工具执行结果。",
            model=model("test-mc", model_name="qwen3.8-max"),
            toolkit=toolkit,
            state=AgentState(permission_context=permissions),
        )
        result = await agent.reply(UserMsg("user", "加载可用 Skill，按其说明完成一个示范。"))
        print(result)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
