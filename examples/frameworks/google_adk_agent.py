"""Google ADK Agent and Runner with managed model, MCP and Skill."""

import asyncio
import logging
from uuid import uuid4

from google.adk import Runner
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.genai import types

from agentcore import AsyncAgentCore
from agentcore.integrations.google_adk import model, skill_tools, tools


async def main():
    async with AsyncAgentCore.auto() as core:
        mcp = await core.mcp("test-mcp")
        skill = await core.skills.managed("test-skill")
        agent = Agent(
            name="resource_assistant",
            model=model("test-mc", model_name="qwen3.8-max"),
            instruction="使用工具完成请求，不要编造工具执行结果。",
            tools=[*tools(await mcp.list_tools()), *skill_tools([skill])],
        )
        sessions = InMemorySessionService()
        session_id = str(uuid4())
        await sessions.create_session(
            app_name="resource_example", user_id="example-user", session_id=session_id
        )
        runner = Runner(app_name="resource_example", agent=agent, session_service=sessions)
        try:
            async for event in runner.run_async(
                user_id="example-user",
                session_id=session_id,
                new_message=types.Content(
                    role="user",
                    parts=[types.Part(text="加载可用 Skill，按其说明完成一个示范。")],
                ),
            ):
                if event.error_code:
                    raise RuntimeError(event.error_message)
                if event.is_final_response() and event.content:
                    for part in event.content.parts or []:
                        if part.text:
                            print(part.text)
        finally:
            await runner.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
