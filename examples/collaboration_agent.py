"""Cloud Worker service: LangChain + collaboration + both Server protocols.

Requires AgentCore platform allowlist access to collaboration.
"""

import logging

from langchain.agents import create_agent

from agentcore import AsyncAgentCore
from agentcore.integrations.langchain import AgentCoreConverter, model, skill_tools, tools
from agentcore.server import AgentCoreServer, AgentRequest

logging.basicConfig(level=logging.INFO)
core = AsyncAgentCore.auto()
agent = None


async def startup():
    global agent
    worker = core.collaboration.worker()
    agent = create_agent(
        model=model("test-mc", model_name="qwen3.8-max"),
        system_prompt=worker.compose_prompt("完成分配的业务任务，简短报告结果。"),
        tools=[*tools(worker.tools()), *skill_tools(worker.skills())],
    )


server = AgentCoreServer(startup=startup, shutdown=core.aclose, readiness=lambda: agent is not None)


@server.invoke
async def invoke(request: AgentRequest, _context):
    if agent is None:
        raise RuntimeError("Agent is not initialized")
    messages = [
        {"role": message.role.value, "content": message.content} for message in request.messages
    ]
    # Server binds the platform collaboration context throughout stream iteration.
    async for event in AgentCoreConverter().stream(
        agent.astream_events({"messages": messages}, version="v2")
    ):
        yield event
