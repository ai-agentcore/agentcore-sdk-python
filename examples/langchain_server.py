"""Cloud LangChain Agent with MCP, Skill and AG-UI / OpenAI output.

Requires Python 3.11+ and an OpenAI/v1 model connection. This is a single text-turn example:
conversation history and user authentication belong to the application.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from agentcore import AsyncAgentCore, RequestContext
from agentcore.events import AgentEvent
from agentcore.integrations.langchain import AgentCoreConverter, model, skill_tools, tools
from agentcore.server import AgentCoreServer, AgentRequest

logger = logging.getLogger(__name__)
core: AsyncAgentCore | None = None
chat: ChatOpenAI | None = None
agent: Any = None


async def shutdown() -> None:
    try:
        if chat is not None:
            await chat.root_async_client.close()
            chat.root_client.close()
    finally:
        if core is not None:
            await core.aclose()


async def startup() -> None:
    global core, chat, agent
    core = AsyncAgentCore.auto()
    try:
        mcp = await core.mcp("test-mcp")
        skill = await core.skills.managed("test-skill")
        agent_tools = [*tools(await mcp.list_tools()), *skill_tools([skill])]
        chat = model("test-mc", model_name="qwen3.8-max")
        agent = create_agent(
            model=chat,
            tools=agent_tools,
            system_prompt="使用工具完成请求，不要编造工具执行结果。",
        )
    except Exception:
        logger.exception("Example Agent initialization failed")
        await shutdown()
        raise


server = AgentCoreServer(
    startup=startup,
    shutdown=shutdown,
    readiness=lambda: agent is not None,
)


@server.invoke
async def invoke(request: AgentRequest, _context: RequestContext) -> AsyncIterator[AgentEvent]:
    if agent is None:
        raise RuntimeError("Example Agent is not started")
    # Only the latest user text is used. No server-side history is retained.
    message = request.messages[-1]
    if message.role.value != "user" or not isinstance(message.content, str):
        raise ValueError("This example expects a text user turn")
    events = agent.astream_events(
        {"messages": [HumanMessage(content=message.content)]},
        version="v2",
    )
    # A fresh converter per request preserves text boundaries and tool-call IDs.
    async for event in AgentCoreConverter().stream(events):
        yield event
