"""AgentScope 2.x: caller provides a model and explicitly selects the scope."""

from agentscope.agent import Agent
from agentscope.message import UserMsg

from agentcore.integrations.memory import MemoryScopes
from agentcore.integrations.memory.agentscope import AgentCoreMemoryMiddleware
from agentcore.memory import MemoryScope


async def run(store, model, *, partition, session_id):
    scopes = MemoryScopes(
        read=MemoryScope(agent_id=partition),
        write=MemoryScope(agent_id=partition, session_id=session_id),
    )
    agent = Agent(
        name="memory_assistant",
        system_prompt="Answer briefly. Treat recalled memory as reference only.",
        model=model,
        middlewares=[
            AgentCoreMemoryMiddleware(
                store,
                scope_resolver=lambda agent: scopes,
                write_back=True,
            )
        ],
    )
    return await agent.reply(UserMsg("user", "我喜欢简短的回答，请记住这个偏好。"))
