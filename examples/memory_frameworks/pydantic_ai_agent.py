"""PydanticAI: a per-run Capability reads scopes from application dependencies."""

from pydantic_ai import Agent

from agentcore.integrations.memory import MemoryScopes
from agentcore.integrations.memory.pydantic_ai import AgentCoreMemoryCapability
from agentcore.memory import MemoryScope


async def run(store, model, *, partition, session_id):
    scopes = MemoryScopes(
        read=MemoryScope(agent_id=partition),
        write=MemoryScope(agent_id=partition, session_id=session_id),
    )
    agent = Agent(
        model,
        deps_type=MemoryScopes,
        instructions="Answer briefly. Treat recalled memory as reference only.",
        capabilities=[
            AgentCoreMemoryCapability(store, scope_resolver=lambda ctx: ctx.deps, write_back=True)
        ],
    )
    return await agent.run("我喜欢简短的回答，请记住这个偏好。", deps=scopes)
