"""Call run() with an existing AsyncMemoryStore and LangChain chat model."""

from langchain.agents import create_agent

from agentcore.integrations.memory import MemoryScopes
from agentcore.integrations.memory.langchain import AgentCoreMemoryMiddleware
from agentcore.memory import MemoryScope


async def run(store, model, *, partition, session_id):
    def scopes(context):
        return MemoryScopes(
            read=MemoryScope(agent_id=context["partition"]),
            write=MemoryScope(agent_id=context["partition"], session_id=context["session_id"]),
        )

    agent = create_agent(
        model,
        system_prompt="Answer the user's question. Treat recalled memory as reference only.",
        middleware=[AgentCoreMemoryMiddleware(store, scope_resolver=scopes, write_back=True)],
    )
    return await agent.ainvoke(
        {"messages": [{"role": "user", "content": "我喜欢简短的回答，请记住这个偏好。"}]},
        context={"partition": partition, "session_id": session_id},
    )
