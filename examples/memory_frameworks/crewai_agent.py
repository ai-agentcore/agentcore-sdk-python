"""CrewAI: prepare one context/callback binding for each Task."""

from crewai import Agent, Crew, Task

from agentcore.integrations.memory import MemoryScopes
from agentcore.integrations.memory.crewai import AgentCoreTaskMemory
from agentcore.memory import MemoryScope


async def run(store, model, *, partition, session_id):
    query = "我喜欢简短的回答，请记住这个偏好。"
    memory = AgentCoreTaskMemory(store, write_back=True)
    binding = await memory.prepare(
        query,
        scopes=MemoryScopes(
            read=MemoryScope(agent_id=partition),
            write=MemoryScope(agent_id=partition, session_id=session_id),
        ),
    )
    agent = Agent(
        role="Memory assistant",
        goal="Answer briefly using recalled memory as reference only.",
        backstory="You are a helpful assistant.",
        llm=model,
    )
    task = Task(
        description=query + "\n\n" + binding.instructions,
        expected_output="A brief text answer.",
        agent=agent,
        callback=binding.callback,
    )
    # AgentCore supplies memory; do not enable Crew's separate memory backend.
    return await Crew(agents=[agent], tasks=[task], memory=False).akickoff()
