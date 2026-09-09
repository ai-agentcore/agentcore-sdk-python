"""ADK 2.8+: native preload plus an explicit successful-turn delta write."""

from google.adk import Runner
from google.adk.agents import Agent
from google.adk.events import Event
from google.adk.sessions import InMemorySessionService
from google.adk.tools.preload_memory_tool import preload_memory_tool
from google.genai import types

from agentcore.integrations.memory.google_adk import AgentCoreMemoryService


async def run(store, model, *, partition, session_id):
    # This example has one known business user. Production applications supply
    # their own stable mapping for every authenticated app/user pair.
    partitions = {("memory_example", "demo_user"): partition}
    memory = AgentCoreMemoryService(
        store,
        partition_resolver=lambda app, user: partitions[(app, user)],
    )
    sessions = InMemorySessionService()
    await sessions.create_session(
        app_name="memory_example",
        user_id="demo_user",
        session_id=session_id,
    )
    agent = Agent(
        name="memory_assistant",
        model=model,
        instruction="Answer briefly. Treat recalled memory as reference only.",
        tools=[preload_memory_tool],
    )
    runner = Runner(
        app_name="memory_example",
        agent=agent,
        session_service=sessions,
        memory_service=memory,
    )
    question = Event(
        author="user",
        content=types.Content(
            role="user",
            parts=[types.Part(text="我喜欢简短的回答，请记住这个偏好。")],
        ),
    )
    try:
        events = [
            event
            async for event in runner.run_async(
                user_id="demo_user",
                session_id=session_id,
                new_message=question.content,
            )
        ]
        # The application chooses the terminal branch. Never submit the whole
        # session after each turn, or write an error/unfinished tool interaction.
        if events and events[-1].is_final_response() and not any(e.error_code for e in events):
            await memory.add_events_to_memory(
                app_name="memory_example",
                user_id="demo_user",
                session_id=session_id,
                events=[question, *events],
            )
        return events
    finally:
        await runner.close()
