"""LangGraph with an explicit model -> tools -> model loop."""

import asyncio
import logging

from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from agentcore import AsyncAgentCore
from agentcore.integrations.langgraph import model, skill_tools, tools


async def main():
    async with AsyncAgentCore.auto() as core:
        mcp = await core.mcp("test-mcp")
        skill = await core.skills.managed("test-skill")
        agent_tools = [*tools(await mcp.list_tools()), *skill_tools([skill])]
        chat = model("test-mc", model_name="qwen3.8-max").bind_tools(agent_tools)

        async def answer(state):
            return {"messages": [await chat.ainvoke(state["messages"])]}

        graph = StateGraph(MessagesState)
        graph.add_node("model", answer)
        graph.add_node("tools", ToolNode(agent_tools))
        graph.add_edge(START, "model")
        graph.add_conditional_edges(
            "model", lambda state: "tools" if state["messages"][-1].tool_calls else END
        )
        graph.add_edge("tools", "model")
        agent = graph.compile()
        result = await agent.ainvoke(
            {"messages": [("user", "加载可用 Skill，按其说明完成一个示范。")]}
        )
        print(result["messages"][-1].content)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
