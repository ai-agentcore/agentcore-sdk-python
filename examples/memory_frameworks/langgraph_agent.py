"""Explicit recall -> model/tool loop -> record graph; caller owns model/store."""

from typing import Annotated

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agentcore.integrations.memory.langgraph import (
    AgentCoreMemoryNodes,
    MemoryNodeState,
    with_memory,
)
from agentcore.memory import MemoryMessage, MemoryScope


class State(MemoryNodeState):
    messages: Annotated[list[BaseMessage], add_messages]


@tool
def add(left: int, right: int) -> int:
    """Add two integers."""
    return left + right


async def run(store, model, *, partition, session_id):
    memory = AgentCoreMemoryNodes(store)
    chat = model.bind_tools([add])

    async def answer(state):
        reply = await chat.ainvoke(
            with_memory(
                [SystemMessage("Use the add tool for addition."), *state["messages"]],
                state["memory_text"],
            )
        )
        pending = []
        if not reply.tool_calls and reply.text.strip():
            pending = [
                MemoryMessage("user", state["memory_query"]),
                MemoryMessage("assistant", reply.text),
            ]
        return {"messages": [reply], "memory_messages": pending}

    graph = StateGraph(State)
    graph.add_node("recall", memory.recall)
    graph.add_node("model", answer)
    graph.add_node("tools", ToolNode([add]))
    graph.add_node("record", memory.record)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "model")
    graph.add_conditional_edges(
        "model", lambda s: "tools" if s["messages"][-1].tool_calls else "record"
    )
    graph.add_edge("tools", "model")
    graph.add_edge("record", END)
    query = "请计算 2 + 3，回答简短一点。"
    return await graph.compile().ainvoke(
        {
            "messages": [HumanMessage(query)],
            "memory_query": query,
            "memory_read_scope": MemoryScope(agent_id=partition),
            "memory_write_scope": MemoryScope(agent_id=partition, session_id=session_id),
        }
    )
