from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired

from agentcore.integrations.memory.langgraph import (
    AgentCoreMemoryNodes,
    MemoryNodeState,
    with_memory,
)
from agentcore.memory import (
    AddMemoriesResult,
    Memory,
    MemoryContent,
    MemoryMessage,
    MemoryScope,
    MemorySearchHit,
    SearchMemoriesResult,
)


class State(MemoryNodeState):
    messages: NotRequired[list]
    step: NotRequired[int]


@pytest.mark.asyncio
async def test_real_graph_explicit_memory_nodes_and_tool_loop():
    store = AsyncMock()
    store.memory_store_name = "memory"
    scope = MemoryScope(agent_id="a", session_id="s")
    store.search_memories.return_value = SearchMemoriesResult(
        memories=[
            MemorySearchHit(Memory("id", MemoryContent("historical fact"), scope), 1, 1),
        ]
    )
    store.add_memories.return_value = AddMemoriesResult()
    nodes = AgentCoreMemoryNodes(store)
    seen = []

    def model(state):
        messages = with_memory(state["messages"], state["memory_text"])
        seen.append(messages)
        return {
            "step": state.get("step", 0) + 1,
            "memory_messages": [
                MemoryMessage("user", state["memory_query"]),
                MemoryMessage("assistant", "done"),
            ],
        }

    graph = StateGraph(State)
    graph.add_node("recall", nodes.recall)
    graph.add_node("model", model)
    graph.add_node("tool", lambda state: {})
    graph.add_node("record", nodes.record)
    graph.add_edge(START, "recall")
    graph.add_edge("recall", "model")
    graph.add_conditional_edges("model", lambda state: "tool" if state["step"] == 1 else "record")
    graph.add_edge("tool", "model")
    graph.add_edge("record", END)
    original = [SystemMessage("original"), HumanMessage("question"), AIMessage("old history")]
    result = await graph.compile().ainvoke(
        {
            "messages": original,
            "memory_query": "explicit current query",
            "memory_read_scope": scope,
            "memory_write_scope": scope,
        }
    )
    assert store.search_memories.await_count == 1
    assert store.search_memories.await_args.args == ("explicit current query",)
    assert store.add_memories.await_count == 1
    assert len(seen) == 2 and "historical fact" in seen[0][0].content
    assert original[0].content == "original"
    assert result["memory_messages"] == []
