"""Exercise the documented tool model using real LangChain and LangGraph."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytest.importorskip("langchain_core")


@pytest.fixture(scope="module")
def character_tool():
    path = Path(__file__).parents[2] / "examples/frameworks/langchain_tool_model.py"
    return runpy.run_path(str(path))["describe_character"]


@pytest.mark.parametrize("field", ["character_id", "characterId"])
def test_tool_model_passes_python_parameter_name(character_tool, field):
    assert character_tool.invoke({field: "alice"}) == "Character: alice"


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["character_id", "characterId"])
async def test_tool_model_passes_python_parameter_name_async(character_tool, field):
    assert await character_tool.ainvoke({field: "alice"}) == "Character: alice"


@pytest.mark.parametrize("field", ["character_id", "characterId"])
def test_tool_model_runs_in_langgraph(character_tool, field):
    pytest.importorskip("langgraph")
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    message = AIMessage(
        content="",
        tool_calls=[
            {"name": character_tool.name, "args": {field: "alice"}, "id": "call-character"},
        ],
    )
    graph = StateGraph(MessagesState)
    graph.add_node("tools", ToolNode([character_tool]))
    graph.add_edge(START, "tools")
    graph.add_edge("tools", END)
    result = graph.compile().invoke({"messages": [message]})

    assert result["messages"][-1].content == "Character: alice"
    assert result["messages"][-1].tool_call_id == "call-character"
