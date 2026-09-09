"""Execution-event contracts, then both wire protocols; no cloud credentials needed."""

import asyncio
import json
from types import SimpleNamespace as NS

import httpx
import pytest

from agentcore.events import AgentEvent, EventType
from agentcore.integrations._adk_events import AgentCoreConverter as ADK
from agentcore.integrations._agentscope_events import AgentCoreConverter as AgentScope
from agentcore.integrations._crewai_events import AgentCoreConverter as CrewAI
from agentcore.integrations._langchain_events import AgentCoreConverter as LangChain
from agentcore.integrations._pydantic_events import AgentCoreConverter as PydanticAI
from agentcore.server import AgentCoreServer


@pytest.mark.parametrize("framework", ["adk", "pydantic"])
async def test_mcp_result_is_json_data_at_event_boundary(framework):
    from mcp.types import CallToolResult, TextContent

    result = CallToolResult(content=[TextContent(type="text", text="杭州")])
    if framework == "adk":
        events = ADK().convert(
            NS(
                content=NS(
                    parts=[
                        NS(
                            function_response=NS(
                                id="mcp-call", name="lookup", response={"result": result}
                            )
                        )
                    ]
                )
            )
        )
        expected = {"result": result.model_dump(mode="json", by_alias=True)}
    else:
        events = PydanticAI().convert(
            NS(
                event_kind="function_tool_result",
                part=NS(tool_call_id="mcp-call", tool_name="lookup", content=result),
            )
        )
        expected = result.model_dump(mode="json", by_alias=True)
    converted = next(e for e in events if e.type == "TOOL_RESULT")
    assert json.loads(json.dumps(converted.data["result"])) == expected
    data = await wire(events, "agui")
    assert data[-1]["type"] == "RUN_FINISHED"
    assert (
        json.loads(next(e["content"] for e in data if e["type"] == "TOOL_CALL_RESULT")) == expected
    )


async def test_langchain_parallel_model_messages_and_scoped_ends():
    from langchain_core.messages import AIMessage, AIMessageChunk

    converter = LangChain()
    events = []
    for run, text in [("a", "A1"), ("b", "B1"), ("a", "A2")]:
        events.extend(
            converter.convert(
                {
                    "event": "on_chat_model_stream",
                    "run_id": run,
                    "data": {"chunk": AIMessageChunk(content=text)},
                }
            )
        )
    events.extend(
        converter.convert(
            {
                "event": "on_chat_model_end",
                "run_id": "a",
                "data": {"output": AIMessage(content="A1A2")},
            }
        )
    )
    events.extend(
        converter.convert(
            {
                "event": "on_chat_model_stream",
                "run_id": "b",
                "data": {"chunk": AIMessageChunk(content="B2")},
            }
        )
    )
    events.extend(
        converter.convert(
            {
                "event": "on_chat_model_end",
                "run_id": "b",
                "data": {"output": AIMessage(content="B1B2")},
            }
        )
    )
    data = await wire(events, "agui")
    starts = [e for e in data if e["type"] == "TEXT_MESSAGE_START"]
    assert len(starts) == 2
    assert [
        "".join(
            e["delta"]
            for e in data
            if e["type"] == "TEXT_MESSAGE_CONTENT" and e["messageId"] == s["messageId"]
        )
        for s in starts
    ] == ["A1A2", "B1B2"]
    for start in starts:
        end = next(
            e
            for e in data
            if e["type"] == "TEXT_MESSAGE_END" and e["messageId"] == start["messageId"]
        )
        assert all(
            data.index(start) < i < data.index(end)
            for i, e in enumerate(data)
            if e["type"] == "TEXT_MESSAGE_CONTENT" and e["messageId"] == start["messageId"]
        )


async def test_actual_langgraph_parallel_models_preserve_messages():
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.runnables import RunnableConfig
    from langgraph.graph import END, START, MessagesState, StateGraph

    gate = asyncio.Event()

    class Model(BaseChatModel):
        label: str

        @property
        def _llm_type(self):
            return "controlled-parallel"

        def _generate(self, *args, **kwargs):
            raise AssertionError("stream expected")

        async def _astream(self, messages, **kwargs):
            yield ChatGenerationChunk(message=AIMessageChunk(content=self.label + "1"))
            await asyncio.wait_for(gate.wait(), timeout=5)
            yield ChatGenerationChunk(message=AIMessageChunk(content=self.label + "2"))

    async def a(state, config: RunnableConfig):
        return {"messages": [await Model(label="A").ainvoke(state["messages"], config)]}

    async def b(state, config: RunnableConfig):
        return {"messages": [await Model(label="B").ainvoke(state["messages"], config)]}

    graph = StateGraph(MessagesState)
    graph.add_node("a", a)
    graph.add_node("b", b)
    for name in ["a", "b"]:
        graph.add_edge(START, name)
        graph.add_edge(name, END)
    converter, events, started = LangChain(), [], set()
    async for native in graph.compile().astream_events(
        {"messages": [("user", "run")]}, version="v2"
    ):
        if native["event"] == "on_chat_model_stream":
            started.add(native["run_id"])
            if len(started) == 2:
                gate.set()
        events.extend(converter.convert(native))
    data = await wire(events, "agui")
    messages = {}
    for event in data:
        if event["type"] == "TEXT_MESSAGE_CONTENT":
            messages.setdefault(event["messageId"], []).append(event["delta"])
    assert sorted("".join(deltas) for deltas in messages.values()) == ["A1A2", "B1B2"]


async def test_langchain_alternating_text_reasoning_has_distinct_closed_segments():
    converter, events = LangChain(), []
    for block in [
        {"type": "text", "text": "checking"},
        {"type": "reasoning", "reasoning": "thought"},
        {"type": "text", "text": "answer"},
    ]:
        events.extend(
            converter.convert(
                {
                    "event": "on_chat_model_stream",
                    "run_id": "r",
                    "data": {"chunk": {"content": [block]}},
                }
            )
        )
    events.extend(converter.convert({"event": "on_chat_model_end", "run_id": "r", "data": {}}))
    data = await wire(events, "agui")
    starts = [e for e in data if e["type"] in {"TEXT_MESSAGE_START", "REASONING_MESSAGE_START"}]
    ends = [e for e in data if e["type"] in {"TEXT_MESSAGE_END", "REASONING_MESSAGE_END"}]
    assert len({e["messageId"] for e in starts}) == 3
    assert [e["messageId"] for e in starts] == [e["messageId"] for e in ends]
    assert data.index(ends[0]) < data.index(starts[1]) < data.index(ends[1]) < data.index(starts[2])


async def test_langgraph_default_handled_validation_result_is_forwarded_once():
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.tools import tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode

    @tool
    async def lookup(value: int) -> str:
        """Look up an integer."""
        return str(value)

    call = {"id": "validation-call", "name": "lookup", "args": {}}
    graph = StateGraph(MessagesState)
    graph.add_node("custom_tool_node", ToolNode([lookup]))
    graph.add_edge(START, "custom_tool_node")
    graph.add_edge("custom_tool_node", END)
    converter = LangChain()
    events = converter.convert(
        {
            "event": "on_chat_model_end",
            "run_id": "model",
            "data": {"output": AIMessage(content="checking", tool_calls=[call])},
        }
    )
    # Historical messages in root snapshots must not be replayed as new results.
    inputs = {
        "messages": [
            ToolMessage("old", tool_call_id="old-call"),
            HumanMessage("lookup"),
            AIMessage(content="", tool_calls=[call]),
        ]
    }
    async for native in graph.compile().astream_events(inputs, version="v2"):
        events.extend(converter.convert(native))
    results = [e for e in events if e.type == "TOOL_RESULT"]
    assert [e.data["id"] for e in results] == ["validation-call"]
    assert "value" in results[0].data["result"]
    data = await wire(events, "agui")
    assert [e["toolCallId"] for e in data if e["type"] == "TOOL_CALL_RESULT"] == ["validation-call"]


def native_events(framework):
    if framework in {"langchain", "langgraph"}:
        from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

        return LangChain(), [
            {
                "event": "on_chat_model_stream",
                "run_id": "first",
                "data": {"chunk": AIMessageChunk(content="checking")},
            },
            {
                "event": "on_chat_model_end",
                "run_id": "first",
                "data": {
                    "output": AIMessage(
                        content="checking",
                        tool_calls=[{"id": "call-1", "name": "lookup", "args": {"city": "杭州"}}],
                    )
                },
            },
            {
                "event": "on_tool_end",
                "run_id": "not-a-call-id",
                "name": "lookup",
                "data": {"output": ToolMessage("result", tool_call_id="call-1")},
            },
            {
                "event": "on_chat_model_stream",
                "run_id": "last",
                "data": {"chunk": AIMessageChunk(content="answer")},
            },
            {
                "event": "on_chat_model_end",
                "run_id": "last",
                "data": {"output": AIMessage(content="answer")},
            },
        ]
    if framework == "google_adk":

        def event(parts, partial=False):
            return NS(author="agent", partial=partial, content=NS(parts=parts))

        return ADK(), [
            event([NS(text="checking")], True),
            event(
                [
                    NS(text="checking"),
                    NS(function_call=NS(id="call-1", name="lookup", args={"city": "杭州"})),
                ]
            ),
            event([NS(function_response=NS(id="call-1", name="lookup", response="result"))]),
            event([NS(text="answer")], True),
            event([NS(text="answer")]),
        ]
    if framework == "agentscope":

        def event(kind, **data):
            return NS(type=kind.upper(), **data)

        return AgentScope(), [
            event("text_block_delta", block_id="first", delta="checking"),
            event("text_block_end"),
            event("tool_call_start", tool_call_id="call-1", tool_call_name="lookup"),
            event("tool_call_delta", tool_call_id="call-1", delta='{"city":"杭州"}'),
            event("tool_call_end", tool_call_id="call-1"),
            event("tool_result_start", tool_call_id="call-1", tool_call_name="lookup"),
            event("tool_result_text_delta", tool_call_id="call-1", delta="result"),
            event("tool_result_end", tool_call_id="call-1"),
            event("text_block_delta", block_id="last", delta="answer"),
            event("text_block_end"),
        ]
    if framework == "pydantic_ai":

        def event(kind, **data):
            return NS(event_kind=kind, **data)

        return PydanticAI(), [
            event("part_start", part=NS(part_kind="text", content="checking")),
            event("part_end", part=NS(part_kind="text")),
            event(
                "function_tool_call",
                part=NS(tool_call_id="call-1", tool_name="lookup", args={"city": "杭州"}),
            ),
            event(
                "function_tool_result",
                part=NS(tool_call_id="call-1", tool_name="lookup", content="result"),
            ),
            event("part_start", part=NS(part_kind="text", content="answer")),
            event("part_end", part=NS(part_kind="text")),
        ]
    # CrewAI ReAct exposes thought explicitly, not ordinary assistant commentary.
    return CrewAI(), [
        NS(thought="checking", tool="lookup", tool_input='{"city":"杭州"}', result="result"),
        NS(thought="", output="answer"),
    ]


async def wire(events, protocol):
    async def invoke(request, context):
        for event in events:
            yield event

    app = AgentCoreServer()
    app.invoke(invoke)
    payload = (
        {
            "threadId": "thread",
            "runId": "run",
            "messages": [],
            "tools": [],
            "context": [],
            "state": {},
            "forwardedProps": {},
        }
        if protocol == "agui"
        else {"model": "app", "messages": [], "stream": True}
    )
    path = "/ag-ui/agent" if protocol == "agui" else "/openai/v1/chat/completions"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as c:
        response = await c.post(path, json=payload)
    assert response.status_code == 200, response.text
    return [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]


@pytest.mark.parametrize(
    "framework",
    [
        "langchain",
        "langgraph",
        "google_adk",
        "agentscope",
        "pydantic_ai",
        "crewai",
    ],
)
@pytest.mark.parametrize("protocol", ["agui", "openai"])
async def test_framework_execution_wire(framework, protocol):
    converter, native = native_events(framework)
    events = [item for event in native for item in converter.convert(event)]
    data = await wire(events, protocol)
    if protocol == "openai":
        deltas = [e["choices"][0]["delta"] for e in data]
        assert "".join(d.get("content", "") for d in deltas) == (
            "answer" if framework == "crewai" else "checkinganswer"
        )
        assert (
            len([call for d in deltas for call in d.get("tool_calls", []) if call.get("id")]) == 1
        )
        # No invented tool-result field or multi-message event in a completion.
        assert all("TOOL_CALL_RESULT" not in str(d) for d in deltas)
        return
    assert data[-1]["type"] == "RUN_FINISHED"
    calls = [e for e in data if e["type"] == "TOOL_CALL_START"]
    results = [e for e in data if e["type"] == "TOOL_CALL_RESULT"]
    assert len(calls) == len(results) == 1
    assert calls[0]["toolCallId"] == results[0]["toolCallId"]
    assert data.index(calls[0]) < data.index(results[0])
    starts = [e for e in data if e["type"] == "TEXT_MESSAGE_START"]
    assert len(starts) == (1 if framework == "crewai" else 2)
    assert len(set(e["messageId"] for e in starts)) == len(starts)
    assert [e["messageId"] for e in data if e["type"] == "TEXT_MESSAGE_END"] == [
        e["messageId"] for e in starts
    ]
    assert data.index(starts[-1]) > data.index(results[0])
    assert "result" in results[0]["content"]


async def test_explicit_boundary_without_tool_and_message_id_change():
    events = [
        AgentEvent(EventType.TEXT, {"delta": "a", "message_id": "a"}),
        AgentEvent(EventType.TEXT_END, {}),
        AgentEvent(EventType.TEXT, {"delta": "b", "message_id": "b"}),
        AgentEvent(EventType.TEXT, {"delta": "c", "message_id": "c"}),
    ]
    data = await wire(events, "agui")
    assert [e["messageId"] for e in data if e["type"] == "TEXT_MESSAGE_START"] == ["a", "b", "c"]
    assert [e["messageId"] for e in data if e["type"] == "TEXT_MESSAGE_END"] == ["a", "b", "c"]


async def test_real_langchain_agent_parallel_tools():
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessageChunk, ToolMessage
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.runnables import RunnableConfig
    from langchain_core.tools import tool
    from langgraph.graph import START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode, tools_condition

    class Model(BaseChatModel):
        @property
        def _llm_type(self):
            return "test"

        def bind_tools(self, tools, **kwargs):
            return self.bind(stream=True)

        def _generate(self, *args, **kwargs):
            raise AssertionError("must stream")

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            for chunk in self._stream(messages, **kwargs):
                yield chunk

        def _stream(self, messages, **kwargs):
            if any(isinstance(m, ToolMessage) for m in messages):
                yield ChatGenerationChunk(message=AIMessageChunk(content="answer"))
            else:
                yield ChatGenerationChunk(message=AIMessageChunk(content="checking"))
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "index": i,
                                "id": f"call-{i}",
                                "name": "lookup",
                                "args": json.dumps({"value": i}),
                            }
                            for i in range(2)
                        ],
                    )
                )

    @tool
    async def lookup(value: int) -> str:
        """Look up a value."""
        return f"result-{value}"

    async def call_model(state: MessagesState, config: RunnableConfig):
        # Explicit config propagation also works on Python 3.10 (no task context=).
        return {
            "messages": [
                await Model().bind_tools([lookup]).ainvoke(state["messages"], config=config)
            ]
        }

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_node("tools", ToolNode([lookup]))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition)
    graph.add_edge("tools", "model")
    agent = graph.compile()
    native = [
        e async for e in agent.astream_events({"messages": [("user", "lookup")]}, version="v2")
    ]
    converter = LangChain()
    events = [item for event in native for item in converter.convert(event)]
    data = await wire(events, "agui")
    assert data[-1]["type"] == "RUN_FINISHED"
    assert {e["toolCallId"] for e in data if e["type"] == "TOOL_CALL_START"} == {"call-0", "call-1"}
    assert {e["toolCallId"] for e in data if e["type"] == "TOOL_CALL_RESULT"} == {
        "call-0",
        "call-1",
    }
    assert len([e for e in data if e["type"] == "TEXT_MESSAGE_START"]) == 2


async def test_tool_result_after_interleaved_text_does_not_repeat_call_start():
    data = await wire(
        [
            AgentEvent(EventType.TOOL_CALL, {"id": "call", "name": "lookup", "args": {}}),
            AgentEvent(EventType.TEXT, {"delta": "waiting"}),
            AgentEvent(EventType.TOOL_RESULT, {"id": "call", "result": "done"}),
        ],
        "agui",
    )
    assert len([e for e in data if e["type"] == "TOOL_CALL_START"]) == 1
    assert len([e for e in data if e["type"] == "TOOL_CALL_END"]) == 1


async def test_native_agentscope_event_models():
    scope = pytest.importorskip("agentscope.event")
    _, native = native_events("agentscope")
    events = []
    for value in native:
        data = vars(value).copy()
        kind = data.pop("type")
        cls = getattr(scope, "".join(p.title() for p in kind.split("_")) + "Event")
        data["reply_id"] = "reply"
        if kind == "TEXT_BLOCK_END":
            data["block_id"] = "block"
        if kind == "TOOL_RESULT_END":
            data["state"] = "success"
        events.append(cls(**data))
    converter = AgentScope()
    data = await wire([item for e in events for item in converter.convert(e)], "agui")
    assert data[-1]["type"] == "RUN_FINISHED"
    assert len([e for e in data if e["type"] == "TEXT_MESSAGE_START"]) == 2
    assert [e["toolCallId"] for e in data if e["type"] == "TOOL_CALL_RESULT"] == ["call-1"]


async def test_real_pydantic_agent_events():
    pytest.importorskip("pydantic_ai")
    from pydantic_ai import Agent
    from pydantic_ai.messages import ToolReturnPart
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    async def respond(messages, info):
        if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
            yield "answer"
        else:
            yield "checking"
            yield {0: DeltaToolCall(name="lookup", json_args='{"value":1}', tool_call_id="call")}

    async def lookup(value: int) -> str:
        return "result"

    agent = Agent(FunctionModel(stream_function=respond), tools=[lookup])
    async with agent.run_stream_events("lookup") as native:
        events = [e async for e in PydanticAI().stream(native)]
    data = await wire(events, "agui")
    assert data[-1]["type"] == "RUN_FINISHED"
    assert len([e for e in data if e["type"] == "TEXT_MESSAGE_START"]) == 2
    assert [e["toolCallId"] for e in data if e["type"] == "TOOL_CALL_RESULT"] == ["call"]


async def test_real_crewai_react_step_callback(monkeypatch):
    pytest.importorskip("crewai")
    monkeypatch.setenv("CREWAI_TRACING_ENABLED", "false")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    from crewai import Agent, Crew, Task
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import tool

    class Model(BaseLLM):
        def __init__(self):
            super().__init__(model="test")
            self.calls = 0

        def call(self, messages, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return 'Thought: checking\nAction: lookup\nAction Input: {"value":1}'
            return "Thought: done\nFinal Answer: answer"

        def supports_function_calling(self):
            return False

    @tool
    def lookup(value: int) -> str:
        """Look up a value."""
        return "result"

    converter = CrewAI()
    events = []
    agent = Agent(role="test", goal="lookup", backstory="test", llm=Model(), tools=[lookup])
    crew = Crew(
        agents=[agent],
        tasks=[Task(description="lookup", expected_output="answer", agent=agent)],
        step_callback=lambda step: events.extend(converter.convert(step)),
        cache=False,
    )
    result = await crew.kickoff_async()
    assert result.raw == "answer"
    data = await wire(events, "agui")
    assert len([e for e in data if e["type"] == "TOOL_CALL_RESULT"]) == 1
    assert data[-1]["type"] == "RUN_FINISHED"


async def test_real_crewai_native_tool_events():
    pytest.importorskip("crewai")
    from crewai import Agent, Crew, Task
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import tool

    class Model(BaseLLM):
        def __init__(self):
            super().__init__(model="test")
            self.calls = 0

        def supports_function_calling(self):
            return True

        def call(self, messages, **kwargs):
            self.calls += 1
            from crewai.events import LLMCallCompletedEvent, LLMStreamChunkEvent, crewai_event_bus
            from crewai.events.types.llm_events import LLMCallType

            if self.calls == 1:
                calls = [
                    {
                        "id": "provider-call",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": '{"value":1}'},
                    }
                ]
                crewai_event_bus.emit(
                    self,
                    LLMStreamChunkEvent(
                        chunk="checking",
                        call_id="first",
                        from_agent=kwargs.get("from_agent"),
                        from_task=kwargs.get("from_task"),
                    ),
                )
                crewai_event_bus.emit(
                    self,
                    LLMCallCompletedEvent(
                        response=calls,
                        call_type=LLMCallType.TOOL_CALL,
                        call_id="first",
                        from_agent=kwargs.get("from_agent"),
                        from_task=kwargs.get("from_task"),
                    ),
                )
                return calls
            # The fake provider also emits the standard completed event, as real LLMs do.

            crewai_event_bus.emit(
                self,
                LLMCallCompletedEvent(
                    response="answer",
                    call_type=LLMCallType.LLM_CALL,
                    call_id="model-call",
                    from_agent=kwargs.get("from_agent"),
                    from_task=kwargs.get("from_task"),
                ),
            )
            return "answer"

    @tool
    def lookup(value: int) -> str:
        """Look up a value."""
        return "result"

    agent = Agent(role="test", goal="lookup", backstory="test", llm=Model(), tools=[lookup])
    crew = Crew(
        agents=[agent],
        tasks=[Task(description="lookup", expected_output="answer", agent=agent)],
        cache=False,
        tracing=False,
    )
    events = [e async for e in CrewAI().run(crew)]
    data = await wire(events, "agui")
    calls = [e for e in data if e["type"] == "TOOL_CALL_START"]
    results = [e for e in data if e["type"] == "TOOL_CALL_RESULT"]
    assert len(calls) == len(results) == 1
    assert calls[0]["toolCallId"] == results[0]["toolCallId"]
    assert data.index(calls[0]) < data.index(results[0])
    assert [e["delta"] for e in data if e["type"] == "TEXT_MESSAGE_CONTENT"] == [
        "checking",
        "answer",
    ]
    starts = [e for e in data if e["type"] == "TEXT_MESSAGE_START"]
    assert len(starts) == 2
    assert data.index(starts[1]) > data.index(results[0])


async def test_langchain_reasoning_snapshot_does_not_hide_non_streamed_answer():
    from langchain_core.messages import AIMessage, AIMessageChunk

    converter = LangChain()
    events = converter.convert(
        {
            "event": "on_chat_model_stream",
            "run_id": "model",
            "data": {
                "chunk": AIMessageChunk(content=[{"type": "reasoning", "reasoning": "thought"}])
            },
        }
    )
    events += converter.convert(
        {
            "event": "on_chat_model_end",
            "run_id": "model",
            "data": {
                "output": AIMessage(
                    content=[
                        {"type": "reasoning", "reasoning": "thought"},
                        {"type": "text", "text": "answer"},
                    ]
                )
            },
        }
    )
    data = await wire(events, "agui")
    assert [e["delta"] for e in data if e["type"] == "REASONING_MESSAGE_CONTENT"] == ["thought"]
    assert [e["delta"] for e in data if e["type"] == "TEXT_MESSAGE_CONTENT"] == ["answer"]


async def test_real_adk_agent_execution_events():
    pytest.importorskip("google.adk")
    from google.adk import Runner
    from google.adk.agents import Agent
    from google.adk.agents.run_config import RunConfig, StreamingMode
    from google.adk.models.base_llm import BaseLlm
    from google.adk.models.llm_response import LlmResponse
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    class Model(BaseLlm):
        async def generate_content_async(self, llm_request, stream=False):
            returned = any(
                part.function_response
                for content in llm_request.contents
                for part in content.parts or []
            )
            message = "answer" if returned else "checking"
            yield LlmResponse(
                partial=True, content=types.Content(role="model", parts=[types.Part(text=message)])
            )
            parts = [types.Part(text=message)]
            if not returned:
                parts.append(
                    types.Part(
                        function_call=types.FunctionCall(
                            id="adk-call", name="lookup", args={"value": 1}
                        )
                    )
                )
            yield LlmResponse(content=types.Content(role="model", parts=parts))

    async def lookup(value: int) -> str:
        return "result"

    sessions = InMemorySessionService()
    await sessions.create_session(app_name="test", user_id="user", session_id="session")
    runner = Runner(
        app_name="test",
        session_service=sessions,
        agent=Agent(name="agent", model=Model(model="test"), tools=[lookup]),
    )
    try:
        events = [
            e
            async for e in ADK().stream(
                runner.run_async(
                    user_id="user",
                    session_id="session",
                    new_message=types.Content(role="user", parts=[types.Part(text="lookup")]),
                    run_config=RunConfig(streaming_mode=StreamingMode.SSE),
                )
            )
        ]
    finally:
        await runner.close()
    data = await wire(events, "agui")
    assert [e["delta"] for e in data if e["type"] == "TEXT_MESSAGE_CONTENT"] == [
        "checking",
        "answer",
    ]
    assert [e["toolCallId"] for e in data if e["type"] == "TOOL_CALL_RESULT"] == ["adk-call"]
    assert len([e for e in data if e["type"] == "TEXT_MESSAGE_START"]) == 2


async def test_crewai_trace_is_emitted_before_execution_failure():
    pytest.importorskip("crewai")
    from crewai.events import LLMCallCompletedEvent, crewai_event_bus
    from crewai.events.types.llm_events import LLMCallType

    class Crew:
        tasks = [NS(id="failed-task")]

        async def kickoff_async(self, inputs=None):
            crewai_event_bus.emit(
                self,
                LLMCallCompletedEvent(
                    task_id="failed-task",
                    call_id="failed-call",
                    response="checking",
                    call_type=LLMCallType.LLM_CALL,
                ),
            )
            raise ValueError("execution failed")

    events = []
    with pytest.raises(ValueError, match="execution failed"):
        async for event in CrewAI().run(Crew()):
            events.append(event)
    assert [e.data["delta"] for e in events if e.type == EventType.TEXT] == ["checking"]
