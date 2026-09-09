"""Convert CrewAI execution events or ReAct step_callback values."""

import asyncio
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from agentcore.events import AgentEvent, EventType
from agentcore.integrations._event_utils import EventConverter, end_text, field


class AgentCoreConverter(EventConverter):
    """Pass ordered CrewAI event-bus events for native function calling.

    ToolUsage events expose execution event IDs, not provider tool-call IDs.
    started_event_id links each result to its start, including same-name calls.
    ReAct step callbacks also work, at step granularity rather than token level.
    Do not mix event-bus events and step callbacks in the same converter.
    """

    def __init__(self) -> None:
        self._streamed: set[tuple[str, str]] = set()

    async def run(
        self, crew: Any, inputs: dict[str, Any] | None = None
    ) -> AsyncIterator[AgentEvent]:
        """Run a request-local Crew and emit its ordered execution trace.

        CrewAI dispatches event handlers on a thread pool. Buffer until execution
        and callbacks finish, then order by emission_sequence, not arrival time.
        This is a completed-run trace, not a token-latency streaming API.
        """
        from crewai.events import (
            LLMCallCompletedEvent,
            LLMCallFailedEvent,
            LLMStreamChunkEvent,
            ToolUsageErrorEvent,
            ToolUsageFinishedEvent,
            ToolUsageStartedEvent,
            crewai_event_bus,
        )

        task_ids = {str(task.id) for task in crew.tasks}
        captured: list[Any] = []

        def receive(source: Any, event: Any) -> None:
            if event.task_id in task_ids:
                captured.append(event)

        kinds = (
            LLMCallCompletedEvent,
            LLMCallFailedEvent,
            LLMStreamChunkEvent,
            ToolUsageStartedEvent,
            ToolUsageFinishedEvent,
            ToolUsageErrorEvent,
        )
        for kind in kinds:
            crewai_event_bus.on(kind)(receive)
        try:
            failure: Exception | None = None
            try:
                await crew.kickoff_async(inputs=inputs)
            except Exception as exc:
                failure = exc
            if not await asyncio.to_thread(crewai_event_bus.flush):
                raise RuntimeError("CrewAI execution event callbacks did not finish")
            for event in sorted(captured, key=lambda event: event.emission_sequence):
                for converted in self.convert(event):
                    yield converted
            if failure is not None:
                raise failure
        finally:
            for kind in kinds:
                crewai_event_bus.off(kind, receive)

    def convert(self, event: Any) -> list[AgentEvent]:
        kind = field(event, "type")
        key = (field(event, "agent_id", ""), field(event, "task_id", ""))
        if kind == "llm_stream_chunk":
            if event.chunk:
                self._streamed.add(key)
                return [AgentEvent(EventType.TEXT, {"delta": event.chunk})]
            return []
        if kind == "llm_call_completed":
            events = []
            if key not in self._streamed and isinstance(event.response, str):
                events.append(AgentEvent(EventType.TEXT, {"delta": event.response}))
            self._streamed.discard(key)
            events.append(end_text())
            return events
        if kind == "tool_usage_started":
            return [
                AgentEvent(
                    EventType.TOOL_CALL,
                    {
                        "id": event.event_id,
                        "name": event.tool_name,
                        "args": event.tool_args,
                    },
                )
            ]
        if kind in {"tool_usage_finished", "tool_usage_error"}:
            return [
                AgentEvent(
                    EventType.TOOL_RESULT,
                    {
                        "id": event.started_event_id,
                        "name": event.tool_name,
                        "result": event.output
                        if kind == "tool_usage_finished"
                        else str(event.error),
                    },
                )
            ]
        if kind == "llm_call_failed":
            return [AgentEvent(EventType.ERROR, {"message": str(event.error)})]
        if kind is not None:
            return []
        if field(event, "tool") is not None:
            call_id = f"crew-{uuid4().hex}"
            events = []
            if event.thought:
                events.extend(
                    [
                        AgentEvent(EventType.REASONING, {"delta": event.thought}),
                        AgentEvent(EventType.REASONING_END, {}),
                    ]
                )
            events.append(
                AgentEvent(
                    EventType.TOOL_CALL,
                    {
                        "id": call_id,
                        "name": event.tool,
                        "args": event.tool_input,
                    },
                )
            )
            if event.result is not None:
                events.append(
                    AgentEvent(
                        EventType.TOOL_RESULT,
                        {
                            "id": call_id,
                            "name": event.tool,
                            "result": event.result,
                        },
                    )
                )
            return events
        if field(event, "output") is not None:
            output = event.output
            if not isinstance(output, str):
                output = output.model_dump_json()
            return [AgentEvent(EventType.TEXT, {"delta": output}), end_text()]
        return []
