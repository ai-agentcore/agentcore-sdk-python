"""Convert PydanticAI run_stream_events, not run_stream().stream_text()."""

from typing import Any

from pydantic_core import to_jsonable_python

from agentcore.events import AgentEvent, EventType
from agentcore.integrations._event_utils import EventConverter, end_text, field


class AgentCoreConverter(EventConverter):
    def convert(self, event: Any) -> list[AgentEvent]:
        kind = field(event, "event_kind")
        part = field(event, "part")
        if kind in {"part_start", "part_delta"}:
            part = part if kind == "part_start" else event.delta
            part_kind = field(part, "part_kind") or field(part, "part_delta_kind")
            if part_kind in {"text", "thinking"}:
                delta = field(part, "content", "") if kind == "part_start" else part.content_delta
                return (
                    [
                        AgentEvent(
                            EventType.TEXT if part_kind == "text" else EventType.REASONING,
                            {"delta": delta},
                        )
                    ]
                    if delta
                    else []
                )
        if kind == "part_end":
            if part.part_kind == "text":
                return [end_text()]
            if part.part_kind == "thinking":
                return [AgentEvent(EventType.REASONING_END, {})]
        if kind in {"function_tool_call", "output_tool_call"}:
            return [
                AgentEvent(
                    EventType.TOOL_CALL,
                    {
                        "id": part.tool_call_id,
                        "name": part.tool_name,
                        "args": part.args,
                    },
                )
            ]
        if kind in {"function_tool_result", "output_tool_result"}:
            return [
                AgentEvent(
                    EventType.TOOL_RESULT,
                    {
                        "id": part.tool_call_id,
                        "name": part.tool_name,
                        "result": to_jsonable_python(part.content),
                    },
                )
            ]
        return []
