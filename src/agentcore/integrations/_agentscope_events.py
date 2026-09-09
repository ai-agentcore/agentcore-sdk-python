"""Convert AgentScope 2 reply_stream events; final Msg snapshots are ignored."""

from typing import Any

from agentcore.events import AgentEvent, EventType
from agentcore.integrations._event_utils import EventConverter, end_text, field


class AgentCoreConverter(EventConverter):
    def __init__(self) -> None:
        self._results: dict[str, list[Any]] = {}
        self._names: dict[str, str] = {}

    def convert(self, event: Any) -> list[AgentEvent]:
        kind = field(event, "type")
        call_id = field(event, "tool_call_id")
        if kind == "TEXT_BLOCK_DELTA":
            return [
                AgentEvent(
                    EventType.TEXT,
                    {
                        "delta": event.delta,
                        "message_id": event.block_id,
                    },
                )
            ]
        if kind == "TEXT_BLOCK_END":
            return [end_text()]
        if kind == "THINKING_BLOCK_DELTA":
            return [AgentEvent(EventType.REASONING, {"delta": event.delta})]
        if kind == "THINKING_BLOCK_END":
            return [AgentEvent(EventType.REASONING_END, {})]
        if kind == "TOOL_CALL_START":
            self._names[call_id] = event.tool_call_name
            return [
                AgentEvent(
                    EventType.TOOL_CALL_CHUNK,
                    {
                        "id": call_id,
                        "name": event.tool_call_name,
                        "args_delta": "",
                    },
                )
            ]
        if kind == "TOOL_CALL_DELTA":
            return [
                AgentEvent(
                    EventType.TOOL_CALL_CHUNK,
                    {
                        "id": call_id,
                        "args_delta": event.delta,
                    },
                )
            ]
        if kind == "TOOL_RESULT_START":
            self._results[call_id] = []
            self._names[call_id] = event.tool_call_name
        elif kind == "TOOL_RESULT_TEXT_DELTA":
            self._results[call_id].append({"type": "text", "text": event.delta})
        elif kind == "TOOL_RESULT_DATA_DELTA":
            self._results[call_id].append(event.model_dump(mode="json"))
        elif kind == "TOOL_RESULT_END":
            return [
                AgentEvent(
                    EventType.TOOL_RESULT,
                    {
                        "id": call_id,
                        "name": self._names.pop(call_id, ""),
                        "result": self._results.pop(call_id, []),
                    },
                )
            ]
        elif kind == "REPLY_END" and field(event, "error"):
            return [AgentEvent(EventType.ERROR, event.error.model_dump(mode="json"))]
        return []
