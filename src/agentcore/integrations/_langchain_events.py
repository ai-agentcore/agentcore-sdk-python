"""Convert LangChain/LangGraph astream_events(version='v2') events."""

from typing import Any

from agentcore.events import AgentEvent, EventType
from agentcore.integrations._event_utils import EventConverter, field


class AgentCoreConverter(EventConverter):
    """Preserve model-turn boundaries and native tool-call IDs.

    Tool arguments are emitted once, when the model response is complete. Tool
    callback run IDs are not tool-call IDs; results use ToolMessage.tool_call_id.
    """

    def __init__(self) -> None:
        self._streamed: dict[str, set[str]] = {}
        self._messages: dict[str, AgentEvent] = {}
        self._segments: dict[str, int] = {}
        self._calls: set[str] = set()
        self._results: set[str] = set()

    def convert(self, event: Any) -> list[AgentEvent]:
        kind, data = event.get("event"), event.get("data", {})
        run_id = event.get("run_id", "")
        if kind == "on_chat_model_stream":
            events = _message_events(data["chunk"])
            self._streamed.setdefault(run_id, set()).update(e.type for e in events)
            return self._with_message_ids(run_id, events)
        elif kind == "on_chat_model_end":
            output = data.get("output")
            streamed = self._streamed.pop(run_id, set())
            events = [e for e in _message_events(output) if e.type not in streamed]
            events = self._with_message_ids(run_id, events)
            events.extend(self._end_message(run_id))
            self._segments.pop(run_id, None)
            for call in field(output, "tool_calls", []) or []:
                self._calls.add(field(call, "id"))
                events.append(
                    AgentEvent(
                        EventType.TOOL_CALL,
                        {
                            "id": field(call, "id"),
                            "name": field(call, "name"),
                            "args": field(call, "args", {}),
                        },
                    )
                )
            return events
        elif kind == "on_tool_end":
            return self._tool_result(data.get("output"))
        elif kind == "on_chain_end":
            # ToolNode emits handled failures as ToolMessages, without on_tool_end.
            # Root snapshots repeat these messages and may include previous turns.
            output = data.get("output")
            messages = output if isinstance(output, list) else field(output, "messages", [])
            return [
                result
                for message in messages or []
                if field(message, "type") == "tool"
                and field(message, "tool_call_id") in self._calls
                for result in self._tool_result(message)
            ]
        return []

    def _tool_result(self, message: Any) -> list[AgentEvent]:
        call_id = field(message, "tool_call_id")
        if not call_id or call_id in self._results:
            return []
        self._results.add(call_id)
        return [
            AgentEvent(
                EventType.TOOL_RESULT,
                {
                    "id": call_id,
                    "name": field(message, "name"),
                    "result": field(message, "content"),
                },
            )
        ]

    def _end_message(self, run_id: str) -> list[AgentEvent]:
        previous = self._messages.pop(run_id, None)
        if previous is None:
            return []
        kind = EventType.TEXT_END if previous.type == "TEXT" else EventType.REASONING_END
        return [AgentEvent(kind, {"message_id": previous.data["message_id"]})]

    def _with_message_ids(self, run_id: str, events: list[AgentEvent]) -> list[AgentEvent]:
        converted = []
        for event in events:
            previous = self._messages.get(run_id)
            if previous is None or previous.type != event.type:
                converted.extend(self._end_message(run_id))
                segment = self._segments.get(run_id, 0) + 1
                self._segments[run_id] = segment
                message_id = f"{run_id}:{segment}"
            else:
                message_id = previous.data["message_id"]
            identified = AgentEvent(event.event, {**event.data, "message_id": message_id})
            self._messages[run_id] = identified
            converted.append(identified)
        return converted


def _message_events(message: Any) -> list[AgentEvent]:
    blocks = field(message, "content_blocks")
    if blocks is None:
        content = field(message, "content")
        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content or []
    events = []
    for block in blocks:
        kind = field(block, "type")
        if kind not in {"text", "reasoning"}:
            continue
        delta = field(block, "text" if kind == "text" else "reasoning", "")
        if delta:
            events.append(
                AgentEvent(
                    EventType.TEXT if kind == "text" else EventType.REASONING,
                    {"delta": delta},
                )
            )
    return events
