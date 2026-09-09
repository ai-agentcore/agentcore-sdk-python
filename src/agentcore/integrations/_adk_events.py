"""Convert Google ADK Runner.run_async events (streaming and non-streaming)."""

from typing import Any

from pydantic_core import to_jsonable_python

from agentcore.events import AgentEvent, EventType
from agentcore.integrations._event_utils import EventConverter, end_text, field


class AgentCoreConverter(EventConverter):
    def __init__(self) -> None:
        self._partial: set[str] = set()
        self._calls: set[str] = set()

    def convert(self, event: Any) -> list[AgentEvent]:
        if field(event, "error_code"):
            return [
                AgentEvent(
                    EventType.ERROR,
                    {
                        "code": field(event, "error_code"),
                        "message": field(event, "error_message"),
                    },
                )
            ]
        author = field(event, "author", "")
        partial = field(event, "partial", False)
        events = []
        for part in field(field(event, "content"), "parts", []) or []:
            content = field(part, "text")
            if content and (partial or author not in self._partial):
                kind = EventType.REASONING if field(part, "thought") else EventType.TEXT
                events.append(AgentEvent(kind, {"delta": content}))
            call = field(part, "function_call")
            if call and field(call, "id") not in self._calls:
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
            result = field(part, "function_response")
            if result:
                events.append(
                    AgentEvent(
                        EventType.TOOL_RESULT,
                        {
                            "id": field(result, "id"),
                            "name": field(result, "name"),
                            "result": to_jsonable_python(field(result, "response")),
                        },
                    )
                )
        if partial:
            self._partial.add(author)
        else:
            self._partial.discard(author)
            events.extend([end_text(), AgentEvent(EventType.REASONING_END, {})])
        return events
