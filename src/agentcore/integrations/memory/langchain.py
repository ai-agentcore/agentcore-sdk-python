"""Async LangChain memory middleware; no checkpoint or history management."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.runtime import Runtime
from typing_extensions import NotRequired

from agentcore.integrations.memory._common import (
    MemoryScopes,
    logger,
    recall,
    record,
    reference_text,
    validate_top_k,
)
from agentcore.memory import AsyncMemoryStore, MemoryMessage, MemoryScope


class MemoryAgentState(AgentState):
    """Separate fields are overwritten per turn, never appended to messages."""

    agentcore_memory_text: NotRequired[str]
    agentcore_memory_input: NotRequired[list[MemoryMessage]]
    agentcore_memory_write_scope: NotRequired[MemoryScope | None]


def message_text(message: BaseMessage) -> str:
    content = message.content
    if isinstance(content, str):
        return content
    return "\n".join(
        part["text"]
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    )


def memory_messages(system: SystemMessage | None, text: str) -> SystemMessage | None:
    """Return a new system message, preserving the original content blocks."""
    note = reference_text(text)
    if not note:
        return system
    if system is None:
        return SystemMessage(content=note)
    content = system.content
    return system.model_copy(
        update={
            "content": content + "\n\n" + note
            if isinstance(content, str)
            else [*content, {"type": "text", "text": note}],
        }
    )


def _incoming_messages(messages: Sequence[BaseMessage]) -> list[MemoryMessage]:
    # Only a fresh trailing user block constitutes a new text turn. A history
    # ending in assistant/tool messages is a continuation, not new user input.
    incoming = []
    for message in reversed(messages):
        if not isinstance(message, HumanMessage):
            break
        text = message_text(message)
        if text.strip():
            incoming.append(MemoryMessage("user", text))
    return list(reversed(incoming))


class AgentCoreMemoryMiddleware(AgentMiddleware[MemoryAgentState, Any]):
    state_schema = MemoryAgentState

    def __init__(
        self,
        store: AsyncMemoryStore,
        *,
        scope_resolver: Callable[[Any], MemoryScopes],
        write_back: bool = False,
        top_k: int = 5,
    ) -> None:
        validate_top_k(top_k)
        self._store = store
        self._scopes = scope_resolver
        self._write_back = write_back
        self._top_k = top_k

    async def abefore_agent(
        self,
        state: MemoryAgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any]:
        scopes = self._scopes(runtime.context)
        write_scope = scopes.require_write() if self._write_back else None
        incoming = _incoming_messages(state["messages"])
        text = await recall(
            self._store,
            "\n".join(msg.content for msg in incoming),
            scopes.read,
            top_k=self._top_k,
            best_effort=True,
        )
        return {
            "agentcore_memory_text": text,
            "agentcore_memory_input": incoming,
            "agentcore_memory_write_scope": write_scope,
        }

    async def awrap_model_call(
        self,
        request: ModelRequest[Any],
        handler: Callable[[ModelRequest[Any]], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        text = cast(MemoryAgentState, request.state).get("agentcore_memory_text", "")
        if text:
            logger.info("agentcore.memory.adapter.injected framework=langchain")
            request = request.override(system_message=memory_messages(request.system_message, text))
        return await handler(request)

    async def aafter_agent(
        self,
        state: MemoryAgentState,
        runtime: Runtime[Any],
    ) -> dict[str, Any]:
        incoming = state.get("agentcore_memory_input", [])
        scope = state.get("agentcore_memory_write_scope")
        final = state["messages"][-1] if state["messages"] else None
        if (
            incoming
            and scope is not None
            and isinstance(final, AIMessage)
            and not final.tool_calls
            and final.response_metadata.get("finish_reason") not in {"length", "max_tokens"}
        ):
            text = message_text(final)
            if text.strip():
                await record(
                    self._store,
                    [*incoming, MemoryMessage("assistant", text)],
                    scope,
                    best_effort=True,
                )
        return {
            "agentcore_memory_input": [],
            "agentcore_memory_text": "",
            "agentcore_memory_write_scope": None,
        }
