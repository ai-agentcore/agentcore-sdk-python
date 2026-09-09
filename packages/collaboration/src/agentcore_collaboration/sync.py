"""Synchronous collaboration facade using the Core-owned portal and lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from functools import partial
from typing import Any, cast

from anyio.from_thread import BlockingPortal

from agentcore.integrations.common import Tool
from agentcore.skill import Skill
from agentcore_collaboration.worker import merge_tools

OperationScope = Callable[[], AbstractContextManager[None]]


class SyncCollaboration:
    def __init__(
        self,
        collaboration: Any,
        portal: BlockingPortal,
        operation: OperationScope,
    ) -> None:
        self._collaboration = collaboration
        self._portal = portal
        self._operation = operation

    def worker(self) -> SyncWorkerCollaboration:
        with self._operation():
            worker = self._portal.call(self._collaboration.worker)
        return SyncWorkerCollaboration(
            worker,
            self._portal,
            self._operation,
        )


class SyncWorkerCollaboration:
    def __init__(
        self,
        worker: Any,
        portal: BlockingPortal,
        operation: OperationScope,
    ) -> None:
        self._worker = worker
        self._portal = portal
        self._operation = operation

    def compose_prompt(self, user_prompt: str) -> str:
        return cast(str, self._worker.compose_prompt(user_prompt))

    def request_context(self, headers: Mapping[str, str]) -> AbstractContextManager[Any]:
        return cast(AbstractContextManager[Any], self._worker.request_context(headers))

    def tools(self) -> list[Tool]:
        with self._operation():
            tools = cast(list[Tool], self._portal.call(self._worker.tools))
        return [
            tool.with_sync_call(partial(self._call_tool, tool))
            for tool in tools
        ]

    def skills(self) -> list[Skill]:
        with self._operation():
            return cast(list[Skill], self._portal.call(self._worker.skills))

    def merge_langchain_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        from agentcore.integrations.langchain import tools

        return merge_tools(user_tools, tools(self.tools()))

    def merge_langgraph_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        return self.merge_langchain_tools(user_tools)

    def merge_agentscope_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        from agentcore.integrations.agentscope import tools

        return merge_tools(user_tools, tools(self.tools()))

    def merge_google_adk_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        from agentcore.integrations.google_adk import tools

        return merge_tools(user_tools, tools(self.tools()))

    def merge_pydantic_ai_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        from agentcore.integrations.pydantic_ai import tools

        return merge_tools(user_tools, tools(self.tools()))

    def merge_crewai_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        from agentcore.integrations.crewai import tools

        return merge_tools(user_tools, tools(self.tools()))

    def _call_tool(self, tool: Tool, arguments: dict[str, Any]) -> Any:
        with self._operation():
            return self._portal.call(partial(tool.ainvoke, arguments))
