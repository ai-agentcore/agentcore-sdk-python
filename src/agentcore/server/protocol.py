"""FastAPI router extension point for Agent HTTP protocols."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fastapi import APIRouter

    from agentcore.server.invoker import AgentInvoker


class ProtocolHandler(ABC):
    """One independently routable Agent HTTP protocol."""

    name: str

    @abstractmethod
    def as_fastapi_router(self, agent_invoker: AgentInvoker) -> APIRouter:
        """Return all routes owned by this protocol."""

    def get_prefix(self) -> str:
        """Return the path prefix used when mounting this protocol."""
        return ""
