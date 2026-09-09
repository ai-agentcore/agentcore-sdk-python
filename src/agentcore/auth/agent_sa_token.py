"""Read the rotating Agent Kubernetes ServiceAccount token."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable
from pathlib import Path
from typing import Protocol

from agentcore.errors import AuthenticationError

DEFAULT_AGENT_SA_TOKEN_PATH = Path("/var/run/agentcore/agent/token")
logger = logging.getLogger(__name__)


class AgentSATokenSource(Protocol):
    def get(self) -> str | Awaitable[str]: ...


class AgentSATokenProvider:
    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or DEFAULT_AGENT_SA_TOKEN_PATH
        self.path = Path(configured)

    def get(self) -> str:
        try:
            token = self.path.read_text(encoding="utf-8").strip()
        except OSError as exc:
            logger.warning(
                "agentcore.sa_token.read.failed path=%s error_type=%s",
                self.path,
                type(exc).__name__,
            )
            raise AuthenticationError("cannot read the Agent SA token file") from exc
        if not token:
            logger.warning("agentcore.sa_token.read.failed path=%s reason=empty", self.path)
            raise AuthenticationError("the Agent SA token file is empty")
        logger.debug("agentcore.sa_token.read.succeeded path=%s", self.path)
        return token


async def get_agent_sa_token(source: AgentSATokenSource) -> str:
    result = source.get()
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, str) or not result.strip():
        logger.warning(
            "agentcore.sa_token.source.failed source_type=%s reason=empty",
            type(source).__name__,
        )
        raise AuthenticationError("the Agent SA token source returned an empty token")
    logger.debug(
        "agentcore.sa_token.source.succeeded source_type=%s",
        type(source).__name__,
    )
    return result.strip()


async def refresh_agent_sa_token(source: AgentSATokenSource, current: str) -> str:
    refresh = getattr(source, "refresh", None)
    result = refresh(current) if callable(refresh) else source.get()
    if inspect.isawaitable(result):
        result = await result
    if not isinstance(result, str) or not result.strip():
        logger.warning(
            "agentcore.sa_token.refresh.failed source_type=%s reason=empty",
            type(source).__name__,
        )
        raise AuthenticationError("the Agent SA token source returned an empty token")
    logger.info(
        "agentcore.sa_token.refresh.succeeded source_type=%s changed=%s",
        type(source).__name__,
        result.strip() != current,
    )
    return result.strip()
