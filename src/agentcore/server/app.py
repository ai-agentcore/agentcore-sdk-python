"""FastAPI application for AgentCore high-code Agents."""

from __future__ import annotations

import inspect
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from agentcore.server.agui_protocol import AGUIProtocolHandler
from agentcore.server.invoker import AgentInvoker, InvokeHandler
from agentcore.server.openai_protocol import OpenAIProtocolHandler
from agentcore.server.protocol import ProtocolHandler

logger = logging.getLogger(__name__)


class AgentCoreServer:
    """Serve one Agent through AG-UI, OpenAI, or custom protocol handlers."""

    def __init__(
        self,
        *,
        protocols: Sequence[ProtocolHandler] | None = None,
        readiness: Callable[[], bool] | None = None,
        startup: Callable[[], Any] | None = None,
        shutdown: Callable[[], Any] | None = None,
    ) -> None:
        self._invoker = AgentInvoker()
        self._startup = startup
        self._shutdown = shutdown
        self._readiness = readiness or self._default_readiness
        self.app = FastAPI(
            docs_url=None,
            redoc_url=None,
            openapi_url=None,
            lifespan=self._lifespan,
        )
        self._mount_health_routes()
        selected_protocols = (
            protocols
            if protocols is not None
            else (OpenAIProtocolHandler(), AGUIProtocolHandler())
        )
        for protocol in selected_protocols:
            self.app.include_router(
                protocol.as_fastapi_router(self._invoker),
                prefix=protocol.get_prefix(),
            )

    def invoke(self, handler: InvokeHandler) -> InvokeHandler:
        self._invoker.set_handler(handler)
        return handler

    def as_fastapi_app(self) -> FastAPI:
        return self.app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        await self.app(scope, receive, send)

    def _default_readiness(self) -> bool:
        return self._invoker.configured

    def _mount_health_routes(self) -> None:
        @self.app.get("/healthz")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self.app.get("/readyz")
        async def readiness() -> JSONResponse:
            try:
                ready = bool(self._readiness())
            except Exception as exc:
                logger.warning(
                    "agentcore.server.readiness.failed error_type=%s",
                    type(exc).__name__,
                )
                ready = False
            return JSONResponse({"ready": ready}, status_code=200 if ready else 503)

    @asynccontextmanager
    async def _lifespan(self, _app: FastAPI) -> AsyncIterator[None]:
        logger.info("agentcore.server.lifecycle.startup.started")
        try:
            await _maybe_call(self._startup)
        except Exception as exc:
            logger.warning(
                "agentcore.server.lifecycle.startup.failed error_type=%s",
                type(exc).__name__,
            )
            raise
        logger.info("agentcore.server.lifecycle.startup.succeeded")
        try:
            yield
        finally:
            logger.info("agentcore.server.lifecycle.shutdown.started")
            try:
                await _maybe_call(self._shutdown)
            except Exception as exc:
                logger.warning(
                    "agentcore.server.lifecycle.shutdown.failed error_type=%s",
                    type(exc).__name__,
                )
                raise
            logger.info("agentcore.server.lifecycle.shutdown.succeeded")


async def _maybe_call(callback: Callable[[], Any] | None) -> None:
    if callback is None:
        return
    result = callback()
    if inspect.isawaitable(result):
        await result
