"""Exchange the rotating Agent SA token for an opaque WAT."""

from __future__ import annotations

import logging
from typing import Any

import anyio
import httpx

from agentcore.auth.agent_sa_token import (
    AgentSATokenSource,
    get_agent_sa_token,
    refresh_agent_sa_token,
)
from agentcore.errors import AuthenticationError, WorkloadIdentityNotConfiguredError

_TOKEN_PATH = "/api/v1/workload/token"
logger = logging.getLogger(__name__)


class WorkloadAccessTokenProvider:
    def __init__(
        self,
        controller_endpoint: str,
        agent_sa_token_provider: AgentSATokenSource,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
        max_attempts: int = 3,
    ) -> None:
        if not controller_endpoint:
            raise AuthenticationError("AgentCore Controller endpoint is not configured")
        self._endpoint = controller_endpoint.rstrip("/")
        self._sa_tokens = agent_sa_token_provider
        self._http = http_client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        self._owns_http = http_client is None
        self._max_attempts = max(1, max_attempts)
        self._token: str | None = None
        self._lock = anyio.Lock()

    async def get(self) -> str:
        if self._token is not None:
            logger.debug("agentcore.wat.cache.hit")
            return self._token
        async with self._lock:
            if self._token is None:
                self._token = await self._exchange()
            else:
                logger.debug("agentcore.wat.cache.hit")
            return self._token

    async def invalidate(self, token: str | None = None) -> None:
        async with self._lock:
            if token is None or self._token == token:
                self._token = None
                logger.info("agentcore.wat.cache.invalidated")

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def _exchange(self) -> str:
        logger.info("agentcore.wat.exchange.started")
        sa_token = await get_agent_sa_token(self._sa_tokens)
        response = await self._request_with_transient_retries(sa_token)
        if response.status_code == 401:
            rotated = await refresh_agent_sa_token(self._sa_tokens, sa_token)
            logger.warning(
                "agentcore.wat.exchange.sa_rejected rotated=%s",
                rotated != sa_token,
            )
            if rotated != sa_token:
                response = await self._request_with_transient_retries(rotated)
        if response.status_code == 401:
            logger.warning("agentcore.wat.exchange.failed status=401")
            raise AuthenticationError("Controller rejected the Agent SA token")
        if response.status_code == 404:
            logger.warning("agentcore.wat.exchange.failed status=404 reason=identity_missing")
            raise WorkloadIdentityNotConfiguredError(
                "the current Agent has no configured Workload Identity"
            )
        if response.status_code >= 400:
            logger.warning("agentcore.wat.exchange.failed status=%s", response.status_code)
            raise AuthenticationError(
                f"Controller workload token request failed with HTTP {response.status_code}"
            )
        try:
            payload: Any = response.json()
        except ValueError as exc:
            logger.warning("agentcore.wat.exchange.failed reason=invalid_response")
            raise AuthenticationError(
                "Controller returned an invalid workload token response"
            ) from exc
        token = payload.get("workloadAccessToken") if isinstance(payload, dict) else None
        if not isinstance(token, str) or not token.strip():
            logger.warning("agentcore.wat.exchange.failed reason=empty_response")
            raise AuthenticationError("Controller returned an empty workload access token")
        logger.info("agentcore.wat.exchange.succeeded")
        return token.strip()

    async def _request_with_transient_retries(self, sa_token: str) -> httpx.Response:
        last_error: httpx.HTTPError | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self._http.post(
                    f"{self._endpoint}{_TOKEN_PATH}",
                    headers={"Authorization": f"Bearer {sa_token}"},
                    content=b"",
                    follow_redirects=False,
                )
            except httpx.TransportError as exc:
                last_error = exc
                retry_reason = type(exc).__name__
            else:
                if response.status_code not in {500, 502, 503, 504}:
                    return response
                last_error = None
                retry_reason = f"http_{response.status_code}"
            if attempt + 1 < self._max_attempts:
                logger.warning(
                    "agentcore.wat.exchange.retry attempt=%s max_attempts=%s reason=%s",
                    attempt + 1,
                    self._max_attempts,
                    retry_reason,
                )
                await anyio.sleep(0.05 * (2**attempt))
        if last_error is not None:
            raise AuthenticationError("cannot reach the AgentCore Controller") from last_error
        raise AuthenticationError("Controller workload token service is unavailable")
