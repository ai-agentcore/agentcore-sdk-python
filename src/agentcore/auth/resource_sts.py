"""Obtain short-lived RAM STS from the AgentCore Controller."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import anyio
import httpx

from agentcore.auth.agent_sa_token import (
    AgentSATokenSource,
    get_agent_sa_token,
    refresh_agent_sa_token,
)
from agentcore.errors import AuthenticationError, CredentialExchangeError

_STS_PATH = "/api/v1/credentials/sts"
_MAX_ERROR_MESSAGE_LENGTH = 1024
HIGH_CODE_SDK_PURPOSE = "highcode_sdk"
AGENT_IDENTITY_DATA_PURPOSE = "agentidentitydata"
logger = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False)
class ResourceCredential:
    access_key_id: str
    access_key_secret: str
    security_token: str
    expiration: datetime
    credential_type: str = "ram_sts"

    def __repr__(self) -> str:
        return (
            "ResourceCredential(credential_type='ram_sts', "
            f"expiration={self.expiration.isoformat()!r})"
        )


class ResourceSTSProvider:
    def __init__(
        self,
        controller_endpoint: str,
        agent_sa_tokens: AgentSATokenSource,
        *,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
        refresh_before: timedelta = timedelta(minutes=5),
        max_attempts: int = 3,
    ) -> None:
        if not controller_endpoint:
            raise CredentialExchangeError("AgentCore Controller endpoint is not configured")
        self._endpoint = controller_endpoint.rstrip("/")
        self._request_url = f"{self._endpoint}{_STS_PATH}"
        self._sa_tokens = agent_sa_tokens
        self._http = http_client or httpx.AsyncClient(timeout=timeout, follow_redirects=False)
        self._owns_http = http_client is None
        self._refresh_before = refresh_before
        self._max_attempts = max(1, max_attempts)
        self._cache: dict[str | None, ResourceCredential] = {}
        self._locks: dict[str | None, anyio.Lock] = {}
        self._locks_guard = anyio.Lock()

    async def get(self, purpose: str | None = None) -> ResourceCredential:
        if purpose is not None:
            purpose = purpose.strip()
            if not purpose:
                raise ValueError("purpose must not be empty")
        cached = self._cache.get(purpose)
        if self._valid(cached):
            assert cached is not None
            logger.debug(
                "agentcore.sts.cache.hit purpose=%s expiration=%s",
                purpose or "default",
                cached.expiration.isoformat(),
            )
            return cached
        lock = await self._lock_for(purpose)
        async with lock:
            cached = self._cache.get(purpose)
            if self._valid(cached):
                assert cached is not None
                logger.debug(
                    "agentcore.sts.cache.hit purpose=%s expiration=%s",
                    purpose or "default",
                    cached.expiration.isoformat(),
                )
                return cached
            credential = await self._exchange(purpose)
            self._cache[purpose] = credential
            return credential

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def _lock_for(self, purpose: str | None) -> anyio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(purpose, anyio.Lock())

    def _valid(self, credential: ResourceCredential | None) -> bool:
        return bool(
            credential and datetime.now(timezone.utc) + self._refresh_before < credential.expiration
        )

    async def _exchange(self, purpose: str | None) -> ResourceCredential:
        purpose_name = purpose or "default"
        logger.info(
            "agentcore.sts.exchange.started purpose=%s url=%s",
            purpose_name,
            self._request_url,
        )
        sa_token = await get_agent_sa_token(self._sa_tokens)
        response = await self._request_with_transient_retries(purpose, sa_token)
        if response.status_code == 401:
            rotated = await refresh_agent_sa_token(self._sa_tokens, sa_token)
            logger.warning(
                "agentcore.sts.exchange.sa_rejected purpose=%s rotated=%s",
                purpose_name,
                rotated != sa_token,
            )
            if rotated != sa_token:
                response = await self._request_with_transient_retries(purpose, rotated)
        if response.status_code == 401:
            message = _response_error_message(response)
            logger.warning(
                "agentcore.sts.exchange.failed purpose=%s url=%s status=401 message=%r",
                purpose_name,
                self._request_url,
                message,
            )
            raise AuthenticationError(
                _with_error_message("Controller rejected the Agent SA token", message)
            )
        if response.status_code >= 400:
            message = _response_error_message(response)
            logger.warning(
                "agentcore.sts.exchange.failed purpose=%s url=%s status=%s message=%r",
                purpose_name,
                self._request_url,
                response.status_code,
                message,
            )
            raise CredentialExchangeError(
                _with_error_message(
                    f"Controller STS request failed with HTTP {response.status_code}",
                    message,
                )
            )
        try:
            credential = self._parse_response(response)
        except CredentialExchangeError:
            logger.warning(
                "agentcore.sts.exchange.failed purpose=%s reason=invalid_response",
                purpose_name,
            )
            raise
        logger.info(
            "agentcore.sts.exchange.succeeded purpose=%s expiration=%s",
            purpose_name,
            credential.expiration.isoformat(),
        )
        return credential

    async def _request_with_transient_retries(
        self,
        purpose: str | None,
        sa_token: str,
    ) -> httpx.Response:
        last_error: httpx.HTTPError | None = None
        last_response: httpx.Response | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self._http.post(
                    self._request_url,
                    params={"purpose": purpose} if purpose is not None else None,
                    headers={"Authorization": f"Bearer {sa_token}"},
                    content=b"",
                    follow_redirects=False,
                )
            except httpx.TransportError as exc:
                last_error = exc
                last_response = None
                retry_reason = type(exc).__name__
            else:
                if response.status_code not in {500, 502, 503, 504}:
                    return response
                last_error = None
                last_response = response
                retry_reason = f"http_{response.status_code}"
            if attempt + 1 < self._max_attempts:
                logger.warning(
                    "agentcore.sts.exchange.retry purpose=%s url=%s attempt=%s "
                    "max_attempts=%s reason=%s message=%r",
                    purpose or "default",
                    self._request_url,
                    attempt + 1,
                    self._max_attempts,
                    retry_reason,
                    _response_error_message(last_response) if last_response is not None else None,
                )
                await anyio.sleep(0.05 * (2**attempt))
        if last_error is not None:
            logger.warning(
                "agentcore.sts.exchange.failed purpose=%s url=%s attempts=%s reason=%s",
                purpose or "default",
                self._request_url,
                self._max_attempts,
                type(last_error).__name__,
            )
            raise CredentialExchangeError("cannot reach the AgentCore Controller") from last_error
        assert last_response is not None
        message = _response_error_message(last_response)
        logger.warning(
            "agentcore.sts.exchange.failed purpose=%s url=%s status=%s attempts=%s message=%r",
            purpose or "default",
            self._request_url,
            last_response.status_code,
            self._max_attempts,
            message,
        )
        raise CredentialExchangeError(
            _with_error_message(
                "Controller STS request failed after "
                f"{self._max_attempts} attempts with HTTP {last_response.status_code}",
                message,
            )
        )

    @staticmethod
    def _parse_response(response: httpx.Response) -> ResourceCredential:
        try:
            payload: Any = response.json()
            if not isinstance(payload, dict):
                raise ValueError
            expiration_text = _required_string(payload, "expiration")
            expiration = datetime.fromisoformat(expiration_text.replace("Z", "+00:00"))
            if expiration.tzinfo is None:
                expiration = expiration.replace(tzinfo=timezone.utc)
            expiration = expiration.astimezone(timezone.utc)
            if expiration <= datetime.now(timezone.utc):
                raise ValueError
            return ResourceCredential(
                access_key_id=_required_string(payload, "access_key_id"),
                access_key_secret=_required_string(payload, "access_key_secret"),
                security_token=_required_string(payload, "security_token"),
                expiration=expiration,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CredentialExchangeError("Controller returned an invalid STS response") from exc


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError
    return value.strip()


def _response_error_message(response: httpx.Response) -> str | None:
    message: object
    try:
        payload: Any = response.json()
    except ValueError:
        message = response.text
    else:
        message = payload.get("message") if isinstance(payload, dict) else None
    if not isinstance(message, str):
        return None
    normalized = " ".join(message.split())
    return normalized[:_MAX_ERROR_MESSAGE_LENGTH] or None


def _with_error_message(summary: str, message: str | None) -> str:
    return f"{summary}: {message}" if message else summary
