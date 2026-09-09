"""Resolve one immutable AgentCore runtime snapshot at process startup."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import anyio
import httpx

from agentcore.auth.agent_sa_token import AgentSATokenProvider, AgentSATokenSource
from agentcore.errors import AuthenticationError, ConfigError, ResourceNotConfiguredError
from agentcore.runtime.config import (
    DEFAULT_CONFIG_PATH,
    AgentConfig,
    MCPConfig,
    ModelConfig,
    load_agent_config,
)
from agentcore.runtime.control_config import ControlConfigLoader
from agentcore.runtime.environment import DEFAULT_ENV_PATH, RuntimeEnvironmentProvider

DEBUG_TOKEN_ENV = "AGENTCORE_DEBUG_TOKEN"
CONFIG_WAIT_TIMEOUT_ENV = "AGENTCORE_CONFIG_WAIT_TIMEOUT"
DEBUG_TOKEN_PATH = "/api/v1/edge/token"
MATRIX_TOKEN_PATH = "/api/v1/credentials/matrix-token"
_MAX_DEBUG_TOKEN_BYTES = 256 * 1024
_SA_REFRESH_BEFORE = timedelta(minutes=1)
_JWT_REFRESH_BEFORE = timedelta(minutes=5)
_JWT_REFRESH_RETRY_DELAY = 30.0
_CONFIG_LOAD_DEFAULT_TIMEOUT = 10
_CONFIG_LOAD_RETRY_DELAY = 0.5
logger = logging.getLogger(__name__)


class _DebugTokenExchangeUnavailable(AuthenticationError):
    pass


@dataclass(frozen=True, repr=False)
class DebugToken:
    product: str
    jwt_token: str
    controller_url: str
    model_gateway_url: str
    matrix_url: str

    def __repr__(self) -> str:
        return (
            "DebugToken(product='agentcore', jwt_token=<redacted>, "
            f"controller_url={self.controller_url!r}, "
            f"model_gateway_url={self.model_gateway_url!r}, "
            f"matrix_url={self.matrix_url!r})"
        )


@dataclass(frozen=True)
class RuntimeBindings:
    config: AgentConfig
    controller_endpoint: str | None
    control_plane_endpoint: str | None
    agent_sa_tokens: AgentSATokenSource | None = None


class RuntimeSource(Protocol):
    initial: RuntimeBindings | None

    async def resolve(self) -> RuntimeBindings: ...

    async def aclose(self) -> None: ...


class ManagedRuntimeSource:
    """Load the mounted runtime configuration only when a managed capability needs it."""

    initial: RuntimeBindings | None = None

    def __init__(
        self,
        config_path: str | Path | None,
        *,
        env_path: str | Path | None,
        control_plane_endpoint: str | None,
    ) -> None:
        self._config_path = config_path
        self._env_path = env_path
        self._control_plane_endpoint = control_plane_endpoint
        self._bindings: RuntimeBindings | None = None
        self._lock = anyio.Lock()

    async def resolve(self) -> RuntimeBindings:
        if self._bindings is not None:
            return self._bindings
        async with self._lock:
            if self._bindings is None:
                config = await _load_managed_config(self._config_path)
                configured_env = self._env_path or os.getenv("AGENTCORE_ENV_PATH")
                configured_env_path = Path(configured_env or DEFAULT_ENV_PATH)
                has_environment = configured_env is not None or await anyio.to_thread.run_sync(
                    configured_env_path.is_file
                )
                runtime = None
                if has_environment:
                    runtime = await anyio.to_thread.run_sync(
                        RuntimeEnvironmentProvider(configured_env_path).snapshot
                    )
                control_plane_endpoint = self._control_plane_endpoint
                if control_plane_endpoint is None:
                    control_plane_endpoint = (
                        runtime.control_plane_endpoint
                        if runtime
                        else os.getenv("AGENTCORE_CONTROL_ENDPOINT")
                    )
                self._bindings = RuntimeBindings(
                    config=config,
                    controller_endpoint=runtime.controller_url if runtime else None,
                    control_plane_endpoint=control_plane_endpoint,
                    agent_sa_tokens=(
                        AgentSATokenProvider(runtime.agent_sa_token_file)
                        if runtime
                        else None
                    ),
                )
                logger.info(
                    "agentcore.runtime.source.resolved mode=managed env_loaded=%s "
                    "auth_configured=%s",
                    runtime is not None,
                    runtime is not None,
                )
            return self._bindings

    async def aclose(self) -> None:
        return None


async def _load_managed_config(config_path: str | Path | None) -> AgentConfig:
    configured = Path(config_path or os.getenv("AGENTCORE_CONFIG_PATH") or DEFAULT_CONFIG_PATH)
    raw_timeout = os.getenv(CONFIG_WAIT_TIMEOUT_ENV, str(_CONFIG_LOAD_DEFAULT_TIMEOUT))
    try:
        timeout = int(raw_timeout)
    except ValueError as exc:
        raise ConfigError(f"{CONFIG_WAIT_TIMEOUT_ENV} must be a non-negative integer") from exc
    if timeout < 0:
        raise ConfigError(f"{CONFIG_WAIT_TIMEOUT_ENV} must be a non-negative integer")

    max_attempts = int(timeout / _CONFIG_LOAD_RETRY_DELAY) + 1
    for attempt in range(1, max_attempts + 1):
        exists = await anyio.to_thread.run_sync(configured.exists)
        if exists or attempt == max_attempts:
            return await anyio.to_thread.run_sync(load_agent_config, configured)
        logger.info(
            "agentcore.runtime.config.waiting_for_mount attempt=%s max_attempts=%s "
            "retry_in_seconds=%s",
            attempt,
            max_attempts,
            _CONFIG_LOAD_RETRY_DELAY,
        )
        await anyio.sleep(_CONFIG_LOAD_RETRY_DELAY)
    raise AssertionError("unreachable")


class UnconfiguredRuntimeSource:
    """Represent a Core that only uses direct or local capabilities."""

    initial: RuntimeBindings | None = None

    async def resolve(self) -> RuntimeBindings:
        raise ResourceNotConfiguredError("AgentCore runtime configuration is not configured")

    async def aclose(self) -> None:
        return None


class DebugRuntimeSource:
    """Exchange and renew local-debug bootstrap JWT and Agent SA tokens."""

    initial: RuntimeBindings | None = None

    def __init__(
        self,
        token: DebugToken,
        *,
        control_plane_endpoint: str | None = None,
        http_client: httpx.AsyncClient | None = None,
        timeout: float = 10.0,
        max_attempts: int = 3,
    ) -> None:
        self._debug_token = token
        self._control_plane_endpoint = control_plane_endpoint
        self._http = http_client
        self._timeout = timeout
        self._owns_http = http_client is None
        self._max_attempts = max(1, max_attempts)
        self._bindings: RuntimeBindings | None = None
        self._control_config: ControlConfigLoader | None = None
        self._sa_token: str | None = None
        self._sa_expires_at: datetime | None = None
        self._jwt_token = token.jwt_token
        self._jwt_expires_at: datetime | None = None
        self._jwt_refresh_task: asyncio.Task[None] | None = None
        self._closed = False
        self._lock = anyio.Lock()

    async def resolve(self) -> RuntimeBindings:
        if self._bindings is not None:
            logger.debug("agentcore.runtime.debug.cache.hit")
            return self._bindings
        async with self._lock:
            if self._bindings is None:
                logger.info("agentcore.runtime.debug.resolve.started")
                if self._sa_token is None or not self._sa_token_is_valid():
                    await self._exchange_sa_locked()
                assert self._sa_token is not None
                if self._control_config is None:
                    self._control_config = ControlConfigLoader(
                        self._debug_token.controller_url,
                        http_client=self._client(),
                        max_attempts=self._max_attempts,
                    )
                config = await self._control_config.load(self._sa_token)
                self._bindings = self._bindings_from(config)
                logger.info(
                    "agentcore.runtime.debug.resolve.succeeded workspace_id=%s region_id=%s",
                    config.workspace_id,
                    config.region_id,
                )
            return self._bindings

    async def get(self) -> str:
        if self._sa_token is not None and self._sa_token_is_valid():
            return self._sa_token
        async with self._lock:
            if self._sa_token is None or not self._sa_token_is_valid():
                await self._exchange_sa_locked()
            assert self._sa_token is not None
            return self._sa_token

    async def refresh(self, current: str) -> str:
        async with self._lock:
            if self._sa_token is not None and self._sa_token != current:
                return self._sa_token
            await self._exchange_sa_locked()
            assert self._sa_token is not None
            return self._sa_token

    @property
    def matrix_url(self) -> str:
        return self._debug_token.matrix_url

    async def load_teams_config(self) -> bytes | None:
        await self.resolve()
        assert self._control_config is not None
        agent_sa_token = await self.get()
        return await self._control_config.load_teams(agent_sa_token)

    async def exchange_matrix_token(self) -> str:
        agent_sa_token = await self.get()
        response = await self._request_matrix_token(agent_sa_token)
        if response.status_code == 401:
            agent_sa_token = await self.refresh(agent_sa_token)
            response = await self._request_matrix_token(agent_sa_token)
        if response.status_code in {401, 403}:
            raise AuthenticationError("Controller rejected the Agent SA token")
        if response.status_code >= 400:
            raise AuthenticationError(
                f"Controller Matrix token request failed with HTTP {response.status_code}"
            )
        try:
            payload: Any = response.json()
            if not isinstance(payload, dict):
                raise ValueError
            return _required_string(payload, "access_token")
        except (KeyError, TypeError, ValueError) as exc:
            raise AuthenticationError(
                "Controller returned an invalid Matrix token response"
            ) from exc

    async def _request_matrix_token(self, agent_sa_token: str) -> httpx.Response:
        try:
            return await self._client().post(
                f"{self._debug_token.controller_url}{MATRIX_TOKEN_PATH}",
                headers={"Authorization": f"Bearer {agent_sa_token}"},
                content=b"",
                follow_redirects=False,
            )
        except httpx.TransportError as exc:
            raise AuthenticationError("cannot reach the AgentCore Controller") from exc

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._jwt_refresh_task is not None:
            self._jwt_refresh_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._jwt_refresh_task
        if self._owns_http and self._http is not None:
            await self._http.aclose()
        logger.debug("agentcore.runtime.debug.closed")

    async def _exchange_sa_locked(self) -> None:
        logger.info("agentcore.runtime.debug.sa_exchange.started")
        response = await self._request_with_transient_retries()
        if response.status_code in {401, 403}:
            logger.warning(
                "agentcore.runtime.debug.sa_exchange.failed status=%s",
                response.status_code,
            )
            raise AuthenticationError("Controller rejected the AgentCore debug token")
        if response.status_code >= 400:
            logger.warning(
                "agentcore.runtime.debug.sa_exchange.failed status=%s",
                response.status_code,
            )
            raise AuthenticationError(
                f"Controller debug token exchange failed with HTTP {response.status_code}"
            )
        try:
            payload: Any = response.json()
        except ValueError as exc:
            logger.warning(
                "agentcore.runtime.debug.sa_exchange.failed reason=invalid_response"
            )
            raise AuthenticationError(
                "Controller returned an invalid debug token response"
            ) from exc
        if not isinstance(payload, dict):
            logger.warning(
                "agentcore.runtime.debug.sa_exchange.failed reason=invalid_response"
            )
            raise AuthenticationError("Controller returned an invalid debug token response")
        try:
            token = _required_string(payload, "token")
            expires_at = _timestamp(_required_string(payload, "expiresAt"))
            jwt_token = _required_string(payload, "jwtToken")
            jwt_expires_at = _timestamp(_required_string(payload, "jwtExpiresAt"))
            now = datetime.now(timezone.utc)
            if expires_at <= now or jwt_expires_at <= now:
                raise ValueError
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(
                "agentcore.runtime.debug.sa_exchange.failed reason=invalid_response"
            )
            raise AuthenticationError(
                "Controller returned an invalid debug token response"
            ) from exc
        self._sa_token = token
        self._sa_expires_at = expires_at
        self._jwt_token = jwt_token
        self._jwt_expires_at = jwt_expires_at
        logger.info(
            "agentcore.runtime.debug.sa_exchange.succeeded expiration=%s "
            "jwt_expiration=%s",
            expires_at.isoformat(),
            jwt_expires_at.isoformat(),
        )
        self._ensure_jwt_refresh_task()

    def _ensure_jwt_refresh_task(self) -> None:
        if self._closed:
            return
        if self._jwt_refresh_task is None or self._jwt_refresh_task.done():
            self._jwt_refresh_task = asyncio.create_task(
                self._refresh_jwt_until_closed(),
                name="agentcore-debug-jwt-refresh",
            )

    async def _refresh_jwt_until_closed(self) -> None:
        while not self._closed:
            expires_at = self._jwt_expires_at
            if expires_at is None:
                return
            now = datetime.now(timezone.utc)
            if expires_at <= now:
                logger.warning("agentcore.runtime.debug.jwt_refresh.expired")
                return
            delay = max(
                0.0,
                (expires_at - _JWT_REFRESH_BEFORE - now).total_seconds(),
            )
            await asyncio.sleep(delay)
            retry = False
            async with self._lock:
                if self._closed:
                    return
                if self._jwt_expires_at != expires_at:
                    continue
                logger.info("agentcore.runtime.debug.jwt_refresh.started")
                try:
                    await self._exchange_sa_locked()
                except _DebugTokenExchangeUnavailable:
                    retry = True
                except AuthenticationError:
                    logger.warning("agentcore.runtime.debug.jwt_refresh.failed")
                    return
                else:
                    logger.info("agentcore.runtime.debug.jwt_refresh.succeeded")
            if retry:
                remaining = (expires_at - datetime.now(timezone.utc)).total_seconds()
                if remaining <= 1.0:
                    logger.warning(
                        "agentcore.runtime.debug.jwt_refresh.retry_window_exhausted"
                    )
                    return
                retry_delay = min(_JWT_REFRESH_RETRY_DELAY, remaining / 2)
                logger.warning(
                    "agentcore.runtime.debug.jwt_refresh.retry retry_in_seconds=%s",
                    retry_delay,
                )
                await asyncio.sleep(retry_delay)

    async def _request_with_transient_retries(self) -> httpx.Response:
        last_error: httpx.HTTPError | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self._client().post(
                    f"{self._debug_token.controller_url}{DEBUG_TOKEN_PATH}",
                    json={"jwtToken": self._jwt_token},
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
                    "agentcore.runtime.debug.sa_exchange.retry attempt=%s "
                    "max_attempts=%s reason=%s",
                    attempt + 1,
                    self._max_attempts,
                    retry_reason,
                )
                await anyio.sleep(0.05 * (2**attempt))
        if last_error is not None:
            raise _DebugTokenExchangeUnavailable(
                "cannot reach the AgentCore Controller"
            ) from last_error
        raise _DebugTokenExchangeUnavailable(
            "Controller debug token service is unavailable"
        )

    def _bindings_from(self, config: AgentConfig) -> RuntimeBindings:
        config = replace(
            config,
            spec=replace(
                config.spec,
                model=ModelConfig(
                    gateway_url=_combine_gateway_url(
                        self._debug_token.model_gateway_url,
                        config.spec.model.gateway_url,
                    )
                ),
                mcp=MCPConfig(
                    gateway_url=_combine_gateway_url(
                        self._debug_token.model_gateway_url,
                        config.spec.mcp.gateway_url,
                    )
                ),
            ),
        )
        return RuntimeBindings(
            config=config,
            controller_endpoint=self._debug_token.controller_url,
            control_plane_endpoint=self._control_plane_endpoint,
            agent_sa_tokens=self,
        )

    def _sa_token_is_valid(self) -> bool:
        return bool(
            self._sa_expires_at
            and datetime.now(timezone.utc) + _SA_REFRESH_BEFORE < self._sa_expires_at
        )

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=self._timeout,
                follow_redirects=False,
            )
        return self._http

    def __repr__(self) -> str:
        return f"DebugRuntimeSource(token={self._debug_token!r})"


def parse_debug_token(value: str) -> DebugToken:
    try:
        raw = base64.b64decode(value.strip(), validate=True)
        if not raw or len(raw) > _MAX_DEBUG_TOKEN_BYTES:
            raise ValueError
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError
        product = _required_string(payload, "product")
        if product != "agentcore":
            raise ValueError
        return DebugToken(
            product=product,
            jwt_token=_required_string(payload, "jwtToken"),
            controller_url=_http_url(
                _required_string(payload, "controllerUrl"),
                "controllerUrl",
            ),
            model_gateway_url=_http_url(
                _required_string(payload, "modelGatewayUrl"),
                "modelGatewayUrl",
            ),
            matrix_url=_http_url(
                _required_string(payload, "matrixUrl"),
                "matrixUrl",
            ),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ConfigError("AGENTCORE_DEBUG_TOKEN is invalid") from exc


def create_runtime_source(
    config_path: str | Path | None = None,
    *,
    env_path: str | Path | None = None,
    control_plane_endpoint: str | None = None,
) -> RuntimeSource:
    debug_token = os.getenv(DEBUG_TOKEN_ENV, "").strip()
    if debug_token:
        if control_plane_endpoint is None:
            control_plane_endpoint = os.getenv("AGENTCORE_CONTROL_ENDPOINT")
        debug_source = DebugRuntimeSource(
            parse_debug_token(debug_token),
            control_plane_endpoint=control_plane_endpoint,
        )
        logger.info("agentcore.runtime.source.selected mode=debug")
        return debug_source

    logger.info("agentcore.runtime.source.selected mode=managed")
    return ManagedRuntimeSource(
        config_path,
        env_path=env_path,
        control_plane_endpoint=control_plane_endpoint,
    )


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError
    return value.strip()


def _timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _http_url(value: str, field: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ValueError from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError
    if field == "controllerUrl" and parsed.path not in {"", "/"}:
        raise ValueError
    return value.rstrip("/")


def _combine_gateway_url(public_gateway_url: str, runtime_gateway_url: str) -> str:
    public = urlsplit(public_gateway_url)
    runtime = urlsplit(runtime_gateway_url)
    path = runtime.path or public.path
    return public._replace(path=path, query=runtime.query, fragment="").geturl().rstrip("/")
