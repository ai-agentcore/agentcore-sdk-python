"""Local-debug collaboration state backed by Controller-scoped configuration."""

from __future__ import annotations

import hashlib
import logging
import os
import time
from typing import Protocol
from urllib.parse import urlsplit

import anyio

from agentcore.collaboration.errors import CollaborationConfigError
from agentcore.errors import AgentCoreError
from agentcore_collaboration.teams import TeamsSnapshot, parse_teams_config

_DEFAULT_REFRESH_INTERVAL = 60.0
logger = logging.getLogger("agentcore.collaboration.debug")


class DebugCollaborationSource(Protocol):
    @property
    def matrix_url(self) -> str: ...

    async def load_teams_config(self) -> bytes | None: ...

    async def exchange_matrix_token(self) -> str: ...


class DebugCollaborationRuntime:
    """Refresh teams.yaml lazily and retain the latest usable debug snapshot."""

    def __init__(
        self,
        source: DebugCollaborationSource,
        *,
        refresh_interval: float = _DEFAULT_REFRESH_INTERVAL,
    ) -> None:
        self._source = source
        self._refresh_interval = max(0.0, refresh_interval)
        self._snapshot: TeamsSnapshot | None = None
        self._digest: bytes | None = None
        self._matrix_token: str | None = None
        self._matrix_user_id: str | None = None
        self._next_refresh = 0.0
        self._initialized = False
        self._lock = anyio.Lock()

    async def teams_snapshot(self) -> TeamsSnapshot | None:
        now = time.monotonic()
        if self._initialized and now < self._next_refresh:
            return self._snapshot
        async with self._lock:
            now = time.monotonic()
            if self._initialized and now < self._next_refresh:
                return self._snapshot
            try:
                data = await self._source.load_teams_config()
                if data is None:
                    self._snapshot = None
                    self._digest = None
                    self._matrix_token = None
                    self._matrix_user_id = None
                else:
                    digest = hashlib.sha256(data).digest()
                    if digest != self._digest:
                        snapshot = parse_teams_config(data)
                        old_user_id = (
                            self._snapshot.self_matrix_user_id
                            if self._snapshot is not None
                            else None
                        )
                        if old_user_id != snapshot.self_matrix_user_id:
                            token = None
                            if self._matrix_token is not None:
                                token = await self._exchange_matrix_token()
                            self._matrix_token = token
                            self._matrix_user_id = (
                                snapshot.self_matrix_user_id if token is not None else None
                            )
                        self._snapshot = snapshot
                        self._digest = digest
            except AgentCoreError as exc:
                if not self._initialized:
                    if isinstance(exc, CollaborationConfigError):
                        raise
                    raise CollaborationConfigError(
                        "cannot load teams.yaml for local debug"
                    ) from exc
                logger.warning(
                    "agentcore.collaboration.debug.teams.update_ignored error_type=%s",
                    type(exc).__name__,
                )
            self._initialized = True
            self._next_refresh = now + self._refresh_interval
            return self._snapshot

    def endpoint(self) -> str:
        override = os.getenv("AGENTCORE_TASK_SERVICE_ENDPOINT", "").strip()
        if override:
            return _http_url(override)
        gateway = self._source.matrix_url
        if gateway.endswith("/agentteams-app"):
            return gateway
        return f"{gateway}/agentteams-app"

    async def token(self, _environment_name: str) -> str:
        async with self._lock:
            snapshot = self._snapshot
            if snapshot is None:
                raise CollaborationConfigError("Worker Matrix token is unavailable.")
            if (
                self._matrix_token is not None
                and self._matrix_user_id == snapshot.self_matrix_user_id
            ):
                return self._matrix_token
            token = await self._exchange_matrix_token()
            self._matrix_token = token
            self._matrix_user_id = snapshot.self_matrix_user_id
            return token

    async def refresh_token(self, _environment_name: str, rejected: str) -> str:
        async with self._lock:
            if self._matrix_token is not None and self._matrix_token != rejected:
                return self._matrix_token
            snapshot = self._snapshot
            if snapshot is None:
                raise CollaborationConfigError("Worker Matrix token is unavailable.")
            token = await self._exchange_matrix_token()
            self._matrix_token = token
            self._matrix_user_id = snapshot.self_matrix_user_id
            return token

    async def _exchange_matrix_token(self) -> str:
        try:
            token = await self._source.exchange_matrix_token()
        except AgentCoreError as exc:
            raise CollaborationConfigError("Worker Matrix token is unavailable.") from exc
        if not isinstance(token, str) or not token.strip():
            raise CollaborationConfigError("Worker Matrix token is unavailable.")
        return token.strip()


def _http_url(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise CollaborationConfigError(
            "Worker Task Service endpoint is unavailable."
        ) from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
    ):
        raise CollaborationConfigError("Worker Task Service endpoint is unavailable.")
    return value.rstrip("/")
