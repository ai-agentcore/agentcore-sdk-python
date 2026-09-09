"""Read the trusted AgentCore runtime environment projection."""

from __future__ import annotations

import logging
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from agentcore.errors import ConfigError

DEFAULT_ENV_PATH = Path("/var/run/agentcore/agent/env")
_EXPORT_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeEnvironment:
    controller_url: str
    agent_sa_token_file: Path
    control_plane_endpoint: str | None


class RuntimeEnvironmentProvider:
    """Parse the Controller-owned env file without evaluating shell code."""

    def __init__(self, path: str | Path = DEFAULT_ENV_PATH) -> None:
        self.path = Path(path)

    def snapshot(self) -> RuntimeEnvironment:
        values = self._read()
        controller_url = _url(
            _required_any(
                values,
                "AGENTCORE_CONTROLLER_URL",
                "AGENTTEAMS_CONTROLLER_URL",
            )
        )
        token_file = Path(
            _required_any(
                values,
                "AGENTCORE_AUTH_TOKEN_FILE",
                "AGENTTEAMS_AUTH_TOKEN_FILE",
            )
        )
        control_plane_endpoint = (
            values.get("AGENTCORE_CONTROL_ENDPOINT")
            or os.getenv("AGENTCORE_CONTROL_ENDPOINT")
            or ""
        ).strip() or None
        environment = RuntimeEnvironment(
            controller_url=controller_url,
            agent_sa_token_file=token_file,
            control_plane_endpoint=control_plane_endpoint,
        )
        logger.info(
            "agentcore.environment.loaded path=%s controller_host=%s token_file=%s "
            "control_endpoint_override=%s",
            self.path,
            urlsplit(controller_url).netloc,
            token_file,
            control_plane_endpoint is not None,
        )
        return environment

    def value(self, name: str) -> str:
        """Read one named value, reloading the runtime projection on every call."""
        if _EXPORT_NAME.fullmatch(name) is None:
            raise ConfigError("runtime environment name is invalid")
        values = self._read()
        value = (values.get(name) or os.getenv(name) or "").strip()
        if not value:
            raise ConfigError(f"runtime env requires {name}")
        return value

    def task_service_endpoint(self) -> str:
        """Resolve the Task Service base URL from the latest runtime projection."""
        values = self._read()
        endpoint = (
            values.get("AGENTCORE_TASK_SERVICE_ENDPOINT")
            or os.getenv("AGENTCORE_TASK_SERVICE_ENDPOINT")
            or ""
        ).strip()
        if endpoint:
            return _http_url(endpoint, "Task Service Endpoint")

        matrix_url = (
            values.get("AGENTTEAMS_MATRIX_URL")
            or os.getenv("AGENTTEAMS_MATRIX_URL")
            or ""
        ).strip()
        if not matrix_url:
            raise ConfigError(
                "runtime env requires AGENTCORE_TASK_SERVICE_ENDPOINT or AGENTTEAMS_MATRIX_URL"
            )
        gateway = _http_url(matrix_url, "Task Service Endpoint")
        if gateway.endswith("/agentteams-app"):
            return gateway
        return f"{gateway}/agentteams-app"

    def _read(self) -> dict[str, str]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            logger.warning(
                "agentcore.environment.load.failed path=%s error_type=%s",
                self.path,
                type(exc).__name__,
            )
            raise ConfigError("cannot read AgentCore runtime env") from exc
        values: dict[str, str] = {}
        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                parts = shlex.split(line, comments=False, posix=True)
            except ValueError as exc:
                raise ConfigError(f"invalid runtime env at line {line_number}") from exc
            if len(parts) != 2 or parts[0] != "export" or "=" not in parts[1]:
                raise ConfigError(f"invalid runtime env at line {line_number}")
            name, value = parts[1].split("=", 1)
            if _EXPORT_NAME.fullmatch(name) is None or name in values:
                raise ConfigError(f"invalid runtime env at line {line_number}")
            values[name] = value
        return values


def _required_any(values: dict[str, str], *names: str) -> str:
    for name in names:
        value = values.get(name, "").strip()
        if value:
            return value
    raise ConfigError(f"runtime env requires {' or '.join(names)}")


def _url(value: str) -> str:
    return _http_url(value, "runtime Controller URL")


def _http_url(value: str, label: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise ConfigError(f"{label} must be a valid HTTP(S) URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"{label} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigError(f"{label} must not contain user info, query, or fragment")
    return value.rstrip("/")
