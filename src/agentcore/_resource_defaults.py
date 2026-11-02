"""Process-environment defaults for resource selection, not runtime authentication."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from typing import Any

from agentcore.errors import ConfigError

_KEYS = (
    "AGENTCORE_MODEL_CONNECTION_NAME",
    "AGENTCORE_MODEL_NAME",
    "AGENTCORE_AGENT_SPEC_NAME",
    "AGENTCORE_AGENT_SPEC_VERSION",
    "AGENTCORE_MCP_NAMES",
    "AGENTCORE_SKILLS",
)


class ResourceDefaults:
    def __init__(self, environment: Mapping[str, str] | None = None) -> None:
        source = os.environ if environment is None else environment
        self._values = {key: source.get(key) for key in _KEYS}

    def model(self, name: str | None, model: str | None) -> tuple[str, str | None]:
        if name is not None:
            return name, model
        return self._required("AGENTCORE_MODEL_CONNECTION_NAME"), (
            model if model is not None else self._optional("AGENTCORE_MODEL_NAME")
        )

    def agent_spec(self, name: str | None, version: str | None) -> tuple[str, str | None]:
        if name is not None:
            return name, version
        return self._required("AGENTCORE_AGENT_SPEC_NAME"), (
            version if version is not None else self._optional("AGENTCORE_AGENT_SPEC_VERSION")
        )

    def mcp_names(self) -> list[str]:
        key = "AGENTCORE_MCP_NAMES"
        return [self._string(value, key) for value in self._list(key)]

    def skills(self) -> list[tuple[str, str | None]]:
        key = "AGENTCORE_SKILLS"
        result = []
        for item in self._list(key):
            if not isinstance(item, dict):
                raise ConfigError(f"{key} must contain objects with a name and optional version")
            name = self._string(item.get("name"), key)
            version = self._string(item["version"], key) if "version" in item else None
            result.append((name, version))
        return result

    def _optional(self, key: str) -> str | None:
        return (self._values[key] or "").strip() or None

    def _required(self, key: str) -> str:
        value = self._optional(key)
        if value is None:
            raise ConfigError(f"Set {key} or pass an explicit resource name")
        return value

    def _list(self, key: str) -> list[Any]:
        raw = self._optional(key)
        if raw is None:
            return []
        try:
            value = json.loads(raw)
        except ValueError as error:
            raise ConfigError(f"{key} must be a JSON array") from error
        if not isinstance(value, list):
            raise ConfigError(f"{key} must be a JSON array")
        return value

    @staticmethod
    def _string(value: Any, key: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} resource names and versions must be non-empty strings")
        return value.strip()
