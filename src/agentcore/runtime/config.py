"""Strict projection of the AgentCore runtime configuration."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit

import yaml
from yaml.constructor import ConstructorError

from agentcore.errors import ConfigError

DEFAULT_CONFIG_PATH = Path("/var/run/agentcore/agent/agent.yaml")
MAX_CONFIG_BYTES = 1024 * 1024
logger = logging.getLogger(__name__)
_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
_SENSITIVE_FIELDS = {
    "apikey",
    "api_key",
    "authorization",
    "accesskeyid",
    "accesskeysecret",
    "cookie",
    "header",
    "headers",
    "password",
    "secret",
    "securitytoken",
    "token",
}


class _UniqueKeyLoader(yaml.SafeLoader):
    pass


def _construct_mapping(
    loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "configuration keys must be strings",
                key_node.start_mark,
            )
        if key in mapping:
            raise ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key: {key}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


@dataclass(frozen=True)
class ModelConfig:
    gateway_url: str


@dataclass(frozen=True)
class MCPConfig:
    gateway_url: str


@dataclass(frozen=True)
class AgentSpec:
    model: ModelConfig
    mcp: MCPConfig
    gateway_headers: Mapping[str, str] = field(repr=False)


@dataclass(frozen=True)
class AgentConfig:
    api_version: str
    kind: str
    name: str
    runtime_name: str
    workspace_id: str
    region_id: str
    spec: AgentSpec


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must be an object")
    return value


def _list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ConfigError(f"{path} must be a list")
    return value


def _string(value: Any, path: str, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{path} must be a non-empty string")
    return value.strip()


def _url(value: Any, path: str) -> str:
    result = _string(value, path)
    assert result is not None
    try:
        parsed = urlsplit(result)
    except ValueError as exc:
        raise ConfigError(f"{path} must be a valid HTTP(S) URL") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ConfigError(f"{path} must be an absolute HTTP(S) URL")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ConfigError(f"{path} must not contain user info, query, or fragment")
    return result if parsed.path == "/" else result.rstrip("/")


def _reject_sensitive_fields(value: Mapping[str, Any], path: str) -> None:
    for key, item in value.items():
        if key.lower() in _SENSITIVE_FIELDS:
            raise ConfigError(f"{path}.{key} must not contain inline credentials")
        if isinstance(item, dict):
            _reject_sensitive_fields(item, f"{path}.{key}")
        elif isinstance(item, list):
            for index, nested in enumerate(item):
                if isinstance(nested, dict):
                    _reject_sensitive_fields(nested, f"{path}.{key}[{index}]")


def _load_yaml(data: bytes) -> Mapping[str, Any]:
    if len(data) > MAX_CONFIG_BYTES:
        raise ConfigError("agent.yaml exceeds the 1 MiB size limit")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError("agent.yaml must be UTF-8") from exc
    try:
        for token in yaml.scan(text):
            if isinstance(token, (yaml.tokens.AnchorToken, yaml.tokens.AliasToken)):
                raise ConfigError("agent.yaml aliases and anchors are not allowed")
        loaded = yaml.load(text, Loader=_UniqueKeyLoader)
    except ConfigError:
        raise
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark is not None else ""
        problem = getattr(exc, "problem", "")
        detail = f": {problem}" if str(problem).startswith("duplicate key:") else ""
        raise ConfigError(f"invalid agent.yaml{location}{detail}") from exc
    return _mapping(loaded, "agent.yaml")


def parse_agent_config(data: bytes) -> AgentConfig:
    return parse_agent_config_mapping(_load_yaml(data))


def parse_agent_config_mapping(data: Mapping[str, Any]) -> AgentConfig:
    """Parse an already-decoded AgentConfig without materializing a YAML file."""

    root = _mapping(data, "agent.yaml")
    api_version = _string(root.get("apiVersion"), "apiVersion")
    kind = _string(root.get("kind"), "kind")
    if api_version != "agentteams.io/v1alpha1" or kind != "AgentConfig":
        raise ConfigError("unsupported agent.yaml apiVersion or kind")

    metadata = _mapping(root.get("metadata"), "metadata")
    runtime_name = _string(metadata.get("runtimeName"), "metadata.runtimeName")
    name = _string(metadata.get("name"), "metadata.name", optional=True) or runtime_name
    workspace_id = _string(metadata.get("workspaceId"), "metadata.workspaceId")
    region_id = _string(metadata.get("regionId"), "metadata.regionId")

    spec_value = _mapping(root.get("spec"), "spec")
    model_value = _mapping(spec_value.get("model"), "spec.model")
    _reject_sensitive_fields(model_value, "spec.model")
    model = ModelConfig(
        gateway_url=_url(model_value.get("gatewayUrl"), "spec.model.gatewayUrl"),
    )
    mcp_value = _mapping(spec_value.get("mcp"), "spec.mcp")
    _reject_sensitive_fields(mcp_value, "spec.mcp")
    mcp = MCPConfig(gateway_url=_url(mcp_value.get("gatewayUrl"), "spec.mcp.gatewayUrl"))

    headers: dict[str, str] = {}
    credentials_value = _mapping(spec_value.get("credentials"), "spec.credentials")
    raw_headers = _list(credentials_value.get("header"), "spec.credentials.header")
    if not raw_headers:
        raise ConfigError("spec.credentials.header must not be empty")
    for index, item in enumerate(raw_headers):
        path = f"spec.credentials.header[{index}]"
        header = _mapping(item, path)
        key = _required_string(header, "key", path)
        value = _required_string(header, "value", path)
        if _HEADER_NAME.fullmatch(key) is None:
            raise ConfigError(f"{path} contains an invalid header name")
        if "\r" in key or "\n" in key or "\r" in value or "\n" in value:
            raise ConfigError(f"{path} contains invalid header characters")
        if key.lower() in {existing.lower() for existing in headers}:
            raise ConfigError(f"duplicate gateway header: {key}")
        headers[key] = value
    authorization = next(
        (value for key, value in headers.items() if key.lower() == "authorization"),
        None,
    )
    if authorization is None or not authorization.lower().startswith("bearer "):
        raise ConfigError("spec.credentials.header requires a Bearer Authorization header")
    if not authorization[7:].strip():
        raise ConfigError("spec.credentials.header requires a Bearer Authorization header")

    assert api_version is not None
    assert kind is not None
    assert name is not None
    assert runtime_name is not None
    assert workspace_id is not None
    assert region_id is not None
    return AgentConfig(
        api_version=api_version,
        kind=kind,
        name=name,
        runtime_name=runtime_name,
        workspace_id=workspace_id,
        region_id=region_id,
        spec=AgentSpec(
            model=model,
            mcp=mcp,
            gateway_headers=MappingProxyType(headers),
        ),
    )


def _required_string(value: Mapping[str, Any], key: str, path: str) -> str:
    result = _string(value.get(key), f"{path}.{key}")
    assert result is not None
    return result


def load_agent_config(path: str | Path | None = None) -> AgentConfig:
    """Load an immutable runtime startup configuration."""

    configured = path or os.getenv("AGENTCORE_CONFIG_PATH") or DEFAULT_CONFIG_PATH
    try:
        data = Path(configured).read_bytes()
    except OSError as exc:
        logger.warning(
            "agentcore.config.load.failed path=%s error_type=%s",
            configured,
            type(exc).__name__,
        )
        raise ConfigError(f"cannot read agent.yaml: {exc}") from exc
    try:
        config = parse_agent_config(data)
    except ConfigError:
        logger.warning("agentcore.config.load.failed path=%s reason=invalid", configured)
        raise
    logger.info(
        "agentcore.config.loaded path=%s runtime_name=%s workspace_id=%s region_id=%s",
        configured,
        config.runtime_name,
        config.workspace_id,
        config.region_id,
    )
    return config
