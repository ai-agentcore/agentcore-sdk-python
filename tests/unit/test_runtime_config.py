from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
from jsonschema import ValidationError, validate

from agentcore.errors import ConfigError
from agentcore.runtime.config import load_agent_config, parse_agent_config

_AGENT_YAML_SCHEMA = Path(__file__).parents[2] / "docs/contracts/agent-yaml.schema.json"


def test_parse_agent_config_projects_supported_fields(agent_config_text: str) -> None:
    config = parse_agent_config(agent_config_text.encode())

    assert config.spec.model.gateway_url == "https://gateway.example.com/v1"
    assert config.spec.mcp.gateway_url == "https://gateway.example.com/mcp"
    assert config.spec.gateway_headers == {"Authorization": "Bearer gateway-secret"}
    assert config.workspace_id == "workspace-a"
    assert config.region_id == "cn-hangzhou"
    assert "gateway-secret" not in repr(config)


def test_parse_high_code_agent_config_projects_only_gateway_contract() -> None:
    config = parse_agent_config(
        b"""apiVersion: agentteams.io/v1alpha1
kind: AgentConfig
metadata:
  name: demo
  runtimeName: runtime-demo
  workspaceId: workspace-a
  regionId: cn-hangzhou
spec:
  model:
    gatewayUrl: https://gateway.example.com/v1
  mcp:
    gatewayUrl: https://gateway.example.com/mcp
  credentials:
    header:
      - key: Authorization
        value: Bearer consumer-token
"""
    )

    assert config.spec.model.gateway_url == "https://gateway.example.com/v1"
    assert config.spec.mcp.gateway_url == "https://gateway.example.com/mcp"
    assert config.spec.gateway_headers == {"Authorization": "Bearer consumer-token"}


@pytest.mark.parametrize(
    "headers",
    [
        "      - key: X-AgentCore-Consumer\n        value: consumer-token",
        "      - key: Authorization\n        value: Basic consumer-token",
        "      - key: Authorization\n        value: Bearer",
    ],
)
def test_parse_agent_config_requires_bearer_authorization(headers: str) -> None:
    text = f"""apiVersion: agentteams.io/v1alpha1
kind: AgentConfig
metadata:
  runtimeName: runtime-demo
  workspaceId: workspace-a
  regionId: cn-hangzhou
spec:
  model:
    gatewayUrl: https://gateway.example.com/v1
  mcp:
    gatewayUrl: https://gateway.example.com/mcp
  credentials:
    header:
{headers}
"""

    with pytest.raises(ConfigError, match="Bearer Authorization"):
        parse_agent_config(text.encode())


def test_agent_yaml_schema_matches_gateway_authorization_contract() -> None:
    headers = [
        {"key": "Authorization", "value": "Bearer consumer-token"},
        {"key": "X-AgentCore-Trace", "value": "trace-value"},
    ]
    document = {
        "apiVersion": "agentteams.io/v1alpha1",
        "kind": "AgentConfig",
        "metadata": {
            "runtimeName": "runtime-demo",
            "workspaceId": "workspace-a",
            "regionId": "cn-hangzhou",
        },
        "spec": {
            "model": {"gatewayUrl": "https://gateway.example.com/v1"},
            "mcp": {"gatewayUrl": "https://gateway.example.com/mcp"},
            "credentials": {"header": headers},
        },
    }
    schema = json.loads(_AGENT_YAML_SCHEMA.read_text(encoding="utf-8"))
    validate(document, schema)

    headers[0]["value"] = "Bearer\tconsumer-token"
    with pytest.raises(ValidationError):
        validate(document, schema)

    headers[0]["value"] = "Bearer consumer-token"
    headers.append({"key": "authorization", "value": "Basic duplicate-token"})

    with pytest.raises(ValidationError):
        validate(document, schema)


@pytest.mark.parametrize(
    "gateway_url",
    [
        "https://user:password@gateway.example.com/ac-agentcore",
        "https://gateway.example.com/ac-agentcore?token=secret",
        "https://gateway.example.com/ac-agentcore#fragment",
    ],
)
def test_agent_yaml_schema_rejects_gateway_urls_not_emitted_by_control_plane(
    gateway_url: str,
) -> None:
    document = {
        "apiVersion": "agentteams.io/v1alpha1",
        "kind": "AgentConfig",
        "metadata": {
            "runtimeName": "runtime-demo",
            "workspaceId": "workspace-a",
            "regionId": "cn-hangzhou",
        },
        "spec": {
            "model": {"gatewayUrl": gateway_url},
            "mcp": {"gatewayUrl": "http://forwarder.internal:10000/mcp"},
            "credentials": {
                "header": [
                    {"key": "Authorization", "value": "Bearer consumer-token"}
                ]
            },
        },
    }
    schema = json.loads(_AGENT_YAML_SCHEMA.read_text(encoding="utf-8"))

    with pytest.raises(ValidationError):
        validate(document, schema)


@pytest.mark.parametrize(
    "fragment",
    [
        "apiKey: secret",
        "gatewayUrl: https://user:password@gateway.example.com/mcp",
        "gatewayUrl: https://gateway.example.com/mcp?token=secret",
    ],
)
def test_parse_agent_config_rejects_sensitive_or_unsafe_mcp_fields(
    agent_config_text: str, fragment: str
) -> None:
    text = agent_config_text.replace("gatewayUrl: https://gateway.example.com/mcp", fragment)

    with pytest.raises(ConfigError):
        parse_agent_config(text.encode())


def test_parse_agent_config_rejects_duplicate_keys(agent_config_text: str) -> None:
    text = agent_config_text.replace("kind: AgentConfig", "kind: AgentConfig\nkind: AgentConfig")

    with pytest.raises(ConfigError, match="duplicate key"):
        parse_agent_config(text.encode())


def test_parse_agent_config_rejects_invalid_gateway_header_name(
    agent_config_text: str,
) -> None:
    text = agent_config_text.replace("key: Authorization", "key: Invalid Header")

    with pytest.raises(ConfigError, match="header name"):
        parse_agent_config(text.encode())


def test_load_agent_config_reads_configured_path(
    agent_config_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="agentcore.runtime.config"):
        config = load_agent_config(agent_config_path)

    assert config.runtime_name == "runtime-demo"
    assert "agentcore.config.loaded" in caplog.text
    assert "gateway-secret" not in caplog.text


def test_load_agent_config_fails_when_config_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "agent.yaml"
    path.write_text("kind: broken", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_agent_config(path)


def test_parse_agent_config_accepts_unified_metadata_without_name(
    agent_config_text: str,
) -> None:
    text = agent_config_text.replace("  name: demo\n", "")

    config = parse_agent_config(text.encode())

    assert config.name == "runtime-demo"


def test_parse_agent_config_requires_high_code_mcp_gateway(
    agent_config_text: str,
) -> None:
    text = agent_config_text.replace(
        "  mcp:\n    gatewayUrl: https://gateway.example.com/mcp\n", ""
    )

    with pytest.raises(ConfigError, match="spec.mcp"):
        parse_agent_config(text.encode())


@pytest.mark.parametrize("field", ["workspaceId", "regionId"])
def test_parse_agent_config_requires_workspace_runtime_identity(
    agent_config_text: str,
    field: str,
) -> None:
    text = "\n".join(
        line for line in agent_config_text.splitlines() if not line.strip().startswith(f"{field}:")
    )

    with pytest.raises(ConfigError, match=rf"metadata\.{field}"):
        parse_agent_config(text.encode())
