from __future__ import annotations

from pathlib import Path

import pytest

from agentcore.controlplane import MCPDescriptor, ModelDescriptor


@pytest.fixture
def agent_config_text() -> str:
    return """
apiVersion: agentteams.io/v1alpha1
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
        value: Bearer gateway-secret
""".lstrip()


@pytest.fixture
def agent_config_path(tmp_path: Path, agent_config_text: str) -> Path:
    path = tmp_path / "agent.yaml"
    path.write_text(agent_config_text, encoding="utf-8")
    return path


@pytest.fixture
def teams_config_text() -> str:
    return """
apiVersion: agentteams.io/v1alpha1
kind: TeamsConfig
metadata:
  runtimeName: worker-a
  generation: "1"
spec:
  self:
    name: worker-a
    runtimeName: worker-a
    matrixUserId: "@worker-a:matrix.example.com"
    personalRoomId: "!worker-a:matrix.example.com"
  matrix:
    homeserver: https://matrix.example.com
    tokenEnv: AGENTTEAMS_WORKER_MATRIX_TOKEN
  defaultTeamName: team-alpha
  teams:
    - name: team-alpha
      teamRoomId: "!alpha:matrix.example.com"
      membership:
        role: worker
      members:
        - type: agent
          name: leader-a
          runtimeName: leader-a
          role: leader
          matrixUserId: "@leader-a:matrix.example.com"
          personalRoomId: "!leader-a:matrix.example.com"
        - type: human
          name: operator-a
          role: human
          matrixUserId: "@operator-a:matrix.example.com"
          personalRoomId: "!operator-a:matrix.example.com"
""".lstrip()


@pytest.fixture
def teams_config_path(tmp_path: Path, teams_config_text: str) -> Path:
    path = tmp_path / "teams.yaml"
    path.write_text(teams_config_text, encoding="utf-8")
    return path


@pytest.fixture
def model_descriptor() -> ModelDescriptor:
    return ModelDescriptor(
        connection_id="model-1",
        connection_name="model-service-a",
        protocol="OpenAI/v1",
        provider_type="QWEN",
        model_id="qwen-plus-id",
        model_name="qwen-plus",
        context_size=32768,
        max_tokens=8192,
        capabilities={"tool_call": True},
    )


@pytest.fixture
def mcp_descriptor() -> MCPDescriptor:
    return MCPDescriptor(
        mcp_server_id="search-id",
        name="search",
        protocol="STREAMABLE_HTTP",
        type="HOSTED",
        status="READY",
    )
