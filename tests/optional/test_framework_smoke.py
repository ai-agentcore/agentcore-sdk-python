from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from agentcore.controlplane import AgentCoreControlPlane, ModelDescriptor
from agentcore.integrations.common import Tool


@pytest.fixture
def canonical_tool() -> Tool:
    async def invoke(arguments: dict[str, Any]) -> Any:
        return arguments

    return Tool(
        "search",
        "Search",
        {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        invoke,
    )


@pytest.mark.parametrize(
    ("dependency", "module_name"),
    [
        ("langchain_openai", "langchain"),
        ("agentscope", "agentscope"),
        ("google.adk", "google_adk"),
        ("pydantic_ai", "pydantic_ai"),
        ("crewai", "crewai"),
    ],
)
def test_framework_current_version_constructs_model_and_tool(
    dependency: str,
    module_name: str,
    agent_config_path: Path,
    canonical_tool: Tool,
    model_descriptor: ModelDescriptor,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip(dependency)
    module = __import__(f"agentcore.integrations.{module_name}", fromlist=["model"])

    token = tmp_path / "token"
    token.write_text("sa-token", encoding="utf-8")
    env = tmp_path / "env"
    env.write_text(
        "export AGENTTEAMS_CONTROLLER_URL='https://controller.example.com'\n"
        f"export AGENTTEAMS_AUTH_TOKEN_FILE='{token}'\n",
        encoding="utf-8",
    )

    async def resolve_model(self, resource_name, model_name=None):  # type: ignore[no-untyped-def]
        return model_descriptor

    monkeypatch.setattr(AgentCoreControlPlane, "resolve_model", resolve_model)

    model = module.model(
        "model-service-a",
        model_name="qwen-plus",
        config_path=agent_config_path,
        env_path=env,
    )
    tools = module.tools([canonical_tool])

    assert model is not None
    assert len(tools) == 1
