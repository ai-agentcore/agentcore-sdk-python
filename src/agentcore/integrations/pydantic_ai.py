"""Pydantic AI adapters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from agentcore.auth import AccessKeyCredential
from agentcore.integrations._pydantic_events import AgentCoreConverter as AgentCoreConverter
from agentcore.integrations._shared import (
    framework_anthropic_client,
    framework_async_http_client,
    framework_model_settings,
    tool_functions,
)
from agentcore.integrations.common import Tool
from agentcore.skill import Skill
from agentcore.skill import skill_tools as canonical_skill_tools


def model(
    resource_name: str,
    *,
    model_name: str | None = None,
    config_path: str | Path | None = None,
    env_path: str | Path | None = None,
    control_plane_endpoint: str | None = None,
    access_key_credential: AccessKeyCredential | None = None,
    **kwargs: Any,
) -> Any:
    try:
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.anthropic import AnthropicProvider
        from pydantic_ai.providers.openai import OpenAIProvider
    except ImportError as exc:
        raise ImportError(
            "Pydantic AI support requires: "
            "pip install alibabacloud-agentcore-sdk[pydantic-ai]"
        ) from exc
    settings = framework_model_settings(
        resource_name,
        model=model_name,
        config_path=config_path,
        env_path=env_path,
        control_plane_endpoint=control_plane_endpoint,
        access_key_credential=access_key_credential,
    )
    if settings.provider == "anthropic":
        anthropic_provider = AnthropicProvider(
            anthropic_client=framework_anthropic_client(settings),
        )
        return AnthropicModel(settings.model, provider=anthropic_provider, **kwargs)
    openai_provider = OpenAIProvider(
        base_url=settings.base_url,
        api_key="agentcore",
        http_client=framework_async_http_client(settings),
    )
    return OpenAIChatModel(settings.model, provider=openai_provider, **kwargs)


def tools(values: Sequence[Tool]) -> list[Any]:
    try:
        from pydantic_ai import Tool as PydanticTool
    except ImportError as exc:
        raise ImportError(
            "Pydantic AI support requires: "
            "pip install alibabacloud-agentcore-sdk[pydantic-ai]"
        ) from exc

    result = []
    for tool, function in zip(values, tool_functions(values), strict=True):
        schema = dict(tool.parameters)

        def prepare(_context: Any, definition: Any, schema: dict[str, Any] = schema) -> Any:
            return replace(definition, parameters_json_schema=schema)

        result.append(
            PydanticTool(
                function,
                name=tool.name,
                description=tool.description,
                prepare=prepare,
            )
        )
    return result


def skill_tools(values: Sequence[Skill]) -> list[Any]:
    return tools(canonical_skill_tools(values))
