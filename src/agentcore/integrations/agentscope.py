"""AgentScope adapters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentcore.auth import AccessKeyCredential
from agentcore.integrations._agentscope_events import AgentCoreConverter as AgentCoreConverter
from agentcore.integrations._shared import (
    framework_async_http_client,
    framework_gateway_headers,
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
        from agentscope.credential import AnthropicCredential, OpenAICredential
        from agentscope.model import AnthropicChatModel, OpenAIChatModel
        from pydantic import SecretStr
    except ImportError as exc:
        raise ImportError(
            "AgentScope support requires Python 3.11+ and: "
            "pip install alibabacloud-agentcore-sdk[agentscope]"
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
        anthropic_credential = AnthropicCredential(
            api_key=SecretStr(""),
            base_url=settings.base_url,
        )
        client_kwargs = dict(kwargs.pop("client_kwargs", {}))
        headers = dict(client_kwargs.pop("default_headers", {}))
        headers.update(framework_gateway_headers(settings))
        client_kwargs.update(
            auth_token=settings.api_key,
            default_headers=headers,
            max_retries=0,
        )
        if settings.descriptor.max_tokens is not None:
            kwargs.setdefault(
                "parameters",
                AnthropicChatModel.Parameters(max_tokens=settings.descriptor.max_tokens),
            )
        if settings.descriptor.context_size is not None:
            kwargs.setdefault("context_size", settings.descriptor.context_size)
        kwargs.setdefault("max_retries", 0)
        return AnthropicChatModel(
            credential=anthropic_credential,
            model=settings.model,
            stream=True,
            client_kwargs=client_kwargs,
            **kwargs,
        )
    http_client = framework_async_http_client(settings)
    openai_credential = OpenAICredential(
        api_key=SecretStr("agentcore"),
        base_url=settings.base_url,
    )
    client_kwargs = dict(kwargs.pop("client_kwargs", {}))
    client_kwargs["http_client"] = http_client
    return OpenAIChatModel(
        credential=openai_credential,
        model=settings.model,
        stream=True,
        client_kwargs=client_kwargs,
        **kwargs,
    )


def tools(values: Sequence[Tool]) -> list[Any]:
    try:
        from agentscope.tool import FunctionTool
    except ImportError as exc:
        raise ImportError(
            "AgentScope support requires Python 3.11+ and: "
            "pip install alibabacloud-agentcore-sdk[agentscope]"
        ) from exc
    return [
        FunctionTool(
            function,
            name=tool.name,
            description=tool.description,
            input_schema=dict(tool.parameters),
        )
        for tool, function in zip(values, tool_functions(values), strict=True)
    ]


def skill_tools(values: Sequence[Skill]) -> list[Any]:
    return tools(canonical_skill_tools(values))
