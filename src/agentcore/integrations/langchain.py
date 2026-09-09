"""LangChain adapters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentcore.auth import AccessKeyCredential
from agentcore.integrations._langchain_events import AgentCoreConverter as AgentCoreConverter
from agentcore.integrations._shared import (
    framework_gateway_headers,
    framework_http_clients,
    framework_model_settings,
    sync_tool_functions,
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
    settings = framework_model_settings(
        resource_name,
        model=model_name,
        config_path=config_path,
        env_path=env_path,
        control_plane_endpoint=control_plane_endpoint,
        access_key_credential=access_key_credential,
    )
    if settings.provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
            from pydantic import SecretStr
        except ImportError as exc:
            raise ImportError(
                "LangChain support requires: "
                "pip install alibabacloud-agentcore-sdk[langchain]"
            ) from exc
        headers = dict(kwargs.pop("default_headers", {}))
        headers.update(framework_gateway_headers(settings))
        if settings.descriptor.max_tokens is not None:
            kwargs.setdefault("max_tokens", settings.descriptor.max_tokens)
        return ChatAnthropic(
            name=settings.model,
            model_name=settings.model,
            base_url=settings.base_url,
            api_key=SecretStr(""),
            default_headers=headers,
            max_retries=0,
            streaming=True,
            stream_usage=True,
            **kwargs,
        )
    try:
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise ImportError(
            "LangChain support requires: "
            "pip install alibabacloud-agentcore-sdk[langchain]"
        ) from exc
    http_client, http_async_client = framework_http_clients(settings)
    return ChatOpenAI(
        name=settings.model,
        model=settings.model,
        base_url=settings.base_url,
        api_key=lambda: "agentcore",
        http_client=http_client,
        http_async_client=http_async_client,
        streaming=True,
        stream_usage=True,
        use_responses_api=False,
        **kwargs,
    )


def tools(values: Sequence[Tool]) -> list[Any]:
    try:
        from langchain_core.tools import StructuredTool
    except ImportError as exc:
        raise ImportError(
            "LangChain support requires: "
            "pip install alibabacloud-agentcore-sdk[langchain]"
        ) from exc
    result = []
    for tool, function, sync_function in zip(
        values, tool_functions(values), sync_tool_functions(values), strict=True
    ):
        result.append(
            StructuredTool(
                name=tool.name,
                description=tool.description,
                args_schema=dict(tool.parameters),
                func=sync_function,
                coroutine=function,
            )
        )
    return result


def skill_tools(values: Sequence[Skill]) -> list[Any]:
    return tools(canonical_skill_tools(values))
