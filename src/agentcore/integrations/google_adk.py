"""Google ADK adapters."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from agentcore.auth import AccessKeyCredential
from agentcore.integrations._adk_events import AgentCoreConverter as AgentCoreConverter
from agentcore.integrations._shared import (
    framework_completion_parameters,
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
        from google.adk.models.lite_llm import LiteLlm, LiteLLMClient
    except ImportError as exc:
        raise ImportError(
            "Google ADK support requires: "
            "pip install alibabacloud-agentcore-sdk[google-adk]"
        ) from exc
    settings = framework_model_settings(
        resource_name,
        model=model_name,
        config_path=config_path,
        env_path=env_path,
        control_plane_endpoint=control_plane_endpoint,
        access_key_credential=access_key_credential,
    )
    delegate = kwargs.pop("llm_client", None) or LiteLLMClient()

    class AgentCoreLiteLLMClient(LiteLLMClient):  # type: ignore[misc,unused-ignore]
        def completion(
            self,
            model: Any,
            messages: Any,
            tools: Any,
            stream: bool = False,
            **call_kwargs: Any,
        ) -> Any:
            parameters = _google_adk_parameters(settings, call_kwargs)
            return delegate.completion(
                model=parameters.pop("model"),
                messages=messages,
                tools=tools,
                stream=stream,
                **parameters,
            )

        async def acompletion(
            self,
            model: Any,
            messages: Any,
            tools: Any,
            **call_kwargs: Any,
        ) -> Any:
            parameters = _google_adk_parameters(settings, call_kwargs)
            return await delegate.acompletion(
                model=parameters.pop("model"),
                messages=messages,
                tools=tools,
                **parameters,
            )

    return LiteLlm(
        model=f"{settings.provider or 'openai'}/{settings.model}",
        llm_client=AgentCoreLiteLLMClient(),
        **kwargs,
    )


def _google_adk_parameters(
    settings: Any,
    call_kwargs: dict[str, Any],
) -> dict[str, Any]:
    configured = framework_completion_parameters(settings)
    parameters = dict(call_kwargs)
    headers = dict(parameters.get("extra_headers", {}))
    headers.update(configured["extra_headers"])
    parameters.update(configured)
    parameters["extra_headers"] = headers
    return parameters


def tools(values: Sequence[Tool]) -> list[Any]:
    try:
        from google.adk.tools.function_tool import FunctionTool
        from google.genai import types
    except ImportError as exc:
        raise ImportError(
            "Google ADK support requires: "
            "pip install alibabacloud-agentcore-sdk[google-adk]"
        ) from exc

    class SchemaFunctionTool(FunctionTool):  # type: ignore[misc,unused-ignore]
        def __init__(self, tool: Tool, function: Any) -> None:
            super().__init__(function)
            self._agentcore_tool = tool

        def _get_declaration(self) -> Any:
            return types.FunctionDeclaration(
                name=self._agentcore_tool.name,
                description=self._agentcore_tool.description,
                parameters_json_schema=dict(self._agentcore_tool.parameters),
            )

    return [
        SchemaFunctionTool(tool, function)
        for tool, function in zip(values, tool_functions(values), strict=True)
    ]


def skill_tools(values: Sequence[Skill]) -> list[Any]:
    return tools(canonical_skill_tools(values))
