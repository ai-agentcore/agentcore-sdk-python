"""CrewAI adapters."""

from __future__ import annotations

import asyncio
import keyword
from collections.abc import Callable, Coroutine, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, TypeVar

from agentcore.auth import AccessKeyCredential
from agentcore.integrations._crewai_events import AgentCoreConverter as AgentCoreConverter
from agentcore.integrations._shared import (
    framework_completion_parameters,
    framework_model_settings,
)
from agentcore.integrations.common import Tool
from agentcore.skill import Skill
from agentcore.skill import skill_tools as canonical_skill_tools

_ResultT = TypeVar("_ResultT")


async def _run_on_loop(
    call: Callable[[], Coroutine[Any, Any, _ResultT]],
    owner_loop: asyncio.AbstractEventLoop | None,
) -> _ResultT:
    """CrewAI tools and task callbacks can arrive on a worker thread's loop."""
    if owner_loop is None or owner_loop is asyncio.get_running_loop():
        return await call()
    if not owner_loop.is_running():
        raise RuntimeError("Keep the original tool event loop running during Crew execution")
    return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(call(), owner_loop))


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
        from crewai import LLM
    except ImportError as exc:
        raise ImportError(
            "CrewAI support requires: pip install alibabacloud-agentcore-sdk[crewai]"
        ) from exc
    settings = framework_model_settings(
        resource_name,
        model=model_name,
        config_path=config_path,
        env_path=env_path,
        control_plane_endpoint=control_plane_endpoint,
        access_key_credential=access_key_credential,
    )
    additional_params = dict(kwargs.pop("additional_params", {}))

    class AgentCoreLLM(LLM):  # type: ignore[misc,unused-ignore]
        def _prepare_completion_params(
            self,
            messages: Any,
            tools: Any = None,
            skip_file_processing: bool = False,
        ) -> dict[str, Any]:
            parameters = super()._prepare_completion_params(
                messages,
                tools,
                skip_file_processing,
            )
            configured = framework_completion_parameters(settings)
            headers = dict(parameters.get("extra_headers", {}))
            headers.update(configured["extra_headers"])
            parameters.update(configured)
            parameters["extra_headers"] = headers
            return dict(parameters)

    return AgentCoreLLM(
        model=f"{settings.provider or 'openai'}/{settings.model}",
        api_key="agentcore",
        base_url=settings.base_url,
        additional_params=additional_params,
        is_litellm=True,
        **kwargs,
    )


def tools(values: Sequence[Tool]) -> list[Any]:
    """Convert tools on their owning loop; keep that loop running during Crew execution."""
    try:
        from crewai.tools.base_tool import Tool as CrewTool
        from pydantic import ConfigDict, Field, create_model
    except ImportError as exc:
        raise ImportError(
            "CrewAI support requires: pip install alibabacloud-agentcore-sdk[crewai]"
        ) from exc

    try:
        owner_loop = asyncio.get_running_loop()
    except RuntimeError:
        owner_loop = None

    result = []
    for index, tool in enumerate(values):
        properties = tool.parameters.get("properties", {})
        required = set(tool.parameters.get("required", []))
        aliases: dict[str, str] = {}
        fields: dict[str, Any] = {}
        used: set[str] = set()
        if isinstance(properties, dict):
            for property_index, name in enumerate(properties):
                if not isinstance(name, str):
                    continue
                safe_name = name
                if (
                    not safe_name.isidentifier()
                    or keyword.iskeyword(safe_name)
                    or safe_name in used
                ):
                    safe_name = f"field_{property_index}"
                while safe_name in used:
                    safe_name = f"_{safe_name}"
                used.add(safe_name)
                aliases[safe_name] = name
                default = ... if name in required else None
                fields[safe_name] = (Any, Field(default, alias=name))

        wire_schema = deepcopy(dict(tool.parameters))

        def schema_replacer(schema: dict[str, Any]) -> Any:
            def replace_schema(generated: dict[str, Any], _model: Any) -> None:
                generated.clear()
                generated.update(deepcopy(schema))

            return replace_schema

        args_schema = create_model(
            f"AgentCoreTool{index}Args",
            __config__=ConfigDict(
                populate_by_name=True,
                extra="forbid",
                json_schema_extra=schema_replacer(wire_schema),
            ),
            **fields,
        )

        async def invoke(
            _tool: Tool = tool,
            _aliases: dict[str, str] = aliases,
            _owner_loop: asyncio.AbstractEventLoop | None = (
                None if tool.supports_sync else owner_loop
            ),
            **kwargs: Any,
        ) -> Any:
            arguments = {_aliases.get(name, name): value for name, value in kwargs.items()}
            return await _run_on_loop(lambda: _tool.ainvoke(arguments), _owner_loop)

        result.append(
            CrewTool(
                name=tool.name,
                description=tool.description,
                func=invoke,
                args_schema=args_schema,
            )
        )
    return result


def skill_tools(values: Sequence[Skill]) -> list[Any]:
    return tools(canonical_skill_tools(values))
