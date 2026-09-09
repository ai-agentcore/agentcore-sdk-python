"""Shared projections used by optional framework packages."""

from __future__ import annotations

import atexit
import inspect
import keyword
import logging
import os
from collections.abc import Callable, Coroutine, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from threading import RLock
from typing import Any

import httpx
from anyio.from_thread import BlockingPortal, start_blocking_portal

from agentcore._logging import safe_url
from agentcore.auth.access_key import AccessKeyCredential
from agentcore.auth.resource_sts import ResourceSTSProvider
from agentcore.controlplane import AgentCoreControlPlane, ModelDescriptor
from agentcore.controlplane.client import _normalize_endpoint as _normalize_control_plane_endpoint
from agentcore.errors import ConfigError
from agentcore.integrations.common import Tool
from agentcore.model.client import (
    agentcore_model_base_url,
    anthropic_gateway_client,
    managed_framework_provider,
    require_managed_model_adapter,
)
from agentcore.runtime.config import AgentConfig
from agentcore.runtime.startup import DEBUG_TOKEN_ENV, RuntimeSource, create_runtime_source

logger = logging.getLogger(__name__)
_framework_debug_runtime: _FrameworkDebugRuntime | None = None
_framework_debug_runtime_lock = RLock()


@dataclass(frozen=True, repr=False)
class FrameworkModelSettings:
    model: str
    provider: str
    base_url: str
    api_key: str
    headers: Mapping[str, str]
    descriptor: ModelDescriptor

    def __repr__(self) -> str:
        return (
            "FrameworkModelSettings("
            f"model={self.model!r}, provider={self.provider!r}, "
            f"base_url={self.base_url!r}, api_key=<redacted>, headers=<redacted>)"
        )


def framework_model_settings(
    resource_name: str,
    *,
    model: str | None = None,
    config_path: str | Path | None = None,
    env_path: str | Path | None = None,
    control_plane_endpoint: str | None = None,
    access_key_credential: AccessKeyCredential | None = None,
) -> FrameworkModelSettings:
    logger.info(
        "agentcore.integration.model.resolve.started resource_name=%s model=%s",
        resource_name,
        model or "auto",
    )
    try:
        if os.getenv(DEBUG_TOKEN_ENV, "").strip():
            config, descriptor = _get_framework_debug_runtime(
                config_path,
                env_path=env_path,
                control_plane_endpoint=control_plane_endpoint,
            ).resolve(resource_name, model, access_key_credential)
        else:
            source = create_runtime_source(
                config_path,
                env_path=env_path,
                control_plane_endpoint=control_plane_endpoint,
            )
            with start_blocking_portal(
                backend="asyncio", name="agentcore-framework-resolver"
            ) as portal:
                config, descriptor = portal.call(
                    partial(
                        _resolve_framework_model,
                        source,
                        resource_name,
                        model,
                        access_key_credential,
                        close_source=True,
                    )
                )
    except Exception as exc:
        logger.warning(
            "agentcore.integration.model.resolve.failed resource_name=%s model=%s "
            "error_type=%s",
            resource_name,
            model or "auto",
            type(exc).__name__,
        )
        raise
    api_key, headers = _project_gateway_credentials(config.spec.gateway_headers)
    settings = FrameworkModelSettings(
        model=descriptor.model_name,
        provider=managed_framework_provider(descriptor.protocol),
        base_url=agentcore_model_base_url(
            config.spec.model.gateway_url,
            descriptor.connection_id,
            descriptor.protocol,
        ),
        api_key=api_key,
        headers=headers,
        descriptor=descriptor,
    )
    logger.info(
        "agentcore.integration.model.resolve.succeeded resource_name=%s model=%s base_url=%s",
        resource_name,
        descriptor.model_name,
        safe_url(settings.base_url),
    )
    return settings


async def _resolve_framework_model(
    source: RuntimeSource,
    resource_name: str,
    model: str | None,
    access_key_credential: AccessKeyCredential | None,
    *,
    close_source: bool,
) -> tuple[AgentConfig, ModelDescriptor]:
    sts: ResourceSTSProvider | None = None
    try:
        runtime = await source.resolve()
        has_automatic_auth = (
            runtime.controller_endpoint is not None and runtime.agent_sa_tokens is not None
        )
        if access_key_credential is None and not has_automatic_auth:
            raise ConfigError("AgentCore control-plane runtime is not configured")
        _normalize_control_plane_endpoint(runtime.control_plane_endpoint)
        if access_key_credential is None:
            assert runtime.controller_endpoint is not None
            assert runtime.agent_sa_tokens is not None
            sts = ResourceSTSProvider(runtime.controller_endpoint, runtime.agent_sa_tokens)
        control_plane = AgentCoreControlPlane(
            workspace_id=runtime.config.workspace_id,
            region_id=runtime.config.region_id,
            resource_sts=sts,
            access_key_credential=access_key_credential,
            endpoint=runtime.control_plane_endpoint,
        )
        descriptor = await control_plane.resolve_model(resource_name, model)
        require_managed_model_adapter(descriptor.protocol)
        return runtime.config, descriptor
    finally:
        if sts is not None:
            await sts.aclose()
        if close_source:
            await source.aclose()


class _FrameworkDebugRuntime:
    def __init__(
        self,
        config_path: str | Path | None,
        *,
        env_path: str | Path | None,
        control_plane_endpoint: str | None,
    ) -> None:
        self._source = create_runtime_source(
            config_path,
            env_path=env_path,
            control_plane_endpoint=control_plane_endpoint,
        )
        self._portal_context = start_blocking_portal(
            backend="asyncio", name="agentcore-framework-runtime"
        )
        self._portal: BlockingPortal = self._portal_context.__enter__()

    def resolve(
        self,
        resource_name: str,
        model: str | None,
        access_key_credential: AccessKeyCredential | None,
    ) -> tuple[AgentConfig, ModelDescriptor]:
        return self._portal.call(
            partial(
                _resolve_framework_model,
                self._source,
                resource_name,
                model,
                access_key_credential,
                close_source=False,
            )
        )

    def close(self) -> None:
        try:
            self._portal.call(self._source.aclose)
        finally:
            self._portal_context.__exit__(None, None, None)


def _get_framework_debug_runtime(
    config_path: str | Path | None,
    *,
    env_path: str | Path | None,
    control_plane_endpoint: str | None,
) -> _FrameworkDebugRuntime:
    global _framework_debug_runtime
    with _framework_debug_runtime_lock:
        if _framework_debug_runtime is None:
            _framework_debug_runtime = _FrameworkDebugRuntime(
                config_path,
                env_path=env_path,
                control_plane_endpoint=control_plane_endpoint,
            )
        return _framework_debug_runtime


def _close_framework_debug_runtime() -> None:
    global _framework_debug_runtime
    with _framework_debug_runtime_lock:
        runtime = _framework_debug_runtime
        _framework_debug_runtime = None
    if runtime is not None:
        runtime.close()


atexit.register(_close_framework_debug_runtime)


def framework_http_clients(
    settings: FrameworkModelSettings,
) -> tuple[httpx.Client, httpx.AsyncClient]:
    return (
        httpx.Client(
            transport=_GatewayCredentialTransport(settings),
            timeout=600.0,
            follow_redirects=False,
        ),
        httpx.AsyncClient(
            transport=_AsyncGatewayCredentialTransport(settings),
            timeout=600.0,
            follow_redirects=False,
        ),
    )


def framework_async_http_client(settings: FrameworkModelSettings) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=_AsyncGatewayCredentialTransport(settings),
        timeout=600.0,
        follow_redirects=False,
    )


def framework_completion_parameters(settings: FrameworkModelSettings) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "model": f"{settings.provider or 'openai'}/{settings.model}",
        "api_base": settings.base_url,
        "base_url": settings.base_url,
        "api_key": settings.api_key,
        "extra_headers": framework_gateway_headers(settings),
    }
    if settings.provider == "anthropic":
        parameters["use_bearer_for_custom_base"] = True
    return parameters


def framework_gateway_headers(settings: FrameworkModelSettings) -> dict[str, str]:
    headers = dict(settings.headers)
    headers["Authorization"] = f"Bearer {settings.api_key}"
    return headers


def framework_anthropic_client(settings: FrameworkModelSettings) -> Any:
    return anthropic_gateway_client(
        auth_token=settings.api_key,
        base_url=settings.base_url,
        default_headers=settings.headers,
        timeout=600.0,
    )


class _GatewayCredentialTransport(httpx.BaseTransport):
    def __init__(
        self,
        settings: FrameworkModelSettings,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        return self._transport.handle_request(_gateway_credential_request(self._settings, request))

    def close(self) -> None:
        self._transport.close()


class _AsyncGatewayCredentialTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        settings: FrameworkModelSettings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport or httpx.AsyncHTTPTransport()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return await self._transport.handle_async_request(
            _gateway_credential_request(self._settings, request)
        )

    async def aclose(self) -> None:
        await self._transport.aclose()


def _gateway_credential_request(
    settings: FrameworkModelSettings,
    request: httpx.Request,
) -> httpx.Request:
    logger.info(
        "agentcore.integration.model.http.request connection_id=%s model=%s method=%s url=%s",
        settings.descriptor.connection_id,
        settings.model,
        request.method,
        safe_url(str(request.url)),
    )
    headers = httpx.Headers(request.headers)
    headers.pop("authorization", None)
    headers["authorization"] = f"Bearer {settings.api_key}"
    headers.update(settings.headers)
    return httpx.Request(
        method=request.method,
        url=request.url,
        headers=headers,
        content=request.content,
        extensions=request.extensions,
    )


def _project_gateway_credentials(headers: Mapping[str, str]) -> tuple[str, dict[str, str]]:
    projected = dict(headers)
    authorization_key = next((key for key in projected if key.lower() == "authorization"), None)
    if authorization_key is None:
        raise ConfigError("gateway credentials require a Bearer Authorization header")
    authorization = projected.pop(authorization_key)
    if not authorization.lower().startswith("bearer ") or not authorization[7:].strip():
        raise ConfigError("gateway credentials require a Bearer Authorization header")
    return authorization[7:].strip(), projected


def tool_functions(tools: Sequence[Tool]) -> list[Callable[..., Coroutine[Any, Any, Any]]]:
    logger.debug("agentcore.integration.tools.projected mode=async count=%s", len(tools))
    return [_tool_function(tool) for tool in tools]


def sync_tool_functions(tools: Sequence[Tool]) -> list[Callable[..., Any] | None]:
    logger.debug("agentcore.integration.tools.projected mode=sync count=%s", len(tools))
    return [_sync_tool_function(tool) if tool.supports_sync else None for tool in tools]


def _tool_function(tool: Tool) -> Callable[..., Coroutine[Any, Any, Any]]:
    async def invoke(**kwargs: Any) -> Any:
        return await tool.ainvoke(kwargs)

    _apply_tool_signature(invoke, tool)
    return invoke


def _sync_tool_function(tool: Tool) -> Callable[..., Any]:
    def invoke(**kwargs: Any) -> Any:
        return tool.invoke(kwargs)

    _apply_tool_signature(invoke, tool)
    return invoke


def _apply_tool_signature(function: Callable[..., Any], tool: Tool) -> None:
    parameters: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {}
    properties = tool.parameters.get("properties", {})
    required = set(tool.parameters.get("required", []))
    if isinstance(properties, Mapping):
        property_names = [name for name in properties if isinstance(name, str)]
        if any(not name.isidentifier() or keyword.iskeyword(name) for name in property_names):
            parameters.append(
                inspect.Parameter(
                    "kwargs",
                    inspect.Parameter.VAR_KEYWORD,
                    annotation=Any,
                )
            )
            annotations["kwargs"] = Any
            properties = {}
        for name, schema in properties.items():
            if not isinstance(name, str) or not isinstance(schema, Mapping):
                continue
            annotation = _schema_type(schema)
            default = inspect.Parameter.empty if name in required else None
            parameters.append(
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=annotation,
                )
            )
            annotations[name] = annotation
    function.__name__ = tool.name
    function.__doc__ = tool.description
    function.__annotations__ = annotations
    object.__setattr__(function, "__signature__", inspect.Signature(parameters))
    object.__setattr__(function, "__agentcore_json_schema__", dict(tool.parameters))


def _schema_type(schema: Mapping[str, Any]) -> Any:
    kind = schema.get("type")
    if not isinstance(kind, str):
        return Any
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }.get(kind, Any)
