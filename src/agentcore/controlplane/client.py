"""Resolve workspace resource names through AgentCore OpenAPI."""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol
from urllib.parse import parse_qs, quote, urlsplit

import httpx
from aiohttp import ClientConnectionError
from alibabacloud_agentcore20260804 import models
from alibabacloud_agentcore20260804.client import Client
from alibabacloud_tea_openapi import models as openapi_models
from alibabacloud_tea_openapi import utils_models as openapi_utils
from darabonba.exceptions import RetryError, UnretryableException  # type: ignore[import-untyped]
from darabonba.runtime import RuntimeOptions  # type: ignore[import-untyped]

from agentcore.auth.access_key import AccessKeyCredential
from agentcore.auth.resource_sts import HIGH_CODE_SDK_PURPOSE, ResourceCredential
from agentcore.errors import (
    ConfigError,
    InvocationError,
    MCPServerNotFoundError,
    ModelConnectionNotFoundError,
    ResourceNotConfiguredError,
)

_MAX_SKILL_ARCHIVE_BYTES = 10 * 1024 * 1024
logger = logging.getLogger(__name__)


class STSProvider(Protocol):
    async def get(self, purpose: str | None = None) -> ResourceCredential: ...


@dataclass(frozen=True)
class ModelDescriptor:
    connection_id: str
    connection_name: str
    protocol: str
    provider_type: str
    model_id: str
    model_name: str
    context_size: int | None
    max_tokens: int | None
    capabilities: Mapping[str, bool]


@dataclass(frozen=True)
class MCPDescriptor:
    mcp_server_id: str
    name: str
    protocol: str
    type: str
    status: str


@dataclass(frozen=True)
class SkillArtifact:
    version: str
    archive: bytes


@dataclass(frozen=True)
class CredentialMetadata:
    credential_id: str
    name: str
    credential_type: str
    resource_scope: str
    resource_refs: tuple[tuple[str, str], ...]

    def require_mcp(self, server_id: str) -> None:
        if self.credential_type != "mcpHeader":
            raise ConfigError("MCP credential must have type mcpHeader")
        if self.resource_scope == "ALL":
            return
        if self.resource_scope == "SPECIFIED" and ("mcpServer", server_id) in self.resource_refs:
            return
        raise ConfigError("MCP credential does not allow the selected MCP server")


class AgentCoreControlPlane:
    def __init__(
        self,
        *,
        workspace_id: str,
        region_id: str,
        resource_sts: STSProvider | None = None,
        access_key_credential: AccessKeyCredential | None = None,
        endpoint: str | None = None,
        _client_factory: Callable[[AccessKeyCredential | ResourceCredential], Any] | None = None,
        _http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if resource_sts is None and access_key_credential is None:
            raise ValueError("AgentCore control plane requires an AccessKey credential source")
        self.workspace_id = workspace_id
        self.region_id = region_id
        self._sts = resource_sts
        self._access_key_credential = access_key_credential
        self._endpoint, self._protocol = _normalize_endpoint(endpoint)
        self._discovered_endpoint: str | None = None
        self._client_factory = _client_factory or self._new_client
        self._http = _http_client

    async def resolve_model(
        self,
        connection_name: str,
        model_name: str | None = None,
    ) -> ModelDescriptor:
        logger.info(
            "agentcore.control_plane.model.resolve.started workspace_id=%s "
            "connection_name=%s model_name=%s",
            self.workspace_id,
            connection_name,
            model_name or "auto",
        )
        client = await self._client()
        connections = await self._list_connections(client, connection_name)
        connection = _one_exact(
            connections,
            "name",
            connection_name,
            "model connection",
            not_found_error=ModelConnectionNotFoundError,
        )
        connection_id = _required_field(connection, "connection_id")
        models_found = await self._list_models(client, connection_id, model_name)
        if model_name is None:
            if len(models_found) != 1:
                raise ResourceNotConfiguredError(
                    f"model connection {connection_name!r} requires an explicit model name"
                )
            model = models_found[0]
        else:
            model = _one_exact(models_found, "model_name", model_name, "model")
        descriptor = ModelDescriptor(
            connection_id=connection_id,
            connection_name=_required_field(connection, "name"),
            protocol=str(_field(connection, "protocol") or ""),
            provider_type=str(_field(connection, "provider_type") or ""),
            model_id=_required_field(model, "model_id"),
            model_name=_required_field(model, "model_name"),
            context_size=_optional_int(_field(model, "context_size")),
            max_tokens=_optional_int(_field(model, "max_tokens")),
            capabilities=_capabilities(_field(model, "capabilities")),
        )
        logger.info(
            "agentcore.control_plane.model.resolve.succeeded workspace_id=%s "
            "connection_name=%s connection_id=%s model_name=%s",
            self.workspace_id,
            descriptor.connection_name,
            descriptor.connection_id,
            descriptor.model_name,
        )
        return descriptor

    async def resolve_mcp(self, name: str) -> MCPDescriptor:
        logger.info(
            "agentcore.control_plane.mcp.resolve.started workspace_id=%s name=%s",
            self.workspace_id,
            name,
        )
        client = await self._client()
        items: list[Any] = []
        next_token: str | None = None
        while True:
            response = await self._invoke(
                client,
                "list_mcps",
                self.workspace_id,
                models.ListMcpsRequest(
                    name=name,
                    search_type="accurate",
                    max_results=100,
                    next_token=next_token,
                ),
            )
            body = _field(response, "body")
            items.extend(_field(body, "items") or [])
            next_token = _field(body, "next_token")
            if not next_token:
                break
        item = _one_exact(
            items,
            "name",
            name,
            "MCP server",
            not_found_error=MCPServerNotFoundError,
        )
        descriptor = MCPDescriptor(
            mcp_server_id=_required_field(item, "mcp_server_id"),
            name=_required_field(item, "name"),
            protocol=str(_field(item, "protocol") or ""),
            type=str(_field(item, "type") or ""),
            status=str(_field(item, "status") or ""),
        )
        logger.info(
            "agentcore.control_plane.mcp.resolve.succeeded workspace_id=%s name=%s "
            "mcp_server_id=%s protocol=%s",
            self.workspace_id,
            descriptor.name,
            descriptor.mcp_server_id,
            descriptor.protocol,
        )
        return descriptor

    async def get_skill(self, name: str, version: str | None = None) -> SkillArtifact:
        logger.info(
            "agentcore.control_plane.skill.get.started workspace_id=%s name=%s version=%s",
            self.workspace_id,
            name,
            version or "latest",
        )
        client = await self._client()
        resolved_version = version
        if resolved_version is None:
            response = await self._invoke(
                client,
                "get_skill_detail",
                self.workspace_id,
                name,
                models.GetSkillDetailRequest(),
            )
            resolved_version = _latest_skill_version(_field(_field(response, "body"), "data"))
        response = await self._invoke(
            client,
            "download_skill_version_via_oss",
            self.workspace_id,
            name,
            resolved_version,
            models.DownloadSkillVersionViaOssRequest(),
        )
        download_url = _field(_field(response, "body"), "data")
        if not isinstance(download_url, str) or not _valid_download_url(download_url):
            raise ConfigError("AgentCore returned an invalid Skill download URL")
        archive = await self._download_skill_archive(download_url)
        logger.info(
            "agentcore.control_plane.skill.get.succeeded workspace_id=%s name=%s "
            "version=%s bytes=%s",
            self.workspace_id,
            name,
            resolved_version,
            len(archive),
        )
        return SkillArtifact(
            version=resolved_version,
            archive=archive,
        )

    async def resolve_credential(self, name: str) -> CredentialMetadata:
        client = await self._client()
        params = openapi_utils.Params(
            action="ListCredentials",
            version="2026-08-04",
            protocol="HTTPS",
            pathname=f"/workspaces/{quote(self.workspace_id, safe='')}/credentials",
            method="GET",
            auth_type="AK",
            style="ROA",
            req_body_type="json",
            body_type="json",
        )
        items: list[Any] = []
        next_token = None
        while True:
            query = {"name": name, "maxResults": "100"}
            if next_token:
                query["nextToken"] = next_token
            request = openapi_utils.OpenApiRequest(query=query)
            response = await self._request(
                client,
                "list_credentials",
                partial(
                    client.call_api_async,
                    params,
                    request,
                    RuntimeOptions(autoretry=False, connect_timeout=10000, read_timeout=30000),
                ),
            )
            body = response["body"]
            items.extend(body.get("items") or [])
            next_token = body.get("nextToken")
            if not next_token:
                break
        item = _one_exact(items, "name", name, "credential")
        return CredentialMetadata(
            credential_id=_required_field(item, "credentialId"),
            name=_required_field(item, "name"),
            credential_type=_required_field(item, "credentialType"),
            resource_scope=str(item.get("resourceScope") or ""),
            resource_refs=tuple(
                (_required_field(ref, "resourceType"), _required_field(ref, "resourceId"))
                for ref in (item.get("resourceRefs") or [])
            ),
        )

    async def _client(self) -> Any:
        credential: AccessKeyCredential | ResourceCredential
        if self._access_key_credential is not None:
            credential = self._access_key_credential
        else:
            assert self._sts is not None
            credential = await self._sts.get(HIGH_CODE_SDK_PURPOSE)
        return self._client_factory(credential)

    async def _download_skill_archive(self, url: str) -> bytes:
        if self._http is not None:
            return await _read_skill_archive(self._http, url)
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
            return await _read_skill_archive(client, url)

    def _new_client(self, credential: AccessKeyCredential | ResourceCredential) -> Client:
        values: dict[str, Any] = {
            "access_key_id": credential.access_key_id,
            "access_key_secret": credential.access_key_secret,
            "region_id": self.region_id,
            "endpoint": self._endpoint or self._discovered_endpoint,
            "protocol": self._protocol,
        }
        if credential.security_token is not None:
            values["security_token"] = credential.security_token
        return Client(openapi_models.Config(**values))

    async def _invoke(self, client: Any, operation: str, *args: Any) -> Any:
        call = getattr(client, f"{operation}_async")
        return await self._request(client, operation, lambda: call(*args))

    async def _request(
        self,
        client: Any,
        operation: str,
        call: Callable[[], Awaitable[Any]],
    ) -> Any:
        """Try the regional private endpoint once for these read-only OpenAPI calls."""
        try:
            return await _invoke_control_plane(operation, call())
        except InvocationError as exc:
            public = f"agentcore.{self.region_id}.aliyuncs.com"
            if (
                self._endpoint is not None
                or not _endpoint_failure(exc.__cause__)
                or client._endpoint != public
            ):
                raise
            private = f"agentcore-vpc.{self.region_id}.aliyuncs.com"
            logger.warning(
                "agentcore.control_plane.endpoint.fallback operation=%s "
                "from_host=%s to_host=%s error_type=%s",
                operation,
                public,
                private,
                type(exc.__cause__).__name__,
            )
            client._endpoint = private
            result = await _invoke_control_plane(operation, call())
            self._discovered_endpoint = private
            logger.info(
                "agentcore.control_plane.endpoint.fallback.succeeded operation=%s host=%s",
                operation,
                private,
            )
            return result

    async def _list_connections(self, client: Any, name: str) -> list[Any]:
        items: list[Any] = []
        next_token: str | None = None
        while True:
            response = await self._invoke(
                client,
                "list_model_connections",
                self.workspace_id,
                models.ListModelConnectionsRequest(
                    name=name,
                    search_type="accurate",
                    max_results=100,
                    next_token=next_token,
                ),
            )
            body = _field(response, "body")
            items.extend(_field(body, "items") or [])
            next_token = _field(body, "next_token")
            if not next_token:
                return items

    async def _list_models(
        self,
        client: Any,
        connection_id: str,
        model_name: str | None,
    ) -> list[Any]:
        items: list[Any] = []
        next_token: str | None = None
        while True:
            response = await self._invoke(
                client,
                "list_models",
                self.workspace_id,
                models.ListModelsRequest(
                    connection_id=connection_id,
                    model_name=model_name,
                    max_results=100,
                    next_token=next_token,
                ),
            )
            body = _field(response, "body")
            items.extend(_field(body, "items") or [])
            next_token = _field(body, "next_token")
            if not next_token:
                return items


def _endpoint_failure(error: BaseException | None) -> bool:
    if isinstance(error, UnretryableException):
        error = error.inner_exception
    if isinstance(error, (RetryError, ClientConnectionError, TimeoutError, ConnectionError)):
        return True
    status = getattr(error, "status_code", None)
    return isinstance(status, int) and 500 <= status < 600


async def _invoke_control_plane(operation: str, awaitable: Awaitable[Any]) -> Any:
    try:
        return await awaitable
    except Exception as exc:
        data = getattr(exc, "data", None)
        upstream_request_id = (
            getattr(exc, "request_id", None)
            or _field(data, "RequestId")
            or _field(data, "requestId")
            or _field(data, "requestid")
        )
        logger.warning(
            "agentcore.control_plane.request.failed operation=%s error_type=%s "
            "status=%s code=%s upstream_request_id=%s",
            operation,
            type(exc).__name__,
            getattr(exc, "status_code", None) or getattr(exc, "statusCode", None) or "-",
            getattr(exc, "code", None) or "-",
            upstream_request_id or "-",
        )
        raise InvocationError("AgentCore control-plane request failed") from exc


def _field(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _required_field(value: Any, name: str) -> str:
    result = _field(value, name)
    if not isinstance(result, str) or not result:
        raise ConfigError(f"AgentCore returned a resource without {name}")
    return result


def _one_exact(
    items: list[Any],
    field: str,
    expected: str,
    resource: str,
    *,
    not_found_error: type[ResourceNotConfiguredError] = ResourceNotConfiguredError,
) -> Any:
    exact = [item for item in items if _field(item, field) == expected]
    if not exact:
        logger.warning(
            "agentcore.control_plane.resource.resolve.failed resource=%s expected=%s "
            "exact_matches=%s candidates=%s",
            resource,
            expected,
            len(exact),
            len(items),
        )
        raise not_found_error(f"{resource} {expected!r} was not found")
    if len(exact) > 1:
        logger.warning(
            "agentcore.control_plane.resource.resolve.failed resource=%s expected=%s "
            "exact_matches=%s candidates=%s",
            resource,
            expected,
            len(exact),
            len(items),
        )
        raise ConfigError(f"{resource} {expected!r} was returned more than once")
    return exact[0]


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _normalize_endpoint(value: str | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    value = value.strip()
    if not value:
        raise ConfigError("AgentCore control endpoint must not be empty")
    has_scheme = "://" in value
    try:
        parsed = urlsplit(value if has_scheme else f"//{value}")
    except ValueError as exc:
        raise ConfigError("AgentCore control endpoint is invalid") from exc
    if has_scheme and parsed.scheme not in {"http", "https"}:
        raise ConfigError("AgentCore control endpoint must use HTTP(S)")
    if not parsed.hostname or parsed.username or parsed.password:
        raise ConfigError("AgentCore control endpoint is invalid")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise ConfigError("AgentCore control endpoint must not contain a path or query")
    return parsed.netloc, parsed.scheme or None


def _capabilities(value: Any) -> Mapping[str, bool]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        raw = value
    else:
        try:
            raw = vars(value)
        except TypeError as exc:
            raise ConfigError("AgentCore returned invalid model capabilities") from exc
    return {
        key: item for key, item in raw.items() if isinstance(key, str) and isinstance(item, bool)
    }


def _latest_skill_version(data: Any) -> str:
    if data is None:
        raise ResourceNotConfiguredError("AgentCore Skill does not exist")
    labels = _field(data, "labels") or {}
    if isinstance(labels, Mapping):
        for key, value in labels.items():
            if str(key).lower() == "latest" and isinstance(value, str) and value:
                return value
    online = [
        item
        for item in (_field(data, "versions") or [])
        if str(_field(item, "status") or "").lower() == "online"
        and isinstance(_field(item, "version"), str)
    ]
    if not online:
        raise ResourceNotConfiguredError("AgentCore Skill has no online version")
    latest = max(
        online,
        key=lambda item: (
            _field(item, "update_time") if isinstance(_field(item, "update_time"), int) else -1,
            str(_field(item, "version")),
        ),
    )
    return str(_field(latest, "version"))


def _valid_download_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.hostname)


async def _read_skill_archive(client: httpx.AsyncClient, url: str) -> bytes:
    candidate = url
    while True:
        try:
            archive = await _read_skill_archive_once(client, candidate)
        except httpx.HTTPError as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            retryable = isinstance(exc, httpx.TransportError) or (
                status is not None and 500 <= status < 600
            )
            fallback = _internal_oss_url(url) if candidate == url and retryable else None
            if fallback is None:
                message = (
                    f"Skill package download failed with HTTP {status}"
                    if status is not None
                    else "Skill package download failed"
                )
                if isinstance(exc, httpx.HTTPStatusError):
                    # HTTPStatusError embeds the presigned URL in its message.
                    raise InvocationError(message) from None
                raise InvocationError(message) from exc
            logger.warning(
                "agentcore.control_plane.skill.download.fallback from_host=%s to_host=%s "
                "error_type=%s",
                urlsplit(url).hostname,
                urlsplit(fallback).hostname,
                type(exc).__name__,
            )
            candidate = fallback
            continue
        if candidate != url:
            logger.info(
                "agentcore.control_plane.skill.download.fallback.succeeded host=%s bytes=%s",
                urlsplit(candidate).hostname,
                len(archive),
            )
        return archive


def _internal_oss_url(url: str) -> str | None:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    match = re.fullmatch(r"([a-z0-9-]+)\.(oss-[a-z]+-[a-z0-9-]+)\.aliyuncs\.com", host)
    if (
        match is None
        or match[2].endswith("-internal")
        or match[2].startswith("oss-accelerate")
        or parsed.username is not None
        or parsed.password is not None
    ):
        return None
    additional_headers = parse_qs(parsed.query).get("x-oss-additional-headers", [])
    if any("host" in value.split(";") for value in additional_headers):
        return None
    internal_host = f"{match[1]}.{match[2]}-internal.aliyuncs.com"
    netloc = f"{internal_host}:{parsed.port}" if parsed.port is not None else internal_host
    return parsed._replace(netloc=netloc).geturl()


async def _read_skill_archive_once(client: httpx.AsyncClient, url: str) -> bytes:
    parsed = urlsplit(url)
    logger.debug(
        "agentcore.control_plane.skill.download.started host=%s path=%s",
        parsed.netloc,
        parsed.path,
    )
    try:
        async with client.stream("GET", url, follow_redirects=False) as response:
            if response.status_code >= 400:
                response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                data.extend(chunk)
                if len(data) > _MAX_SKILL_ARCHIVE_BYTES:
                    raise ConfigError("Skill package exceeds the 10 MiB size limit")
    except httpx.HTTPError as exc:
        error_response = exc.response if isinstance(exc, httpx.HTTPStatusError) else None
        logger.warning(
            "agentcore.control_plane.skill.download.failed host=%s error_type=%s "
            "status=%s request_id=%s",
            parsed.netloc,
            type(exc).__name__,
            error_response.status_code if error_response is not None else "-",
            error_response.headers.get("x-oss-request-id", "-")
            if error_response is not None
            else "-",
        )
        raise
    if not data:
        raise ConfigError("Skill package is empty")
    logger.debug(
        "agentcore.control_plane.skill.download.succeeded host=%s bytes=%s",
        parsed.netloc,
        len(data),
    )
    return bytes(data)
