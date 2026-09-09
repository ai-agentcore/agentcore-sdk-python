"""Resolve credential metadata by name, then fetch the secret from AgentIdentity."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from urllib.parse import urlsplit

from agentcore.auth.resource_sts import AGENT_IDENTITY_DATA_PURPOSE, ResourceCredential
from agentcore.errors import (
    ConfigError,
    CredentialExchangeError,
    WorkloadAccessTokenRejectedError,
)

if TYPE_CHECKING:
    from agentcore.controlplane.client import AgentCoreControlPlane, CredentialMetadata

_WAT_ERROR_CODES = {"WORKLOAD_ACCESS_TOKEN_EXPIRED", "WORKLOAD_ACCESS_TOKEN_INVALID"}
logger = logging.getLogger(__name__)


class STSProvider(Protocol):
    async def get(self, purpose: str) -> ResourceCredential: ...


class WATProvider(Protocol):
    async def get(self) -> str: ...

    async def invalidate(self, token: str | None = None) -> None: ...


class BoundCredentialTransport(Protocol):
    async def get_api_key(
        self,
        *,
        region_id: str,
        credential: ResourceCredential,
        provider_name: str,
        workload_access_token: str,
    ) -> str: ...


CredentialRuntimeProvider = Callable[
    [],
    Awaitable[
        "tuple[str, str, WATProvider | None, STSProvider | None, AgentCoreControlPlane | None]"
    ],
]


@dataclass(frozen=True, repr=False)
class BoundCredential:
    provider_name: str
    value: str
    metadata: CredentialMetadata

    @property
    def credential_type(self) -> str:
        return self.metadata.credential_type

    def as_headers(self) -> dict[str, str]:
        if self.credential_type != "mcpHeader":
            raise ConfigError("Credential must have type mcpHeader to read headers")
        try:
            data = json.loads(self.value)
        except (ValueError, TypeError):
            raise ConfigError("Invalid MCP Header credential payload") from None
        entries = data.get("headers") if isinstance(data, dict) else None
        if not isinstance(entries, list) or not entries:
            raise ConfigError("MCP Header credential must contain a non-empty headers array")
        headers: dict[str, str] = {}
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else None
            value = entry.get("value") if isinstance(entry, dict) else None
            if not isinstance(name, str) or not re.fullmatch(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+", name):
                raise ConfigError("Invalid credential header name")
            if not isinstance(value, str) or any(ord(c) < 32 or ord(c) == 127 for c in value):
                raise ConfigError("Invalid credential header value")
            if name.lower() in headers:
                raise ConfigError("Duplicate credential header name")
            headers[name.lower()] = value
        return headers

    def __repr__(self) -> str:
        return f"BoundCredential(provider_name={self.provider_name!r}, value=<redacted>)"


class AsyncBoundCredentials:
    def __init__(
        self,
        workspace_id: str,
        region_id: str,
        workload_access_tokens: WATProvider | None,
        resource_sts: STSProvider | None,
        *,
        control_plane: AgentCoreControlPlane | None = None,
        transport: BoundCredentialTransport | None = None,
        _runtime_provider: CredentialRuntimeProvider | None = None,
    ) -> None:
        self._workspace_id = workspace_id
        self._region_id = region_id
        self._wat = workload_access_tokens
        self._sts = resource_sts
        self._control_plane = control_plane
        self._transport = transport or AgentIdentityDataTransport()
        self._runtime_provider = _runtime_provider

    async def get(self, credential_name: str) -> BoundCredential:
        return await self._get(credential_name)

    async def _get(
        self,
        credential_name: str,
        *,
        mcp_server_id: str | None = None,
    ) -> BoundCredential:
        credential_name = credential_name.strip()
        if not credential_name:
            raise ValueError("credential name must not be empty")
        workspace_id = self._workspace_id
        region_id = self._region_id
        wat_provider = self._wat
        sts_provider = self._sts
        control_plane = self._control_plane
        if self._runtime_provider is not None:
            (
                workspace_id,
                region_id,
                wat_provider,
                sts_provider,
                control_plane,
            ) = await self._runtime_provider()
        if wat_provider is None or sts_provider is None:
            raise CredentialExchangeError("AgentCore credential runtime is not configured")
        if control_plane is None:
            raise CredentialExchangeError("AgentCore credential metadata runtime is not configured")
        metadata = await control_plane.resolve_credential(credential_name)
        if mcp_server_id is not None:
            metadata.require_mcp(mcp_server_id)
        # Matches the control plane's CreateCredential provider naming contract.
        provider_name = f"{workspace_id}-{credential_name}"
        logger.info(
            "agentcore.credential.resolve.started workspace_id=%s credential_name=%s",
            workspace_id,
            credential_name,
        )
        credential = await sts_provider.get(AGENT_IDENTITY_DATA_PURPOSE)
        wat = await wat_provider.get()
        try:
            value = await self._transport.get_api_key(
                region_id=region_id,
                credential=credential,
                provider_name=provider_name,
                workload_access_token=wat,
            )
        except WorkloadAccessTokenRejectedError:
            logger.warning(
                "agentcore.credential.resolve.wat_rejected workspace_id=%s "
                "credential_name=%s retry=true",
                workspace_id,
                credential_name,
            )
            await wat_provider.invalidate(wat)
            wat = await wat_provider.get()
            value = await self._transport.get_api_key(
                region_id=region_id,
                credential=credential,
                provider_name=provider_name,
                workload_access_token=wat,
            )
        if not value:
            raise CredentialExchangeError("Agent Identity Data returned an empty API key")
        logger.info(
            "agentcore.credential.resolve.succeeded workspace_id=%s credential_name=%s",
            workspace_id,
            credential_name,
        )
        return BoundCredential(provider_name=provider_name, value=value, metadata=metadata)


class AgentIdentityDataTransport:
    async def get_api_key(
        self,
        *,
        region_id: str,
        credential: ResourceCredential,
        provider_name: str,
        workload_access_token: str,
    ) -> str:
        try:
            from alibabacloud_agentidentitydata20251127 import (  # type: ignore[import-untyped]
                models,
            )
            from alibabacloud_agentidentitydata20251127.client import (  # type: ignore[import-untyped]
                Client,
            )
            from alibabacloud_tea_openapi import (
                models as openapi_models,
            )
        except ImportError as exc:
            raise ImportError(
                "bound credentials require: pip install alibabacloud-agentcore-sdk[credentials]"
            ) from exc
        endpoint = f"https://agentidentitydata.{region_id}.aliyuncs.com"
        parsed = urlsplit(endpoint)
        configuration = openapi_models.Config(
            access_key_id=credential.access_key_id,
            access_key_secret=credential.access_key_secret,
            security_token=credential.security_token,
            region_id=region_id,
            endpoint=parsed.netloc or parsed.path,
            protocol=parsed.scheme or "https",
        )
        client = Client(configuration)
        try:
            response = await client.get_resource_apikey_async(
                models.GetResourceAPIKeyRequest(
                    resource_credential_provider_name=provider_name,
                    workload_access_token=workload_access_token,
                )
            )
        except Exception as exc:
            if _is_wat_rejection(exc):
                raise WorkloadAccessTokenRejectedError(
                    "Agent Identity Data rejected the Workload Access Token"
                ) from exc
            logger.warning(
                "agentcore.credential.api.failed provider_name=%s error_type=%s",
                provider_name,
                type(exc).__name__,
            )
            raise CredentialExchangeError("Agent Identity Data API key request failed") from exc
        value = getattr(getattr(response, "body", None), "apikey", None)
        if not isinstance(value, str):
            raise CredentialExchangeError("Agent Identity Data returned an invalid response")
        return value


def _is_wat_rejection(exc: Exception) -> bool:
    current: object | None = exc
    for _ in range(3):
        code = getattr(current, "code", None)
        if isinstance(code, str) and code.upper() in _WAT_ERROR_CODES:
            return True
        data = getattr(current, "data", None)
        if isinstance(data, dict):
            nested_code = data.get("code", data.get("Code"))
            if isinstance(nested_code, str) and nested_code.upper() in _WAT_ERROR_CODES:
                return True
        current = getattr(current, "inner_exception", None)
        if current is None:
            break
    return False
