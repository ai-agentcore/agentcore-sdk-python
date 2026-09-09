"""Load the high-code runtime configuration from Controller-scoped OSS."""

from __future__ import annotations

import base64
import hashlib
import hmac
import ipaddress
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import anyio
import httpx

from agentcore.errors import AuthenticationError, ConfigError, CredentialExchangeError
from agentcore.runtime.config import MAX_CONFIG_BYTES, AgentConfig, parse_agent_config

_CONTROL_STS_PATH = "/api/v1/credentials/sts"
_CREDENTIAL_REFRESH_BEFORE = timedelta(minutes=1)
logger = logging.getLogger(__name__)


@dataclass(frozen=True, repr=False)
class _ControlConfigCredential:
    access_key_id: str
    access_key_secret: str
    security_token: str
    oss_endpoint: str
    oss_bucket: str
    agent_config_path: str
    teams_config_path: str | None
    expiration: datetime


class ControlConfigLoader:
    """Read Agent and Team configuration from Controller-scoped OSS."""

    def __init__(
        self,
        controller_endpoint: str,
        *,
        http_client: httpx.AsyncClient,
        max_attempts: int = 3,
    ) -> None:
        self._controller_endpoint = controller_endpoint.rstrip("/")
        self._http = http_client
        self._max_attempts = max(1, max_attempts)
        self._credential: _ControlConfigCredential | None = None

    async def load(self, agent_sa_token: str) -> AgentConfig:
        logger.info("agentcore.control_config.load.started")
        try:
            credential = await self._credential_for(agent_sa_token)
            data = await self._download(
                credential,
                credential.agent_config_path,
                label="agent.yaml",
            )
            assert data is not None
            config = parse_agent_config(data)
        except Exception as exc:
            logger.warning(
                "agentcore.control_config.load.failed error_type=%s",
                type(exc).__name__,
            )
            raise
        logger.info(
            "agentcore.control_config.load.succeeded workspace_id=%s region_id=%s bytes=%s",
            config.workspace_id,
            config.region_id,
            len(data),
        )
        return config

    async def load_teams(
        self,
        agent_sa_token: str,
    ) -> bytes | None:
        credential = await self._credential_for(agent_sa_token)
        if credential.teams_config_path is None:
            credential = await self._control_credential(agent_sa_token)
            self._credential = credential
            if credential.teams_config_path is None:
                return None
        object_key = _oss_object_key(
            credential.teams_config_path,
            "teams_config_path",
        )
        return await self._download(
            credential,
            object_key,
            label="teams.yaml",
            missing_ok=True,
        )

    async def _credential_for(
        self,
        agent_sa_token: str,
    ) -> _ControlConfigCredential:
        if (
            self._credential is not None
            and datetime.now(timezone.utc) + _CREDENTIAL_REFRESH_BEFORE
            < self._credential.expiration
        ):
            return self._credential
        self._credential = await self._control_credential(agent_sa_token)
        return self._credential

    async def _control_credential(self, agent_sa_token: str) -> _ControlConfigCredential:
        response = await self._request_sts(agent_sa_token)
        if response.status_code == 401:
            logger.warning("agentcore.control_config.sts.failed status=401")
            raise AuthenticationError("Controller rejected the Agent SA token")
        if response.status_code >= 400:
            logger.warning(
                "agentcore.control_config.sts.failed status=%s",
                response.status_code,
            )
            raise CredentialExchangeError(
                f"Controller control-config STS request failed with HTTP {response.status_code}"
            )
        try:
            payload: Any = response.json()
            if not isinstance(payload, dict):
                raise ValueError
            expiration = _timestamp(_required_string(payload, "expiration"))
            if expiration <= datetime.now(timezone.utc):
                raise ValueError
            credential = _ControlConfigCredential(
                access_key_id=_required_string(payload, "access_key_id"),
                access_key_secret=_required_string(payload, "access_key_secret"),
                security_token=_required_string(payload, "security_token"),
                oss_endpoint=_required_string(payload, "oss_endpoint"),
                oss_bucket=_oss_bucket(_required_string(payload, "oss_bucket")),
                agent_config_path=_oss_object_key(
                    _required_string(payload, "agent_config_path"),
                    "agent_config_path",
                ),
                teams_config_path=_optional_string(payload, "teams_config_path"),
                expiration=expiration,
            )
            logger.info(
                "agentcore.control_config.sts.succeeded bucket=%s object_key=%s",
                credential.oss_bucket,
                credential.agent_config_path,
            )
            return credential
        except (KeyError, TypeError, ValueError) as exc:
            raise CredentialExchangeError(
                "Controller returned an invalid control-config STS response"
            ) from exc

    async def _request_sts(self, agent_sa_token: str) -> httpx.Response:
        last_error: httpx.HTTPError | None = None
        for attempt in range(self._max_attempts):
            try:
                response = await self._http.post(
                    f"{self._controller_endpoint}{_CONTROL_STS_PATH}",
                    params={"purpose": "oss", "target": "control"},
                    headers={"Authorization": f"Bearer {agent_sa_token}"},
                    content=b"",
                    follow_redirects=False,
                )
            except httpx.TransportError as exc:
                last_error = exc
                retry_reason = type(exc).__name__
            else:
                if response.status_code not in {500, 502, 503, 504}:
                    return response
                last_error = None
                retry_reason = f"http_{response.status_code}"
            if attempt + 1 < self._max_attempts:
                logger.warning(
                    "agentcore.control_config.sts.retry attempt=%s max_attempts=%s reason=%s",
                    attempt + 1,
                    self._max_attempts,
                    retry_reason,
                )
                await anyio.sleep(0.05 * (2**attempt))
        if last_error is not None:
            raise CredentialExchangeError("cannot reach the AgentCore Controller") from last_error
        raise CredentialExchangeError("Controller control-config STS service is unavailable")

    async def _download(
        self,
        credential: _ControlConfigCredential,
        object_key: str,
        *,
        label: str,
        missing_ok: bool = False,
    ) -> bytes | None:
        endpoint = _normalize_oss_endpoint(credential.oss_endpoint)
        candidates = [endpoint]
        fallback = _public_oss_endpoint(endpoint)
        if fallback is not None:
            candidates.append(fallback)
        last_error: httpx.HTTPError | None = None
        last_status: int | None = None
        for index, candidate in enumerate(candidates):
            url = _oss_url(
                candidate,
                credential.oss_bucket,
                object_key,
            )
            headers = _oss_headers(credential, object_key)
            logger.debug(
                "agentcore.control_config.download.started endpoint_host=%s bucket=%s "
                "object_key=%s fallback=%s",
                urlsplit(candidate).netloc,
                credential.oss_bucket,
                object_key,
                index > 0,
            )
            try:
                async with self._http.stream(
                    "GET",
                    url,
                    headers=headers,
                    follow_redirects=False,
                ) as response:
                    if response.status_code < 400:
                        data = bytearray()
                        async for chunk in response.aiter_bytes():
                            data.extend(chunk)
                            if len(data) > MAX_CONFIG_BYTES:
                                raise ConfigError(f"{label} exceeds the 1 MiB size limit")
                        logger.info(
                            "agentcore.control_config.download.succeeded endpoint_host=%s "
                            "bytes=%s fallback=%s",
                            urlsplit(candidate).netloc,
                            len(data),
                            index > 0,
                        )
                        return bytes(data)
                    if missing_ok and response.status_code == 404:
                        logger.info(
                            "agentcore.control_config.download.not_found object=%s",
                            label,
                        )
                        return None
                    last_status = response.status_code
                    logger.warning(
                        "agentcore.control_config.download.failed endpoint_host=%s status=%s "
                        "fallback_available=%s",
                        urlsplit(candidate).netloc,
                        response.status_code,
                        index + 1 < len(candidates),
                    )
            except httpx.TransportError as exc:
                last_error = exc
                logger.warning(
                    "agentcore.control_config.download.failed endpoint_host=%s error_type=%s "
                    "fallback_available=%s",
                    urlsplit(candidate).netloc,
                    type(exc).__name__,
                    index + 1 < len(candidates),
                )
                continue
        if last_error is not None and last_status is None:
            raise ConfigError("cannot reach the AgentCore configuration store") from last_error
        raise ConfigError(
            f"AgentCore configuration download failed with HTTP {last_status or 502}"
        )


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError
    return value.strip()


def _optional_string(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError
    return value.strip()


def _normalize_oss_endpoint(value: str) -> str:
    candidate = value if "://" in value else f"https://{value}"
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise CredentialExchangeError("Controller returned an invalid OSS endpoint") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or (port is not None and not 1 <= port <= 65535)
    ):
        raise CredentialExchangeError("Controller returned an invalid OSS endpoint")
    return urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))


def _timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _oss_bucket(value: str) -> str:
    if (
        len(value) < 3
        or len(value) > 63
        or not value[0].isalnum()
        or not value[-1].isalnum()
        or any(
            not (character.islower() or character.isdigit() or character == "-")
            for character in value
        )
    ):
        raise CredentialExchangeError("Controller returned an invalid OSS bucket")
    return value


def _oss_object_key(value: str, field: str) -> str:
    parts = value.split("/")
    if (
        value.startswith("/")
        or "\\" in value
        or "\r" in value
        or "\n" in value
        or any(part in {"", ".", ".."} for part in parts)
    ):
        raise CredentialExchangeError(f"Controller returned an invalid {field}")
    return value


def _public_oss_endpoint(endpoint: str) -> str | None:
    parsed = urlsplit(endpoint)
    hostname = parsed.hostname or ""
    suffix = "-internal.aliyuncs.com"
    if "oss-" not in hostname or not hostname.endswith(suffix):
        return None
    public_hostname = f"{hostname[: -len(suffix)]}.aliyuncs.com"
    port = f":{parsed.port}" if parsed.port is not None else ""
    return urlunsplit((parsed.scheme, f"{public_hostname}{port}", "", "", ""))


def _oss_url(endpoint: str, bucket: str, object_key: str) -> str:
    parsed = urlsplit(endpoint)
    hostname = parsed.hostname or ""
    encoded_key = "/".join(quote(part, safe="") for part in object_key.split("/"))
    path_style = hostname == "localhost" or _is_ip_address(hostname)
    if path_style:
        path = f"/{quote(bucket, safe='')}/{encoded_key}"
        netloc = parsed.netloc
    elif hostname.split(".", 1)[0] == bucket:
        path = f"/{encoded_key}"
        netloc = parsed.netloc
    else:
        port = f":{parsed.port}" if parsed.port is not None else ""
        netloc = f"{bucket}.{hostname}{port}"
        path = f"/{encoded_key}"
    return urlunsplit((parsed.scheme, netloc, path, "", ""))


def _is_ip_address(value: str) -> bool:
    try:
        ipaddress.ip_address(value)
    except ValueError:
        return False
    return True


def _oss_headers(
    credential: _ControlConfigCredential,
    object_key: str,
) -> dict[str, str]:
    date = format_datetime(datetime.now(timezone.utc), usegmt=True)
    canonical_headers = f"x-oss-security-token:{credential.security_token}\n"
    canonical_resource = f"/{credential.oss_bucket}/{object_key}"
    string_to_sign = f"GET\n\n\n{date}\n{canonical_headers}{canonical_resource}"
    signature = base64.b64encode(
        hmac.new(
            credential.access_key_secret.encode(),
            string_to_sign.encode(),
            hashlib.sha1,
        ).digest()
    ).decode()
    return {
        "Date": date,
        "Authorization": f"OSS {credential.access_key_id}:{signature}",
        "x-oss-security-token": credential.security_token,
    }
