from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pytest

from agentcore.auth.agent_sa_token import AgentSATokenProvider
from agentcore.auth.resource_sts import ResourceSTSProvider
from agentcore.auth.workload_access_token import WorkloadAccessTokenProvider
from agentcore.errors import (
    AuthenticationError,
    CredentialExchangeError,
    WorkloadIdentityNotConfiguredError,
)


@pytest.mark.asyncio
async def test_workload_token_request_has_empty_body(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token-a\n", encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/workload/token"
        assert request.headers["Authorization"] == "Bearer sa-token-a"
        assert request.content == b""
        return httpx.Response(200, json={"workloadAccessToken": "wat-a"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = WorkloadAccessTokenProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        assert await provider.get() == "wat-a"


@pytest.mark.asyncio
async def test_workload_token_reloads_rotated_sa_token_on_401(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token-old", encoding="utf-8")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers["Authorization"]
        seen.append(authorization)
        if len(seen) == 1:
            token_path.write_text("sa-token-new", encoding="utf-8")
            return httpx.Response(401)
        return httpx.Response(200, json={"workloadAccessToken": "wat-new"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = WorkloadAccessTokenProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        assert await provider.get() == "wat-new"

    assert seen == ["Bearer sa-token-old", "Bearer sa-token-new"]


@pytest.mark.asyncio
async def test_workload_token_does_not_retry_unchanged_sa_token_on_401(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token", encoding="utf-8")
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = WorkloadAccessTokenProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        with pytest.raises(AuthenticationError):
            await provider.get()

    assert calls == 1


@pytest.mark.asyncio
async def test_workload_token_refreshes_async_sa_token_source_on_401() -> None:
    class AsyncTokenSource:
        token = "sa-token-old"

        async def get(self) -> str:
            return self.token

        async def refresh(self, current: str) -> str:
            assert current == "sa-token-old"
            self.token = "sa-token-new"
            return self.token

    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers["Authorization"]
        seen.append(authorization)
        if authorization == "Bearer sa-token-old":
            return httpx.Response(401)
        return httpx.Response(200, json={"workloadAccessToken": "wat-new"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = WorkloadAccessTokenProvider(
            "https://controller.example.com",
            AsyncTokenSource(),
            http_client=http,
        )
        assert await provider.get() == "wat-new"

    assert seen == ["Bearer sa-token-old", "Bearer sa-token-new"]


@pytest.mark.asyncio
async def test_workload_token_maps_missing_identity(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token", encoding="utf-8")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(404))
    ) as http:
        provider = WorkloadAccessTokenProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        with pytest.raises(WorkloadIdentityNotConfiguredError):
            await provider.get()


@pytest.mark.asyncio
async def test_workload_token_single_flight(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token", encoding="utf-8")
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return httpx.Response(200, json={"workloadAccessToken": "wat"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = WorkloadAccessTokenProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        values = await asyncio.gather(*(provider.get() for _ in range(10)))

    assert values == ["wat"] * 10
    assert calls == 1


@pytest.mark.asyncio
async def test_workload_token_retries_gateway_and_transport_failures(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token", encoding="utf-8")
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(502)
        if calls == 2:
            raise httpx.ReadError("connection interrupted", request=request)
        return httpx.Response(200, json={"workloadAccessToken": "wat"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = WorkloadAccessTokenProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        assert await provider.get() == "wat"

    assert calls == 3


@pytest.mark.asyncio
async def test_resource_sts_uses_sa_token_controller_contract(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token", encoding="utf-8")
    calls = 0
    expiration = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.url.path == "/api/v1/credentials/sts"
        assert dict(request.url.params) == {}
        assert request.headers["Authorization"] == "Bearer sa-token"
        assert request.content == b""
        return httpx.Response(
            200,
            json={
                "access_key_id": "ak",
                "access_key_secret": "sk",
                "security_token": "sts",
                "expiration": expiration,
                "expires_in_sec": 3599,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = ResourceSTSProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        credential = await provider.get()
        cached = await provider.get()

    assert credential.access_key_id == "ak"
    assert cached is credential
    assert calls == 1


@pytest.mark.asyncio
async def test_resource_sts_retries_gateway_and_transport_failures(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token", encoding="utf-8")
    expiration = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(504)
        if calls == 2:
            raise httpx.RemoteProtocolError("connection interrupted", request=request)
        return httpx.Response(
            200,
            json={
                "access_key_id": "ak",
                "access_key_secret": "sk",
                "security_token": "sts",
                "expiration": expiration,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = ResourceSTSProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        credential = await provider.get("highcode_sdk")

    assert credential.security_token == "sts"
    assert calls == 3


@pytest.mark.asyncio
async def test_resource_sts_reports_controller_error_after_retries(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token", encoding="utf-8")

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                503,
                json={
                    "message": "STS provider is unavailable",
                    "access_key_secret": "must-not-be-logged",
                },
            )
        )
    ) as http:
        provider = ResourceSTSProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        with caplog.at_level(logging.WARNING, logger="agentcore.auth.resource_sts"):
            with pytest.raises(
                CredentialExchangeError,
                match=(
                    "Controller STS request failed after 3 attempts with HTTP 503: "
                    "STS provider is unavailable"
                ),
            ):
                await provider.get("highcode_sdk")

    assert "status=503" in caplog.text
    assert "attempts=3" in caplog.text
    assert "url=https://controller.example.com/api/v1/credentials/sts" in caplog.text
    assert "message='STS provider is unavailable'" in caplog.text
    assert "must-not-be-logged" not in caplog.text


@pytest.mark.asyncio
async def test_resource_sts_logs_lifecycle_without_credentials(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token-secret", encoding="utf-8")
    expiration = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={
                    "access_key_id": "access-key-secret",
                    "access_key_secret": "access-secret",
                    "security_token": "security-token-secret",
                    "expiration": expiration,
                },
            )
        )
    ) as http:
        provider = ResourceSTSProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        with caplog.at_level(logging.DEBUG, logger="agentcore.auth.resource_sts"):
            await provider.get("agentcore")
            await provider.get("agentcore")

    assert "agentcore.sts.exchange.started" in caplog.text
    assert "agentcore.sts.exchange.succeeded" in caplog.text
    assert "agentcore.sts.cache.hit" in caplog.text
    for secret in (
        "sa-token-secret",
        "access-key-secret",
        "access-secret",
        "security-token-secret",
    ):
        assert secret not in caplog.text


@pytest.mark.asyncio
async def test_resource_sts_sends_explicit_purpose(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token", encoding="utf-8")
    expiration = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    def handler(request: httpx.Request) -> httpx.Response:
        assert dict(request.url.params) == {"purpose": "agentidentitydata"}
        return httpx.Response(
            200,
            json={
                "access_key_id": "ak",
                "access_key_secret": "sk",
                "security_token": "sts",
                "expiration": expiration,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = ResourceSTSProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        await provider.get("agentidentitydata")


@pytest.mark.asyncio
async def test_resource_sts_rejects_already_expired_response(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token", encoding="utf-8")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_key_id": "ak",
                "access_key_secret": "sk",
                "security_token": "sts",
                "expiration": (
                    datetime.now(timezone.utc) - timedelta(seconds=1)
                ).isoformat(),
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = ResourceSTSProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        with pytest.raises(CredentialExchangeError, match="invalid STS response"):
            await provider.get()


@pytest.mark.asyncio
async def test_resource_sts_reloads_rotated_sa_token_on_401(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token-old", encoding="utf-8")
    calls: list[str] = []
    expiration = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    def handler(request: httpx.Request) -> httpx.Response:
        authorization = request.headers["Authorization"]
        calls.append(authorization)
        if authorization == "Bearer sa-token-old":
            token_path.write_text("sa-token-new", encoding="utf-8")
            return httpx.Response(401)
        return httpx.Response(
            200,
            json={
                "access_key_id": "ak",
                "access_key_secret": "sk",
                "security_token": "sts",
                "expiration": expiration,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = ResourceSTSProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        assert (await provider.get()).access_key_id == "ak"

    assert calls == ["Bearer sa-token-old", "Bearer sa-token-new"]


@pytest.mark.asyncio
async def test_resource_sts_does_not_retry_unchanged_sa_token(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token", encoding="utf-8")
    calls = 0

    def unauthorized(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(unauthorized)) as http:
        provider = ResourceSTSProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        with pytest.raises(AuthenticationError):
            await provider.get()

    assert calls == 1


@pytest.mark.asyncio
async def test_resource_sts_single_flight_per_purpose(tmp_path: Path) -> None:
    token_path = tmp_path / "token"
    token_path.write_text("sa-token", encoding="utf-8")
    calls = 0
    expiration = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return httpx.Response(
            200,
            json={
                "access_key_id": "ak",
                "access_key_secret": "sk",
                "security_token": "sts",
                "expiration": expiration,
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        provider = ResourceSTSProvider(
            "https://controller.example.com",
            AgentSATokenProvider(token_path),
            http_client=http,
        )
        values = await asyncio.gather(*(provider.get() for _ in range(10)))

    assert [value.access_key_id for value in values] == ["ak"] * 10
    assert calls == 1
