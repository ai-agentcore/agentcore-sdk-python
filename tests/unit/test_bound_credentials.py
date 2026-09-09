from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from agentcore.auth.bound_credentials import AgentIdentityDataTransport, AsyncBoundCredentials
from agentcore.auth.resource_sts import ResourceCredential
from agentcore.controlplane.client import CredentialMetadata
from agentcore.errors import (
    CredentialExchangeError,
    WorkloadAccessTokenRejectedError,
)


class FakeWAT:
    async def get(self) -> str:
        return "wat"

    async def invalidate(self, token=None):  # type: ignore[no-untyped-def]
        return None


class FakeSTS:
    def __init__(self) -> None:
        self.purposes = []

    async def get(self, purpose):  # type: ignore[no-untyped-def]
        self.purposes.append(purpose)
        return ResourceCredential(
            credential_type="ram_sts",
            access_key_id="ak",
            access_key_secret="sk",
            security_token="sts",
            expiration=datetime.now(timezone.utc) + timedelta(hours=1),
        )


class FakeTransport:
    async def get_api_key(self, **kwargs):  # type: ignore[no-untyped-def]
        assert kwargs["workload_access_token"] == "wat"
        assert kwargs["provider_name"] == "ws-1-github"
        assert kwargs["region_id"] == "cn-hangzhou"
        return "third-party-secret"


class FakeControlPlane:
    async def resolve_credential(self, name):
        return CredentialMetadata("cred-1", name, "apiKey", "ALL", ())


@pytest.mark.asyncio
async def test_bound_credential_derives_provider_from_workspace_and_name() -> None:
    sts = FakeSTS()
    credentials = AsyncBoundCredentials(
        "ws-1",
        "cn-hangzhou",
        FakeWAT(),
        sts,
        transport=FakeTransport(),
        control_plane=FakeControlPlane(),
    )

    value = await credentials.get("github")

    assert value.value == "third-party-secret"
    assert "third-party-secret" not in repr(value)
    assert value.provider_name == "ws-1-github"
    assert sts.purposes == ["agentidentitydata"]


@pytest.mark.asyncio
async def test_bound_credential_rejects_empty_name() -> None:
    credentials = AsyncBoundCredentials(
        "ws-1",
        "cn-hangzhou",
        FakeWAT(),
        FakeSTS(),
        transport=FakeTransport(),
        control_plane=FakeControlPlane(),
    )

    with pytest.raises(ValueError):
        await credentials.get("  ")


@pytest.mark.asyncio
async def test_bound_credential_refreshes_rejected_wat_and_retries_once() -> None:
    class RotatingWAT:
        def __init__(self) -> None:
            self.tokens = iter(("wat-expired", "wat-current"))
            self.invalidated: list[str | None] = []

        async def get(self) -> str:
            return next(self.tokens)

        async def invalidate(self, token=None):  # type: ignore[no-untyped-def]
            self.invalidated.append(token)

    class RejectOnceTransport:
        def __init__(self) -> None:
            self.tokens: list[str] = []

        async def get_api_key(self, **kwargs):  # type: ignore[no-untyped-def]
            token = kwargs["workload_access_token"]
            self.tokens.append(token)
            if token == "wat-expired":
                raise WorkloadAccessTokenRejectedError("WAT expired")
            return "third-party-secret"

    wat = RotatingWAT()
    sts = FakeSTS()
    transport = RejectOnceTransport()
    credentials = AsyncBoundCredentials(
        "ws-1",
        "cn-hangzhou",
        wat,
        sts,
        transport=transport,
        control_plane=FakeControlPlane(),
    )

    value = await credentials.get("github")

    assert value.value == "third-party-secret"
    assert transport.tokens == ["wat-expired", "wat-current"]
    assert wat.invalidated == ["wat-expired"]
    assert sts.purposes == ["agentidentitydata"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_code",
    ["WORKLOAD_ACCESS_TOKEN_EXPIRED", "WORKLOAD_ACCESS_TOKEN_INVALID"],
)
async def test_agent_identity_transport_classifies_explicit_wat_rejection(
    monkeypatch: pytest.MonkeyPatch,
    error_code: str,
) -> None:
    pytest.importorskip("alibabacloud_agentidentitydata20251127.client")
    from alibabacloud_agentidentitydata20251127.client import Client
    from darabonba.exceptions import TeaException

    async def reject_wat(client, request):  # type: ignore[no-untyped-def]
        raise TeaException({"code": error_code, "message": "rejected", "data": {}})

    monkeypatch.setattr(Client, "get_resource_apikey_async", reject_wat)
    transport = AgentIdentityDataTransport()

    with pytest.raises(WorkloadAccessTokenRejectedError):
        await transport.get_api_key(
            region_id="cn-hangzhou",
            credential=ResourceCredential(
                credential_type="ram_sts",
                access_key_id="ak",
                access_key_secret="sk",
                security_token="sts",
                expiration=datetime.now(timezone.utc) + timedelta(hours=1),
            ),
            provider_name="github-provider",
            workload_access_token="wat-expired",
        )


@pytest.mark.asyncio
async def test_bound_credential_does_not_retry_non_wat_exchange_failure() -> None:
    class CountingWAT(FakeWAT):
        def __init__(self) -> None:
            self.get_calls = 0
            self.invalidated: list[str | None] = []

        async def get(self) -> str:
            self.get_calls += 1
            return "wat"

        async def invalidate(self, token=None):  # type: ignore[no-untyped-def]
            self.invalidated.append(token)

    class FailingTransport:
        async def get_api_key(self, **kwargs):  # type: ignore[no-untyped-def]
            raise CredentialExchangeError("provider is unavailable")

    wat = CountingWAT()
    credentials = AsyncBoundCredentials(
        "ws-1",
        "cn-hangzhou",
        wat,
        FakeSTS(),
        transport=FailingTransport(),
        control_plane=FakeControlPlane(),
    )

    with pytest.raises(CredentialExchangeError):
        await credentials.get("github")

    assert wat.get_calls == 1
    assert wat.invalidated == []


@pytest.mark.asyncio
async def test_bound_credential_retries_wat_rejection_only_once() -> None:
    class RotatingWAT:
        def __init__(self) -> None:
            self.tokens = iter(("wat-1", "wat-2", "wat-3"))
            self.invalidated: list[str | None] = []

        async def get(self) -> str:
            return next(self.tokens)

        async def invalidate(self, token=None):  # type: ignore[no-untyped-def]
            self.invalidated.append(token)

    class RejectingTransport:
        def __init__(self) -> None:
            self.tokens: list[str] = []

        async def get_api_key(self, **kwargs):  # type: ignore[no-untyped-def]
            self.tokens.append(kwargs["workload_access_token"])
            raise WorkloadAccessTokenRejectedError("WAT rejected")

    wat = RotatingWAT()
    transport = RejectingTransport()
    credentials = AsyncBoundCredentials(
        "ws-1",
        "cn-hangzhou",
        wat,
        FakeSTS(),
        transport=transport,
        control_plane=FakeControlPlane(),
    )

    with pytest.raises(WorkloadAccessTokenRejectedError):
        await credentials.get("github")

    assert transport.tokens == ["wat-1", "wat-2"]
    assert wat.invalidated == ["wat-1"]
