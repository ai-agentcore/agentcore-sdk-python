from __future__ import annotations

import io
import zipfile
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from agentcore.auth import AccessKeyCredential
from agentcore.auth.resource_sts import ResourceCredential
from agentcore.controlplane import AgentCoreControlPlane
from agentcore.errors import (
    ConfigError,
    InvocationError,
    MCPServerNotFoundError,
    ModelConnectionNotFoundError,
    ResourceNotConfiguredError,
)


class FakeSTS:
    def __init__(self) -> None:
        self.purposes: list[str | None] = []

    async def get(self, purpose: str | None = None) -> ResourceCredential:
        self.purposes.append(purpose)
        return ResourceCredential(
            access_key_id="ak",
            access_key_secret="sk",
            security_token="sts",
            expiration=datetime.now(timezone.utc) + timedelta(hours=1),
        )


class FakeClient:
    def __init__(self) -> None:
        self.connection_requests = []
        self.model_requests = []
        self.mcp_requests = []

    async def list_model_connections_async(self, workspace_id, request):  # type: ignore[no-untyped-def]
        self.connection_requests.append((workspace_id, request))
        return SimpleNamespace(
            body=SimpleNamespace(
                items=[
                    SimpleNamespace(
                        connection_id="mc-1",
                        name="production",
                        protocol="OpenAI/v1",
                        provider_type="CUSTOM",
                    )
                ],
                next_token=None,
            )
        )

    async def list_models_async(self, workspace_id, request):  # type: ignore[no-untyped-def]
        self.model_requests.append((workspace_id, request))
        return SimpleNamespace(
            body=SimpleNamespace(
                items=[
                    SimpleNamespace(
                        model_id="model-1",
                        model_name="qwen-plus",
                        connection_id="mc-1",
                        context_size=32768,
                        max_tokens=8192,
                        capabilities=SimpleNamespace(tool_call=True, vision=False),
                    )
                ],
                next_token=None,
            )
        )

    async def list_mcps_async(self, workspace_id, request):  # type: ignore[no-untyped-def]
        self.mcp_requests.append((workspace_id, request))
        return SimpleNamespace(
            body=SimpleNamespace(
                items=[
                    SimpleNamespace(
                        mcp_server_id="mcp-1",
                        name="web-search",
                        protocol="STREAMABLE_HTTP",
                        type="HOSTED",
                        status="READY",
                    )
                ],
                next_token=None,
            )
        )


def _credential() -> ResourceCredential:
    return ResourceCredential(
        access_key_id="ak",
        access_key_secret="sk",
        security_token="sts",
        expiration=datetime.now(timezone.utc) + timedelta(hours=1),
    )


def test_control_plane_uses_generated_regional_endpoint_by_default() -> None:
    control_plane = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        resource_sts=FakeSTS(),
    )

    client = control_plane._new_client(_credential())

    assert client._endpoint == "agentcore.cn-hangzhou.aliyuncs.com"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "credential",
    [
        AccessKeyCredential("ak", "sk"),
        AccessKeyCredential("sts-ak", "sts-sk", "sts-token"),
    ],
)
async def test_control_plane_uses_explicit_access_key_without_sts(
    credential: AccessKeyCredential,
) -> None:
    captured: list[AccessKeyCredential | ResourceCredential] = []
    client = FakeClient()
    control_plane = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        access_key_credential=credential,
        _client_factory=lambda value: captured.append(value) or client,
    )

    await control_plane.resolve_mcp("web-search")

    assert captured == [credential]


@pytest.mark.parametrize(
    ("value", "endpoint", "protocol"),
    [
        (
            "https://agentcore-vpc.cn-hangzhou.aliyuncs.com/",
            "agentcore-vpc.cn-hangzhou.aliyuncs.com",
            "https",
        ),
        ("control-plane.internal:8443", "control-plane.internal:8443", None),
    ],
)
def test_control_plane_accepts_endpoint_override(
    value: str,
    endpoint: str,
    protocol: str | None,
) -> None:
    control_plane = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        resource_sts=FakeSTS(),
        endpoint=value,
    )

    client = control_plane._new_client(_credential())

    assert client._endpoint == endpoint
    assert client._protocol == protocol


def test_control_plane_rejects_endpoint_with_path() -> None:
    with pytest.raises(ConfigError, match="endpoint"):
        AgentCoreControlPlane(
            workspace_id="ws-1",
            region_id="cn-hangzhou",
            resource_sts=FakeSTS(),
            endpoint="https://agentcore.example.com/api",
        )


@pytest.mark.asyncio
async def test_control_plane_resolves_exact_model_and_mcp_names() -> None:
    sts = FakeSTS()
    client = FakeClient()
    control_plane = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        resource_sts=sts,
        _client_factory=lambda _: client,
    )

    model = await control_plane.resolve_model("production", "qwen-plus")
    mcp = await control_plane.resolve_mcp("web-search")

    assert model.connection_id == "mc-1"
    assert model.model_name == "qwen-plus"
    assert model.context_size == 32768
    assert model.capabilities == {"tool_call": True, "vision": False}
    assert mcp.mcp_server_id == "mcp-1"
    assert client.connection_requests[0][1].name == "production"
    assert client.connection_requests[0][1].search_type == "accurate"
    assert client.mcp_requests[0][1].search_type == "accurate"
    assert sts.purposes == ["highcode_sdk", "highcode_sdk"]


@pytest.mark.asyncio
async def test_control_plane_rejects_non_exact_connection_result() -> None:
    class FuzzyClient(FakeClient):
        async def list_model_connections_async(self, workspace_id, request):  # type: ignore[no-untyped-def]
            response = await super().list_model_connections_async(workspace_id, request)
            response.body.items[0].name = "production-copy"
            return response

    control_plane = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        resource_sts=FakeSTS(),
        _client_factory=lambda _: FuzzyClient(),
    )

    with pytest.raises(ModelConnectionNotFoundError, match="production"):
        await control_plane.resolve_model("production", "qwen-plus")


@pytest.mark.asyncio
async def test_control_plane_keeps_missing_model_distinct_from_missing_connection() -> None:
    class MissingModelClient(FakeClient):
        async def list_models_async(self, workspace_id, request):  # type: ignore[no-untyped-def]
            response = await super().list_models_async(workspace_id, request)
            response.body.items = []
            return response

    control_plane = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        resource_sts=FakeSTS(),
        _client_factory=lambda _: MissingModelClient(),
    )

    with pytest.raises(ResourceNotConfiguredError, match="qwen-plus") as raised:
        await control_plane.resolve_model("production", "qwen-plus")

    assert not isinstance(raised.value, ModelConnectionNotFoundError)


@pytest.mark.asyncio
async def test_control_plane_rejects_duplicate_exact_connections_as_invalid() -> None:
    class DuplicateConnectionClient(FakeClient):
        async def list_model_connections_async(self, workspace_id, request):  # type: ignore[no-untyped-def]
            response = await super().list_model_connections_async(workspace_id, request)
            response.body.items.append(response.body.items[0])
            return response

    control_plane = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        resource_sts=FakeSTS(),
        _client_factory=lambda _: DuplicateConnectionClient(),
    )

    with pytest.raises(ConfigError, match="more than once"):
        await control_plane.resolve_model("production", "qwen-plus")


@pytest.mark.asyncio
async def test_control_plane_reports_missing_mcp_by_resource_type() -> None:
    class MissingMCPClient(FakeClient):
        async def list_mcps_async(self, workspace_id, request):  # type: ignore[no-untyped-def]
            response = await super().list_mcps_async(workspace_id, request)
            response.body.items = []
            return response

    control_plane = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        resource_sts=FakeSTS(),
        _client_factory=lambda _: MissingMCPClient(),
    )

    with pytest.raises(MCPServerNotFoundError, match="web-search"):
        await control_plane.resolve_mcp("web-search")


@pytest.mark.asyncio
async def test_control_plane_wraps_generated_sdk_failures() -> None:
    class FailingClient(FakeClient):
        async def list_model_connections_async(self, workspace_id, request):  # type: ignore[no-untyped-def]
            raise RuntimeError("generated SDK implementation detail")

    control_plane = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        resource_sts=FakeSTS(),
        _client_factory=lambda _: FailingClient(),
    )

    with pytest.raises(InvocationError, match="control-plane request failed") as raised:
        await control_plane.resolve_model("production", "qwen-plus")

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "implementation detail" not in str(raised.value)


@pytest.mark.asyncio
async def test_control_plane_rejects_invalid_model_capabilities() -> None:
    class InvalidCapabilitiesClient(FakeClient):
        async def list_models_async(self, workspace_id, request):  # type: ignore[no-untyped-def]
            response = await super().list_models_async(workspace_id, request)
            response.body.items[0].capabilities = "invalid"
            return response

    control_plane = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        resource_sts=FakeSTS(),
        _client_factory=lambda _: InvalidCapabilitiesClient(),
    )

    with pytest.raises(ConfigError, match="capabilities"):
        await control_plane.resolve_model("production", "qwen-plus")


@pytest.mark.asyncio
async def test_control_plane_downloads_skill_archive_via_oss() -> None:
    archive = _skill_archive()

    class SkillClient:
        async def download_skill_version_via_oss_async(  # type: ignore[no-untyped-def]
            self, workspace_id, name, version, request
        ):
            assert (workspace_id, name, version) == ("ws-1", "review", "1.0.0")
            return SimpleNamespace(body=SimpleNamespace(data="https://oss.example.com/review.zip"))

    def download(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://oss.example.com/review.zip"
        assert "authorization" not in request.headers
        return httpx.Response(200, content=archive)

    sts = FakeSTS()
    async with httpx.AsyncClient(transport=httpx.MockTransport(download)) as http:
        control_plane = AgentCoreControlPlane(
            workspace_id="ws-1",
            region_id="cn-hangzhou",
            resource_sts=sts,
            _client_factory=lambda _: SkillClient(),
            _http_client=http,
        )

        skill = await control_plane.get_skill("review", "1.0.0")

    assert skill.version == "1.0.0"
    assert skill.archive == archive
    assert sts.purposes == ["highcode_sdk"]


@pytest.mark.asyncio
async def test_control_plane_rejects_invalid_skill_download_url() -> None:
    class InvalidSkillClient:
        async def download_skill_version_via_oss_async(  # type: ignore[no-untyped-def]
            self, workspace_id, name, version, request
        ):
            return SimpleNamespace(body=SimpleNamespace(data="oss://bucket/review.zip"))

    control_plane = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        resource_sts=FakeSTS(),
        _client_factory=lambda _: InvalidSkillClient(),
    )

    with pytest.raises(ConfigError, match="download URL"):
        await control_plane.get_skill("review", "1.0.0")


@pytest.mark.asyncio
async def test_control_plane_latest_skill_tolerates_missing_update_time() -> None:
    class SkillClient:
        async def get_skill_detail_async(  # type: ignore[no-untyped-def]
            self, workspace_id, name, request
        ):
            return SimpleNamespace(
                body=SimpleNamespace(
                    data=SimpleNamespace(
                        labels={},
                        versions=[
                            SimpleNamespace(
                                version="1.0.0",
                                status="ONLINE",
                                update_time=None,
                            ),
                            SimpleNamespace(
                                version="1.1.0",
                                status="ONLINE",
                                update_time=2,
                            ),
                        ],
                    )
                )
            )

        async def download_skill_version_via_oss_async(  # type: ignore[no-untyped-def]
            self, workspace_id, name, version, request
        ):
            assert version == "1.1.0"
            return SimpleNamespace(body=SimpleNamespace(data="https://oss.example.com/review.zip"))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=_skill_archive()))
    ) as http:
        control_plane = AgentCoreControlPlane(
            workspace_id="ws-1",
            region_id="cn-hangzhou",
            resource_sts=FakeSTS(),
            _client_factory=lambda _: SkillClient(),
            _http_client=http,
        )

        skill = await control_plane.get_skill("review")

    assert skill.version == "1.1.0"


def _skill_archive() -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        archive.writestr("SKILL.md", "# Review\n")
    return result.getvalue()
