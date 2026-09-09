"""Real local HTTP serialization/signing; no cloud credentials or services."""

import json
import socket
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiohttp import web

from agentcore.auth import AccessKeyCredential
from agentcore.auth.bound_credentials import AsyncBoundCredentials
from agentcore.auth.resource_sts import ResourceCredential
from agentcore.controlplane import AgentCoreControlPlane


@pytest.mark.asyncio
async def test_list_credentials_common_request_over_http():
    queries = []

    async def handle(request):
        assert request.headers["x-acs-action"] == "ListCredentials"
        assert request.headers["Authorization"]
        assert request.headers["x-acs-security-token"] == "test-sts"
        queries.append(dict(request.query))
        if "nextToken" not in request.query:
            return web.json_response({"items": [{"name": "key-other"}], "nextToken": "page2"})
        return web.json_response(
            {
                "items": [
                    {
                        "name": "key",
                        "credentialId": "id",
                        "credentialType": "mcpHeader",
                        "resourceScope": "SPECIFIED",
                        "resourceRefs": [{"resourceType": "mcpServer", "resourceId": "mcp-1"}],
                    }
                ]
            }
        )

    app = web.Application()
    app.router.add_get("/workspaces/ws-1/credentials", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        endpoint = f"http://127.0.0.1:{listener.getsockname()[1]}"
        await web.SockSite(runner, listener).start()
        control = AgentCoreControlPlane(
            workspace_id="ws-1",
            region_id="cn-hangzhou",
            endpoint=endpoint,
            access_key_credential=AccessKeyCredential("test-ak", "test-sk", "test-sts"),
        )
        try:
            metadata = await control.resolve_credential("key")
            assert metadata.resource_scope == "SPECIFIED"
            assert metadata.resource_refs == (("mcpServer", "mcp-1"),)
            assert queries == [
                {"name": "key", "maxResults": "100"},
                {"name": "key", "maxResults": "100", "nextToken": "page2"},
            ]
        finally:
            await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("credential_type", ["apiKey", "mcpHeader"])
async def test_list_then_identity_returns_real_secret_without_get_credential(
    credential_type,
    monkeypatch,
):
    identity_module = pytest.importorskip("alibabacloud_agentidentitydata20251127.client")
    actions = []
    secret = (
        "real-api-key"
        if credential_type == "apiKey"
        else json.dumps({"headers": [{"name": "X-API-Key", "value": "real-header-value"}]})
    )

    async def handle(request):
        action = request.headers["x-acs-action"]
        actions.append(action)
        assert request.headers["Authorization"]
        assert request.headers["x-acs-security-token"] == "test-sts"
        if action == "ListCredentials":
            assert request.method == "GET"
            assert request.path == "/workspaces/ws-1/credentials"
            assert request.query["name"] == "my-key"
            return web.json_response(
                {
                    "items": [
                        {
                            "workspaceId": "ws-1",
                            "credentialId": "credential-id-not-provider-name",
                            "name": "my-key",
                            "credentialType": credential_type,
                            "credentialMetadata": json.dumps({"apiKey": "********"})
                            if credential_type == "apiKey"
                            else json.dumps({"headers": [{"name": "X-API-Key"}]}),
                            "resourceScope": "SPECIFIED"
                            if credential_type == "mcpHeader"
                            else "ALL",
                            "resourceRefs": [
                                {
                                    "resourceType": "mcpServer",
                                    "resourceId": "mcp-1",
                                    "resourceName": "search",
                                }
                            ]
                            if credential_type == "mcpHeader"
                            else [],
                            "boundAgentsCounts": 1,
                        }
                    ]
                }
            )
        assert action == "GetResourceAPIKey"
        body = await request.post()
        assert body["ResourceCredentialProviderName"] == "ws-1-my-key"
        assert body["WorkloadAccessToken"] == "test-wat"
        return web.json_response({"APIKey": secret})

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        endpoint = f"127.0.0.1:{listener.getsockname()[1]}"
        await web.SockSite(runner, listener).start()
        call_api = identity_module.Client.call_api_async

        async def local_identity(client, params, request, runtime):
            assert client._endpoint == "agentidentitydata.cn-hangzhou.aliyuncs.com"
            client._endpoint, client._protocol = endpoint, "http"
            return await call_api(client, params, request, runtime)

        monkeypatch.setattr(identity_module.Client, "call_api_async", local_identity)
        control = AgentCoreControlPlane(
            workspace_id="ws-1",
            region_id="cn-hangzhou",
            endpoint=f"http://{endpoint}",
            access_key_credential=AccessKeyCredential("test-ak", "test-sk", "test-sts"),
        )
        sts = SimpleNamespace(
            get=AsyncMock(
                return_value=ResourceCredential(
                    access_key_id="test-ak",
                    access_key_secret="test-sk",
                    security_token="test-sts",
                    expiration=datetime.now(timezone.utc) + timedelta(hours=1),
                )
            )
        )
        credentials = AsyncBoundCredentials(
            "ws-1",
            "cn-hangzhou",
            SimpleNamespace(get=AsyncMock(return_value="test-wat")),
            sts,
            control_plane=control,
        )
        try:
            result = (
                await credentials.get(" my-key ")
                if credential_type == "apiKey"
                else (await credentials._get(" my-key ", mcp_server_id="mcp-1"))
            )
            assert result.provider_name == "ws-1-my-key"
            assert result.value == secret
            if credential_type == "mcpHeader":
                assert result.as_headers() == {"x-api-key": "real-header-value"}
            assert actions == ["ListCredentials", "GetResourceAPIKey"]
            sts.get.assert_awaited_once_with("agentidentitydata")
            assert secret not in repr(result)
        finally:
            await runner.cleanup()
