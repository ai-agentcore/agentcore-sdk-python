from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentcore.auth import AccessKeyCredential
from agentcore.auth.bound_credentials import AsyncBoundCredentials, BoundCredential
from agentcore.controlplane.client import AgentCoreControlPlane, CredentialMetadata
from agentcore.errors import ConfigError, ResourceNotConfiguredError


@pytest.mark.asyncio
async def test_list_credentials_common_request_paginates_and_matches_exact_name():
    call = AsyncMock(
        side_effect=[
            {"body": {"items": [{"name": "key-other"}], "nextToken": "next"}},
            {
                "body": {
                    "items": [
                        {
                            "name": "key",
                            "credentialId": "cred-1",
                            "credentialType": "mcpHeader",
                            "resourceScope": "SPECIFIED",
                            "resourceRefs": [{"resourceType": "mcpServer", "resourceId": "mcp-1"}],
                        }
                    ]
                }
            },
        ]
    )
    control = AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        access_key_credential=AccessKeyCredential("ak", "sk"),
        _client_factory=lambda _: SimpleNamespace(call_api_async=call),
    )
    metadata = await control.resolve_credential("key")
    assert metadata.credential_type == "mcpHeader"
    assert metadata.resource_refs == (("mcpServer", "mcp-1"),)
    first, second = call.call_args_list
    assert first.args[0].action == "ListCredentials"
    assert first.args[0].pathname == "/workspaces/ws-1/credentials"
    assert first.args[1].query == {"name": "key", "maxResults": "100"}
    assert second.args[1].query["nextToken"] == "next"


@pytest.mark.asyncio
@pytest.mark.parametrize("items", [[], [{"name": "key-other"}], [{"name": "key"}] * 2])
async def test_credential_lookup_does_not_accept_missing_or_ambiguous_name(items):
    control = AgentCoreControlPlane(
        workspace_id="ws",
        region_id="cn-hangzhou",
        access_key_credential=AccessKeyCredential("ak", "sk"),
        _client_factory=lambda _: SimpleNamespace(
            call_api_async=AsyncMock(return_value={"body": {"items": items}})
        ),
    )
    with pytest.raises((ConfigError, ResourceNotConfiguredError)):
        await control.resolve_credential("key")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,scope,refs,allowed",
    [
        ("mcpHeader", "ALL", (), True),
        ("mcpHeader", "SPECIFIED", (("mcpServer", "mcp-1"),), True),
        ("apiKey", "ALL", (), False),
        ("mcpHeader", "SPECIFIED", (("mcpServer", "other"),), False),
        ("mcpHeader", "SPECIFIED", (("modelConnection", "mcp-1"),), False),
        ("mcpHeader", "", (), False),
    ],
)
async def test_mcp_scope_checked_before_secret_fetch(kind, scope, refs, allowed):
    metadata = CredentialMetadata("cred-1", "key", kind, scope, refs)
    wat = SimpleNamespace(get=AsyncMock(return_value="wat"))
    sts = SimpleNamespace(get=AsyncMock(return_value=object()))
    transport = SimpleNamespace(
        get_api_key=AsyncMock(return_value='{"headers":[{"name":"X-Key","value":"secret"}]}')
    )
    control = SimpleNamespace(resolve_credential=AsyncMock(return_value=metadata))
    credentials = AsyncBoundCredentials(
        "ws", "cn-hangzhou", wat, sts, control_plane=control, transport=transport
    )
    if allowed:
        result = await credentials._get("key", mcp_server_id="mcp-1")
        assert result.as_headers() == {"x-key": "secret"}
        assert result.metadata is metadata
        assert "secret" not in repr(result)
    else:
        with pytest.raises(ConfigError):
            await credentials._get("key", mcp_server_id="mcp-1")
        wat.get.assert_not_called()
        sts.get.assert_not_called()
        transport.get_api_key.assert_not_called()


@pytest.mark.parametrize(
    "value",
    [
        "not-json",
        '{"headers":[]}',
        '{"headers":[{"name":"X-Key"}]}',
        '{"headers":[{"name":"X-Key","value":"a"},{"name":"x-key","value":"b"}]}',
    ],
)
def test_header_credential_parser_rejects_invalid_payload_without_exposing_secret(value):
    credential = BoundCredential(
        "provider", value, CredentialMetadata("id", "key", "mcpHeader", "ALL", ())
    )
    with pytest.raises(ConfigError) as failure:
        credential.as_headers()
    assert value not in str(failure.value)


def test_api_key_is_not_implicitly_parsed_as_headers():
    credential = BoundCredential(
        "provider", "raw-key", CredentialMetadata("id", "key", "apiKey", "ALL", ())
    )
    assert credential.value == "raw-key"
    with pytest.raises(ConfigError):
        credential.as_headers()
