from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agentcore import AsyncAgentCore
from agentcore.auth.bound_credentials import BoundCredential
from agentcore.controlplane import MCPDescriptor
from agentcore.controlplane.client import CredentialMetadata
from agentcore.errors import ConfigError
from agentcore.mcp import AsyncMCPClient
from agentcore.runtime.config import load_agent_config


@pytest.mark.asyncio
async def test_managed_headers_are_copied_and_do_not_replace_auth(
    agent_config_path: Path,
    mcp_descriptor: MCPDescriptor,
) -> None:
    headers = {"X-API-Key": "custom-secret"}
    client = AsyncMCPClient.platform(
        load_agent_config(agent_config_path),
        mcp_descriptor,
        headers=headers,
    )
    headers["X-API-Key"] = "changed"
    assert await client._connection.resolve_headers() == {
        "Authorization": "Bearer gateway-secret",
        "x-api-key": "custom-secret",
    }
    assert "custom-secret" not in repr(client._connection)
    await client.aclose()


@pytest.mark.parametrize(
    "headers",
    [
        {"aUtHoRiZaTiOn": "replace"},
        {"MCP-Session-ID": "replace"},
        {"Host": "other-host"},
        {"X-Test": "one", "x-test": "two"},
        {"X-Test": "unsafe\r\nvalue"},
    ],
)
def test_managed_headers_reject_conflicts_and_invalid_values(
    agent_config_path: Path,
    mcp_descriptor: MCPDescriptor,
    headers: dict[str, str],
) -> None:
    with pytest.raises(ConfigError):
        AsyncMCPClient.platform(
            load_agent_config(agent_config_path),
            mcp_descriptor,
            headers=headers,
        )


@pytest.mark.asyncio
async def test_core_mcp_cache_distinguishes_headers(
    agent_config_path: Path,
    mcp_descriptor: MCPDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = AsyncAgentCore.auto(config_path=agent_config_path)
    await core._ensure_runtime()

    async def resolve(_name: str) -> MCPDescriptor:
        return mcp_descriptor

    monkeypatch.setattr(core, "_control_plane", SimpleNamespace(resolve_mcp=resolve, aclose=_close))
    try:
        first = await core.mcp("search", headers={"X-User": "a"})
        same = await core.mcp("search", headers={"x-user": "a"})
        second = await core.mcp("search", headers={"X-User": "b"})
        default = await core.mcp("search")
        bound_a = await core.mcp("search", credential_name="key-a")
        bound_b = await core.mcp("search", credential_name="key-b")
        assert bound_a is not bound_b and bound_a is not default
        assert await core.mcp("search", credential_name=" key-a ") is bound_a
        get = AsyncMock(
            return_value=BoundCredential(
                "provider",
                '{"headers":[{"name":"X-Key","value":"secret"}]}',
                CredentialMetadata("id", "key-a", "mcpHeader", "ALL", ()),
            )
        )
        monkeypatch.setattr(core.credentials, "_get", get)
        assert (await bound_a._connection.resolve_headers())["x-key"] == "secret"
        get.assert_awaited_once_with("key-a", mcp_server_id=mcp_descriptor.mcp_server_id)
        assert first is same
        assert first is not second and first is not default
        assert (await second._connection.resolve_headers())["x-user"] == "b"
        assert "x-user" not in await default._connection.resolve_headers()
    finally:
        await core.aclose()


async def _close() -> None:
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize("bound", [{"aUtHoRiZaTiOn": "replace"}, {"X-API-Key": "bound"}])
async def test_bound_header_conflicts_cannot_override_platform_or_credential(
    agent_config_path,
    mcp_descriptor,
    bound,
):
    async def credentials():
        return bound

    client = AsyncMCPClient.platform(
        load_agent_config(agent_config_path),
        mcp_descriptor,
        headers={"x-api-key": "custom"},
        credential_headers_provider=credentials,
    )
    with pytest.raises(ConfigError, match="protected"):
        await client._connection.resolve_headers()
    await client.aclose()
