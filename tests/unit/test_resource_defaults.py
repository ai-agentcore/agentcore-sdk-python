from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from agentcore._resource_defaults import ResourceDefaults
from agentcore.client import AgentCore, AsyncAgentCore
from agentcore.errors import ConfigError
from agentcore.skill import AsyncSkills
from agentcore.templates import AsyncTemplates


def test_defaults_are_paired_and_explicit_names_do_not_inherit_versions():
    defaults = ResourceDefaults(
        {
            "AGENTCORE_MODEL_CONNECTION_NAME": "default-mc",
            "AGENTCORE_MODEL_NAME": "default-model",
            "AGENTCORE_AGENT_SPEC_NAME": "default-spec",
            "AGENTCORE_AGENT_SPEC_VERSION": "1.0",
        }
    )
    assert defaults.model(None, None) == ("default-mc", "default-model")
    assert defaults.model(None, "override") == ("default-mc", "override")
    assert defaults.model("explicit-mc", None) == ("explicit-mc", None)
    assert defaults.agent_spec(None, None) == ("default-spec", "1.0")
    assert defaults.agent_spec(None, "2.0") == ("default-spec", "2.0")
    assert defaults.agent_spec("explicit-spec", None) == ("explicit-spec", None)


def test_json_lists_preserve_order_and_each_skill_version():
    defaults = ResourceDefaults(
        {
            "AGENTCORE_MCP_NAMES": '["a", "b"]',
            "AGENTCORE_SKILLS": '[{"name":"s1"},{"name":"s2","version":"2.0"}]',
        }
    )
    assert defaults.mcp_names() == ["a", "b"]
    assert defaults.skills() == [("s1", None), ("s2", "2.0")]
    assert ResourceDefaults({}).mcp_names() == []
    assert ResourceDefaults({}).skills() == []


@pytest.mark.parametrize(
    "key,value",
    [
        ("AGENTCORE_MCP_NAMES", "bad"),
        ("AGENTCORE_MCP_NAMES", "{}"),
        ("AGENTCORE_MCP_NAMES", '[""]'),
        ("AGENTCORE_MCP_NAMES", "[1]"),
        ("AGENTCORE_SKILLS", "null"),
        ("AGENTCORE_SKILLS", '["s"]'),
        ("AGENTCORE_SKILLS", '[{"name":"s","version":1}]'),
        ("AGENTCORE_SKILLS", '[{"version":"1"}]'),
    ],
)
def test_invalid_list_is_rejected_only_when_used(key, value):
    defaults = ResourceDefaults({key: value})
    with pytest.raises(ConfigError, match=key):
        defaults.mcp_names() if key == "AGENTCORE_MCP_NAMES" else defaults.skills()
    assert defaults.model("explicit", None) == ("explicit", None)


def test_missing_default_names_have_actionable_errors():
    defaults = ResourceDefaults({})
    with pytest.raises(ConfigError, match="AGENTCORE_MODEL_CONNECTION_NAME"):
        defaults.model(None, None)
    with pytest.raises(ConfigError, match="AGENTCORE_AGENT_SPEC_NAME"):
        defaults.agent_spec(None, None)


@pytest.mark.asyncio
async def test_core_default_model_uses_existing_resolution_and_cache(monkeypatch):
    monkeypatch.setenv("AGENTCORE_MODEL_CONNECTION_NAME", "mc")
    monkeypatch.setenv("AGENTCORE_MODEL_NAME", "model-a")
    async with AsyncAgentCore() as core:
        core._ensure_runtime = AsyncMock(return_value=SimpleNamespace(config=object()))
        resolve = AsyncMock(return_value=SimpleNamespace(model_name="model-a"))
        core._control_plane = SimpleNamespace(resolve_model=resolve)
        client = SimpleNamespace(aclose=AsyncMock())
        with patch("agentcore.client.AsyncModelClient.platform", return_value=client):
            assert await core.model() is client
            monkeypatch.setenv("AGENTCORE_MODEL_NAME", "changed-after-startup")
            assert await core.model() is client
            assert await core.model("other") is client
        assert resolve.call_args_list[0].args == ("mc", "model-a")
        assert resolve.call_args_list[1].args == ("other", None)


@pytest.mark.asyncio
async def test_bulk_loaders_delegate_to_single_resource_methods(monkeypatch):
    monkeypatch.setenv("AGENTCORE_MCP_NAMES", '["a","b"]')
    monkeypatch.setenv("AGENTCORE_SKILLS", '[{"name":"s1"},{"name":"s2","version":"2"}]')
    async with AsyncAgentCore() as core:
        core.mcp = AsyncMock(side_effect=["client-a", "client-b"])
        core.skills.managed = AsyncMock(side_effect=["skill-1", "skill-2"])
        assert await core.mcps_from_env() == ["client-a", "client-b"]
        assert await core.skills.from_env() == ["skill-1", "skill-2"]
        assert [call.args for call in core.mcp.call_args_list] == [("a",), ("b",)]
        assert core.skills.managed.call_args_list[1].kwargs == {"version": "2"}


@pytest.mark.asyncio
async def test_template_defaults_are_passed_to_existing_provider(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENTCORE_AGENT_SPEC_NAME", "spec")
    monkeypatch.setenv("AGENTCORE_AGENT_SPEC_VERSION", "v1")
    provider = SimpleNamespace(get_template=AsyncMock(side_effect=RuntimeError("provider reached")))
    runtime = AsyncMock(return_value=(provider, "ws-test"))
    templates = AsyncTemplates(runtime, AsyncSkills(None, None), workspace_dir=tmp_path)
    with pytest.raises(RuntimeError, match="provider reached"):
        await templates.get()
    provider.get_template.assert_awaited_once_with("spec", "v1")
    with pytest.raises(RuntimeError, match="provider reached"):
        await templates.get("explicit-spec")
    assert provider.get_template.call_args.args == ("explicit-spec", None)


def test_sync_facade_supports_bulk_resource_methods(monkeypatch):
    monkeypatch.setenv("AGENTCORE_MCP_NAMES", '["a"]')
    monkeypatch.setenv("AGENTCORE_SKILLS", '[{"name":"s"}]')
    with AgentCore() as core:
        async_core = core._get()
        async_core.mcp = AsyncMock(return_value=SimpleNamespace())
        async_core.skills.managed = AsyncMock(return_value="skill")
        assert len(core.mcps_from_env()) == 1
        assert core.skills.from_env() == ["skill"]


def test_sync_bulk_entry_can_be_the_first_call(monkeypatch):
    monkeypatch.delenv("AGENTCORE_MCP_NAMES", raising=False)
    core = AgentCore()
    try:
        assert core.mcps_from_env() == []
    finally:
        core.close()


def test_sync_model_and_template_forward_omitted_names():
    with AgentCore() as core:
        async_core = core._get()
        async_core.model = AsyncMock(return_value=SimpleNamespace())
        async_core.templates.get = AsyncMock(
            return_value=SimpleNamespace(
                name="spec",
                version="v1",
                instruction="prompt",
                load_skills=AsyncMock(return_value=[]),
            )
        )
        core.model()
        template = core.templates.get()
        assert template.instruction == "prompt"
        assert template.load_skills() == []
        async_core.model.assert_awaited_once_with(None, model=None)
        async_core.templates.get.assert_awaited_once_with(None, version=None)


def test_framework_convenience_entry_resolves_environment_defaults(
    monkeypatch, agent_config_path, model_descriptor
):
    from agentcore.auth import AccessKeyCredential
    from agentcore.integrations import _shared

    monkeypatch.setenv("AGENTCORE_MODEL_CONNECTION_NAME", "default-mc")
    monkeypatch.setenv("AGENTCORE_MODEL_NAME", "default-model")
    resolve = AsyncMock(return_value=model_descriptor)
    monkeypatch.setattr(_shared.AgentCoreControlPlane, "resolve_model", resolve)
    _shared.framework_model_settings(
        config_path=agent_config_path,
        access_key_credential=AccessKeyCredential("fixture-ak", "fixture-sk"),
    )
    assert resolve.call_args.args == ("default-mc", "default-model")
