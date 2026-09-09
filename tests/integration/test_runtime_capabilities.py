from __future__ import annotations

import io
import zipfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentcore import AsyncAgentCore
from agentcore.controlplane import MCPDescriptor, ModelDescriptor, SkillArtifact
from agentcore.mcp import MCPConnection
from agentcore.runtime.context import RequestContext, use_context
from agentcore.skill import AsyncSkills


class StubControlPlane:
    def __init__(
        self,
        model_descriptor: ModelDescriptor,
        mcp_descriptor: MCPDescriptor,
    ) -> None:
        self.model_descriptor = model_descriptor
        self.mcp_descriptor = mcp_descriptor

    async def resolve_model(self, resource_name: str, model: str | None):  # type: ignore[no-untyped-def]
        return self.model_descriptor

    async def resolve_mcp(self, name: str):  # type: ignore[no-untyped-def]
        return self.mcp_descriptor

    async def get_skill(self, name: str, version: str | None = None) -> SkillArtifact:
        skill_md = "---\nname: review\n---\nReview carefully.\n"
        package = io.BytesIO()
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr(f"{name}/SKILL.md", skill_md)
        return SkillArtifact(version or "1.0.0", package.getvalue())


class StubMCPSession:
    async def list_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search",
                "description": "Search",
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            }
        ]

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        return {"name": name, "query": arguments["query"]}


@pytest.mark.asyncio
async def test_async_core_model_mcp_and_managed_skill_share_runtime_projection(
    agent_config_path: Path,
    tmp_path: Path,
    model_descriptor: ModelDescriptor,
    mcp_descriptor: MCPDescriptor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def model_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    async def handle_model_request(
        _transport: httpx.AsyncHTTPTransport,
        request: httpx.Request,
    ) -> httpx.Response:
        return model_handler(request)

    @asynccontextmanager
    async def mcp_factory(_connection: MCPConnection) -> AsyncIterator[StubMCPSession]:
        yield StubMCPSession()

    monkeypatch.setattr("agentcore.mcp.client._official_session", mcp_factory)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport,
        "handle_async_request",
        handle_model_request,
    )
    core = AsyncAgentCore(agent_config_path, skill_workspace_dir=tmp_path / "skills")
    await core._ensure_runtime()
    control_plane = StubControlPlane(model_descriptor, mcp_descriptor)
    core._control_plane = control_plane  # type: ignore[assignment]
    core._skills = AsyncSkills(
        control_plane,
        "workspace-a",
        workspace_dir=tmp_path / "skills",
    )
    async with core:
        with use_context(RequestContext({"x-agentcore-session-id": "session-a"})):
            model_client = await core.model("model-service-a", model="qwen-plus")
            model_response = await model_client.invoke([{"role": "user", "content": "hello"}])
            mcp_client = await core.mcp("search")
            tools = await mcp_client.tools()
            tool_response = await tools[0].ainvoke({"query": "agentcore"})
            skill = await core.skills.managed("review", version="1.0.0")

    assert model_response["choices"][0]["message"]["content"] == "ok"
    assert "X-AgentCore-Session-ID" not in requests[0].headers
    assert tool_response == {"name": "search", "query": "agentcore"}
    assert skill.instruction.endswith("Review carefully.\n")
    assert skill.root.is_dir()
