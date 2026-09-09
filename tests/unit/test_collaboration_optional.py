"""Exercise the base SDK without the optional collaboration distribution."""

from __future__ import annotations

import subprocess
import sys
import textwrap


def test_base_sdk_and_server_work_without_collaboration_package() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent("""
            import asyncio
            import base64
            import importlib.abc
            import json
            import sys

            class WithoutCollaboration(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    if (fullname == "agentcore_collaboration"
                            or fullname.startswith("agentcore_collaboration.")):
                        raise ModuleNotFoundError("not installed", name="agentcore_collaboration")

            sys.meta_path.insert(0, WithoutCollaboration())
            from agentcore import AgentCore, AsyncAgentCore
            from agentcore.collaboration import current_collaboration_context
            from agentcore.server import AgentCoreServer
            import httpx

            server = AgentCoreServer()
            @server.invoke
            async def invoke(request, context):
                return current_collaboration_context().team_id

            async def check():
                core = AsyncAgentCore()
                try:
                    assert not any(n.startswith("agentcore_collaboration") for n in sys.modules)
                    try:
                        core.collaboration.worker()
                    except ImportError as exc:
                        assert "alibabacloud-agentcore-sdk[collaboration]" in str(exc), str(exc)
                    else:
                        raise AssertionError("missing installation must be reported")
                    context = base64.urlsafe_b64encode(json.dumps({
                        "version": 1, "teamId": "team-a", "roomId": "!room",
                        "roomKind": "task", "eventId": "$event",
                    }).encode()).decode()
                    async with httpx.AsyncClient(
                        transport=httpx.ASGITransport(app=server), base_url="http://test"
                    ) as client:
                        response = await client.post("/openai/v1/chat/completions",
                            json={"model": "worker", "messages": []},
                            headers={"X-AgentCore-Collaboration-Context": context})
                    assert response.status_code == 200, response.text
                    assert response.json()["choices"][0]["message"]["content"] == "team-a"
                    assert current_collaboration_context(required=False) is None
                finally:
                    await core.aclose()

            asyncio.run(check())
            with AgentCore() as core:
                try:
                    core.collaboration.worker()
                except ImportError as exc:
                    assert "alibabacloud-agentcore-sdk[collaboration]" in str(exc)
                else:
                    raise AssertionError("missing installation must be reported")
            try:
                from agentcore.collaboration import Collaboration
            except ImportError as exc:
                assert "alibabacloud-agentcore-sdk[collaboration]" in str(exc)
            else:
                raise AssertionError("legacy import must require the optional package")
        """),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_legacy_imports_share_context_and_types_with_standalone_package() -> None:
    import agentcore_collaboration as standalone

    from agentcore.collaboration import Collaboration, current_collaboration_context
    from agentcore.collaboration.debug import DebugCollaborationRuntime
    from agentcore.collaboration.task_service import TaskServiceClient
    from agentcore.collaboration.teams import TeamsProvider
    from agentcore.collaboration.worker import WorkerCollaboration

    assert standalone.Collaboration is Collaboration
    assert standalone.WorkerCollaboration is WorkerCollaboration
    assert standalone.TeamsProvider is TeamsProvider
    assert standalone.current_collaboration_context is current_collaboration_context
    assert DebugCollaborationRuntime.__module__ == "agentcore_collaboration.debug"
    assert TaskServiceClient.__module__ == "agentcore_collaboration.task_service"
