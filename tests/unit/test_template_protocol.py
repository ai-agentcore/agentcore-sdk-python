from __future__ import annotations

import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from urllib.parse import parse_qs, urlsplit
from zipfile import ZipFile

import pytest

from agentcore import AgentCore, AsyncAgentCore
from agentcore.auth import AccessKeyCredential
from agentcore.errors import ResourceNotConfiguredError


def archive(files: dict[str, str | bytes]) -> bytes:
    result = io.BytesIO()
    with ZipFile(result, "w") as target:
        for path, content in files.items():
            target.writestr(path, content)
    return result.getvalue()


@pytest.fixture
def registry():
    calls = []
    state = {
        "labels": {"stable": "v3", "latest": "v2"},
        "selector": {"type": "LABEL", "value": "stable"},
    }
    skill = archive({"remote/SKILL.md": "Remote instructions"})

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path)
            action = self.headers.get("x-acs-action")
            if path.path == "/template.zip":
                template = archive(
                    {
                        "manifest.json": json.dumps(
                            {
                                "skills": [
                                    {
                                        "name": "remote",
                                        "sourceType": "REFERENCE",
                                        "versionSelector": state["selector"],
                                    }
                                ]
                            }
                        ),
                        "config/SOUL.md": "Hello",
                    }
                )
                body = template
            elif path.path == "/skill.zip":
                body = skill
            else:
                assert self.headers.get("Authorization")
                calls.append((action, path.path, parse_qs(path.query)))
                if action == "GetAgentSpec":
                    data = {"labels": {"latest": "v1"}}
                elif action == "DownloadAgentSpecViaOss":
                    data = endpoint + "/template.zip"
                elif action == "GetSkillDetail":
                    data = {
                        "labels": state["labels"],
                        "versions": [{"version": "v9", "status": "online"}],
                    }
                elif action == "DownloadSkillVersionViaOss":
                    data = endpoint + "/skill.zip"
                else:
                    raise AssertionError(action)
                body = json.dumps({"data": data}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json" if action else "application/zip")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    endpoint = f"http://127.0.0.1:{server.server_port}"
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield endpoint, calls, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def options(endpoint: str, root: Path) -> dict:
    return {
        "workspace_id": "ws-test",
        "region_id": "cn-hangzhou",
        "control_plane_endpoint": endpoint,
        "skill_workspace_dir": root,
        "access_key_credential": AccessKeyCredential("fixture-ak", "fixture-sk"),
    }


@pytest.mark.asyncio
async def test_async_template_uses_signed_openapi_and_explicit_selected_versions(
    registry, tmp_path
):
    endpoint, calls, _ = registry
    async with AsyncAgentCore.auto(**options(endpoint, tmp_path)) as core:
        template = await core.templates.get("assistant")
        assert template.version == "v1" and template.instruction == "Hello"
        assert [call[0] for call in calls] == ["GetAgentSpec", "DownloadAgentSpecViaOss"]
        assert calls[-1][2]["agentSpecVersion"] == ["v1"]
        assert (await template.load_skills())[0].version == "v3"
        assert "/versions/v3/" in calls[-1][1]


def test_sync_template_facade_uses_the_same_core_portal(registry, tmp_path):
    endpoint, calls, state = registry
    state["selector"] = {"type": "VERSION", "value": "v4"}
    with AgentCore.auto(**options(endpoint, tmp_path)) as core:
        template = core.templates.get("assistant", version="v1")
        assert template.instruction == "Hello"
        assert template.name == "assistant" and template.version == "v1"
        assert template.load_skills()[0].version == "v4"
        assert template.load_skills()[0].version == "v4"
    assert [call[0] for call in calls] == ["DownloadAgentSpecViaOss", "DownloadSkillVersionViaOss"]


@pytest.mark.asyncio
async def test_missing_label_does_not_use_latest_online_fallback(registry, tmp_path):
    endpoint, calls, state = registry
    state["labels"] = {}
    async with AsyncAgentCore.auto(**options(endpoint, tmp_path)) as core:
        template = await core.templates.get("assistant", version="v1")
        with pytest.raises(ResourceNotConfiguredError, match="no label"):
            await template.load_skills()
        assert template.instruction == "Hello"
    assert "DownloadSkillVersionViaOss" not in [call[0] for call in calls]
