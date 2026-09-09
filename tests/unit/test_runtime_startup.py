from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from agentcore.errors import AuthenticationError, ConfigError
from agentcore.runtime import startup
from agentcore.runtime.startup import (
    DebugRuntimeSource,
    ManagedRuntimeSource,
    create_runtime_source,
    parse_debug_token,
)


def _debug_token(**overrides: str) -> str:
    payload = {
        "product": "agentcore",
        "jwtToken": "bootstrap-jwt",
        "controllerUrl": "https://public-controller.example.com",
        "modelGatewayUrl": "https://public-model.example.com/v1",
        "matrixUrl": "https://public-matrix.example.com",
        **overrides,
    }
    return base64.b64encode(json.dumps(payload).encode()).decode()


def _exchange_response(*, token: str = "agent-sa-token") -> dict[str, Any]:
    return {
        "token": token,
        "expiresAt": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "jwtToken": "refreshed-bootstrap-jwt",
        "jwtExpiresAt": (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat(),
    }


def _control_sts_response(
    *, endpoint: str = "https://oss-cn-hangzhou.aliyuncs.com"
) -> dict[str, str]:
    return {
        "access_key_id": "sts-ak",
        "access_key_secret": "sts-sk",
        "security_token": "sts-token",
        "expiration": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        "oss_endpoint": endpoint,
        "oss_bucket": "control-bucket",
        "agent_config_path": "agents/runtime-debug/agent.yaml",
        "teams_config_path": "agents/runtime-debug/teams.yaml",
    }


def _agent_yaml() -> bytes:
    return b"""apiVersion: agentteams.io/v1alpha1
kind: AgentConfig
metadata:
  name: debug-agent
  runtimeName: runtime-debug
  workspaceId: workspace-a
  regionId: cn-hangzhou
spec:
  model:
    gatewayUrl: http://private-model.internal/v1
  mcp:
    gatewayUrl: http://private-mcp.internal/mcp
  credentials:
    header:
      - key: Authorization
        value: Bearer gateway-secret
"""


@pytest.mark.asyncio
async def test_managed_runtime_retries_until_agent_config_is_mounted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "agent.yaml"
    delays: list[float] = []

    async def mount_after_delay(delay: float) -> None:
        delays.append(delay)
        config_path.write_bytes(_agent_yaml())

    monkeypatch.setattr(startup.anyio, "sleep", mount_after_delay)
    source = ManagedRuntimeSource(
        config_path,
        env_path=None,
        control_plane_endpoint=None,
    )

    runtime = await source.resolve()

    assert runtime.config.workspace_id == "workspace-a"
    assert delays == [0.5]


@pytest.mark.asyncio
async def test_default_runtime_source_waits_for_delayed_agent_config_mount(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "agent.yaml"
    monkeypatch.delenv("AGENTCORE_DEBUG_TOKEN", raising=False)
    monkeypatch.delenv("AGENTCORE_CONFIG_PATH", raising=False)
    monkeypatch.delenv("AGENTCORE_ENV_PATH", raising=False)
    monkeypatch.setattr(startup, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.setattr(startup, "DEFAULT_ENV_PATH", tmp_path / "missing-env")

    async def mount_after_delay(_delay: float) -> None:
        config_path.write_bytes(_agent_yaml())

    monkeypatch.setattr(startup.anyio, "sleep", mount_after_delay)

    source = create_runtime_source()

    assert isinstance(source, ManagedRuntimeSource)
    runtime = await source.resolve()
    assert runtime.config.workspace_id == "workspace-a"


@pytest.mark.asyncio
async def test_managed_runtime_stops_retrying_when_agent_config_remains_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []
    monkeypatch.delenv("AGENTCORE_CONFIG_WAIT_TIMEOUT", raising=False)

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(startup.anyio, "sleep", record_delay)
    source = ManagedRuntimeSource(
        tmp_path / "missing-agent.yaml",
        env_path=None,
        control_plane_endpoint=None,
    )

    with pytest.raises(ConfigError, match="cannot read agent.yaml"):
        await source.resolve()

    assert delays == [0.5] * 20


@pytest.mark.asyncio
async def test_managed_runtime_uses_configured_config_wait_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delays: list[float] = []

    async def record_delay(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setenv("AGENTCORE_CONFIG_WAIT_TIMEOUT", "1")
    monkeypatch.setattr(startup.anyio, "sleep", record_delay)
    source = ManagedRuntimeSource(
        tmp_path / "missing-agent.yaml",
        env_path=None,
        control_plane_endpoint=None,
    )

    with pytest.raises(ConfigError, match="cannot read agent.yaml"):
        await source.resolve()

    assert delays == [0.5, 0.5]


@pytest.mark.asyncio
async def test_managed_runtime_does_not_retry_invalid_agent_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "agent.yaml"
    config_path.write_text("kind: broken", encoding="utf-8")

    async def unexpected_sleep(_delay: float) -> None:
        pytest.fail("invalid agent.yaml must fail without retry")

    monkeypatch.setattr(startup.anyio, "sleep", unexpected_sleep)
    source = ManagedRuntimeSource(
        config_path,
        env_path=None,
        control_plane_endpoint=None,
    )

    with pytest.raises(ConfigError):
        await source.resolve()


def test_parse_debug_token_uses_external_agent_envelope_without_secret_repr() -> None:
    token = parse_debug_token(_debug_token())

    assert token.product == "agentcore"
    assert token.controller_url == "https://public-controller.example.com"
    assert token.model_gateway_url == "https://public-model.example.com/v1"
    assert token.matrix_url == "https://public-matrix.example.com"
    assert "bootstrap-jwt" not in repr(token)


def test_debug_runtime_uses_process_control_plane_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AGENTCORE_DEBUG_TOKEN", _debug_token())
    monkeypatch.setenv(
        "AGENTCORE_CONTROL_ENDPOINT",
        "https://pre-agentcore.example.com",
    )

    source = create_runtime_source()

    assert isinstance(source, DebugRuntimeSource)
    assert source._control_plane_endpoint == "https://pre-agentcore.example.com"


@pytest.mark.parametrize(
    "value",
    [
        "not-base64",
        base64.b64encode(b"not-json").decode(),
        _debug_token(product="wrong-product"),
        _debug_token(controllerUrl="file:///tmp/controller"),
        _debug_token(matrixUrl=""),
    ],
)
def test_parse_debug_token_rejects_invalid_envelopes_without_echoing_token(value: str) -> None:
    with pytest.raises(ConfigError) as error:
        parse_debug_token(value)

    assert value not in str(error.value)
    assert "bootstrap-jwt" not in str(error.value)


@pytest.mark.asyncio
async def test_debug_runtime_exchanges_once_and_keeps_config_in_memory() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path == "/api/v1/edge/token":
            return httpx.Response(200, json=_exchange_response())
        if request.url.path == "/api/v1/credentials/sts":
            return httpx.Response(200, json=_control_sts_response())
        if request.url.path == "/agents/runtime-debug/agent.yaml":
            return httpx.Response(200, content=_agent_yaml())
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        first, second = await asyncio.gather(source.resolve(), source.resolve())
        await source.aclose()

    assert first is second
    assert len(requests) == 3
    assert requests[0].url == "https://public-controller.example.com/api/v1/edge/token"
    assert json.loads(requests[0].content) == {"jwtToken": "bootstrap-jwt"}
    assert requests[1].url == (
        "https://public-controller.example.com/api/v1/credentials/sts"
        "?purpose=oss&target=control"
    )
    assert requests[1].headers["Authorization"] == "Bearer agent-sa-token"
    assert requests[2].url == (
        "https://control-bucket.oss-cn-hangzhou.aliyuncs.com/"
        "agents/runtime-debug/agent.yaml"
    )
    assert requests[2].headers["x-oss-security-token"] == "sts-token"
    assert requests[2].headers["Authorization"].startswith("OSS sts-ak:")
    assert first.config.workspace_id == "workspace-a"
    assert first.config.region_id == "cn-hangzhou"
    assert first.controller_endpoint == "https://public-controller.example.com"
    assert first.config.spec.model.gateway_url == "https://public-model.example.com/v1"
    assert first.config.spec.mcp.gateway_url == "https://public-model.example.com/mcp"
    assert first.agent_sa_tokens is source


@pytest.mark.asyncio
async def test_debug_runtime_loads_teams_config_with_cached_control_sts() -> None:
    sts_calls = 0
    teams_yaml = b"""apiVersion: agentteams.io/v1alpha1
kind: TeamsConfig
metadata:
  runtimeName: runtime-debug
spec:
  self:
    name: worker-a
    runtimeName: runtime-debug
    matrixUserId: "@worker-a:matrix.example.com"
  matrix:
    tokenEnv: AGENTTEAMS_WORKER_MATRIX_TOKEN
  teams: []
"""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sts_calls
        if request.url.path == "/api/v1/edge/token":
            return httpx.Response(200, json=_exchange_response())
        if request.url.path == "/api/v1/credentials/sts":
            sts_calls += 1
            return httpx.Response(200, json=_control_sts_response())
        if request.url.path == "/agents/runtime-debug/agent.yaml":
            return httpx.Response(200, content=_agent_yaml())
        if request.url.path == "/agents/runtime-debug/teams.yaml":
            return httpx.Response(200, content=teams_yaml)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        await source.resolve()
        loaded = await source.load_teams_config()
        await source.aclose()

    assert loaded == teams_yaml
    assert sts_calls == 1


@pytest.mark.asyncio
async def test_debug_runtime_refreshes_control_sts_when_teams_path_appears() -> None:
    sts_calls = 0
    teams_yaml = b"""apiVersion: agentteams.io/v1alpha1
kind: TeamsConfig
metadata:
  runtimeName: runtime-debug
spec:
  self:
    name: worker-a
    runtimeName: runtime-debug
    matrixUserId: "@worker-a:matrix.example.com"
  matrix:
    tokenEnv: AGENTTEAMS_WORKER_MATRIX_TOKEN
  teams: []
"""

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sts_calls
        if request.url.path == "/api/v1/edge/token":
            return httpx.Response(200, json=_exchange_response())
        if request.url.path == "/api/v1/credentials/sts":
            sts_calls += 1
            response = _control_sts_response()
            if sts_calls == 1:
                response.pop("teams_config_path")
            return httpx.Response(200, json=response)
        if request.url.path == "/agents/runtime-debug/agent.yaml":
            return httpx.Response(200, content=_agent_yaml())
        if request.url.path == "/agents/runtime-debug/teams.yaml":
            return httpx.Response(200, content=teams_yaml)
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        await source.resolve()
        loaded = await source.load_teams_config()
        await source.aclose()

    assert loaded == teams_yaml
    assert sts_calls == 2


@pytest.mark.asyncio
async def test_debug_runtime_refreshes_agent_sa_once_for_matrix_token() -> None:
    matrix_authorizations: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/edge/token":
            token = "agent-sa-token-1" if not matrix_authorizations else "agent-sa-token-2"
            return httpx.Response(200, json=_exchange_response(token=token))
        if request.url.path == "/api/v1/credentials/matrix-token":
            authorization = request.headers["Authorization"]
            matrix_authorizations.append(authorization)
            if authorization == "Bearer agent-sa-token-1":
                return httpx.Response(401)
            return httpx.Response(200, json={"access_token": "matrix-worker-token"})
        return httpx.Response(404)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        token = await source.exchange_matrix_token()
        await source.aclose()

    assert token == "matrix-worker-token"
    assert matrix_authorizations == [
        "Bearer agent-sa-token-1",
        "Bearer agent-sa-token-2",
    ]


@pytest.mark.asyncio
async def test_debug_runtime_refresh_uses_latest_bootstrap_jwt() -> None:
    requests: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        response = _exchange_response(token=f"agent-sa-token-{len(requests)}")
        response["jwtToken"] = "server-returned-rotated-jwt"
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        assert await source.get() == "agent-sa-token-1"
        assert await source.refresh("agent-sa-token-1") == "agent-sa-token-2"
        await source.aclose()

    assert requests == [
        {"jwtToken": "bootstrap-jwt"},
        {"jwtToken": "server-returned-rotated-jwt"},
    ]


@pytest.mark.asyncio
async def test_debug_runtime_refreshes_jwt_in_background_before_expiration() -> None:
    requests: list[dict[str, Any]] = []
    refreshed = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        response = _exchange_response(token=f"agent-sa-token-{len(requests)}")
        response["jwtToken"] = f"refreshed-bootstrap-jwt-{len(requests)}"
        if len(requests) == 1:
            response["jwtExpiresAt"] = (
                datetime.now(timezone.utc) + timedelta(milliseconds=50)
            ).isoformat()
        else:
            refreshed.set()
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        assert await source.get() == "agent-sa-token-1"
        await asyncio.wait_for(refreshed.wait(), timeout=1)
        assert await source.get() == "agent-sa-token-2"
        await source.aclose()

    assert requests == [
        {"jwtToken": "bootstrap-jwt"},
        {"jwtToken": "refreshed-bootstrap-jwt-1"},
    ]


@pytest.mark.asyncio
async def test_debug_runtime_retries_transient_background_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[dict[str, Any]] = []
    refreshed = asyncio.Event()
    monkeypatch.setattr(startup, "_JWT_REFRESH_RETRY_DELAY", 0.01)

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        if len(requests) == 2:
            return httpx.Response(503)
        response = _exchange_response(token=f"agent-sa-token-{len(requests)}")
        response["jwtToken"] = f"refreshed-bootstrap-jwt-{len(requests)}"
        if len(requests) == 1:
            response["jwtExpiresAt"] = (
                datetime.now(timezone.utc) + timedelta(seconds=2)
            ).isoformat()
        else:
            refreshed.set()
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(
            parse_debug_token(_debug_token()),
            http_client=http,
            max_attempts=1,
        )
        assert await source.get() == "agent-sa-token-1"
        await asyncio.wait_for(refreshed.wait(), timeout=1)
        assert await source.get() == "agent-sa-token-3"
        await source.aclose()

    assert requests == [
        {"jwtToken": "bootstrap-jwt"},
        {"jwtToken": "refreshed-bootstrap-jwt-1"},
        {"jwtToken": "refreshed-bootstrap-jwt-1"},
    ]


@pytest.mark.asyncio
async def test_debug_runtime_does_not_restart_rejected_background_refresh() -> None:
    calls = 0

    async def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            response = _exchange_response()
            response["jwtExpiresAt"] = (
                datetime.now(timezone.utc) + timedelta(seconds=2)
            ).isoformat()
            return httpx.Response(200, json=response)
        return httpx.Response(401)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(
            parse_debug_token(_debug_token()),
            http_client=http,
            max_attempts=1,
        )
        assert await source.get() == "agent-sa-token"
        assert source._jwt_refresh_task is not None
        await asyncio.wait_for(source._jwt_refresh_task, timeout=1)
        assert calls == 2

        assert await source.get() == "agent-sa-token"
        await asyncio.sleep(0.05)
        assert calls == 2
        await source.aclose()


@pytest.mark.asyncio
async def test_debug_runtime_retries_gateway_and_transport_failures() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(502)
        if calls == 2:
            raise httpx.ReadError("connection interrupted", request=request)
        return httpx.Response(200, json=_exchange_response())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        assert await source.get() == "agent-sa-token"
        await source.aclose()

    assert calls == 3


@pytest.mark.asyncio
async def test_control_config_sts_retries_gateway_and_transport_failures() -> None:
    sts_calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal sts_calls
        if request.url.path == "/api/v1/edge/token":
            return httpx.Response(200, json=_exchange_response())
        if request.url.path == "/api/v1/credentials/sts":
            sts_calls += 1
            if sts_calls == 1:
                return httpx.Response(504)
            if sts_calls == 2:
                raise httpx.RemoteProtocolError("connection interrupted", request=request)
            return httpx.Response(200, json=_control_sts_response())
        return httpx.Response(200, content=_agent_yaml())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        runtime = await source.resolve()
        await source.aclose()

    assert runtime.config.workspace_id == "workspace-a"
    assert sts_calls == 3


@pytest.mark.asyncio
async def test_debug_runtime_falls_back_from_internal_to_public_oss() -> None:
    oss_hosts: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/edge/token":
            return httpx.Response(200, json=_exchange_response())
        if request.url.path == "/api/v1/credentials/sts":
            return httpx.Response(
                200,
                json=_control_sts_response(
                    endpoint="https://oss-cn-hangzhou-internal.aliyuncs.com"
                ),
            )
        oss_hosts.append(request.url.host)
        if request.url.host.endswith("-internal.aliyuncs.com"):
            raise httpx.ConnectError("private endpoint is unreachable", request=request)
        return httpx.Response(200, content=_agent_yaml())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        runtime = await source.resolve()
        await source.aclose()

    assert runtime.config.workspace_id == "workspace-a"
    assert oss_hosts == [
        "control-bucket.oss-cn-hangzhou-internal.aliyuncs.com",
        "control-bucket.oss-cn-hangzhou.aliyuncs.com",
    ]


@pytest.mark.asyncio
async def test_debug_runtime_falls_back_when_oss_response_body_is_interrupted() -> None:
    oss_hosts: list[str] = []

    class InterruptedBody(httpx.AsyncByteStream):
        def __init__(self, request: httpx.Request) -> None:
            self._request = request

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"partial"
            raise httpx.ReadError("connection interrupted", request=self._request)

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/edge/token":
            return httpx.Response(200, json=_exchange_response())
        if request.url.path == "/api/v1/credentials/sts":
            return httpx.Response(
                200,
                json=_control_sts_response(
                    endpoint="https://oss-cn-hangzhou-internal.aliyuncs.com"
                ),
            )
        oss_hosts.append(request.url.host)
        if request.url.host.endswith("-internal.aliyuncs.com"):
            return httpx.Response(200, stream=InterruptedBody(request))
        return httpx.Response(200, content=_agent_yaml())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        runtime = await source.resolve()
        await source.aclose()

    assert runtime.config.workspace_id == "workspace-a"
    assert oss_hosts == [
        "control-bucket.oss-cn-hangzhou-internal.aliyuncs.com",
        "control-bucket.oss-cn-hangzhou.aliyuncs.com",
    ]


@pytest.mark.asyncio
async def test_debug_runtime_preserves_explicit_root_gateway_path() -> None:
    root_gateway_yaml = _agent_yaml().replace(
        b"http://private-model.internal/v1",
        b"http://private-model.internal/",
    ).replace(
        b"http://private-mcp.internal/mcp",
        b"http://private-mcp.internal/",
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/edge/token":
            return httpx.Response(200, json=_exchange_response())
        if request.url.path == "/api/v1/credentials/sts":
            return httpx.Response(200, json=_control_sts_response())
        return httpx.Response(200, content=root_gateway_yaml)

    token = parse_debug_token(
        _debug_token(modelGatewayUrl="https://public-model.example.com/public-prefix")
    )
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(token, http_client=http)
        runtime = await source.resolve()
        await source.aclose()

    assert runtime.config.spec.model.gateway_url == "https://public-model.example.com"
    assert runtime.config.spec.mcp.gateway_url == "https://public-model.example.com"


@pytest.mark.asyncio
async def test_debug_runtime_rejects_unsafe_control_config_path() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/edge/token":
            return httpx.Response(200, json=_exchange_response())
        response = _control_sts_response()
        response["agent_config_path"] = "../agent.yaml"
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        with pytest.raises(AuthenticationError, match="invalid agent_config_path"):
            await source.resolve()


@pytest.mark.asyncio
async def test_invalid_teams_config_path_only_fails_when_collaboration_is_used() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/edge/token":
            return httpx.Response(200, json=_exchange_response())
        if request.url.path == "/api/v1/credentials/sts":
            response = _control_sts_response()
            response["teams_config_path"] = "../teams.yaml"
            return httpx.Response(200, json=response)
        return httpx.Response(200, content=_agent_yaml())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        runtime = await source.resolve()
        with pytest.raises(AuthenticationError, match="invalid teams_config_path"):
            await source.load_teams_config()
        await source.aclose()

    assert runtime.config.workspace_id == "workspace-a"


@pytest.mark.asyncio
async def test_debug_runtime_rejects_expired_control_config_sts() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/edge/token":
            return httpx.Response(200, json=_exchange_response())
        response = _control_sts_response()
        response["expiration"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        return httpx.Response(200, json=response)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        with pytest.raises(AuthenticationError, match="invalid control-config STS response"):
            await source.resolve()


@pytest.mark.asyncio
async def test_debug_runtime_maps_invalid_oss_port_to_configured_error() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v1/edge/token":
            return httpx.Response(200, json=_exchange_response())
        return httpx.Response(
            200,
            json=_control_sts_response(endpoint="https://oss.example.com:not-a-port"),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        with pytest.raises(AuthenticationError, match="invalid OSS endpoint"):
            await source.resolve()


@pytest.mark.asyncio
async def test_debug_runtime_maps_exchange_failure_without_exposing_credentials() -> None:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(401))
    ) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        with pytest.raises(AuthenticationError) as error:
            await source.resolve()

    assert "bootstrap-jwt" not in str(error.value)
    assert "gateway-secret" not in str(error.value)


@pytest.mark.asyncio
async def test_debug_runtime_rejects_expired_agent_sa_token() -> None:
    response = _exchange_response()
    response["expiresAt"] = (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as http:
        source = DebugRuntimeSource(parse_debug_token(_debug_token()), http_client=http)
        with pytest.raises(AuthenticationError, match="invalid debug token response"):
            await source.resolve()


def test_debug_runtime_takes_priority_over_managed_runtime_files(
    monkeypatch: pytest.MonkeyPatch,
    agent_config_path: Path,
) -> None:
    monkeypatch.setenv("AGENTCORE_DEBUG_TOKEN", _debug_token())
    monkeypatch.setenv("AGENTCORE_CONFIG_PATH", str(agent_config_path))
    monkeypatch.setenv("AGENTCORE_ENV_PATH", str(agent_config_path.parent / "runtime.env"))

    source = create_runtime_source(
        config_path=agent_config_path,
        env_path=agent_config_path.parent / "explicit-runtime.env",
    )

    assert isinstance(source, DebugRuntimeSource)


def test_empty_debug_token_does_not_enable_debug_mode(
    monkeypatch: pytest.MonkeyPatch,
    agent_config_path: Path,
) -> None:
    monkeypatch.setenv("AGENTCORE_DEBUG_TOKEN", "  ")
    source = create_runtime_source(config_path=agent_config_path)

    assert source.initial is None
