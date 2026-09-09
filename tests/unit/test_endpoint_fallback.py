from __future__ import annotations

import json
import logging
import traceback
from typing import Any

import httpx
import pytest
from darabonba.core import DaraCore
from darabonba.exceptions import RetryError
from darabonba.response import DaraResponse

from agentcore.auth import AccessKeyCredential
from agentcore.controlplane import AgentCoreControlPlane
from agentcore.errors import ConfigError, InvocationError

PUBLIC = "agentcore.cn-hangzhou.aliyuncs.com"
PRIVATE = "agentcore-vpc.cn-hangzhou.aliyuncs.com"
OSS_PUBLIC = "skills.oss-cn-hangzhou.aliyuncs.com"
OSS_PRIVATE = "skills.oss-cn-hangzhou-internal.aliyuncs.com"
SIGNED_URL = f"https://{OSS_PUBLIC}/skill%20test.zip?Signature=secret%2Bvalue&Expires=99"


def control_plane(**kwargs: Any) -> AgentCoreControlPlane:
    return AgentCoreControlPlane(
        workspace_id="ws-1",
        region_id="cn-hangzhou",
        access_key_credential=AccessKeyCredential("ak", "sk"),
        **kwargs,
    )


def pop_response(status: int = 200) -> DaraResponse:
    response = DaraResponse()
    response.status_code = status
    response.headers = {"content-type": "application/json", "x-acs-request-id": "req-1"}
    response.body = json.dumps(
        {"items": [{"mcpServerId": "mcp-1", "name": "test-mcp"}]}
        if status == 200
        else {"Code": "TestFailure", "Message": "failure", "RequestId": "req-1"}
    ).encode()
    return response


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connection", "timeout", 500, 502, 503, 504])
async def test_generated_sdk_falls_back_and_reuses_private_endpoint(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str | int
) -> None:
    hosts = []

    async def send(request: Any, runtime: Any) -> DaraResponse:
        host = request.headers["host"]
        hosts.append(host)
        if host == PUBLIC:
            if failure == "connection":
                raise RetryError("connection failed")
            if failure == "timeout":
                raise TimeoutError("timeout")
            return pop_response(int(failure))
        return pop_response()

    monkeypatch.setattr(DaraCore, "async_do_action", send)
    caplog.set_level(logging.INFO, logger="agentcore")
    cp = control_plane()
    assert (await cp.resolve_mcp("test-mcp")).mcp_server_id == "mcp-1"
    await cp.resolve_mcp("test-mcp")
    assert hosts == [PUBLIC, PRIVATE, PRIVATE]
    assert "fallback" in caplog.text and PUBLIC in caplog.text and PRIVATE in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 403, 404, 429])
async def test_control_plane_does_not_switch_on_service_rejection(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    hosts = []

    async def send(request: Any, runtime: Any) -> DaraResponse:
        hosts.append(request.headers["host"])
        return pop_response(status)

    monkeypatch.setattr(DaraCore, "async_do_action", send)
    with pytest.raises(InvocationError):
        await control_plane().resolve_mcp("test-mcp")
    assert hosts == [PUBLIC]


@pytest.mark.asyncio
@pytest.mark.parametrize("endpoint", [PUBLIC, "https://agentcore-pre.aliyuncs.com", PRIVATE])
async def test_explicit_control_endpoint_never_switches(
    monkeypatch: pytest.MonkeyPatch, endpoint: str
) -> None:
    hosts = []

    async def send(request: Any, runtime: Any) -> DaraResponse:
        hosts.append(request.headers["host"])
        raise RetryError("unreachable")

    monkeypatch.setattr(DaraCore, "async_do_action", send)
    with pytest.raises(InvocationError):
        await control_plane(endpoint=endpoint).resolve_mcp("test-mcp")
    assert hosts == [endpoint.removeprefix("https://")]


@pytest.mark.asyncio
async def test_failed_private_attempt_is_not_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    hosts = []

    async def send(request: Any, runtime: Any) -> DaraResponse:
        hosts.append(request.headers["host"])
        raise RetryError("unreachable")

    monkeypatch.setattr(DaraCore, "async_do_action", send)
    cp = control_plane()
    for _ in range(2):
        with pytest.raises(InvocationError):
            await cp.resolve_mcp("test-mcp")
    assert hosts == [PUBLIC, PRIVATE, PUBLIC, PRIVATE]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failing_action",
    ["ListModelConnections", "ListModels", "GetSkillDetail", "DownloadSkillVersionViaOss"],
)
async def test_model_and_skill_calls_use_endpoint_fallback(
    monkeypatch: pytest.MonkeyPatch, failing_action: str
) -> None:
    calls = []
    bodies = {
        "ListModelConnections": {"items": [{"name": "test-mc", "connectionId": "mc-1"}]},
        "ListModels": {"items": [{"modelId": "model-1", "modelName": "qwen-plus"}]},
        "GetSkillDetail": {"data": {"labels": {"latest": "1.0.0"}}},
        "DownloadSkillVersionViaOss": {"data": SIGNED_URL},
    }

    async def send(request: Any, runtime: Any) -> DaraResponse:
        host, action = request.headers["host"], request.headers["x-acs-action"]
        calls.append((host, action))
        if host == PUBLIC and action == failing_action:
            raise RetryError("unreachable")
        response = pop_response()
        response.body = json.dumps(bodies[action]).encode()
        return response

    monkeypatch.setattr(DaraCore, "async_do_action", send)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b"archive"))
    ) as http:
        cp = control_plane(_http_client=http)
        if failing_action.startswith("List"):
            assert (await cp.resolve_model("test-mc", "qwen-plus")).model_id == "model-1"
        else:
            assert (await cp.get_skill("test-skill")).archive == b"archive"
    index = calls.index((PUBLIC, failing_action))
    assert calls[index + 1] == (PRIVATE, failing_action)
    assert all(host == PRIVATE for host, _ in calls[index + 1 :])


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["connection", "timeout", 500, 502, 503, 504])
async def test_oss_fallback_preserves_signed_query(
    caplog: pytest.LogCaptureFixture, failure: str | int
) -> None:
    requests = []

    def download(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert "authorization" not in request.headers
        if request.url.host == OSS_PUBLIC:
            if failure == "connection":
                raise httpx.ConnectError("unreachable")
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout")
            return httpx.Response(int(failure))
        return httpx.Response(200, content=b"archive")

    caplog.set_level(logging.INFO, logger="agentcore")
    async with httpx.AsyncClient(transport=httpx.MockTransport(download)) as http:
        result = await control_plane(_http_client=http)._download_skill_archive(SIGNED_URL)
    assert result == b"archive"
    assert [r.url.host for r in requests] == [OSS_PUBLIC, OSS_PRIVATE]
    assert requests[0].url.raw_path == requests[1].url.raw_path
    assert "secret" not in caplog.text
    assert "fallback" in caplog.text and OSS_PRIVATE in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        SIGNED_URL + "&x-oss-additional-headers=host",
        SIGNED_URL + "&x-oss-additional-headers=content-type%3Bhost",
        "https://download.example.com/skill.zip",
        f"https://{OSS_PRIVATE}/skill.zip",
        "https://skills.oss-accelerate-overseas.aliyuncs.com/skill.zip",
    ],
)
async def test_oss_does_not_rewrite_signed_host_or_non_regional_url(url: str) -> None:
    requests = []

    def download(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("unreachable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(download)) as http:
        with pytest.raises(InvocationError):
            await control_plane(_http_client=http)._download_skill_archive(url)
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_oss_v4_without_signed_host_can_fallback() -> None:
    hosts = []

    def download(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        if request.url.host == OSS_PUBLIC:
            raise httpx.ConnectError("unreachable")
        return httpx.Response(200, content=b"archive")

    url = f"https://{OSS_PUBLIC}/skill.zip?x-oss-signature-version=OSS4-HMAC-SHA256&x-oss-signature=secret"
    async with httpx.AsyncClient(transport=httpx.MockTransport(download)) as http:
        assert await control_plane(_http_client=http)._download_skill_archive(url) == b"archive"
    assert hosts == [OSS_PUBLIC, OSS_PRIVATE]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 404, 429, 200])
async def test_oss_does_not_retry_rejection_or_empty_archive(status: int) -> None:
    hosts = []

    def download(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(status)

    async with httpx.AsyncClient(transport=httpx.MockTransport(download)) as http:
        with pytest.raises(ConfigError if status == 200 else InvocationError):
            await control_plane(_http_client=http)._download_skill_archive(SIGNED_URL)
    assert hosts == [OSS_PUBLIC]


@pytest.mark.asyncio
async def test_oss_failure_attempts_each_endpoint_only_once() -> None:
    hosts = []

    def download(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        raise httpx.ConnectError("unreachable")

    async with httpx.AsyncClient(transport=httpx.MockTransport(download)) as http:
        with pytest.raises(InvocationError):
            await control_plane(_http_client=http)._download_skill_archive(SIGNED_URL)
    assert hosts == [OSS_PUBLIC, OSS_PRIVATE]


@pytest.mark.asyncio
async def test_oss_retry_discards_partial_public_download() -> None:
    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):  # type: ignore[no-untyped-def]
            yield b"partial-public-bytes"
            raise httpx.ReadError("connection lost")

    def download(request: httpx.Request) -> httpx.Response:
        if request.url.host == OSS_PUBLIC:
            return httpx.Response(200, stream=BrokenStream())
        return httpx.Response(200, content=b"complete-private-archive")

    async with httpx.AsyncClient(transport=httpx.MockTransport(download)) as http:
        assert await control_plane(_http_client=http)._download_skill_archive(SIGNED_URL) == (
            b"complete-private-archive"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [403, 503])
@pytest.mark.parametrize("signature_version", ["v1", "v4"])
async def test_oss_error_traceback_does_not_expose_signed_url(
    caplog: pytest.LogCaptureFixture, status: int, signature_version: str
) -> None:
    query = (
        "Signature=signature-secret&security-token=sts-secret"
        if signature_version == "v1"
        else "x-oss-signature=signature-secret&x-oss-security-token=sts-secret"
        "&x-oss-signature-version=OSS4-HMAC-SHA256"
    )
    url = f"https://{OSS_PUBLIC}/skill.zip?{query}"
    hosts = []

    def download(request: httpx.Request) -> httpx.Response:
        hosts.append(request.url.host)
        return httpx.Response(status, headers={"x-oss-request-id": "oss-request-123"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(download)) as http:
        with pytest.raises(InvocationError) as raised:
            try:
                await control_plane(_http_client=http)._download_skill_archive(url)
            except InvocationError:
                logging.getLogger("test.caller").exception("Skill download failed")
                raise

    error = raised.value
    trace = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    for output in (trace, caplog.text):
        assert "signature-secret" not in output
        assert "sts-secret" not in output
        assert url not in output
    assert f"HTTP {status}" in str(error)
    assert f"status={status}" in caplog.text
    assert "request_id=oss-request-123" in caplog.text
    assert OSS_PUBLIC in caplog.text
    assert hosts == ([OSS_PUBLIC] if status == 403 else [OSS_PUBLIC, OSS_PRIVATE])


@pytest.mark.asyncio
async def test_oss_transport_failure_retains_diagnostic_cause() -> None:
    failure = httpx.ConnectError("connection refused")

    def download(request: httpx.Request) -> httpx.Response:
        raise failure

    async with httpx.AsyncClient(transport=httpx.MockTransport(download)) as http:
        with pytest.raises(InvocationError) as raised:
            await control_plane(_http_client=http)._download_skill_archive(SIGNED_URL)
    assert raised.value.__cause__ is failure
