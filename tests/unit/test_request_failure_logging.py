import logging
from dataclasses import replace
from types import SimpleNamespace

import httpx
import openai
import pytest

from agentcore.auth.workload_access_token import WorkloadAccessTokenProvider
from agentcore.errors import AuthenticationError, CredentialExchangeError, InvocationError
from agentcore.model.client import _ManagedGatewayBackendBase


@pytest.mark.parametrize("message, expected", [
    ("Incorrect API key provided: sk-PRIVATE.", "Incorrect <redacted>"),
    ("Incorrect API KEY PROVIDED : 'sk-PRIVATE'.", "Incorrect <redacted>"),
    ("Invalid API Key: sk-PRIVATE", "Invalid <redacted>"),
    ("Invalid api-key=sk-PRIVATE", "Invalid <redacted>"),
    ("Invalid api_key: sk-PRIVATE", "Invalid <redacted>"),
    ("API key is missing", "API key is missing"),
])
def test_api_key_error_redaction_in_both_packages(message, expected):
    from agentcore_collaboration._logging import safe_error_message as collaboration_message

    from agentcore._logging import safe_error_message

    assert safe_error_message(message) == expected
    assert collaboration_message(message) == expected


@pytest.mark.parametrize("header", ["x-acs-request-id", "X-Request-ID", "request-id",
                                    "x-oss-request-id"])
@pytest.mark.parametrize("shape", ["flat", "nested", "empty", "non-json"])
def test_response_diagnostics_variants(header, shape):
    from agentcore._logging import response_diagnostics
    body = {"code": "Denied", "message": "Access denied; token=PRIVATE"}
    if shape == "nested":
        body = {"error": body}
    if shape == "empty":
        body = {}
    response = (httpx.Response(503, text="PRIVATE_RAW", headers={header: "req-test"})
                if shape == "non-json" else
                httpx.Response(503, json=body, headers={header: "req-test"}))
    details = response_diagnostics(response)
    assert "req-test" in details and "503" in details
    assert "PRIVATE" not in details
    if shape in {"flat", "nested"}:
        assert "Access denied" in details and "Denied" in details


@pytest.mark.asyncio
@pytest.mark.parametrize("auth_error", [False, True])
@pytest.mark.parametrize("protocol, operation", [
    ("OpenAI/v1", "completion"), ("OpenAI/v1", "stream"),
    ("OpenAI/v1", "responses_stream"), ("Anthropic", "completion"),
    ("Anthropic", "stream"),
])
async def test_native_model_error_logs_for_plain_and_stream_calls(
    protocol, operation, auth_error, model_descriptor, agent_config_path, monkeypatch, caplog,
):
    import httpx2

    from agentcore.model import AsyncModelClient
    from agentcore.runtime.config import load_agent_config

    status = 401 if auth_error else 403
    code = "invalid_api_key" if auth_error else "Forbidden"
    message = ("Incorrect API key provided: sk-PRIVATE." if auth_error
               else "Access denied; token=PRIVATE")

    async def fail(_transport, request):
        module = httpx2 if isinstance(request, httpx2.Request) else httpx
        return module.Response(status, request=request, headers={"x-request-id": "req-native"},
            json={"error": {"code": code, "message": message}})

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", fail)
    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", fail)
    client = AsyncModelClient(load_agent_config(agent_config_path),
        replace(model_descriptor, protocol=protocol, max_tokens=512))
    try:
        with pytest.raises(InvocationError):
            if operation == "completion":
                await client.completion([{"role": "user", "content": "PRIVATE_INPUT"}])
            elif operation == "responses_stream":
                async for _ in client.responses_stream("PRIVATE_INPUT"):
                    pass
            else:
                async for _ in client.stream([{"role": "user", "content": "PRIVATE_INPUT"}]):
                    pass
        for field in ("req-native", str(status), code,
                      "Incorrect" if auth_error else "Access denied"):
            assert field in caplog.text
        assert "PRIVATE" not in caplog.text
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_model_failure_logs_upstream_details_without_changing_cause(caplog):
    backend = _ManagedGatewayBackendBase.__new__(_ManagedGatewayBackendBase)
    backend._base_url = "https://model.example/v1"
    backend.descriptor = SimpleNamespace(connection_id="mc-test", model_name="test")
    response = httpx.Response(429, headers={"x-request-id": "req-model"},
                             request=httpx.Request("POST", "https://model.example/v1"))
    original = openai.RateLimitError("Quota exhausted", response=response,
        body={"code": "QuotaExceeded", "message": "Quota exhausted; token=PRIVATE"})

    async def fail():
        raise original

    with pytest.raises(InvocationError) as caught:
        await backend._request("completion", fail())
    assert caught.value.__cause__ is original
    for value in ("req-model", "QuotaExceeded", "Quota exhausted", "429"):
        assert value in caplog.text
    assert "PRIVATE" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 404, 503])
async def test_wat_failures_log_last_response_even_without_retry(status, caplog):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(
        status, json={"requestId": "req-wat", "code": "Rejected",
                      "message": "Identity unavailable; token=PRIVATE"}, request=req,
    ))) as client:
        provider = WorkloadAccessTokenProvider("https://controller.example",
            SimpleNamespace(get=lambda: "PRIVATE_SA"), http_client=client, max_attempts=1)
        with pytest.raises(AuthenticationError):
            await provider.get()
    for value in ("req-wat", "Rejected", "Identity unavailable", str(status),
                  "https://controller.example/api/v1/workload/token"):
        assert value in caplog.text
    assert "PRIVATE" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["bootstrap", "config", "matrix"])
@pytest.mark.parametrize("network_error", [False, True])
async def test_runtime_exchange_failure_diagnostics(operation, network_error, caplog):
    from agentcore.runtime.control_config import ControlConfigLoader
    from agentcore.runtime.startup import DebugRuntimeSource, DebugToken

    def fail(request):
        if network_error:
            raise httpx.ConnectError("PRIVATE_INTERNAL_DETAIL", request=request)
        return httpx.Response(503, json={"requestId": "req-runtime", "code": "Unavailable",
            "message": "STS signing failed; token=PRIVATE"}, request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
        source = DebugRuntimeSource(DebugToken("agentcore", "PRIVATE_JWT",
            "https://controller.example", "https://gateway.example", "https://matrix.example"),
            http_client=client, max_attempts=1)
        try:
            if operation == "config":
                with pytest.raises(CredentialExchangeError):
                    await ControlConfigLoader("https://controller.example", http_client=client,
                        max_attempts=1).load("PRIVATE_SA")
            elif operation == "bootstrap":
                with pytest.raises(AuthenticationError):
                    await source.get()
            elif network_error:
                with pytest.raises(AuthenticationError):
                    await source._request_matrix_token("PRIVATE_SA")
            else:
                assert (await source._request_matrix_token("PRIVATE_SA")).status_code == 503
        finally:
            await source.aclose()
    assert "https://controller.example/api/v1/" in caplog.text
    if network_error:
        assert "ConnectError" in caplog.text
    else:
        for field in ("req-runtime", "503", "Unavailable", "STS signing failed"):
            assert field in caplog.text
    assert "PRIVATE" not in caplog.text


def test_task_rejection_logs_before_conversion_to_tool_result(caplog):
    from agentcore_collaboration.task_service import _raise_for_status
    from agentcore_collaboration.worker import _error_result

    response = httpx.Response(503, json={"code": "Unavailable", "message": "Storage down",
                                        "requestId": "req-task"},
        request=httpx.Request("POST", "https://task.example/tasks?token=PRIVATE"))
    with caplog.at_level(logging.WARNING), pytest.raises(Exception) as caught:
        _raise_for_status(response)
    result = _error_result(caught.value)
    assert result["code"] == "COLLABORATION_TASK_UNAVAILABLE"
    assert "Storage down" not in result["message"]
    for value in ("req-task", "Storage down", "Unavailable", "503", "/tasks"):
        assert value in caplog.text
    assert "PRIVATE" not in caplog.text
