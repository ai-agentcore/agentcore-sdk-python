from __future__ import annotations

import asyncio
from collections.abc import Awaitable

import pytest

from agentcore.runtime.context import (
    current_context,
    request_context_from_headers,
    use_context,
)


def test_context_preserves_request_headers_without_interpreting_them() -> None:
    context = request_context_from_headers(
        {
            "x-agentcore-session-id": "session-a",
            "X-AgentCore-User-ID": "user-a",
            "X-Request-ID": "request-a",
        }
    )

    assert context.headers == {
        "x-agentcore-session-id": "session-a",
        "x-agentcore-user-id": "user-a",
        "x-request-id": "request-a",
    }


def test_context_accepts_requests_without_agentcore_business_headers() -> None:
    context = request_context_from_headers({"X-Request-ID": "request-a"})

    assert context.headers == {"x-request-id": "request-a"}


def test_context_repr_does_not_expose_header_values() -> None:
    context = request_context_from_headers({"Authorization": "Bearer secret"})

    assert repr(context) == "RequestContext(headers=<redacted>)"


def test_context_headers_are_a_read_only_copy() -> None:
    source = {"X-Request-ID": "request-a"}
    context = request_context_from_headers(source)

    source["X-Request-ID"] = "changed"
    with pytest.raises(TypeError):
        context.headers["x-request-id"] = "changed"  # type: ignore[index]

    assert context.headers["x-request-id"] == "request-a"


def test_context_does_not_assign_special_precedence_to_agentrun_headers() -> None:
    context = request_context_from_headers(
        {
            "X-AgentCore-Session-ID": "agentcore-a",
            "X-AgentRun-Conversation-ID": "conversation-a",
            "X-AgentRun-User-ID": "user-a",
        }
    )

    assert context.headers["x-agentcore-session-id"] == "agentcore-a"
    assert context.headers["x-agentrun-conversation-id"] == "conversation-a"
    assert context.headers["x-agentrun-user-id"] == "user-a"


def test_context_is_request_scoped() -> None:
    context = request_context_from_headers({"X-Request-ID": "request-a"})

    with use_context(context):
        assert current_context() is context

    assert current_context(required=False) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["sync", "async", "cancelled"])
async def test_cleanup_failure_still_waits_for_other_resources(failure: str) -> None:
    context = request_context_from_headers({})
    release = asyncio.Event()
    cleaned: list[str] = []

    async def fail_async() -> None:
        if failure == "cancelled":
            raise asyncio.CancelledError()
        raise RuntimeError("cleanup failed")

    def fail() -> Awaitable[None]:
        if failure == "sync":
            raise RuntimeError("cleanup failed")
        return fail_async()

    async def close_resource() -> None:
        await release.wait()
        cleaned.append("closed")

    context.on_close(fail)
    context.on_close(close_resource)
    closing = asyncio.create_task(context.aclose())
    try:
        done, _ = await asyncio.wait({closing}, timeout=0.01)
        assert not done, "Cleanup must wait for the remaining resource even after a failure"
    finally:
        release.set()
        error = asyncio.CancelledError if failure == "cancelled" else RuntimeError
        with pytest.raises(error):
            await closing
    assert cleaned == ["closed"]
    await context.aclose()
    assert cleaned == ["closed"], "A second close must not run cleanups again"


@pytest.mark.asyncio
async def test_cleanup_waits_for_all_failures_and_preserves_first_error() -> None:
    context = request_context_from_headers({})
    failures = [RuntimeError("first"), ValueError("second")]
    cleaned: list[int] = []

    async def fail(index: int) -> None:
        cleaned.append(index)
        raise failures[index]

    context.on_close(lambda: fail(0))
    context.on_close(lambda: fail(1))
    with pytest.raises(RuntimeError) as raised:
        await context.aclose()
    assert raised.value is failures[0]
    assert cleaned == [0, 1]
