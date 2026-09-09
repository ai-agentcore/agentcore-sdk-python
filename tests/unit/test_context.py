from __future__ import annotations

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
