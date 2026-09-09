from __future__ import annotations

import asyncio
import base64
import json

import pytest

from agentcore.collaboration import (
    bind_collaboration_context,
    current_collaboration_context,
)
from agentcore.collaboration.errors import CollaborationContextError


def _header(**overrides: object) -> str:
    value = {
        "version": 1,
        "teamId": "team-alpha",
        "roomId": "!room:matrix.example.com",
        "eventId": "$event",
        "roomKind": "task",
        **overrides,
    }
    return (
        base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )


def _header_without_team(*, room_kind: str) -> str:
    value = {
        "version": 1,
        "roomId": "!room:matrix.example.com",
        "eventId": "$event",
        "roomKind": room_kind,
    }
    return (
        base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode())
        .decode()
        .rstrip("=")
    )


def test_collaboration_context_binds_header_for_one_request() -> None:
    with bind_collaboration_context(
        {
            "X-AgentCore-Session-ID": "session-1",
            "X-AgentCore-Collaboration-Context": _header(),
        }
    ):
        context = current_collaboration_context()
        assert context is not None
        assert context.session_id == "session-1"
        assert context.team_id == "team-alpha"
        assert context.room_id == "!room:matrix.example.com"
        assert context.event_id == "$event"
        assert context.room_kind == "task"

    assert current_collaboration_context(required=False) is None


@pytest.mark.parametrize("room_kind", ["dm", "group"])
def test_collaboration_context_allows_team_to_be_omitted_for_ordinary_rooms(
    room_kind: str,
) -> None:
    with bind_collaboration_context(
        {"X-AgentCore-Collaboration-Context": _header_without_team(room_kind=room_kind)}
    ):
        context = current_collaboration_context()
        assert context.team_id is None
        assert context.room_kind == room_kind


def test_collaboration_context_requires_team_for_task_room() -> None:
    with pytest.raises(CollaborationContextError, match="requires teamId"):
        with bind_collaboration_context(
            {"X-AgentCore-Collaboration-Context": _header_without_team(room_kind="task")}
        ):
            pass


@pytest.mark.asyncio
async def test_collaboration_context_is_isolated_between_concurrent_requests() -> None:
    async def read(team_id: str) -> str:
        with bind_collaboration_context(
            {"X-AgentCore-Collaboration-Context": _header(teamId=team_id)}
        ):
            await asyncio.sleep(0)
            return current_collaboration_context().team_id

    assert await asyncio.gather(read("team-a"), read("team-b")) == [
        "team-a",
        "team-b",
    ]
    assert current_collaboration_context(required=False) is None


@pytest.mark.parametrize("room_kind", [None, "", "team", "unknown"])
def test_collaboration_context_requires_supported_room_kind(room_kind: object) -> None:
    with pytest.raises(CollaborationContextError):
        with bind_collaboration_context(
            {"X-AgentCore-Collaboration-Context": _header(roomKind=room_kind)}
        ):
            pass


def test_collaboration_context_requires_integer_version() -> None:
    with pytest.raises(CollaborationContextError):
        with bind_collaboration_context(
            {"X-AgentCore-Collaboration-Context": _header(version=True)}
        ):
            pass
