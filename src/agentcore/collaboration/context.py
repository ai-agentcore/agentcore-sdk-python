"""Request-scoped collaboration context."""

from __future__ import annotations

import base64
import binascii
import json
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, overload

from agentcore.collaboration.errors import (
    CollaborationContextError,
    CollaborationContextRequiredError,
)

_HEADER = "x-agentcore-collaboration-context"
_SESSION_HEADER = "x-agentcore-session-id"
_MAX_HEADER_LENGTH = 8192
_MAX_FIELD_LENGTH = 1024


@dataclass(frozen=True, repr=False)
class CollaborationTurnContext:
    session_id: str | None
    team_id: str | None
    room_id: str
    event_id: str
    room_kind: str

    def __repr__(self) -> str:
        return "CollaborationTurnContext(<redacted>)"


_current_context: ContextVar[CollaborationTurnContext | None] = ContextVar(
    "agentcore_collaboration_context",
    default=None,
)


@overload
def current_collaboration_context(
    *, required: Literal[True] = True
) -> CollaborationTurnContext: ...


@overload
def current_collaboration_context(
    *, required: Literal[False]
) -> CollaborationTurnContext | None: ...


def current_collaboration_context(*, required: bool = True) -> CollaborationTurnContext | None:
    context = _current_context.get()
    if context is None and required:
        raise CollaborationContextRequiredError("No collaboration request context is active.")
    return context


@contextmanager
def bind_collaboration_context(
    headers: Mapping[str, str],
) -> Iterator[CollaborationTurnContext | None]:
    normalized = {key.lower(): value for key, value in headers.items()}
    context = _parse_header(
        normalized.get(_HEADER),
        session_id=_optional_text(normalized.get(_SESSION_HEADER)),
    )
    token = _current_context.set(context)
    try:
        yield context
    finally:
        _current_context.reset(token)


def _parse_header(
    value: str | None,
    *,
    session_id: str | None,
) -> CollaborationTurnContext | None:
    if value is None:
        return None
    if not value or len(value) > _MAX_HEADER_LENGTH:
        raise CollaborationContextError("collaboration context header is invalid")
    try:
        padding = "=" * (-len(value) % 4)
        decoded = base64.b64decode(value + padding, altchars=b"-_", validate=True)
        payload = json.loads(decoded)
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollaborationContextError("collaboration context header is invalid") from exc
    if (
        not isinstance(payload, dict)
        or type(payload.get("version")) is not int
        or payload["version"] != 1
    ):
        raise CollaborationContextError("collaboration context schema is invalid")
    room_kind = _required_field(payload, "roomKind")
    if room_kind not in {"dm", "group", "task"}:
        raise CollaborationContextError("collaboration context roomKind is invalid")
    team_id = _optional_field(payload, "teamId")
    if room_kind == "task" and team_id is None:
        raise CollaborationContextError("collaboration context requires teamId")
    return CollaborationTurnContext(
        session_id=session_id,
        team_id=team_id,
        room_id=_required_field(payload, "roomId"),
        event_id=_required_field(payload, "eventId"),
        room_kind=room_kind,
    )


def _required_field(payload: Mapping[str, object], name: str) -> str:
    value = _optional_field(payload, name)
    if value is None:
        raise CollaborationContextError(f"collaboration context requires {name}")
    return value


def _optional_field(payload: Mapping[str, object], name: str) -> str | None:
    value = payload.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > _MAX_FIELD_LENGTH:
        raise CollaborationContextError(f"collaboration context {name} is invalid")
    return value.strip()


def _optional_text(value: str | None) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
