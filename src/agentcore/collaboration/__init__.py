"""Stable context contract and lazy legacy imports for optional collaboration."""

from typing import TYPE_CHECKING, Any

from agentcore.collaboration._optional import require_collaboration
from agentcore.collaboration.context import (
    CollaborationTurnContext,
    bind_collaboration_context,
    current_collaboration_context,
)

if TYPE_CHECKING:
    from agentcore_collaboration import (
        DEFAULT_TEAMS_PATH,
        Collaboration,
        TeamMemberSnapshot,
        TeamSnapshot,
        TeamsProvider,
        TeamsSnapshot,
        WorkerCollaboration,
    )

__all__ = [
    "CollaborationTurnContext",
    "Collaboration",
    "DEFAULT_TEAMS_PATH",
    "TeamMemberSnapshot",
    "TeamSnapshot",
    "TeamsProvider",
    "TeamsSnapshot",
    "WorkerCollaboration",
    "bind_collaboration_context",
    "current_collaboration_context",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(require_collaboration(), name)
