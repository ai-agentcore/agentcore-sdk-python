"""Legacy imports; implementation lives in agentcore_collaboration.worker."""

from agentcore_collaboration.worker import (
    WORKER_PROMPT,
    Collaboration,
    WorkerCollaboration,
    merge_tools,
)

__all__ = ["WORKER_PROMPT", "Collaboration", "WorkerCollaboration", "merge_tools"]
