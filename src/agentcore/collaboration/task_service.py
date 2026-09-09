"""Legacy imports; implementation lives in agentcore_collaboration.task_service."""

from agentcore_collaboration.task_service import (
    EndpointProvider,
    TaskServiceClient,
    TokenProvider,
    TokenRefresher,
)

__all__ = ["EndpointProvider", "TaskServiceClient", "TokenProvider", "TokenRefresher"]
