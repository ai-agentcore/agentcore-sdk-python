"""AgentCore credential providers."""

from agentcore.auth.access_key import AccessKeyCredential
from agentcore.auth.agent_sa_token import AgentSATokenProvider
from agentcore.auth.bound_credentials import AsyncBoundCredentials, BoundCredential
from agentcore.auth.resource_sts import ResourceCredential, ResourceSTSProvider
from agentcore.auth.workload_access_token import WorkloadAccessTokenProvider

__all__ = [
    "AccessKeyCredential",
    "AgentSATokenProvider",
    "AsyncBoundCredentials",
    "BoundCredential",
    "ResourceCredential",
    "ResourceSTSProvider",
    "WorkloadAccessTokenProvider",
]
