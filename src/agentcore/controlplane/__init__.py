"""Small typed facade over the generated AgentCore control-plane SDK."""

from agentcore.controlplane.client import (
    AgentCoreControlPlane,
    CredentialMetadata,
    MCPDescriptor,
    ModelDescriptor,
    SkillArtifact,
)

__all__ = [
    "AgentCoreControlPlane",
    "CredentialMetadata",
    "MCPDescriptor",
    "ModelDescriptor",
    "SkillArtifact",
]
