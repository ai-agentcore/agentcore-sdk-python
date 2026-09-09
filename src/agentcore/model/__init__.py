"""Managed AgentCore Gateway and direct multi-provider model clients."""

from agentcore.controlplane import ModelDescriptor
from agentcore.model.client import AsyncModelClient

__all__ = ["AsyncModelClient", "ModelDescriptor"]
