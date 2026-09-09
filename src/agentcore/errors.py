"""Stable, secret-safe SDK errors."""

from __future__ import annotations


class AgentCoreError(Exception):
    """Base error carrying a stable machine-readable code."""

    code = "AGENTCORE_ERROR"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigError(AgentCoreError):
    code = "CONFIG_INVALID"


class ContextError(AgentCoreError):
    code = "CONTEXT_INVALID"


class AuthenticationError(AgentCoreError):
    code = "AUTHENTICATION_FAILED"


class WorkloadIdentityNotConfiguredError(AuthenticationError):
    code = "WORKLOAD_IDENTITY_NOT_CONFIGURED"


class CredentialExchangeError(AuthenticationError):
    code = "CREDENTIAL_EXCHANGE_FAILED"


class WorkloadAccessTokenRejectedError(CredentialExchangeError):
    code = "WORKLOAD_ACCESS_TOKEN_REJECTED"


class ResourceNotConfiguredError(AgentCoreError):
    code = "RESOURCE_NOT_CONFIGURED"


class ModelConnectionNotFoundError(ResourceNotConfiguredError):
    code = "MODEL_CONNECTION_NOT_FOUND"


class MCPServerNotFoundError(ResourceNotConfiguredError):
    code = "MCP_SERVER_NOT_FOUND"


class InvocationError(AgentCoreError):
    code = "INVOCATION_FAILED"


class UnsupportedFeatureError(AgentCoreError):
    code = "UNSUPPORTED_FEATURE"


class MemoryValidationError(AgentCoreError):
    code = "MEMORY_VALIDATION_FAILED"


class MemoryAPIError(AgentCoreError):
    code = "MEMORY_API_FAILED"

    def __init__(
        self,
        operation: str,
        *,
        service_code: str | None = None,
        http_status_code: int | None = None,
        request_id: str | None = None,
    ) -> None:
        self.operation = operation
        self.service_code = service_code
        self.http_status_code = http_status_code
        self.request_id = request_id
        super().__init__(self._safe_message(operation))

    def _safe_message(self, operation: str) -> str:
        return f"AgentCore Memory operation {operation} failed"

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(operation={self.operation!r}, "
            f"service_code={self.service_code!r}, "
            f"http_status_code={self.http_status_code!r}, "
            f"request_id={self.request_id!r})"
        )


class MemoryContractError(AgentCoreError):
    code = "MEMORY_RESPONSE_INVALID"

    def __init__(self, operation: str, detail: str) -> None:
        self.operation = operation
        self.detail = detail
        super().__init__(f"AgentCore Memory {operation} returned an invalid response: {detail}")


class AddMemoriesOutcomeUnknownError(MemoryAPIError):
    code = "ADD_MEMORIES_OUTCOME_UNKNOWN"

    def _safe_message(self, operation: str) -> str:
        return "AgentCore Memory AddMemories outcome is unknown; the write may have succeeded"
