"""Stable collaboration errors."""

from agentcore.errors import AgentCoreError, ConfigError, ContextError


class CollaborationError(AgentCoreError):
    code = "COLLABORATION_ERROR"
    retryable = False


class CollaborationConfigError(ConfigError, CollaborationError):
    code = "COLLABORATION_CONFIG_INVALID"
    retryable = True


class CollaborationDisabledError(CollaborationError):
    code = "COLLABORATION_DISABLED"


class CollaborationContextRequiredError(CollaborationError):
    code = "COLLABORATION_CONTEXT_REQUIRED"


class CollaborationContextError(ContextError, CollaborationError):
    code = "COLLABORATION_CONTEXT_INVALID"


class CollaborationTeamUnavailableError(CollaborationError):
    code = "COLLABORATION_TEAM_UNAVAILABLE"


class CollaborationRoleUnsupportedError(CollaborationError):
    code = "COLLABORATION_ROLE_UNSUPPORTED"


class CollaborationToolArgumentError(CollaborationError):
    code = "COLLABORATION_ARGUMENT_INVALID"


class CollaborationTaskUnauthorizedError(CollaborationError):
    code = "COLLABORATION_TASK_UNAUTHORIZED"


class CollaborationTaskInvalidError(CollaborationError):
    code = "COLLABORATION_TASK_INVALID"


class CollaborationTaskNotFoundError(CollaborationError):
    code = "COLLABORATION_TASK_NOT_FOUND"


class CollaborationTaskConflictError(CollaborationError):
    code = "COLLABORATION_TASK_CONFLICT"


class CollaborationFileTooLargeError(CollaborationError):
    code = "COLLABORATION_FILE_TOO_LARGE"


class CollaborationTaskUnavailableError(CollaborationError):
    code = "COLLABORATION_TASK_UNAVAILABLE"
    retryable = True
