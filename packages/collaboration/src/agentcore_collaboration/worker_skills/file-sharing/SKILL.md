---
name: file-sharing
description: Synchronize non-Task Team files; Task and Subtask files belong to task-execution.
version: "1"
---

# File sharing

Use this Skill only for non-Task Team files. Task and Subtask files are covered by `task-execution`.

## Non-Task Team files

Use `agentteams_filesync_push`, `agentteams_filesync_pull`, `agentteams_filesync_list`, and
`agentteams_filesync_stat`. The SDK selects and validates the current Team; never supply or infer a
Team ID. Use only `shared/**` paths outside `shared/tasks/**` and `shared/subtasks/**`.

Local staging is not durable or automatically synchronized. Pull required Team files before using
them after a Runtime restart or in a new Session. Push and pull never delete unrelated files.
Local input and output paths must stay inside the configured collaboration workspace, and symbolic
links are not synchronized.

Before pushing, inspect selected content for credentials, tokens, authorization headers, signed
URLs, private keys, or passwords. Remove prohibited material through an authorized redaction or
stop the upload. Upload only required files.

Team paths do not create Task Service `fileRef` values and must not be attached to a Result.
Task and Subtask files remain separate from the non-Task Team shared space.

## Failure handling

Report success only after the tool returns `ok: true`; do not re-read or compare state solely to
reconfirm a successful write. Use the structured error to choose the next step. Check paths only
when the error points to them; do not guess another Team or change credentials to bypass a failure.
The Task Service owns file-size enforcement. Do not repeat an unchanged non-retryable operation,
including a file rejected as too large.
