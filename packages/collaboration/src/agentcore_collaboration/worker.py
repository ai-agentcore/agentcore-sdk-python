"""Explicit Worker prompt and tool composition."""

from __future__ import annotations

import base64
import binascii
import inspect
import mimetypes
import os
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path, PurePosixPath
from typing import Any

import anyio

from agentcore.collaboration.context import (
    CollaborationTurnContext,
    bind_collaboration_context,
    current_collaboration_context,
)
from agentcore.collaboration.errors import (
    CollaborationConfigError,
    CollaborationContextRequiredError,
    CollaborationDisabledError,
    CollaborationError,
    CollaborationRoleUnsupportedError,
    CollaborationTaskConflictError,
    CollaborationTaskNotFoundError,
    CollaborationTaskUnauthorizedError,
    CollaborationTaskUnavailableError,
    CollaborationTeamUnavailableError,
    CollaborationToolArgumentError,
)
from agentcore.integrations.common import Tool
from agentcore.runtime.environment import DEFAULT_ENV_PATH, RuntimeEnvironmentProvider
from agentcore.skill import Skill
from agentcore_collaboration.debug import DebugCollaborationRuntime
from agentcore_collaboration.skills import worker_skills
from agentcore_collaboration.teams import DEFAULT_TEAMS_PATH, TeamsProvider, TeamsSnapshot

WORKER_PROMPT = """
## Team collaboration capabilities

You can use the team collaboration capabilities described below. These rules apply to collaboration
operations and do not change the application's identity or the current user request.

Use task-execution for incoming task assignments and revision notifications. Use team and task
queries when the current request needs that information. Handle other requests through the
application's normal workflow; available collaboration tools are not an instruction to look for
work.
Explain a tool failure only when it affects the current request, and never infer a business result
from a failed query.

For assigned Subtask work, execute only the Subtask assigned to you. Do not create or manage
top-level Tasks, reassign Subtasks, or approve your own Results. Treat Task Service state and
authorization as authoritative.

At the start of every newly assigned Subtask or revision turn, you MUST load and follow
`task-execution`, including its Task/Subtask file workflow. Before synchronizing non-Task Team
files, load and follow `file-sharing`. If a required Skill is unavailable, do not perform
collaboration writes.

For assigned Subtask work, produce only the requested deliverables and perform only explicitly
required checks or checks necessary to establish usability. Once they pass, submit a short Result.
After submission succeeds, reply with one short sentence that it awaits review; do not restate the
Result, deliverables, checks, or identifiers. Disclose material limitations in the Result.

When configured Team membership, roles, or Matrix identities matter, call
`agentteams_get_team_context`; do not infer them from messages. It returns the configured Runtime
roster, not live Matrix room membership.

Do not inspect or expose credentials, tokens, authorization headers, signed URLs, or routing
metadata. Never include them in prompts, messages, Task state, Results, Events, or uploaded files.
""".strip()

_OBJECT = {"type": "object", "additionalProperties": True}
_STRING = {"type": "string", "minLength": 1}
_WORKSPACE_PATH = {
    "type": "string",
    "minLength": 1,
    "description": (
        "Path inside the collaboration workspace. Relative paths are resolved from its root."
    ),
}


TeamsSnapshotProvider = Callable[
    [],
    TeamsSnapshot | None | Awaitable[TeamsSnapshot | None],
]


class WorkerCollaboration:
    def __init__(
        self,
        teams: TeamsProvider | TeamsSnapshotProvider,
        task_client: Any,
        workspace: Path,
    ) -> None:
        self._teams = teams
        self._task_client = task_client
        self._workspace = workspace
        self._skills = worker_skills()
        self._tools = self._create_tools()

    def compose_prompt(self, user_prompt: str) -> str:
        collaboration_prompt = (
            f"{WORKER_PROMPT}\n\n"
            "## Collaboration Workspace\n\n"
            f"The local collaboration workspace is `{self._workspace}`. Pass `local_path` and "
            "`output_path` relative to this directory unless an absolute path is already known. "
            "Before uploading an artifact created elsewhere, stage it in this workspace using "
            "only filesystem capabilities already provided by the application."
        )
        return (
            f"{user_prompt.rstrip()}\n\n{collaboration_prompt}"
            if user_prompt.strip()
            else collaboration_prompt
        )

    def tools(self) -> list[Tool]:
        return list(self._tools)

    def skills(self) -> list[Skill]:
        return list(self._skills)

    def request_context(self, headers: Mapping[str, str]) -> AbstractContextManager[Any]:
        return bind_collaboration_context(headers)

    def merge_langchain_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        from agentcore.integrations.langchain import tools

        return merge_tools(user_tools, tools(self.tools()))

    def merge_langgraph_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        return self.merge_langchain_tools(user_tools)

    def merge_agentscope_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        from agentcore.integrations.agentscope import tools

        return merge_tools(user_tools, tools(self.tools()))

    def merge_google_adk_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        from agentcore.integrations.google_adk import tools

        return merge_tools(user_tools, tools(self.tools()))

    def merge_pydantic_ai_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        from agentcore.integrations.pydantic_ai import tools

        return merge_tools(user_tools, tools(self.tools()))

    def merge_crewai_tools(self, user_tools: Sequence[Any] = ()) -> list[Any]:
        from agentcore.integrations.crewai import tools

        return merge_tools(user_tools, tools(self.tools()))

    def _create_tools(self) -> list[Tool]:
        return [
            _tool(
                "agentteams_get_team_context",
                "Read-only. Get the current AgentCore Worker identity, Team memberships, roles, "
                "and members.",
                {},
                self._get_team_context,
            ),
            _tool(
                "agentteams_list_tasks",
                "Read-only. List authoritative Tasks visible to this Worker.",
                {
                    "status": _STRING,
                    "team_id": _STRING,
                    "assigned_to": _STRING,
                    "search": _STRING,
                    "cursor": _STRING,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                self._list_tasks,
            ),
            _tool(
                "agentteams_get_task",
                "Read-only. Get authoritative Task details by exact ID.",
                {"task_id": _STRING},
                self._get_task,
                required=("task_id",),
            ),
            _tool(
                "agentteams_get_subtask",
                "Read-only. Get authoritative Subtask details by exact ID.",
                {"subtask_id": _STRING},
                self._get_subtask,
                required=("subtask_id",),
            ),
            _tool(
                "agentteams_list_subtasks",
                "Read-only. List authoritative Subtasks visible to this Worker.",
                {"task_id": _STRING, "status": _STRING, "assigned_to": _STRING},
                self._list_subtasks,
            ),
            _tool(
                "agentteams_ack_subtask",
                "Acknowledge an assigned Subtask by its exact ID.",
                {"subtask_id": _STRING},
                self._ack_subtask,
                required=("subtask_id",),
            ),
            _tool(
                "agentteams_report_subtask_progress",
                "Report useful evidence of progress for an assigned Subtask by its exact ID.",
                {"subtask_id": _STRING, "content": _OBJECT},
                self._report_subtask_progress,
                required=("subtask_id", "content"),
            ),
            _tool(
                "agentteams_heartbeat_subtask",
                "Report liveness for an assigned Subtask by its exact ID.",
                {"subtask_id": _STRING},
                self._heartbeat_subtask,
                required=("subtask_id",),
            ),
            _tool(
                "agentteams_block_subtask",
                "Block an assigned Subtask by its exact ID with a concrete reason.",
                {"subtask_id": _STRING, "reason": _STRING, "evidence": _OBJECT},
                self._block_subtask,
                required=("subtask_id", "reason"),
            ),
            _file_list_tool("task", self._list_task_files),
            _file_list_tool("subtask", self._list_subtask_files),
            _file_read_tool("task", self._read_task_file),
            _file_read_tool("subtask", self._read_subtask_file),
            _file_download_tool("task", self._download_task_file),
            _file_download_tool("subtask", self._download_subtask_file),
            _tool(
                "agentteams_write_subtask_file",
                "Persist UTF-8 or Base64 content under an assigned Subtask by its exact ID.",
                {
                    "subtask_id": _STRING,
                    "path": _STRING,
                    "content": {"type": "string"},
                    "encoding": {"type": "string", "enum": ["utf-8", "base64"]},
                    "content_type": _STRING,
                },
                self._write_subtask_file,
                required=("subtask_id", "path", "content"),
            ),
            _tool(
                "agentteams_write_subtask_file_from_path",
                "Persist a workspace file under an assigned Subtask by its exact ID.",
                {
                    "subtask_id": _STRING,
                    "path": _STRING,
                    "local_path": _WORKSPACE_PATH,
                    "content_type": _STRING,
                },
                self._write_subtask_file_from_path,
                required=("subtask_id", "path", "local_path"),
            ),
            _tool(
                "agentteams_submit_subtask_result",
                "Submit a Result for an assigned Subtask by its exact ID.",
                {
                    "subtask_id": _STRING,
                    "summary": _STRING,
                    "file_refs": {"type": "array", "items": _STRING},
                },
                self._submit_subtask_result,
                required=("subtask_id", "summary"),
            ),
            _tool(
                "agentteams_list_results",
                "Read-only. List authoritative Result history for a Task by exact Task ID.",
                {"task_id": _STRING, "subtask_id": _STRING},
                self._list_results,
                required=("task_id",),
            ),
            _tool(
                "agentteams_list_task_events",
                "Read-only. List authoritative Event history for a Task by exact Task ID.",
                {
                    "task_id": _STRING,
                    "subtask_id": _STRING,
                    "event_type": _STRING,
                    "cursor": _STRING,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                self._list_task_events,
                required=("task_id",),
            ),
            _tool(
                "agentteams_filesync_push",
                "Persist a file or directory from the configured collaboration workspace in the "
                "current Team shared space. The SDK selects and validates the Team.",
                {"path": _STRING, "local_path": _WORKSPACE_PATH},
                self._filesync_push,
                required=("path", "local_path"),
            ),
            _tool(
                "agentteams_filesync_pull",
                "Download files from the current Team shared space into the configured "
                "collaboration workspace. The SDK selects and validates the Team.",
                {
                    "path": _STRING,
                    "local_path": _WORKSPACE_PATH,
                    "overwrite": {"type": "boolean"},
                },
                self._filesync_pull,
                required=("path", "local_path"),
            ),
            _tool(
                "agentteams_filesync_list",
                "Read-only. List files in the current Team shared space. The SDK selects and "
                "validates the Team.",
                {
                    "path": _STRING,
                    "cursor": _STRING,
                    "limit": {"type": "integer", "minimum": 1, "maximum": 100},
                },
                self._filesync_list,
                required=("path",),
            ),
            _tool(
                "agentteams_filesync_stat",
                "Read-only. Inspect a file or directory in the current Team shared space. The SDK "
                "selects and validates the Team.",
                {"path": _STRING},
                self._filesync_stat,
                required=("path",),
            ),
        ]

    async def _get_team_context(self, _arguments: dict[str, Any]) -> dict[str, Any]:
        try:
            snapshot, _context = await self._require_read()
            return {"ok": True, "data": _team_context(snapshot)}
        except CollaborationError as exc:
            return _error_result(exc)

    async def _list_tasks(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_read(
            lambda snapshot, _context: self._task_client.list_tasks(
                snapshot,
                status=_optional_string(arguments, "status"),
                team_id=_optional_string(arguments, "team_id"),
                assigned_to=_optional_string(arguments, "assigned_to"),
                search=_optional_string(arguments, "search"),
                cursor=_optional_string(arguments, "cursor"),
                limit=_limit(arguments),
            )
        )

    async def _get_task(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_read(
            lambda snapshot, _context: self._task_client.get_task(
                snapshot,
                _required_string(arguments, "task_id"),
            )
        )

    async def _get_subtask(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_read(
            lambda snapshot, _context: self._task_client.get_subtask(
                snapshot,
                _required_string(arguments, "subtask_id"),
            )
        )

    async def _list_subtasks(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_read(
            lambda snapshot, _context: self._task_client.list_subtasks(
                snapshot,
                task_id=_optional_string(arguments, "task_id"),
                status=_optional_string(arguments, "status"),
                assigned_to=_optional_string(arguments, "assigned_to"),
            )
        )

    async def _ack_subtask(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_subtask_write(
            arguments,
            lambda snapshot, context, subtask_id: self._task_client.ack_subtask(
                snapshot,
                subtask_id,
                event_id=context.event_id,
            )
        )

    async def _report_subtask_progress(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_subtask_write(
            arguments,
            lambda snapshot, context, subtask_id: self._task_client.report_subtask_progress(
                snapshot,
                subtask_id,
                content=_required_mapping(arguments, "content"),
                event_id=context.event_id,
            )
        )

    async def _heartbeat_subtask(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_subtask_write(
            arguments,
            lambda snapshot, _context, subtask_id: self._task_client.heartbeat_subtask(
                snapshot,
                subtask_id,
            )
        )

    async def _block_subtask(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_subtask_write(
            arguments,
            lambda snapshot, context, subtask_id: self._task_client.block_subtask(
                snapshot,
                subtask_id,
                reason=_required_string(arguments, "reason"),
                evidence=_optional_mapping(arguments, "evidence"),
                event_id=context.event_id,
            )
        )

    async def _list_task_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_read(
            lambda snapshot, _context: self._task_client.list_task_files(
                snapshot,
                _required_string(arguments, "task_id"),
                prefix=_optional_string(arguments, "prefix"),
                cursor=_optional_string(arguments, "cursor"),
                limit=_limit(arguments),
            )
        )

    async def _list_subtask_files(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_read(
            lambda snapshot, _context: self._task_client.list_subtask_files(
                snapshot,
                _required_string(arguments, "subtask_id"),
                prefix=_optional_string(arguments, "prefix"),
                cursor=_optional_string(arguments, "cursor"),
                limit=_limit(arguments),
            )
        )

    async def _read_task_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._execute_read(
            lambda snapshot, _context: self._task_client.read_task_file(
                snapshot,
                _required_string(arguments, "task_id"),
                file_ref=_required_string(arguments, "file_ref"),
            )
        )
        return _encode_file_result(result)

    async def _read_subtask_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        result = await self._execute_read(
            lambda snapshot, _context: self._task_client.read_subtask_file(
                snapshot,
                _required_string(arguments, "subtask_id"),
                file_ref=_required_string(arguments, "file_ref"),
            )
        )
        return _encode_file_result(result)

    async def _download_task_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        async def operation(
            snapshot: TeamsSnapshot, _context: CollaborationTurnContext | None
        ) -> dict[str, Any]:
            task_id = _required_string(arguments, "task_id")
            return await _download_local_file(
                _output_path(arguments, self._workspace),
                self._workspace,
                overwrite=_optional_bool(arguments, "overwrite", default=False),
                download=lambda destination: self._task_client.download_task_file_to_path(
                    snapshot,
                    task_id,
                    file_ref=_required_string(arguments, "file_ref"),
                    destination=destination,
                ),
            )

        return await self._execute_read(operation)

    async def _download_subtask_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        async def operation(
            snapshot: TeamsSnapshot, _context: CollaborationTurnContext | None
        ) -> dict[str, Any]:
            subtask_id = _required_string(arguments, "subtask_id")
            return await _download_local_file(
                _output_path(arguments, self._workspace),
                self._workspace,
                overwrite=_optional_bool(arguments, "overwrite", default=False),
                download=lambda destination: self._task_client.download_subtask_file_to_path(
                    snapshot,
                    subtask_id,
                    file_ref=_required_string(arguments, "file_ref"),
                    destination=destination,
                ),
            )

        return await self._execute_read(operation)

    async def _write_subtask_file(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_subtask_write(
            arguments,
            lambda snapshot, _context, subtask_id: self._task_client.write_subtask_file(
                snapshot,
                subtask_id,
                path=_required_string(arguments, "path"),
                content=_file_content(arguments),
                content_type=_optional_string(arguments, "content_type")
                or "application/octet-stream",
            )
        )

    async def _write_subtask_file_from_path(self, arguments: dict[str, Any]) -> dict[str, Any]:
        async def operation(
            snapshot: TeamsSnapshot,
            _context: CollaborationTurnContext,
            subtask_id: str,
        ) -> Any:
            source = _source_path(arguments, self._workspace, allow_directory=False)
            return await self._task_client.write_subtask_file_from_path(
                snapshot,
                subtask_id,
                path=_required_string(arguments, "path"),
                source=source,
                content_type=_content_type(arguments, source),
            )

        return await self._execute_subtask_write(arguments, operation)

    async def _submit_subtask_result(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_subtask_write(
            arguments,
            lambda snapshot, context, subtask_id: self._task_client.submit_subtask_result(
                snapshot,
                subtask_id,
                summary=_required_string(arguments, "summary"),
                file_refs=_string_list(arguments, "file_refs"),
                event_id=context.event_id,
            )
        )

    async def _list_results(self, arguments: dict[str, Any]) -> dict[str, Any]:
        async def operation(
            snapshot: TeamsSnapshot, context: CollaborationTurnContext | None
        ) -> Any:
            task_id, subtask_id = _history_target(arguments)
            return await self._task_client.list_results(
                snapshot,
                task_id,
                subtask_id=subtask_id,
            )

        return await self._execute_read(operation)

    async def _list_task_events(self, arguments: dict[str, Any]) -> dict[str, Any]:
        async def operation(
            snapshot: TeamsSnapshot, context: CollaborationTurnContext | None
        ) -> Any:
            task_id, subtask_id = _history_target(arguments)
            return await self._task_client.list_events(
                snapshot,
                task_id,
                subtask_id=subtask_id,
                event_type=_optional_string(arguments, "event_type"),
                cursor=_optional_string(arguments, "cursor"),
                limit=_limit(arguments, default=50),
            )

        return await self._execute_read(operation)

    async def _filesync_push(self, arguments: dict[str, Any]) -> dict[str, Any]:
        async def operation(
            snapshot: TeamsSnapshot, team_id: str
        ) -> dict[str, Any]:
            remote_path = _team_path(arguments)
            source = _source_path(arguments, self._workspace, allow_directory=True)
            files = await anyio.to_thread.run_sync(_collect_local_files, source, remote_path)
            transferred: list[dict[str, Any]] = []
            for local_file, target_path in files:
                result = await self._task_client.write_team_file_from_path(
                    snapshot,
                    team_id,
                    path=target_path,
                    source=local_file,
                    content_type=mimetypes.guess_type(local_file.name)[0]
                    or "application/octet-stream",
                )
                transferred.append(
                    {
                        "path": target_path,
                        "localPath": str(local_file),
                        "result": result,
                    }
                )
            return {
                "action": "push",
                "path": remote_path,
                "localPath": str(source),
                "transferred": len(transferred),
                "files": transferred,
            }

        return await self._execute_team_turn(operation)

    async def _filesync_pull(self, arguments: dict[str, Any]) -> dict[str, Any]:
        async def operation(
            snapshot: TeamsSnapshot, team_id: str
        ) -> dict[str, Any]:
            remote_path = _team_path(arguments)
            local_path = _workspace_path(
                _local_path(arguments, "local_path", self._workspace),
                self._workspace,
            )
            if remote_path == "shared":
                selected = await self._list_all_team_files(snapshot, team_id, remote_path)
                exact_file = False
            else:
                try:
                    metadata = await self._task_client.stat_team_file(
                        snapshot,
                        team_id,
                        path=remote_path,
                    )
                    if not isinstance(metadata, dict):
                        raise CollaborationTaskUnavailableError(
                            "Task Service returned invalid Team file metadata."
                        )
                    selected = [{**metadata, "path": remote_path}]
                    exact_file = True
                except (
                    CollaborationTaskConflictError,
                    CollaborationTaskNotFoundError,
                ) as original:
                    selected = await self._list_all_team_files(
                        snapshot, team_id, remote_path
                    )
                    if not selected:
                        raise original
                    exact_file = False
            overwrite = _optional_bool(arguments, "overwrite", default=True)
            transferred: list[dict[str, Any]] = []
            for entry in selected:
                entry_path = entry["path"]
                output = (
                    local_path
                    if exact_file
                    else local_path / PurePosixPath(entry_path).relative_to(remote_path)
                )
                content = await self._task_client.read_team_file(
                    snapshot,
                    team_id,
                    path=entry_path,
                )
                written = await _write_local_file(
                    output,
                    content,
                    self._workspace,
                    overwrite=overwrite,
                )
                transferred.append({"path": entry_path, "output": written["path"]})
            return {
                "action": "pull",
                "path": remote_path,
                "localPath": str(local_path),
                "transferred": len(transferred),
                "files": transferred,
            }

        return await self._execute_team_turn(operation)

    async def _filesync_list(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self._execute_team_turn(
            lambda snapshot, team_id: self._task_client.list_team_files(
                snapshot,
                team_id,
                path=_team_path(arguments),
                cursor=_optional_string(arguments, "cursor"),
                limit=_limit(arguments),
            )
        )

    async def _filesync_stat(self, arguments: dict[str, Any]) -> dict[str, Any]:
        async def operation(
            snapshot: TeamsSnapshot, team_id: str
        ) -> dict[str, Any]:
            path = _team_path(arguments)
            if path == "shared":
                entries = await self._list_all_team_files(snapshot, team_id, path)
                return {"kind": "directory", "path": path, "entries": len(entries)}
            try:
                result = await self._task_client.stat_team_file(
                    snapshot,
                    team_id,
                    path=path,
                )
                if not isinstance(result, dict):
                    raise CollaborationTaskUnavailableError(
                        "Task Service returned invalid Team file metadata."
                    )
                return {"kind": "file", **result}
            except (CollaborationTaskConflictError, CollaborationTaskNotFoundError) as original:
                entries = await self._list_all_team_files(snapshot, team_id, path)
                if not entries:
                    raise original
                return {"kind": "directory", "path": path, "entries": len(entries)}

        return await self._execute_team_turn(operation)

    async def _list_all_team_files(
        self,
        snapshot: TeamsSnapshot,
        team_id: str,
        path: str,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            result = await self._task_client.list_team_files(
                snapshot,
                team_id,
                path=path,
                cursor=cursor,
                limit=100,
            )
            if not isinstance(result, dict) or not isinstance(result.get("items"), list):
                raise CollaborationTaskUnavailableError(
                    "Task Service returned an invalid Team file list."
                )
            for item in result["items"]:
                if not isinstance(item, dict) or not isinstance(item.get("path"), str):
                    raise CollaborationTaskUnavailableError(
                        "Task Service returned an invalid Team file list."
                    )
                try:
                    item_path = _validated_team_path(item["path"])
                except CollaborationToolArgumentError as exc:
                    raise CollaborationTaskUnavailableError(
                        "Task Service returned an invalid Team file list."
                    ) from exc
                if item_path != path and not item_path.startswith(f"{path}/"):
                    raise CollaborationTaskUnavailableError(
                        "Task Service returned an invalid Team file list."
                    )
                entries.append({**item, "path": item_path})
            next_cursor = result.get("nextCursor")
            if next_cursor is None:
                break
            if not isinstance(next_cursor, str) or not next_cursor:
                raise CollaborationTaskUnavailableError(
                    "Task Service returned an invalid Team file list."
                )
            cursor = next_cursor
        return sorted(entries, key=lambda item: item["path"])

    async def _execute_read(
        self,
        operation: Callable[[TeamsSnapshot, CollaborationTurnContext | None], Awaitable[Any]],
    ) -> dict[str, Any]:
        try:
            snapshot, context = await self._require_read()
            return {"ok": True, "data": await operation(snapshot, context)}
        except CollaborationError as exc:
            return _error_result(exc)

    async def _execute_subtask_write(
        self,
        arguments: Mapping[str, Any],
        operation: Callable[[TeamsSnapshot, CollaborationTurnContext, str], Awaitable[Any]],
    ) -> dict[str, Any]:
        try:
            snapshot, context = await self._require_write()
            subtask_id = _required_string(arguments, "subtask_id")
            await self._require_subtask_task_room(snapshot, context, subtask_id)
            return {"ok": True, "data": await operation(snapshot, context, subtask_id)}
        except CollaborationError as exc:
            return _error_result(exc)

    async def _require_subtask_task_room(
        self,
        snapshot: TeamsSnapshot,
        context: CollaborationTurnContext,
        subtask_id: str,
    ) -> None:
        subtask = await self._task_client.get_subtask(snapshot, subtask_id)
        if not isinstance(subtask, dict):
            raise CollaborationTaskUnavailableError(
                "Task Service returned an invalid Subtask response."
            )
        task_id = subtask.get("taskId")
        if not isinstance(task_id, str) or not task_id.strip():
            raise CollaborationTaskUnavailableError(
                "Task Service returned an invalid Subtask response."
            )
        task = await self._task_client.get_task(snapshot, task_id)
        if not isinstance(task, dict):
            raise CollaborationTaskUnavailableError(
                "Task Service returned an invalid Task response."
            )
        room_id = task.get("roomId")
        if not isinstance(room_id, str) or room_id != context.room_id:
            raise CollaborationTaskUnauthorizedError(
                "This operation must run in the owning Task Room."
            )

    async def _execute_team_turn(
        self,
        operation: Callable[[TeamsSnapshot, str], Awaitable[Any]],
    ) -> dict[str, Any]:
        try:
            snapshot, team_id = await self._require_team_turn()
            return {"ok": True, "data": await operation(snapshot, team_id)}
        except CollaborationError as exc:
            return _error_result(exc)

    async def _require_read(
        self,
    ) -> tuple[TeamsSnapshot, CollaborationTurnContext | None]:
        snapshot = await self._require_enabled()
        context = current_collaboration_context(required=False)
        if context is not None and context.team_id is not None:
            self._validate_worker_team(snapshot, context.team_id)
        elif not snapshot.teams:
            raise CollaborationTeamUnavailableError("This Agent has no configured Team membership.")
        elif not any(team.role == "worker" for team in snapshot.teams.values()):
            raise CollaborationRoleUnsupportedError(
                "The current Agent is not a Worker in any configured Team."
            )
        return snapshot, context

    async def _require_write(self) -> tuple[TeamsSnapshot, CollaborationTurnContext]:
        snapshot = await self._require_enabled()
        context = current_collaboration_context(required=False)
        if context is None:
            raise CollaborationContextRequiredError(
                "This action is only available from a collaboration request in its Task Room."
            )
        if context.room_kind != "task" or context.team_id is None:
            raise CollaborationTaskUnauthorizedError(
                "This operation must run in the owning Task Room."
            )
        self._validate_worker_team(snapshot, context.team_id)
        return snapshot, context

    async def _require_team_turn(self) -> tuple[TeamsSnapshot, str]:
        snapshot = await self._require_enabled()
        context = current_collaboration_context(required=False)
        if context is None:
            raise CollaborationContextRequiredError(
                "The SDK could not select a current Team for this operation."
            )
        if context.team_id is not None:
            self._validate_worker_team(snapshot, context.team_id)
            return snapshot, context.team_id
        worker_teams = [team.name for team in snapshot.teams.values() if team.role == "worker"]
        if len(worker_teams) != 1:
            raise CollaborationContextRequiredError(
                "The SDK could not select a current Team for this operation."
            )
        return snapshot, worker_teams[0]

    async def _require_enabled(self) -> TeamsSnapshot:
        provider = self._teams.snapshot if isinstance(self._teams, TeamsProvider) else self._teams
        snapshot = provider()
        if inspect.isawaitable(snapshot):
            snapshot = await snapshot
        if snapshot is None:
            raise CollaborationDisabledError("Collaboration is not enabled for this Agent.")
        return snapshot

    @staticmethod
    def _validate_worker_team(snapshot: TeamsSnapshot, team_id: str) -> None:
        team = snapshot.teams.get(team_id)
        if team is None:
            raise CollaborationTeamUnavailableError("The collaboration Team is unavailable.")
        if team.role != "worker":
            raise CollaborationRoleUnsupportedError(
                "The current Agent is not a Worker in this Team."
            )


class Collaboration:
    def __init__(
        self,
        *,
        teams_path: str | Path = DEFAULT_TEAMS_PATH,
        env_path: str | Path | None = None,
        workspace_dir: str | Path | None = None,
        task_client: Any = None,
        _debug_runtime: DebugCollaborationRuntime | None = None,
        _use_debug_teams: bool = True,
    ) -> None:
        self._teams = TeamsProvider(teams_path)
        self._env_path = env_path or os.getenv("AGENTCORE_ENV_PATH") or DEFAULT_ENV_PATH
        self._workspace_dir = workspace_dir
        self._task_client = task_client
        self._debug_runtime = _debug_runtime
        self._use_debug_teams = _use_debug_teams
        self._worker: WorkerCollaboration | None = None

    def worker(self) -> WorkerCollaboration:
        if self._worker is None:
            teams: TeamsProvider | TeamsSnapshotProvider = self._teams
            if self._debug_runtime is not None and self._use_debug_teams:
                teams = self._debug_runtime.teams_snapshot
            else:
                self._teams.snapshot()
            workspace = _workspace_root(
                self._workspace_dir or os.getenv("AGENT_WORKSPACE") or Path.cwd()
            )
            if self._task_client is None:
                from agentcore_collaboration.task_service import TaskServiceClient

                if self._debug_runtime is not None:
                    self._task_client = TaskServiceClient(
                        self._debug_runtime.token,
                        endpoint_provider=self._debug_runtime.endpoint,
                        token_refresher=self._debug_runtime.refresh_token,
                    )
                else:
                    runtime_environment = RuntimeEnvironmentProvider(self._env_path)
                    self._task_client = TaskServiceClient(
                        runtime_environment.value,
                        endpoint_provider=runtime_environment.task_service_endpoint,
                    )
            self._worker = WorkerCollaboration(teams, self._task_client, workspace)
        return self._worker

    async def aclose(self) -> None:
        close = getattr(self._task_client, "aclose", None)
        if close is not None:
            await close()


def _tool(
    name: str,
    description: str,
    properties: Mapping[str, Any],
    call: Callable[[dict[str, Any]], Awaitable[Any]],
    *,
    required: Sequence[str] = (),
) -> Tool:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": dict(properties),
        "additionalProperties": False,
    }
    if required:
        parameters["required"] = list(required)
    return Tool(name=name, description=description, parameters=parameters, _call=call)


def _file_list_tool(
    target: str,
    call: Callable[[dict[str, Any]], Awaitable[Any]],
) -> Tool:
    return _tool(
        f"agentteams_list_{target}_files",
        f"Read-only. List files visible in a {target.capitalize()} shared space by exact ID.",
        {
            f"{target}_id": _STRING,
            "prefix": {"type": "string"},
            "cursor": _STRING,
            "limit": {"type": "integer", "minimum": 1, "maximum": 100},
        },
        call,
        required=(f"{target}_id",),
    )


def _file_read_tool(
    target: str,
    call: Callable[[dict[str, Any]], Awaitable[Any]],
) -> Tool:
    return _tool(
        f"agentteams_read_{target}_file",
        f"Read-only. Read a file from a {target.capitalize()} shared space by exact ID.",
        {f"{target}_id": _STRING, "file_ref": _STRING},
        call,
        required=(f"{target}_id", "file_ref"),
    )


def _file_download_tool(
    target: str,
    call: Callable[[dict[str, Any]], Awaitable[Any]],
) -> Tool:
    return _tool(
        f"agentteams_download_{target}_file",
        f"Read-only remote operation. Download a {target.capitalize()} file into the configured "
        f"collaboration workspace by exact ID without exposing its signed URL.",
        {
            f"{target}_id": _STRING,
            "file_ref": _STRING,
            "output_path": _WORKSPACE_PATH,
            "overwrite": {"type": "boolean"},
        },
        call,
        required=(f"{target}_id", "file_ref", "output_path"),
    )


def _history_target(
    arguments: Mapping[str, Any],
) -> tuple[str, str | None]:
    return (
        _required_string(arguments, "task_id"),
        _optional_string(arguments, "subtask_id"),
    )


def _required_string(
    arguments: Mapping[str, Any],
    name: str,
    *,
    allow_empty: bool = False,
) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise CollaborationToolArgumentError(f"{name} must be a string.")
    return value if allow_empty else value.strip()


def _optional_string(arguments: Mapping[str, Any], name: str) -> str | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise CollaborationToolArgumentError(f"{name} must be a non-empty string.")
    return value.strip()


def _required_mapping(arguments: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = arguments.get(name)
    if not isinstance(value, dict):
        raise CollaborationToolArgumentError(f"{name} must be an object.")
    return value


def _optional_mapping(arguments: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    value = arguments.get(name)
    if value is None:
        return None
    if not isinstance(value, dict):
        raise CollaborationToolArgumentError(f"{name} must be an object.")
    return value


def _string_list(arguments: Mapping[str, Any], name: str) -> list[str]:
    value = arguments.get(name, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise CollaborationToolArgumentError(f"{name} must be an array of strings.")
    return [item.strip() for item in value]


def _optional_bool(
    arguments: Mapping[str, Any],
    name: str,
    *,
    default: bool,
) -> bool:
    value = arguments.get(name, default)
    if not isinstance(value, bool):
        raise CollaborationToolArgumentError(f"{name} must be a boolean.")
    return value


def _file_content(arguments: Mapping[str, Any]) -> bytes:
    content = _required_string(arguments, "content", allow_empty=True)
    encoding = _optional_string(arguments, "encoding") or "utf-8"
    if encoding == "utf-8":
        return content.encode()
    if encoding == "base64":
        try:
            return base64.b64decode(content, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise CollaborationToolArgumentError("content is not valid Base64.") from exc
    raise CollaborationToolArgumentError("encoding must be utf-8 or base64.")


def _limit(arguments: Mapping[str, Any], *, default: int = 100) -> int:
    value = arguments.get("limit", default)
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 100:
        raise CollaborationToolArgumentError("limit must be between 1 and 100.")
    return value


def _local_path(arguments: Mapping[str, Any], name: str, workspace: Path) -> Path:
    path = Path(_required_string(arguments, name))
    return path if path.is_absolute() else workspace / path


def _source_path(
    arguments: Mapping[str, Any],
    workspace: Path,
    *,
    allow_directory: bool,
) -> Path:
    path = _workspace_path(
        _local_path(arguments, "local_path", workspace),
        workspace,
        must_exist=True,
    )
    try:
        path.lstat()
    except OSError as exc:
        raise CollaborationToolArgumentError("local_path is unavailable.") from exc
    if path.is_symlink():
        raise CollaborationToolArgumentError("local_path must not be a symbolic link.")
    if path.is_file():
        return path
    if allow_directory and path.is_dir():
        return path
    expected = "a file or directory" if allow_directory else "a file"
    raise CollaborationToolArgumentError(f"local_path must be {expected}.")


def _output_path(arguments: Mapping[str, Any], workspace: Path) -> Path:
    return _workspace_path(_local_path(arguments, "output_path", workspace), workspace)


def _workspace_root(value: str | Path) -> Path:
    try:
        root = Path(value).resolve(strict=True)
    except OSError as exc:
        raise CollaborationConfigError("The collaboration workspace is unavailable.") from exc
    if not root.is_dir():
        raise CollaborationConfigError("The collaboration workspace must be a directory.")
    return root


def _workspace_path(path: Path, workspace: Path, *, must_exist: bool = False) -> Path:
    lexical = Path(os.path.abspath(path))
    try:
        relative = lexical.relative_to(workspace)
    except ValueError as exc:
        raise CollaborationToolArgumentError(
            "The local path must stay within the collaboration workspace."
        ) from exc
    current = workspace
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise CollaborationToolArgumentError("The local path must not contain symbolic links.")
    try:
        resolved = lexical.resolve(strict=must_exist)
        resolved.relative_to(workspace)
    except (OSError, ValueError) as exc:
        raise CollaborationToolArgumentError("The local path is unavailable.") from exc
    return resolved


def _content_type(arguments: Mapping[str, Any], source: Path) -> str:
    return (
        _optional_string(arguments, "content_type")
        or mimetypes.guess_type(source.name)[0]
        or "application/octet-stream"
    )


def _team_path(arguments: Mapping[str, Any]) -> str:
    return _validated_team_path(_required_string(arguments, "path"))


def _validated_team_path(value: str) -> str:
    normalized = value[:-1] if value.endswith("/") else value
    segments = normalized.split("/")
    if (
        not normalized
        or normalized.startswith("/")
        or "\\" in normalized
        or segments[0] != "shared"
        or any(not segment or segment in {".", ".."} for segment in segments)
    ):
        raise CollaborationToolArgumentError("path must stay within shared/**.")
    if len(segments) > 1 and segments[1] in {"tasks", "subtasks"}:
        raise CollaborationToolArgumentError(
            "shared/tasks/** and shared/subtasks/** are owned by Task Service."
        )
    return normalized


def _collect_local_files(source: Path, remote_path: str) -> list[tuple[Path, str]]:
    if source.is_file():
        return [(source, remote_path)]
    files: list[tuple[Path, str]] = []
    for candidate in sorted(source.rglob("*")):
        if candidate.is_symlink():
            raise CollaborationToolArgumentError("Team filesync does not support symbolic links.")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise CollaborationToolArgumentError(
                "Team filesync supports only ordinary files and directories."
            )
        relative = candidate.relative_to(source).as_posix()
        target_path = _validated_team_path(f"{remote_path}/{relative}")
        files.append((candidate, target_path))
    return files


async def _write_local_file(
    output: Path,
    content: bytes,
    workspace: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    if not isinstance(content, bytes):
        raise CollaborationTaskUnavailableError("Task Service returned invalid file content.")
    return await anyio.to_thread.run_sync(
        lambda: _write_local_file_sync(output, content, workspace, overwrite=overwrite)
    )


def _write_local_file_sync(
    output: Path,
    content: bytes,
    workspace: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    output = _workspace_path(output, workspace)
    if output.is_symlink() or output.is_dir():
        raise CollaborationToolArgumentError("The local output path is unsafe.")
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("wb" if overwrite else "xb") as target:
            target.write(content)
    except FileExistsError as exc:
        raise CollaborationToolArgumentError(
            "The local output already exists; set overwrite=true to replace it."
        ) from exc
    except OSError as exc:
        raise CollaborationToolArgumentError("The local output path is unavailable.") from exc
    return {"path": str(output), "size": len(content)}


async def _download_local_file(
    output: Path,
    workspace: Path,
    *,
    overwrite: bool,
    download: Callable[[Path], Awaitable[int]],
) -> dict[str, Any]:
    output, temporary = await anyio.to_thread.run_sync(
        lambda: _prepare_download(output, workspace, overwrite=overwrite)
    )
    try:
        size = await download(temporary)
        await anyio.to_thread.run_sync(
            lambda: _commit_download(temporary, output, overwrite=overwrite)
        )
        return {"path": str(output), "size": size}
    finally:
        await anyio.to_thread.run_sync(lambda: temporary.unlink(missing_ok=True))


def _prepare_download(output: Path, workspace: Path, *, overwrite: bool) -> tuple[Path, Path]:
    output = _workspace_path(output, workspace)
    if output.is_symlink() or output.is_dir():
        raise CollaborationToolArgumentError("The local output path is unsafe.")
    if output.exists() and not overwrite:
        raise CollaborationToolArgumentError(
            "The local output already exists; set overwrite=true to replace it."
        )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output = _workspace_path(output, workspace)
        descriptor, name = tempfile.mkstemp(prefix=".agentcore-download-", dir=output.parent)
        os.close(descriptor)
    except OSError as exc:
        raise CollaborationToolArgumentError("The local output path is unavailable.") from exc
    return output, Path(name)


def _commit_download(temporary: Path, output: Path, *, overwrite: bool) -> None:
    try:
        if overwrite:
            os.replace(temporary, output)
        else:
            os.link(temporary, output)
            temporary.unlink()
    except FileExistsError as exc:
        raise CollaborationToolArgumentError(
            "The local output already exists; set overwrite=true to replace it."
        ) from exc
    except OSError as exc:
        raise CollaborationToolArgumentError("The local output path is unavailable.") from exc


def _encode_file_result(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("ok") or not isinstance(result.get("data"), bytes):
        return result
    content = result["data"]
    try:
        data = {"encoding": "utf-8", "content": content.decode()}
    except UnicodeDecodeError:
        data = {"encoding": "base64", "content": base64.b64encode(content).decode()}
    return {"ok": True, "data": data}


def _team_context(snapshot: TeamsSnapshot) -> dict[str, Any]:
    member = _without_none(
        {
            "name": snapshot.self_name,
            "runtimeName": snapshot.runtime_name,
            "matrixUserId": snapshot.self_matrix_user_id,
            "personalRoomId": snapshot.self_personal_room_id,
        }
    )
    teams: list[dict[str, Any]] = []
    for team in snapshot.teams.values():
        members = [
            _without_none(
                {
                    "type": item.type,
                    "name": item.name,
                    "runtimeName": item.runtime_name,
                    "role": item.role,
                    "matrixUserId": item.matrix_user_id,
                    "personalRoomId": item.personal_room_id,
                }
            )
            for item in team.members
        ]
        teams.append(
            _without_none(
                {
                    "name": team.name,
                    "teamRoomId": team.room_id,
                    "role": team.role,
                    "members": members,
                }
            )
        )
    return _without_none(
        {
            "member": member,
            "defaultTeamName": snapshot.default_team_name,
            "teams": teams,
        }
    )


def _without_none(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _error_result(error: CollaborationError) -> dict[str, Any]:
    return {
        "ok": False,
        "code": error.code,
        "retryable": error.retryable,
        "message": error.message,
    }


def merge_tools(user_tools: Sequence[Any], worker_tools: Sequence[Any]) -> list[Any]:
    names: set[str] = set()
    merged: list[Any] = []
    for tool in [*user_tools, *worker_tools]:
        name = getattr(tool, "name", None)
        if not isinstance(name, str) or not name:
            raise ValueError("tools must expose a non-empty name")
        if name in names:
            raise ValueError(f"duplicate tool name: {name}")
        names.add(name)
        merged.append(tool)
    return merged
