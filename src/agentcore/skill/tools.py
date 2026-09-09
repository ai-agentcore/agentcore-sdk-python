"""Canonical tools for reading and executing selected Skills."""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path, PurePosixPath
from typing import Any

import anyio

from agentcore.errors import ConfigError
from agentcore.integrations.common import Tool
from agentcore.skill._command import CommandResult, run_command, run_command_sync
from agentcore.skill.loader import Skill

_MAX_COMMAND_OUTPUT_BYTES = 100 * 1024
_DEFAULT_COMMAND_TIMEOUT_SECONDS = 300
CommandApproval = Callable[[str, str], bool]


def skill_tools(
    skills: Sequence[Skill],
    *,
    command_approval: CommandApproval | None = None,
    command_timeout: int = _DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> list[Tool]:
    if command_timeout <= 0:
        raise ConfigError("command_timeout must be positive")
    selected = {skill.name: skill for skill in skills}
    if len(selected) != len(skills):
        raise ConfigError("selected Skills must have unique names")

    def load_selected(arguments: dict[str, Any]) -> str:
        name = arguments.get("name")
        if name is None or name == "":
            return _json_result(
                skills=[
                    {"name": skill.name, "description": skill.description}
                    for skill in selected.values()
                ]
            )
        skill = selected.get(name) if isinstance(name, str) else None
        if skill is None:
            return _skill_not_found(selected, name)
        return _json_result(
            name=skill.name,
            description=skill.description,
            instruction=skill.instruction,
            files=_top_level_files(skill),
        )

    def read_selected(arguments: dict[str, Any]) -> str:
        name = arguments.get("name")
        skill = selected.get(name) if isinstance(name, str) else None
        if skill is None:
            return _skill_not_found(selected, name)
        resource = arguments.get("relative_path")
        if not isinstance(resource, str):
            return _json_result(error="relative_path must be a string")
        return _read_skill_resource(skill, resource)

    async def read_selected_async(arguments: dict[str, Any]) -> str:
        return await anyio.to_thread.run_sync(read_selected, arguments)

    tools = [
        Tool(
            name="load_skills",
            description=_load_skills_description(tuple(selected.values())),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "The name of the skill to load. If omitted, returns a list of "
                            "all available skills."
                        ),
                    }
                },
            },
            _call=load_selected,
            _sync_call=load_selected,
        ),
        Tool(
            name="read_skill_file",
            description=(
                "Read a file from a skill's directory, or list the directory when the "
                "relative path points to a directory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the skill containing the file.",
                    },
                    "relative_path": {
                        "type": "string",
                        "description": "Relative path to the file within the skill directory.",
                    },
                },
                "required": ["name", "relative_path"],
            },
            _call=read_selected_async,
            _sync_call=read_selected,
        ),
    ]
    if _execute_command_allowed():
        run_command_sync_call = partial(
            _execute_command,
            default_cwd=_default_command_cwd(skills),
            approval=command_approval,
            default_timeout=command_timeout,
        )

        async def execute_command(arguments: dict[str, Any]) -> str:
            return await _execute_command_async(
                arguments,
                default_cwd=_default_command_cwd(skills),
                approval=command_approval,
                default_timeout=command_timeout,
            )

        tools.append(
            Tool(
                name="execute_command",
                description=(
                    "Execute a shell command in the Agent container for a selected Skill. "
                    "Before calling this tool, display the exact command and ask the user for "
                    "confirmation. Returns stdout, stderr, exit_code, and timeout status."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "cwd": {
                            "type": "string",
                            "description": "Optional working directory.",
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Optional timeout in seconds.",
                        },
                    },
                    "required": ["command"],
                },
                _call=execute_command,
                _sync_call=run_command_sync_call,
            )
        )
    return tools


def _load_skills_description(skills: Sequence[Skill]) -> str:
    if not skills:
        return "Load skill instructions for the agent. No skills are currently selected."
    available = "\n".join(
        f"- {skill.name}{f': {skill.description}' if skill.description else ''}"
        for skill in skills
    )
    return (
        "Load skill instructions for the agent. Call without arguments to list all "
        "available skills, or with a skill name to get detailed instructions.\n\n"
        f"Available skills:\n{available}"
    )


def _skill_not_found(selected: Mapping[str, Skill], name: Any) -> str:
    available = ", ".join(selected) if selected else "none"
    return _json_result(error=f"Skill '{name}' not found. Available skills: {available}")


def _top_level_files(skill: Skill) -> list[str]:
    try:
        return [
            f"{entry.name}/" if entry.is_dir() else entry.name
            for entry in sorted(skill.root.iterdir())
            if not entry.name.startswith(".")
        ]
    except OSError:
        return []


def _read_skill_resource(skill: Skill, resource: str) -> str:
    relative = PurePosixPath(resource)
    if relative.is_absolute() or ".." in relative.parts:
        return _json_result(
            error=f"Path '{resource}' is outside the skill directory. Access denied."
        )
    target = skill.root.joinpath(*relative.parts)
    try:
        target.resolve().relative_to(skill.root.resolve())
    except (OSError, ValueError):
        return _json_result(
            error=f"Path '{resource}' is outside the skill directory. Access denied."
        )
    if target.is_symlink():
        return _json_result(
            error=f"Path '{resource}' is outside the skill directory. Access denied."
        )
    if target.is_dir():
        try:
            entries = [
                f"{entry.name}/" if entry.is_dir() else entry.name
                for entry in sorted(target.iterdir())
                if not entry.name.startswith(".")
            ]
        except OSError as exc:
            return _json_result(error=f"Failed to list directory: {exc}")
        return _json_result(files=entries)
    if not target.is_file():
        return _json_result(error=f"File '{resource}' not found in skill '{skill.name}'.")
    try:
        return _json_result(content=target.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return _json_result(
            error=f"File '{resource}' cannot be read as text. It may be a binary file."
        )
    except OSError as exc:
        return _json_result(error=f"Failed to read file: {exc}")


def _execute_command_allowed() -> bool:
    return os.getenv("ALLOW_EXECUTE_COMMAND", "true").lower() != "false"


def _default_command_cwd(skills: Sequence[Skill]) -> Path:
    roots = [skill.root for skill in skills]
    if len(roots) == 1:
        return roots[0]
    workspace_id = skills[0].workspace_id if skills else None
    if workspace_id is not None and all(
        skill.source == "agentcore" and skill.workspace_id == workspace_id for skill in skills
    ):
        return Path(os.path.commonpath(roots))
    if roots and len({root.parent for root in roots}) == 1:
        return roots[0].parent
    return Path.cwd()


def _execute_command(
    arguments: dict[str, Any],
    *,
    default_cwd: Path,
    approval: CommandApproval | None,
    default_timeout: int,
) -> str:
    parsed = _parse_command(arguments, default_cwd, default_timeout)
    if isinstance(parsed, str):
        return parsed
    command, cwd, timeout = parsed
    approval_error = _check_approval(approval, command, cwd)
    if approval_error is not None:
        return approval_error
    try:
        result = run_command_sync(
            command,
            cwd=cwd,
            timeout_seconds=timeout,
            output_limit=_MAX_COMMAND_OUTPUT_BYTES,
        )
    except OSError as exc:
        return _json_result(error=f"Failed to execute command: {exc}")
    return _command_result(result)


async def _execute_command_async(
    arguments: dict[str, Any],
    *,
    default_cwd: Path,
    approval: CommandApproval | None,
    default_timeout: int,
) -> str:
    parsed = _parse_command(arguments, default_cwd, default_timeout)
    if isinstance(parsed, str):
        return parsed
    command, cwd, timeout = parsed
    if approval is not None:
        approval_error = await anyio.to_thread.run_sync(_check_approval, approval, command, cwd)
        if approval_error is not None:
            return approval_error
    try:
        result = await run_command(
            command,
            cwd=cwd,
            timeout_seconds=timeout,
            output_limit=_MAX_COMMAND_OUTPUT_BYTES,
        )
    except OSError as exc:
        return _json_result(error=f"Failed to execute command: {exc}")
    return _command_result(result)


def _parse_command(
    arguments: dict[str, Any],
    default_cwd: Path,
    default_timeout: int,
) -> tuple[str, str, int] | str:
    command = arguments.get("command")
    if not isinstance(command, str) or not command:
        return _json_result(error="command must be a non-empty string")
    cwd = arguments.get("cwd", str(default_cwd))
    if not isinstance(cwd, str) or not Path(cwd).is_dir():
        return _json_result(error=f"Working directory {cwd!r} does not exist.")
    timeout = arguments.get("timeout", default_timeout)
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        return _json_result(error="timeout must be a positive integer")
    return command, cwd, timeout


def _check_approval(
    approval: CommandApproval | None,
    command: str,
    cwd: str,
) -> str | None:
    if approval is None:
        return None
    try:
        approved = approval(command, cwd)
    except Exception as exc:
        return _json_result(error=f"Command approval callback failed: {exc}")
    if not approved:
        return _json_result(error="Command execution rejected by user.")
    return None


def _command_result(result: CommandResult) -> str:
    return _json_result(
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.exit_code,
        timed_out=result.timed_out,
    )


def _json_result(**value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)
