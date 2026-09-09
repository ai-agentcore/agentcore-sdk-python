from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import anyio
import pytest

from agentcore import AgentCore, AsyncAgentCore
from agentcore.collaboration import Collaboration
from agentcore.collaboration.errors import (
    CollaborationTaskInvalidError,
    CollaborationTaskNotFoundError,
)


def _context_header(
    *,
    room_id: str = "!alpha:matrix.example.com",
    room_kind: str = "task",
    team_id: str | None = "team-alpha",
) -> str:
    context = {
        "version": 1,
        "roomId": room_id,
        "eventId": "$event-1",
        "roomKind": room_kind,
        **({} if team_id is None else {"teamId": team_id}),
    }
    payload = json.dumps(
        context
    ).encode()
    return base64.urlsafe_b64encode(payload).decode().rstrip("=")


class _TaskClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def __getattr__(self, name: str) -> Any:
        async def call(_snapshot: object, *args: Any, **kwargs: Any) -> dict[str, Any]:
            self.calls.append((name, args, kwargs))
            if name == "get_subtask":
                return {
                    "operation": name,
                    "subTaskId": args[0],
                    "taskId": "task-1",
                }
            if name == "get_task":
                return {
                    "operation": name,
                    "taskId": args[0],
                    "roomId": "!alpha:matrix.example.com",
                }
            return {"operation": name}

        return call

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_runtime_worker_tools_are_explicit_and_disabled_without_teams(
    agent_config_path: Path,
    tmp_path: Path,
) -> None:
    async with AsyncAgentCore(
        agent_config_path,
        teams_path=tmp_path / "missing-teams.yaml",
    ) as runtime:
        worker = runtime.collaboration.worker()
        get_task = next(tool for tool in worker.tools() if tool.name == "agentteams_get_task")

        assert worker.compose_prompt("You are a code reviewer.").startswith(
            "You are a code reviewer."
        )
        result = await get_task.ainvoke({"task_id": "task-1"})

    assert result == {
        "ok": False,
        "code": "COLLABORATION_DISABLED",
        "retryable": False,
        "message": "Collaboration is not enabled for this Agent.",
    }


@pytest.mark.asyncio
async def test_worker_starts_before_matrix_registration(tmp_path: Path) -> None:
    path = tmp_path / "teams.yaml"
    path.write_text(
        "apiVersion: agentteams.io/v1alpha1\nkind: TeamsConfig\n"
        "metadata:\n  runtimeName: worker-a\nspec:\n"
        "  self:\n    name: worker-a\n    runtimeName: worker-a\n  teams: []\n",
        encoding="utf-8",
    )
    task_client = _TaskClient()
    collaboration = Collaboration(teams_path=path, task_client=task_client)
    try:
        worker = collaboration.worker()
        assert worker.compose_prompt("You are a code reviewer.").startswith(
            "You are a code reviewer."
        )
        assert worker.skills()
        get_task = next(tool for tool in worker.tools() if tool.name == "agentteams_get_task")
        result = await get_task.ainvoke({"task_id": "task-1"})
        assert result["code"] == "COLLABORATION_TEAM_UNAVAILABLE"
        assert result["message"] == "This Agent has no configured Team membership."
        assert not task_client.calls
    finally:
        await collaboration.aclose()


@pytest.mark.asyncio
async def test_worker_read_tool_works_without_request_context(
    agent_config_path: Path,
    teams_config_path: Path,
) -> None:
    task_client = _TaskClient()
    collaboration = Collaboration(
        teams_path=teams_config_path,
        task_client=task_client,
    )
    try:
        get_task = next(
            tool for tool in collaboration.worker().tools() if tool.name == "agentteams_get_task"
        )
        result = await get_task.ainvoke({"task_id": "task-1"})
    finally:
        await collaboration.aclose()

    assert result["ok"] is True
    assert result["data"]["operation"] == "get_task"
    assert task_client.calls == [("get_task", ("task-1",), {})]


@pytest.mark.asyncio
async def test_worker_write_tool_requires_managed_collaboration_turn(
    teams_config_path: Path,
) -> None:
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    ).worker()
    ack = next(tool for tool in worker.tools() if tool.name == "agentteams_ack_subtask")

    result = await ack.ainvoke({"subtask_id": "subtask-1"})
    write = next(tool for tool in worker.tools() if tool.name == "agentteams_write_subtask_file")
    invalid_write = await write.ainvoke({"encoding": "invalid"})

    assert result == {
        "ok": False,
        "code": "COLLABORATION_CONTEXT_REQUIRED",
        "retryable": False,
        "message": (
            "This action is only available from a collaboration request in its Task Room."
        ),
    }
    assert invalid_write == result


@pytest.mark.asyncio
async def test_worker_exposes_supported_tools_and_writes_explicit_subtask_in_task_room(
    teams_config_path: Path,
) -> None:
    task_client = _TaskClient()
    collaboration = Collaboration(
        teams_path=teams_config_path,
        task_client=task_client,
    )
    worker = collaboration.worker()
    tools = {tool.name: tool for tool in worker.tools()}

    assert set(tools) == {
        "agentteams_get_team_context",
        "agentteams_list_tasks",
        "agentteams_get_task",
        "agentteams_get_subtask",
        "agentteams_list_subtasks",
        "agentteams_ack_subtask",
        "agentteams_report_subtask_progress",
        "agentteams_heartbeat_subtask",
        "agentteams_block_subtask",
        "agentteams_list_task_files",
        "agentteams_list_subtask_files",
        "agentteams_read_task_file",
        "agentteams_read_subtask_file",
        "agentteams_write_subtask_file",
        "agentteams_submit_subtask_result",
        "agentteams_list_results",
        "agentteams_list_task_events",
        "agentteams_download_task_file",
        "agentteams_download_subtask_file",
        "agentteams_write_subtask_file_from_path",
        "agentteams_filesync_push",
        "agentteams_filesync_pull",
        "agentteams_filesync_list",
        "agentteams_filesync_stat",
    }
    assert "agentteams_resume_subtask" not in tools
    subtask_write_tools = (
        "agentteams_ack_subtask",
        "agentteams_report_subtask_progress",
        "agentteams_heartbeat_subtask",
        "agentteams_block_subtask",
        "agentteams_write_subtask_file",
        "agentteams_write_subtask_file_from_path",
        "agentteams_submit_subtask_result",
    )
    for name in subtask_write_tools:
        assert "subtask_id" in tools[name].parameters["properties"]
        assert "subtask_id" in tools[name].parameters["required"]
    for name, target in (
        ("agentteams_get_task", "task_id"),
        ("agentteams_get_subtask", "subtask_id"),
        ("agentteams_list_task_files", "task_id"),
        ("agentteams_list_subtask_files", "subtask_id"),
        ("agentteams_read_task_file", "task_id"),
        ("agentteams_read_subtask_file", "subtask_id"),
        ("agentteams_download_task_file", "task_id"),
        ("agentteams_download_subtask_file", "subtask_id"),
        ("agentteams_list_results", "task_id"),
        ("agentteams_list_task_events", "task_id"),
    ):
        assert target in tools[name].parameters["required"]
    for name in (
        "agentteams_filesync_push",
        "agentteams_filesync_pull",
        "agentteams_filesync_list",
        "agentteams_filesync_stat",
    ):
        assert "subtask_id" not in tools[name].parameters["properties"]
    for name in (
        "agentteams_filesync_push",
        "agentteams_filesync_pull",
        "agentteams_filesync_list",
        "agentteams_filesync_stat",
    ):
        assert "team_id" not in tools[name].parameters["properties"]
    assert all(
        "ordinary request" not in tool.description.lower()
        and "managed collaboration" not in tool.description.lower()
        for tool in tools.values()
    )

    headers = {
        "X-AgentCore-Session-ID": "session-1",
        "X-AgentCore-Collaboration-Context": _context_header(),
    }
    with worker.request_context(headers):
        ack = await tools["agentteams_ack_subtask"].ainvoke(
            {"subtask_id": "caller-controlled-subtask"}
        )
        progress = await tools["agentteams_report_subtask_progress"].ainvoke(
            {"subtask_id": "caller-controlled-subtask", "content": {"percent": 50}}
        )
        result = await tools["agentteams_submit_subtask_result"].ainvoke(
            {
                "subtask_id": "caller-controlled-subtask",
                "summary": "done",
                "file_refs": ["shared/result.md"],
            }
        )

    assert ack == {"ok": True, "data": {"operation": "ack_subtask"}}
    assert progress["ok"] is True
    assert result["ok"] is True
    assert task_client.calls == [
        ("get_subtask", ("caller-controlled-subtask",), {}),
        ("get_task", ("task-1",), {}),
        ("ack_subtask", ("caller-controlled-subtask",), {"event_id": "$event-1"}),
        ("get_subtask", ("caller-controlled-subtask",), {}),
        ("get_task", ("task-1",), {}),
        (
            "report_subtask_progress",
            ("caller-controlled-subtask",),
            {"content": {"percent": 50}, "event_id": "$event-1"},
        ),
        ("get_subtask", ("caller-controlled-subtask",), {}),
        ("get_task", ("task-1",), {}),
        (
            "submit_subtask_result",
            ("caller-controlled-subtask",),
            {
                "summary": "done",
                "file_refs": ["shared/result.md"],
                "event_id": "$event-1",
            },
        ),
    ]


@pytest.mark.asyncio
async def test_worker_write_uses_explicit_subtask_id_with_room_context(
    teams_config_path: Path,
) -> None:
    task_client = _TaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=task_client,
    ).worker()
    ack = next(tool for tool in worker.tools() if tool.name == "agentteams_ack_subtask")

    with worker.request_context(
        {"X-AgentCore-Collaboration-Context": _context_header()}
    ):
        result = await ack.ainvoke({"subtask_id": "subtask-2"})

    assert result["ok"] is True
    assert task_client.calls[-1] == (
        "ack_subtask",
        ("subtask-2",),
        {"event_id": "$event-1"},
    )


@pytest.mark.asyncio
async def test_worker_write_rejects_subtask_from_another_task_room(
    teams_config_path: Path,
) -> None:
    task_client = _TaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=task_client,
    ).worker()
    ack = next(tool for tool in worker.tools() if tool.name == "agentteams_ack_subtask")

    with worker.request_context(
        {
            "X-AgentCore-Collaboration-Context": _context_header(
                room_id="!other:matrix.example.com",
            )
        }
    ):
        result = await ack.ainvoke({"subtask_id": "subtask-2"})

    assert result == {
        "ok": False,
        "code": "COLLABORATION_TASK_UNAUTHORIZED",
        "retryable": False,
        "message": "This operation must run in the owning Task Room.",
    }
    assert [call[0] for call in task_client.calls] == ["get_subtask", "get_task"]


@pytest.mark.asyncio
async def test_worker_write_rejects_non_task_room(
    teams_config_path: Path,
) -> None:
    task_client = _TaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=task_client,
    ).worker()
    ack = next(tool for tool in worker.tools() if tool.name == "agentteams_ack_subtask")

    with worker.request_context(
        {
            "X-AgentCore-Collaboration-Context": _context_header(
                room_kind="group"
            )
        }
    ):
        result = await ack.ainvoke({"subtask_id": "subtask-2"})

    assert result == {
        "ok": False,
        "code": "COLLABORATION_TASK_UNAUTHORIZED",
        "retryable": False,
        "message": "This operation must run in the owning Task Room.",
    }
    assert task_client.calls == []


@pytest.mark.asyncio
async def test_all_task_queries_are_available_without_request_context(
    teams_config_path: Path,
) -> None:
    task_client = _TaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=task_client,
    ).worker()
    tools = {tool.name: tool for tool in worker.tools()}

    calls = (
        ("agentteams_list_tasks", {}),
        ("agentteams_get_task", {"task_id": "task-1"}),
        ("agentteams_get_subtask", {"subtask_id": "subtask-1"}),
        ("agentteams_list_subtasks", {"task_id": "task-1"}),
        ("agentteams_list_task_files", {"task_id": "task-1"}),
        ("agentteams_list_subtask_files", {"subtask_id": "subtask-1"}),
        (
            "agentteams_read_task_file",
            {"task_id": "task-1", "file_ref": "shared/input.txt"},
        ),
        (
            "agentteams_read_subtask_file",
            {"subtask_id": "subtask-1", "file_ref": "shared/output.txt"},
        ),
        ("agentteams_list_results", {"task_id": "task-1"}),
        ("agentteams_list_task_events", {"task_id": "task-1"}),
    )
    for name, arguments in calls:
        result = await tools[name].ainvoke(arguments)
        assert result["ok"] is True

    assert [call[0] for call in task_client.calls] == [
        "list_tasks",
        "get_task",
        "get_subtask",
        "list_subtasks",
        "list_task_files",
        "list_subtask_files",
        "read_task_file",
        "read_subtask_file",
        "list_results",
        "list_events",
    ]


@pytest.mark.asyncio
async def test_read_by_id_requires_explicit_id_with_or_without_request_context(
    teams_config_path: Path,
) -> None:
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    ).worker()
    get_task = next(tool for tool in worker.tools() if tool.name == "agentteams_get_task")

    without_context = await get_task.ainvoke({})
    with worker.request_context(
        {"X-AgentCore-Collaboration-Context": _context_header()}
    ):
        with_context = await get_task.ainvoke({})

    assert without_context["code"] == "COLLABORATION_ARGUMENT_INVALID"
    assert without_context["retryable"] is False
    assert with_context == without_context


@pytest.mark.asyncio
async def test_worker_returns_complete_runtime_team_context_without_request_data(
    teams_config_path: Path,
) -> None:
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    ).worker()
    get_context = next(
        tool for tool in worker.tools() if tool.name == "agentteams_get_team_context"
    )

    result = await get_context.ainvoke({})

    assert result == {
        "ok": True,
        "data": {
            "member": {
                "name": "worker-a",
                "runtimeName": "worker-a",
                "matrixUserId": "@worker-a:matrix.example.com",
                "personalRoomId": "!worker-a:matrix.example.com",
            },
            "defaultTeamName": "team-alpha",
            "teams": [
                {
                    "name": "team-alpha",
                    "teamRoomId": "!alpha:matrix.example.com",
                    "role": "worker",
                    "members": [
                        {
                            "type": "agent",
                            "name": "leader-a",
                            "runtimeName": "leader-a",
                            "role": "leader",
                            "matrixUserId": "@leader-a:matrix.example.com",
                            "personalRoomId": "!leader-a:matrix.example.com",
                        },
                        {
                            "type": "human",
                            "name": "operator-a",
                            "role": "human",
                            "matrixUserId": "@operator-a:matrix.example.com",
                            "personalRoomId": "!operator-a:matrix.example.com",
                        },
                    ],
                }
            ],
        },
    }
    rendered = json.dumps(result)
    assert "managedCollaborationTurn" not in rendered
    assert "taskId" not in rendered
    assert "subTaskId" not in rendered
    assert "tokenEnv" not in rendered
    assert "taskService" not in rendered


def test_worker_prompt_preserves_application_identity_and_subtask_workflow(
    teams_config_path: Path,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "collaboration-workspace"
    workspace.mkdir()
    worker = Collaboration(
        teams_path=teams_config_path,
        workspace_dir=workspace,
        task_client=_TaskClient(),
    ).worker()

    prompt = worker.compose_prompt("You are a writing assistant.")
    normalized_prompt = " ".join(prompt.split())
    skills = worker.skills()
    normalized_task_skill = " ".join(skills[0].instruction.split())

    assert prompt.startswith("You are a writing assistant.")
    assert "You are an AgentCore Worker" not in normalized_prompt
    assert "do not change the application's identity" in normalized_prompt
    assert "not an instruction to look for work" in normalized_prompt
    assert "MUST load and follow `task-execution`" in normalized_prompt
    assert "Before synchronizing non-Task Team files" in normalized_prompt
    assert "load and follow `file-sharing`" in normalized_prompt
    assert "every newly assigned Subtask or revision turn" in normalized_prompt
    assert "reply with one short sentence" in normalized_prompt
    assert "agentteams_get_team_context" in normalized_prompt
    assert "not live Matrix room membership" in normalized_prompt
    assert str(workspace) in prompt
    assert "relative to this directory" in normalized_prompt
    assert [skill.name for skill in skills] == ["task-execution", "file-sharing"]
    assert all(skill.source == "agentcore-collaboration" for skill in skills)
    assert "agentteams_ack_subtask" in normalized_task_skill
    assert "a heartbeat is not progress evidence" in normalized_task_skill
    assert "re-read the Subtask once" in normalized_task_skill
    assert "stop and wait for new input" in normalized_task_skill
    assert "pending_review" in normalized_task_skill
    assert "agentteams_write_subtask_file" in skills[0].instruction
    assert "agentteams_write_subtask_file" not in skills[1].instruction
    assert "agentteams_list_task_events" in normalized_task_skill
    assert "agentteams_filesync_push" in skills[1].instruction


def test_worker_uses_injected_agent_workspace(
    teams_config_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "injected-workspace"
    process_dir = tmp_path / "process-dir"
    workspace.mkdir()
    process_dir.mkdir()
    monkeypatch.setenv("AGENT_WORKSPACE", str(workspace))
    monkeypatch.chdir(process_dir)

    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    ).worker()

    prompt = worker.compose_prompt("")
    assert str(workspace) in prompt
    assert str(process_dir) not in prompt


def test_worker_prompt_keeps_request_classification_inside_sdk(
    teams_config_path: Path,
) -> None:
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    ).worker()

    prompt = " ".join(worker.compose_prompt("").split()).lower()

    assert "managed collaboration turn" not in prompt
    assert "ordinary request" not in prompt
    assert "do not inspect or expose credentials" in prompt


def test_worker_tool_descriptions_explain_business_usage_without_turn_detection(
    teams_config_path: Path,
) -> None:
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    ).worker()
    tools = {tool.name: tool for tool in worker.tools()}

    get_task = tools["agentteams_get_task"].description.lower()
    ack = tools["agentteams_ack_subtask"].description.lower()
    filesync = tools["agentteams_filesync_push"].description.lower()
    local_path = tools["agentteams_filesync_push"].parameters["properties"]["local_path"]

    assert "read-only" in get_task
    assert "exact id" in get_task
    assert "exact id" in ack
    assert "sdk selects" in filesync
    assert "relative paths" in local_path["description"].lower()
    assert all(
        "managed collaboration" not in tool.description.lower()
        and "ordinary request" not in tool.description.lower()
        for tool in tools.values()
    )

    for name in (
        "agentteams_list_task_files",
        "agentteams_list_subtask_files",
        "agentteams_read_task_file",
        "agentteams_read_subtask_file",
        "agentteams_download_task_file",
        "agentteams_download_subtask_file",
    ):
        assert "read-only" in tools[name].description.lower()


def test_task_execution_skill_is_state_aware_and_evidence_driven(
    teams_config_path: Path,
) -> None:
    skill = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    ).worker().skills()[0].instruction
    normalized = " ".join(skill.split())

    assert normalized.index("agentteams_get_subtask") < normalized.index("agentteams_get_task")
    assert "`assigned`: call `agentteams_ack_subtask` with the exact `subtask_id`" in normalized
    assert "Never infer an ID from a title or generate one" in normalized
    assert "`in_progress`: continue without acknowledging again" in normalized
    assert "`blocked`, `pending_review`, `completed`, or `cancelled`: stop" in normalized
    assert "`assignedTo.userId`" in normalized
    assert "`metadata.language`" in normalized
    assert "For every exact Subtask ID in `dependsOn`" in normalized
    assert "call `agentteams_list_results`" in normalized
    assert "`status` is `accepted`" in normalized
    assert "Do not execute until every dependency" in normalized
    assert "roughly 10 minutes" in normalized
    assert "every acceptance criterion" in normalized
    assert "Do not call a Subtask resume operation" in normalized
    assert "confirmed the Subtask is assigned to you" in normalized
    assert "one deliberate verification pass" in normalized
    assert "at most three short sentences" in normalized
    assert "Do not independently revalidate accepted dependency" in normalized
    assert "Leader owns further coordination" in normalized
    assert "Do not list, re-read, hash" in normalized


def test_task_execution_skill_covers_task_file_ownership_and_durability(
    teams_config_path: Path,
) -> None:
    skill = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    ).worker().skills()[0].instruction
    normalized = " ".join(skill.split())

    assert "do not try both" in normalized
    assert "Local staging paths are not durable" in normalized
    assert "`nextCursor`" in normalized
    assert "Before every upload" in normalized
    assert "`shared/subtasks/{subTaskId}/input/`" in normalized
    assert "`shared/subtasks/{subTaskId}/result/`" in normalized
    assert "Do not stage Worker files under `shared/tasks/{taskId}/`" in normalized
    assert "local workspace directories" in normalized
    assert "one ownership or path hypothesis" not in normalized
    assert "Do not repeat an unchanged non-retryable operation" in normalized
    assert "managed collaboration" not in normalized.lower()
    assert "team-bound collaboration turn" not in normalized.lower()


def test_file_sharing_skill_only_covers_non_task_team_files(
    teams_config_path: Path,
) -> None:
    collaboration = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    )
    skill = collaboration.worker().skills()[1]
    assert "non-Task Team files" in skill.description
    assert skill.description in skill.instruction
    assert "agentteams_filesync_push" in skill.instruction
    assert "agentteams_write_subtask_file" not in skill.instruction
    assert "agentteams_read_task_file" not in skill.instruction
    assert "Task and Subtask files are covered by `task-execution`" in skill.instruction
    assert "one ownership or path hypothesis" not in skill.instruction


def test_sync_runtime_exposes_sync_worker_tools(
    agent_config_path: Path,
    tmp_path: Path,
) -> None:
    with AgentCore(
        agent_config_path,
        teams_path=tmp_path / "missing-teams.yaml",
    ) as runtime:
        worker = runtime.collaboration.worker()
        assert [skill.name for skill in worker.skills()] == [
            "task-execution",
            "file-sharing",
        ]
        get_task = next(tool for tool in worker.tools() if tool.name == "agentteams_get_task")

        result = get_task.invoke({"task_id": "task-1"})

    assert result["code"] == "COLLABORATION_DISABLED"


def test_sync_runtime_propagates_request_context_to_worker_tool(
    agent_config_path: Path,
    teams_config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from agentcore_collaboration import task_service

    task_client = _TaskClient()
    monkeypatch.setattr(
        task_service,
        "TaskServiceClient",
        lambda _token_provider, *, endpoint_provider: task_client,
    )

    with AgentCore(agent_config_path, teams_path=teams_config_path) as runtime:
        worker = runtime.collaboration.worker()
        ack = next(tool for tool in worker.tools() if tool.name == "agentteams_ack_subtask")
        with worker.request_context({"X-AgentCore-Collaboration-Context": _context_header()}):
            result = ack.invoke({"subtask_id": "subtask-1"})

    assert result["ok"] is True
    assert result["data"]["operation"] == "ack_subtask"
    assert task_client.calls == [
        ("get_subtask", ("subtask-1",), {}),
        ("get_task", ("task-1",), {}),
        ("ack_subtask", ("subtask-1",), {"event_id": "$event-1"}),
    ]


@pytest.mark.parametrize(
    ("method", "dependency"),
    [
        ("merge_langchain_tools", "langchain_core"),
        ("merge_langgraph_tools", "langchain_core"),
        ("merge_agentscope_tools", "agentscope"),
        ("merge_google_adk_tools", "google.adk"),
        ("merge_pydantic_ai_tools", "pydantic_ai"),
        ("merge_crewai_tools", "crewai"),
    ],
)
def test_worker_framework_adapters_expose_the_same_static_toolset(
    method: str,
    dependency: str,
    teams_config_path: Path,
) -> None:
    pytest.importorskip(dependency)
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    ).worker()

    adapted = getattr(worker, method)()

    assert len(adapted) == 24
    assert {tool.name for tool in adapted} == {tool.name for tool in worker.tools()}


@pytest.mark.asyncio
async def test_worker_local_file_tools_keep_bytes_out_of_model_results(
    teams_config_path: Path,
    tmp_path: Path,
) -> None:
    class FileTaskClient(_TaskClient):
        async def download_task_file_to_path(
            self,
            _snapshot: object,
            task_id: str,
            *,
            file_ref: str,
            destination: Path,
        ) -> int:
            self.calls.append(
                (
                    "download_task_file_to_path",
                    (task_id,),
                    {"file_ref": file_ref, "destination": destination},
                )
            )
            content = b"\x00downloaded\xff"
            await anyio.to_thread.run_sync(destination.write_bytes, content)
            return len(content)

    task_client = FileTaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        workspace_dir=tmp_path,
        task_client=task_client,
    ).worker()
    tools = {tool.name: tool for tool in worker.tools()}
    output = tmp_path / "downloads" / "artifact.bin"

    result = await tools["agentteams_download_task_file"].ainvoke(
        {
            "task_id": "task-1",
            "file_ref": "shared/artifact.bin",
            "output_path": str(output),
        }
    )

    assert result == {
        "ok": True,
        "data": {"path": str(output), "size": len(b"\x00downloaded\xff")},
    }
    assert output.read_bytes() == b"\x00downloaded\xff"
    assert "downloaded" not in json.dumps(result)


@pytest.mark.asyncio
async def test_worker_file_tools_resolve_relative_paths_from_workspace(
    teams_config_path: Path,
    tmp_path: Path,
) -> None:
    class FileTaskClient(_TaskClient):
        async def download_task_file_to_path(
            self,
            _snapshot: object,
            task_id: str,
            *,
            file_ref: str,
            destination: Path,
        ) -> int:
            self.calls.append(
                (
                    "download_task_file_to_path",
                    (task_id,),
                    {"file_ref": file_ref, "destination": destination},
                )
            )
            content = b"downloaded"
            await anyio.to_thread.run_sync(destination.write_bytes, content)
            return len(content)

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "shared" / "subtasks" / "subtask-1" / "result.md"
    source.parent.mkdir(parents=True)
    source.write_text("result", encoding="utf-8")
    task_client = FileTaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        workspace_dir=workspace,
        task_client=task_client,
    ).worker()
    tools = {tool.name: tool for tool in worker.tools()}

    with worker.request_context({"X-AgentCore-Collaboration-Context": _context_header()}):
        uploaded = await tools["agentteams_write_subtask_file_from_path"].ainvoke(
            {
                "subtask_id": "subtask-1",
                "path": "result/result.md",
                "local_path": "shared/subtasks/subtask-1/result.md",
            }
        )
        downloaded = await tools["agentteams_download_task_file"].ainvoke(
            {
                "task_id": "task-1",
                "file_ref": "shared/input.txt",
                "output_path": "shared/tasks/task-1/input.txt",
            }
        )

    output = workspace / "shared" / "tasks" / "task-1" / "input.txt"
    assert uploaded["ok"] is True
    assert downloaded == {
        "ok": True,
        "data": {"path": str(output), "size": len(b"downloaded")},
    }
    assert output.read_bytes() == b"downloaded"
    upload_call = next(
        call for call in task_client.calls if call[0] == "write_subtask_file_from_path"
    )
    assert upload_call[2]["source"] == source


@pytest.mark.asyncio
async def test_worker_file_tools_reject_relative_workspace_escape(
    teams_config_path: Path,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    worker = Collaboration(
        teams_path=teams_config_path,
        workspace_dir=workspace,
        task_client=_TaskClient(),
    ).worker()
    download = next(
        tool for tool in worker.tools() if tool.name == "agentteams_download_task_file"
    )

    result = await download.ainvoke(
        {
            "task_id": "task-1",
            "file_ref": "shared/input.txt",
            "output_path": "../outside.txt",
        }
    )

    assert result["code"] == "COLLABORATION_ARGUMENT_INVALID"


@pytest.mark.asyncio
async def test_worker_file_tools_reject_paths_outside_workspace(
    teams_config_path: Path,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    worker = Collaboration(
        teams_path=teams_config_path,
        workspace_dir=workspace,
        task_client=_TaskClient(),
    ).worker()
    tools = {tool.name: tool for tool in worker.tools()}

    with worker.request_context({"X-AgentCore-Collaboration-Context": _context_header()}):
        upload = await tools["agentteams_write_subtask_file_from_path"].ainvoke(
            {
                "subtask_id": "subtask-1",
                "path": "result.txt",
                "local_path": str(outside),
            }
        )
        download = await tools["agentteams_download_task_file"].ainvoke(
            {
                "task_id": "task-1",
                "file_ref": "shared/input.txt",
                "output_path": str(outside),
                "overwrite": True,
            }
        )

    assert upload["code"] == "COLLABORATION_ARGUMENT_INVALID"
    assert download["code"] == "COLLABORATION_ARGUMENT_INVALID"
    assert outside.read_text(encoding="utf-8") == "secret"


@pytest.mark.asyncio
async def test_worker_cross_task_history_does_not_inherit_current_subtask(
    teams_config_path: Path,
) -> None:
    task_client = _TaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=task_client,
    ).worker()
    tools = {tool.name: tool for tool in worker.tools()}

    with worker.request_context({"X-AgentCore-Collaboration-Context": _context_header()}):
        await tools["agentteams_list_results"].ainvoke({"task_id": "task-2"})
        await tools["agentteams_list_task_events"].ainvoke({"task_id": "task-2"})

    assert task_client.calls == [
        ("list_results", ("task-2",), {"subtask_id": None}),
        (
            "list_events",
            ("task-2",),
            {"subtask_id": None, "event_type": None, "cursor": None, "limit": 50},
        ),
    ]


@pytest.mark.asyncio
async def test_worker_filesync_push_and_pull_use_context_team(
    teams_config_path: Path,
    tmp_path: Path,
) -> None:
    class FilesyncTaskClient(_TaskClient):
        async def stat_team_file(
            self, _snapshot: object, team_id: str, *, path: str
        ) -> dict[str, Any]:
            self.calls.append(("stat_team_file", (team_id,), {"path": path}))
            raise CollaborationTaskNotFoundError("not a file")

        async def list_team_files(
            self,
            _snapshot: object,
            team_id: str,
            *,
            path: str,
            cursor: str | None,
            limit: int,
        ) -> dict[str, Any]:
            self.calls.append(
                (
                    "list_team_files",
                    (team_id,),
                    {"path": path, "cursor": cursor, "limit": limit},
                )
            )
            return {
                "items": [
                    {"path": "shared/knowledge/a.txt", "size": 1},
                    {"path": "shared/knowledge/nested/b.txt", "size": 1},
                ],
                "nextCursor": None,
            }

        async def read_team_file(self, _snapshot: object, team_id: str, *, path: str) -> bytes:
            self.calls.append(("read_team_file", (team_id,), {"path": path}))
            return path.rsplit("/", 1)[-1][:1].encode()

    task_client = FilesyncTaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        workspace_dir=tmp_path,
        task_client=task_client,
    ).worker()
    tools = {tool.name: tool for tool in worker.tools()}
    source = tmp_path / "source"
    source.mkdir()
    (source / "a.txt").write_text("a")
    nested = source / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("b")
    output = tmp_path / "pulled"

    with worker.request_context({"X-AgentCore-Collaboration-Context": _context_header()}):
        pushed = await tools["agentteams_filesync_push"].ainvoke(
            {"path": "shared/knowledge", "local_path": "source"}
        )
        pulled = await tools["agentteams_filesync_pull"].ainvoke(
            {"path": "shared/knowledge", "local_path": "pulled"}
        )

    assert pushed["ok"] is True
    assert pushed["data"]["transferred"] == 2
    assert pulled["ok"] is True
    assert pulled["data"]["transferred"] == 2
    assert (output / "a.txt").read_bytes() == b"a"
    assert (output / "nested" / "b.txt").read_bytes() == b"b"
    assert [
        (call[1][0], call[2]["path"])
        for call in task_client.calls
        if call[0] == "write_team_file_from_path"
    ] == [
        ("team-alpha", "shared/knowledge/a.txt"),
        ("team-alpha", "shared/knowledge/nested/b.txt"),
    ]


@pytest.mark.asyncio
async def test_worker_filesync_uses_unique_worker_team_when_context_omits_team(
    teams_config_path: Path,
) -> None:
    task_client = _TaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=task_client,
    ).worker()
    tool = next(tool for tool in worker.tools() if tool.name == "agentteams_filesync_list")

    with worker.request_context(
        {"X-AgentCore-Collaboration-Context": _context_header(room_kind="group", team_id=None)}
    ):
        result = await tool.ainvoke({"path": "shared/knowledge"})

    assert result["ok"] is True
    assert task_client.calls == [
        (
            "list_team_files",
            ("team-alpha",),
            {"path": "shared/knowledge", "cursor": None, "limit": 100},
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_team_count", [0, 2])
async def test_worker_filesync_rejects_missing_or_ambiguous_worker_team(
    tmp_path: Path,
    teams_config_text: str,
    worker_team_count: int,
) -> None:
    if worker_team_count == 0:
        config = teams_config_text.replace("role: worker", "role: leader", 1)
    else:
        config = teams_config_text.replace(
            "  teams:\n",
            "  teams:\n"
            "    - name: team-beta\n"
            "      membership:\n"
            "        role: worker\n"
            "      members: []\n",
            1,
        )
    teams_path = tmp_path / "teams.yaml"
    teams_path.write_text(config, encoding="utf-8")
    worker = Collaboration(teams_path=teams_path, task_client=_TaskClient()).worker()
    tool = next(tool for tool in worker.tools() if tool.name == "agentteams_filesync_list")

    with worker.request_context(
        {"X-AgentCore-Collaboration-Context": _context_header(room_kind="dm", team_id=None)}
    ):
        result = await tool.ainvoke({"path": "shared/knowledge"})

    assert result == {
        "ok": False,
        "code": "COLLABORATION_CONTEXT_REQUIRED",
        "retryable": False,
        "message": "The SDK could not select a current Team for this operation.",
    }


@pytest.mark.asyncio
async def test_worker_filesync_uses_latest_unique_worker_team(
    teams_config_path: Path,
    teams_config_text: str,
) -> None:
    task_client = _TaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=task_client,
    ).worker()
    tool = next(tool for tool in worker.tools() if tool.name == "agentteams_filesync_list")
    headers = {
        "X-AgentCore-Collaboration-Context": _context_header(room_kind="group", team_id=None)
    }

    with worker.request_context(headers):
        first = await tool.ainvoke({"path": "shared/knowledge"})
    await anyio.to_thread.run_sync(
        teams_config_path.write_text,
        teams_config_text.replace("team-alpha", "team-beta"),
        "utf-8",
    )
    with worker.request_context(headers):
        second = await tool.ainvoke({"path": "shared/knowledge"})

    assert first["ok"] is True
    assert second["ok"] is True
    assert [call[1][0] for call in task_client.calls] == ["team-alpha", "team-beta"]


@pytest.mark.asyncio
async def test_worker_filesync_stat_and_pull_support_shared_root(
    teams_config_path: Path,
    tmp_path: Path,
) -> None:
    class FilesyncTaskClient(_TaskClient):
        async def stat_team_file(
            self, _snapshot: object, team_id: str, *, path: str
        ) -> dict[str, Any]:
            self.calls.append(("stat_team_file", (team_id,), {"path": path}))
            raise CollaborationTaskInvalidError("Team shared path is invalid")

        async def list_team_files(
            self,
            _snapshot: object,
            team_id: str,
            *,
            path: str,
            cursor: str | None,
            limit: int,
        ) -> dict[str, Any]:
            self.calls.append(
                (
                    "list_team_files",
                    (team_id,),
                    {"path": path, "cursor": cursor, "limit": limit},
                )
            )
            return {
                "items": [{"path": "shared/knowledge/guide.md", "size": 5}],
                "nextCursor": None,
            }

        async def read_team_file(self, _snapshot: object, team_id: str, *, path: str) -> bytes:
            self.calls.append(("read_team_file", (team_id,), {"path": path}))
            return b"guide"

    task_client = FilesyncTaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        workspace_dir=tmp_path,
        task_client=task_client,
    ).worker()
    tools = {tool.name: tool for tool in worker.tools()}

    with worker.request_context({"X-AgentCore-Collaboration-Context": _context_header()}):
        stat = await tools["agentteams_filesync_stat"].ainvoke({"path": "shared"})
        pull = await tools["agentteams_filesync_pull"].ainvoke(
            {"path": "shared", "local_path": "pulled"}
        )

    assert stat == {
        "ok": True,
        "data": {"kind": "directory", "path": "shared", "entries": 1},
    }
    assert pull["ok"] is True
    assert pull["data"]["transferred"] == 1
    assert (tmp_path / "pulled" / "knowledge" / "guide.md").read_bytes() == b"guide"
    assert all(call[0] != "stat_team_file" for call in task_client.calls)


@pytest.mark.asyncio
async def test_worker_filesync_pull_resolves_an_exact_remote_file(
    teams_config_path: Path,
    tmp_path: Path,
) -> None:
    class FilesyncTaskClient(_TaskClient):
        async def stat_team_file(
            self, _snapshot: object, team_id: str, *, path: str
        ) -> dict[str, Any]:
            self.calls.append(("stat_team_file", (team_id,), {"path": path}))
            return {"path": path, "size": 4}

        async def read_team_file(self, _snapshot: object, team_id: str, *, path: str) -> bytes:
            self.calls.append(("read_team_file", (team_id,), {"path": path}))
            return b"data"

    task_client = FilesyncTaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        workspace_dir=tmp_path,
        task_client=task_client,
    ).worker()
    pull = next(tool for tool in worker.tools() if tool.name == "agentteams_filesync_pull")
    output = tmp_path / "guide.md"

    with worker.request_context({"X-AgentCore-Collaboration-Context": _context_header()}):
        result = await pull.ainvoke(
            {
                "path": "shared/knowledge/guide.md",
                "local_path": str(output),
            }
        )

    assert result["ok"] is True
    assert result["data"]["transferred"] == 1
    assert output.read_bytes() == b"data"
    assert [call[0] for call in task_client.calls] == [
        "stat_team_file",
        "read_team_file",
    ]


@pytest.mark.asyncio
async def test_worker_filesync_leaves_file_size_enforcement_to_task_service(
    teams_config_path: Path,
    tmp_path: Path,
) -> None:
    class FilesyncTaskClient(_TaskClient):
        async def stat_team_file(
            self, _snapshot: object, team_id: str, *, path: str
        ) -> dict[str, Any]:
            self.calls.append(("stat_team_file", (team_id,), {"path": path}))
            return {"path": path, "size": 1024 * 1024 + 1}

        async def read_team_file(self, _snapshot: object, team_id: str, *, path: str) -> bytes:
            self.calls.append(("read_team_file", (team_id,), {"path": path}))
            return b"remote content"

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    large = workspace / "large.bin"
    large.write_bytes(b"x" * (1024 * 1024 + 1))
    task_client = FilesyncTaskClient()
    worker = Collaboration(
        teams_path=teams_config_path,
        workspace_dir=workspace,
        task_client=task_client,
    ).worker()
    tools = {tool.name: tool for tool in worker.tools()}

    with worker.request_context({"X-AgentCore-Collaboration-Context": _context_header()}):
        pushed = await tools["agentteams_filesync_push"].ainvoke(
            {"path": "shared/large.bin", "local_path": str(large)}
        )
        pulled = await tools["agentteams_filesync_pull"].ainvoke(
            {"path": "shared/large.bin", "local_path": str(workspace / "pull.bin")}
        )

    assert pushed["ok"] is True
    assert pulled["ok"] is True
    assert (workspace / "pull.bin").read_bytes() == b"remote content"
    assert [call[0] for call in task_client.calls] == [
        "write_team_file_from_path",
        "stat_team_file",
        "read_team_file",
    ]


@pytest.mark.asyncio
async def test_worker_filesync_reports_missing_current_team_without_turn_classification(
    teams_config_path: Path,
) -> None:
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    ).worker()
    tool = next(tool for tool in worker.tools() if tool.name == "agentteams_filesync_list")

    result = await tool.ainvoke({"path": "shared/knowledge"})

    assert result["code"] == "COLLABORATION_CONTEXT_REQUIRED"
    assert result["retryable"] is False
    assert result["message"] == "The SDK could not select a current Team for this operation."


@pytest.mark.asyncio
async def test_worker_filesync_rejects_task_owned_team_paths(
    teams_config_path: Path,
) -> None:
    worker = Collaboration(
        teams_path=teams_config_path,
        task_client=_TaskClient(),
    ).worker()
    tool = next(tool for tool in worker.tools() if tool.name == "agentteams_filesync_list")

    with worker.request_context({"X-AgentCore-Collaboration-Context": _context_header()}):
        result = await tool.ainvoke({"path": "shared/tasks/task-1"})

    assert result["code"] == "COLLABORATION_ARGUMENT_INVALID"
    assert result["retryable"] is False
