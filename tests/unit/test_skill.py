from __future__ import annotations

import io
import json
import logging
import shlex
import sys
import time
import zipfile
from pathlib import Path

import anyio
import pytest

from agentcore.controlplane import SkillArtifact
from agentcore.errors import ConfigError
from agentcore.skill import AsyncSkills, Skill, skill_tools


class FakeRegistry:
    def __init__(self) -> None:
        self.calls = 0

    async def get_skill(self, name: str, version: str | None = None):  # type: ignore[no-untyped-def]
        self.calls += 1
        skill_md = "---\nname: review\ndescription: Reviews code\n---\nUse tests.\n"
        return SkillArtifact(
            version=version or "1.2.0",
            archive=_skill_archive(
                skill_md,
                {"references/checklist.md": "Check behavior.\n"},
            ),
        )


class StaticRegistry:
    def __init__(
        self,
        skill_md: str,
        resources: dict[str, str | bytes] | None = None,
    ) -> None:
        self.skill_md = skill_md
        self.resources = resources or {}

    async def get_skill(self, name: str, version: str | None = None):  # type: ignore[no-untyped-def]
        return SkillArtifact(
            version=version or "1.0.0",
            archive=_skill_archive(self.skill_md, self.resources),
        )


class ArchiveRegistry:
    def __init__(self, archive: bytes) -> None:
        self.archive = archive

    async def get_skill(self, name: str, version: str | None = None) -> SkillArtifact:
        return SkillArtifact(version=version or "1.0.0", archive=self.archive)


def _skill_archive(
    skill_md: str,
    resources: dict[str, str | bytes] | None = None,
    *,
    root: str = "review",
) -> bytes:
    result = io.BytesIO()
    with zipfile.ZipFile(result, "w") as archive:
        archive.writestr(f"{root}/SKILL.md", skill_md)
        for name, content in (resources or {}).items():
            archive.writestr(f"{root}/{name}", content)
    return result.getvalue()


@pytest.mark.asyncio
async def test_local_skills_are_loaded_without_remote_access(tmp_path: Path) -> None:
    directory = tmp_path / "skills" / "review"
    directory.mkdir(parents=True)
    directory.joinpath("SKILL.md").write_text(
        "---\nname: review\ndescription: Reviews code\nversion: 1.0.0\n---\nDo it.\n",
        encoding="utf-8",
    )
    directory.joinpath("notes.txt").write_text("note", encoding="utf-8")

    skills = AsyncSkills(None, None)
    loaded = await skills.local(tmp_path / "skills")

    assert loaded[0].name == "review"
    assert loaded[0].source == "local"
    assert loaded[0].version == "1.0.0"
    assert loaded[0].files == ("SKILL.md", "notes.txt")


@pytest.mark.asyncio
async def test_managed_skill_uses_agentcore_and_client_pin(tmp_path: Path) -> None:
    registry = FakeRegistry()
    skills = AsyncSkills(
        registry,
        "workspace-a",
        workspace_dir=tmp_path / "workspace",
    )

    first = await skills.managed("review")
    second = await skills.managed("review")

    assert first is second
    assert first.source == "agentcore"
    assert first.workspace_id == "workspace-a"
    assert first.version == "1.2.0"
    assert first.root == tmp_path / "workspace" / "workspace-a" / "review" / "1.2.0"
    assert first.root.joinpath("references/checklist.md").read_text() == "Check behavior.\n"
    assert registry.calls == 1


@pytest.mark.asyncio
async def test_managed_skill_preserves_binary_resources(tmp_path: Path) -> None:
    skills = AsyncSkills(
        StaticRegistry("---\nname: review\n---\nReview.\n", {"assets/logo.png": b"\x89PNG"}),
        "workspace-a",
        workspace_dir=tmp_path / "workspace",
    )

    skill = await skills.managed("review")

    assert skill.root.joinpath("assets/logo.png").read_bytes() == b"\x89PNG"


@pytest.mark.asyncio
async def test_managed_skill_uses_requested_resource_name(tmp_path: Path) -> None:
    skills = AsyncSkills(
        StaticRegistry("---\nname: package-name\n---\nReview.\n"),
        "workspace-a",
        workspace_dir=tmp_path / "workspace",
    )

    skill = await skills.managed("review")

    assert skill.name == "review"


@pytest.mark.asyncio
async def test_managed_artifact_rejects_path_traversal(tmp_path: Path) -> None:
    skills = AsyncSkills(
        StaticRegistry("# skill", {"../secret": "bad"}),
        "workspace-a",
        workspace_dir=tmp_path / "workspace",
    )

    with pytest.raises(ConfigError):
        await skills.managed("review")


@pytest.mark.asyncio
async def test_managed_artifact_requires_the_requested_skill_root(tmp_path: Path) -> None:
    archive = _skill_archive("# skill", root="another-skill")
    skills = AsyncSkills(
        ArchiveRegistry(archive),
        "workspace-a",
        workspace_dir=tmp_path / "workspace",
    )

    with pytest.raises(ConfigError, match="review/SKILL.md"):
        await skills.managed("review")


@pytest.mark.asyncio
async def test_managed_artifact_rejects_blank_version(tmp_path: Path) -> None:
    skills = AsyncSkills(
        StaticRegistry("# skill"),
        "workspace-a",
        workspace_dir=tmp_path / "workspace",
    )

    with pytest.raises(ConfigError, match="version is invalid"):
        await skills.managed("review", version=" ")


@pytest.mark.asyncio
async def test_managed_artifact_cannot_replace_materialization_metadata(
    tmp_path: Path,
) -> None:
    skills = AsyncSkills(
        StaticRegistry("# skill", {".agentcore-skill.json": "package data"}),
        "workspace-a",
        workspace_dir=tmp_path / "workspace",
    )

    with pytest.raises(ConfigError, match="unsafe resource path"):
        await skills.managed("review")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resources",
    [
        {"references": "file", "references/checklist.md": "nested"},
        {"references//checklist.md": "non-canonical"},
    ],
)
async def test_managed_artifact_rejects_colliding_resource_paths(
    tmp_path: Path,
    resources: dict[str, str],
) -> None:
    skills = AsyncSkills(
        StaticRegistry("# skill", resources),
        "workspace-a",
        workspace_dir=tmp_path / "workspace",
    )

    with pytest.raises(ConfigError, match="resource path"):
        await skills.managed("review")


@pytest.mark.asyncio
async def test_skill_tools_keep_agentrun_protocol_and_only_read_selected_files(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "skills" / "review"
    directory.mkdir(parents=True)
    directory.joinpath("SKILL.md").write_text(
        "---\nname: review\ndescription: Reviews code\n---\nReview carefully.\n",
        encoding="utf-8",
    )
    directory.joinpath("checklist.md").write_text("Run tests.\n", encoding="utf-8")
    references = directory / "references"
    references.mkdir()
    references.joinpath("guide.md").write_text("Follow the guide.\n", encoding="utf-8")
    loaded = await AsyncSkills(None, None).local(tmp_path / "skills")
    tools = skill_tools(loaded)
    tool_map = {tool.name: tool for tool in tools}

    assert list(tool_map) == ["load_skills", "read_skill_file", "execute_command"]

    listed = json.loads(await tool_map["load_skills"].ainvoke({}))
    detail = json.loads(await tool_map["load_skills"].ainvoke({"name": "review"}))
    checklist = json.loads(
        await tool_map["read_skill_file"].ainvoke(
            {"name": "review", "relative_path": "checklist.md"}
        )
    )
    references_list = json.loads(
        await tool_map["read_skill_file"].ainvoke(
            {"name": "review", "relative_path": "references"}
        )
    )
    traversal = json.loads(
        await tool_map["read_skill_file"].ainvoke(
            {"name": "review", "relative_path": "../secret"}
        )
    )
    missing_skill = json.loads(
        await tool_map["load_skills"].ainvoke({"name": "missing"})
    )
    missing_file = json.loads(
        await tool_map["read_skill_file"].ainvoke(
            {"name": "review", "relative_path": "missing.md"}
        )
    )

    assert listed == {"skills": [{"name": "review", "description": "Reviews code"}]}
    assert detail == {
        "name": "review",
        "description": "Reviews code",
        "instruction": loaded[0].instruction,
        "files": ["SKILL.md", "checklist.md", "references/"],
    }
    assert checklist == {"content": "Run tests.\n"}
    assert references_list == {"files": ["guide.md"]}
    assert "outside the skill directory" in traversal["error"]
    assert missing_skill == {
        "error": "Skill 'missing' not found. Available skills: review"
    }
    assert missing_file == {
        "error": "File 'missing.md' not found in skill 'review'."
    }


def test_skill_tools_keep_agentrun_parameter_schema(tmp_path: Path) -> None:
    tool_map = {tool.name: tool for tool in skill_tools([_skill(tmp_path)])}

    assert tool_map["load_skills"].parameters.get("required", []) == []
    assert set(tool_map["load_skills"].parameters["properties"]) == {"name"}
    assert tool_map["read_skill_file"].parameters["required"] == [
        "name",
        "relative_path",
    ]


@pytest.mark.asyncio
async def test_skill_tools_execute_command_is_enabled_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.delenv("ALLOW_EXECUTE_COMMAND", raising=False)
    directory = tmp_path / "skills" / "review"
    directory.mkdir(parents=True)
    directory.joinpath("SKILL.md").write_text("---\nname: review\n---\nRun it.\n", encoding="utf-8")
    loaded = await AsyncSkills(None, None).local(tmp_path / "skills")

    tools = skill_tools(loaded)
    execute = next(tool for tool in tools if tool.name == "execute_command")
    with caplog.at_level(logging.INFO, logger="agentcore.skill._command"):
        result = json.loads(await execute.ainvoke({"command": "printf command-secret"}))

    assert result == {
        "stdout": "command-secret",
        "stderr": "",
        "exit_code": 0,
        "timed_out": False,
    }
    assert "agentcore.skill.command.started" in caplog.text
    assert "agentcore.skill.command.completed" in caplog.text
    assert "command-secret" not in caplog.text


def test_skill_tools_execute_command_can_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ALLOW_EXECUTE_COMMAND", "false")
    skill = _skill(tmp_path)

    assert [tool.name for tool in skill_tools([skill])] == ["load_skills", "read_skill_file"]


def test_skill_tools_execute_command_enforces_approval(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    approved: list[tuple[str, str]] = []

    def reject(command: str, cwd: str) -> bool:
        approved.append((command, cwd))
        return False

    execute = next(
        tool
        for tool in skill_tools([skill], command_approval=reject)
        if tool.name == "execute_command"
    )
    result = json.loads(execute.invoke({"command": "printf rejected"}))

    assert result == {"error": "Command execution rejected by user."}
    assert approved == [("printf rejected", str(skill.root))]


def test_multiple_managed_skills_execute_from_their_common_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "managed" / "workspace-a"
    skills = []
    for name in ("review", "deploy"):
        root = workspace / name / "1.0.0"
        root.mkdir(parents=True)
        instruction = f"---\nname: {name}\n---\nRun it.\n"
        root.joinpath("SKILL.md").write_text(instruction, encoding="utf-8")
        skills.append(
            Skill(
                name=name,
                description="",
                version="1.0.0",
                instruction=instruction,
                root=root,
                files=("SKILL.md",),
                source="agentcore",
                workspace_id="workspace-a",
            )
        )
    approved: list[str] = []

    def reject(_command: str, cwd: str) -> bool:
        approved.append(cwd)
        return False

    execute = next(
        tool
        for tool in skill_tools(skills, command_approval=reject)
        if tool.name == "execute_command"
    )

    result = json.loads(execute.invoke({"command": "pwd"}))

    assert result == {"error": "Command execution rejected by user."}
    assert approved == [str(workspace)]


def test_skill_tools_execute_command_times_out(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    execute = next(
        tool for tool in skill_tools([skill], command_timeout=1) if tool.name == "execute_command"
    )

    result = json.loads(execute.invoke({"command": "sleep 2"}))

    assert result["timed_out"] is True
    assert result["exit_code"] == -1


def test_skill_tools_execute_command_timeout_stops_child_processes(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    marker = tmp_path / "child-survived"
    script = (
        "import time; from pathlib import Path; "
        f"time.sleep(1.2); Path({str(marker)!r}).write_text('survived')"
    )
    child = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    execute = next(
        tool for tool in skill_tools([skill], command_timeout=1) if tool.name == "execute_command"
    )

    result = json.loads(execute.invoke({"command": f"{child} & wait"}))
    time.sleep(0.4)

    assert result["timed_out"] is True
    assert not marker.exists()


@pytest.mark.asyncio
async def test_skill_tools_async_execute_command_times_out(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    execute = next(
        tool for tool in skill_tools([skill], command_timeout=1) if tool.name == "execute_command"
    )

    result = json.loads(await execute.ainvoke({"command": "sleep 2"}))

    assert result["timed_out"] is True
    assert result["exit_code"] == -1


@pytest.mark.asyncio
async def test_skill_tools_execute_command_cancellation_stops_process(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    marker = tmp_path / "cancelled-child-survived"
    script = (
        "import time; from pathlib import Path; "
        f"time.sleep(0.5); Path({str(marker)!r}).write_text('survived')"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"
    execute = next(tool for tool in skill_tools([skill]) if tool.name == "execute_command")

    with anyio.move_on_after(0.05) as scope:
        await execute.ainvoke({"command": command})
    await anyio.sleep(0.6)

    assert scope.cancel_called
    assert not marker.exists()


def test_skill_tools_execute_command_truncates_large_output(tmp_path: Path) -> None:
    skill = _skill(tmp_path)
    execute = next(tool for tool in skill_tools([skill]) if tool.name == "execute_command")
    script = "print('x' * 110000, end='')"
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    result = json.loads(execute.invoke({"command": command}))

    assert result["stdout"].endswith("[output truncated, exceeded 102400 bytes]")


@pytest.mark.asyncio
async def test_managed_skill_workspace_is_shared_across_clients(
    tmp_path: Path,
) -> None:
    registry = StaticRegistry("---\nname: review\n---\nUse tests.\n")
    workspace = tmp_path / "workspace"
    first = AsyncSkills(
        registry,
        "workspace-a",
        workspace_dir=workspace,
    )
    loaded = await first.managed("review", version="1.0.0")
    loaded.root.joinpath("generated.txt").write_text("runtime output", encoding="utf-8")

    second = AsyncSkills(
        registry,
        "workspace-a",
        workspace_dir=workspace,
    )
    reused = await second.managed("review", version="1.0.0")

    assert reused.root == loaded.root
    assert reused.root.joinpath("generated.txt").read_text(encoding="utf-8") == "runtime output"


@pytest.mark.asyncio
async def test_managed_skill_same_version_replaces_changed_content(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    first = AsyncSkills(
        StaticRegistry("---\nname: review\n---\nVersion one.\n"),
        "workspace-a",
        workspace_dir=workspace,
    )
    second = AsyncSkills(
        StaticRegistry("---\nname: review\n---\nVersion two.\n"),
        "workspace-a",
        workspace_dir=workspace,
    )

    version_one = await first.managed("review", version="1.0.0")
    version_two = await second.managed("review", version="1.0.0")

    assert version_one.root == version_two.root
    assert version_one.instruction.endswith("Version one.\n")
    assert version_two.instruction.endswith("Version two.\n")
    assert version_two.root.joinpath("SKILL.md").read_text().endswith("Version two.\n")


@pytest.mark.asyncio
async def test_managed_skill_rejects_oversize_artifact_before_publishing_workspace(
    tmp_path: Path,
) -> None:
    registry = StaticRegistry("# Skill\n" + "x" * (10 * 1024 * 1024))
    workspace = tmp_path / "workspace"
    skills = AsyncSkills(
        registry,
        "workspace-a",
        workspace_dir=workspace,
    )

    with pytest.raises(ConfigError, match="10 MiB"):
        await skills.managed("review", version="1.0.0")

    assert not workspace.exists() or list(workspace.iterdir()) == []


def _skill(tmp_path: Path) -> Skill:
    directory = tmp_path / "skills" / "review"
    directory.mkdir(parents=True)
    instruction = "---\nname: review\n---\nRun it.\n"
    directory.joinpath("SKILL.md").write_text(instruction, encoding="utf-8")
    return Skill(
        name="review",
        description="",
        version="",
        instruction=instruction,
        root=directory,
        files=("SKILL.md",),
        source="local",
    )
