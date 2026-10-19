from __future__ import annotations

import io
import json
import shlex
import sys
import zipfile
from pathlib import Path

import anyio
import pytest

from agentcore.controlplane import SkillArtifact, TemplateArtifact
from agentcore.errors import ConfigError
from agentcore.skill import AsyncSkills, skill_tools
from agentcore.templates import AsyncTemplates


def archive(files: dict[str, str | bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as target:
        for name, content in files.items():
            target.writestr(name, content)
    return output.getvalue()


class Registry:
    def __init__(self, files: dict[str, str | bytes]) -> None:
        self.files = files
        self.calls: list[tuple] = []
        self.fail = False

    async def get_template(self, name, version=None):
        self.calls.append(("template", name, version))
        return TemplateArtifact(version or "1.0.0", archive(self.files))

    async def resolve_skill_version(self, name, selector_type, value):
        self.calls.append(("selector", name, selector_type, value))
        return value if selector_type == "VERSION" else "2.0.0"

    async def get_skill(self, name, version=None):
        self.calls.append(("skill", name, version))
        if self.fail:
            raise RuntimeError("Skill service unavailable")
        return SkillArtifact(version, archive({f"{name}/SKILL.md": "Referenced instructions"}))


def templates(registry: Registry, root: Path) -> AsyncTemplates:
    async def runtime():
        return registry, "ws-test"

    return AsyncTemplates(
        runtime, AsyncSkills(registry, "ws-test", workspace_dir=root), workspace_dir=root
    )


@pytest.mark.asyncio
async def test_prompt_and_mixed_skills_are_lazy_and_pinned(tmp_path):
    registry = Registry(
        {
            "manifest.json": json.dumps(
                {
                    "skills": [
                        {"name": "local"},
                        {"name": "remote", "sourceType": "REFERENCE"},
                        {
                            "name": "fixed",
                            "sourceType": "REFERENCE",
                            "versionSelector": {"type": "VERSION", "value": "v3"},
                        },
                    ]
                }
            ),
            "config/AGENTS.md": "Agent instructions",
            "config/SOUL.md": "Personality",
            "config/subagents/helper/AGENTS.md": "Not the main prompt",
            "skills/local/SKILL.md": "---\nname: local\n---\nStatic instructions",
            "skills/local/assets/data.bin": b"\x00\xff",
            "skills/local/scripts/run.py": "print('ok')",
        }
    )
    template = await templates(registry, tmp_path).get("assistant")
    assert template.name == "assistant" and template.version == "1.0.0"
    assert template.instruction == "## AGENTS.md\n\nAgent instructions\n\n## SOUL.md\n\nPersonality"
    assert registry.calls == [("template", "assistant", None)]
    results = []

    async def load():
        results.append(await template.load_skills())

    async with anyio.create_task_group() as group:
        group.start_soon(load)
        group.start_soon(load)
    assert [s.name for s in results[0]] == ["local", "remote", "fixed"]
    assert [s.version for s in results[0][1:]] == ["2.0.0", "v3"]
    assert results[0][0].root.joinpath("assets/data.bin").read_bytes() == b"\x00\xff"
    assert all(a is b for a, b in zip(results[0], results[1], strict=True))
    assert len([c for c in registry.calls if c[0] == "skill"]) == 2
    assert len(skill_tools(results[0])) == 3


@pytest.mark.asyncio
async def test_failed_skill_load_does_not_break_prompt_or_poison_retry(tmp_path):
    registry = Registry(
        {
            "manifest.json": json.dumps(
                {"skills": [{"name": "remote", "sourceType": "REFERENCE"}]}
            ),
            "config/SOUL.md": "Hello",
        }
    )
    template = await templates(registry, tmp_path).get("assistant", version="v1")
    registry.fail = True
    with pytest.raises(RuntimeError, match="unavailable"):
        await template.load_skills()
    assert template.instruction == "Hello"
    registry.fail = False
    assert (await template.load_skills())[0].name == "remote"


@pytest.mark.asyncio
async def test_prompt_only_and_discovered_static_skill(tmp_path):
    service = templates(Registry({"manifest.json": "{}"}), tmp_path)
    assert await (await service.get("empty")).load_skills() == []
    template = await templates(
        Registry({"manifest.json": "{}", "skills/local/SKILL.md": "Instructions"}), tmp_path
    ).get("static")
    assert (await template.load_skills())[0].name == "local"


@pytest.mark.asyncio
@pytest.mark.parametrize("names", [("one",), ("one", "two")])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_static_skill_commands_use_the_skill_collection_directory(
    tmp_path, monkeypatch, names, asynchronous
):
    monkeypatch.delenv("ALLOW_EXECUTE_COMMAND", raising=False)
    monkeypatch.chdir(tmp_path)
    files = {"manifest.json": "{}"}
    for name in names:
        files[f"skills/{name}/SKILL.md"] = f"Instructions for {name}"
        files[f"skills/{name}/scripts/run.py"] = f"print({name!r})"
    template = await templates(Registry(files), tmp_path / "workspace").get("static")
    loaded = await template.load_skills()
    execute = next(tool for tool in skill_tools(loaded) if tool.name == "execute_command")

    for name in names:
        relative_path = f"{name}/scripts/run.py" if len(names) > 1 else "scripts/run.py"
        arguments = {"command": f"{shlex.quote(sys.executable)} {relative_path}"}
        output = await execute.ainvoke(arguments) if asynchronous else execute.invoke(arguments)
        result = json.loads(output)
        assert result["exit_code"] == 0, result
        assert result["stdout"] == f"{name}\n"
    assert len({skill.root.parent for skill in loaded}) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "files",
    [
        {"manifest.json": "{broken"},
        {"manifest.json": "{}", "../escape": "bad"},
        {"manifest.json": json.dumps({"skills": [{"name": "missing"}]})},
        {
            "manifest.json": json.dumps({"skills": [{"name": "x", "sourceType": "REFERENCE"}]}),
            "skills/x/SKILL.md": "Conflict",
        },
        {
            "manifest.json": json.dumps(
                {
                    "skills": [
                        {
                            "name": "x",
                            "sourceType": "STATIC",
                            "versionSelector": {"type": "VERSION", "value": "v1"},
                        }
                    ]
                }
            )
        },
    ],
)
async def test_invalid_template_reports_configuration_error(tmp_path, files):
    with pytest.raises(ConfigError):
        template = await templates(Registry(files), tmp_path).get("invalid")
        await template.load_skills()
