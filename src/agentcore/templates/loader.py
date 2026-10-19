"""Load AgentSpec packages without taking over Agent execution."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import zipfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import anyio

from agentcore.controlplane import TemplateArtifact
from agentcore.errors import ConfigError, ResourceNotConfiguredError
from agentcore.skill import AsyncSkills, Skill
from agentcore.skill.loader import (
    _load_skill_directory,
    _materialize_directory,
    _PreparedSkill,
    _safe_resource_path,
    _validate_directory_component,
)

logger = logging.getLogger(__name__)


class TemplateProvider(Protocol):
    async def get_template(self, name: str, version: str | None = None) -> TemplateArtifact: ...

    async def resolve_skill_version(self, name: str, selector_type: str, value: str) -> str: ...


TemplateRuntime = Callable[[], Awaitable[tuple[TemplateProvider | None, str | None]]]


@dataclass(frozen=True)
class _SkillDefinition:
    name: str
    source: str
    selector_type: str = "LABEL"
    selector_value: str = "latest"


class AgentTemplate:
    """One fetched template; referenced Skills are loaded only on explicit request."""

    def __init__(
        self, name: str, version: str, instruction: str, load: Callable[[], Awaitable[list[Skill]]]
    ) -> None:
        self.name = name
        self.version = version
        self.instruction = instruction
        self._load = load
        self._skills: tuple[Skill, ...] | None = None
        self._lock = anyio.Lock()

    async def load_skills(self) -> list[Skill]:
        async with self._lock:
            if self._skills is None:
                logger.info(
                    "agentcore.template.skills.load.started name=%s version=%s",
                    self.name,
                    self.version,
                )
                try:
                    self._skills = tuple(await self._load())
                except Exception:
                    logger.exception(
                        "agentcore.template.skills.load.failed name=%s version=%s",
                        self.name,
                        self.version,
                    )
                    raise
                logger.info(
                    "agentcore.template.skills.load.succeeded name=%s version=%s count=%s",
                    self.name,
                    self.version,
                    len(self._skills),
                )
            return list(self._skills)


class AsyncTemplates:
    def __init__(
        self,
        runtime: TemplateRuntime,
        skills: AsyncSkills,
        *,
        workspace_dir: str | Path | None = None,
    ) -> None:
        self._runtime = runtime
        self._skills = skills
        self._workspace = (
            Path(
                workspace_dir
                or os.getenv("AGENTCORE_SKILL_WORKSPACE_DIR")
                or ".skills/.agentcore-managed"
            )
            / ".templates"
        )

    async def get(self, name: str, *, version: str | None = None) -> AgentTemplate:
        _validate_directory_component(name, "template name")
        if version is not None:
            _validate_directory_component(version, "template version")
        provider, workspace_id = await self._runtime()
        if provider is None or workspace_id is None:
            raise ResourceNotConfiguredError("Agent template runtime is not configured")
        _validate_directory_component(workspace_id, "workspace ID")
        logger.info("agentcore.template.get.started name=%s version=%s", name, version or "latest")
        artifact = await provider.get_template(name, version)
        _validate_directory_component(artifact.version, "template version")
        files, definitions, instruction = await anyio.to_thread.run_sync(_parse, artifact.archive)
        digest = hashlib.sha256(artifact.archive).hexdigest()

        async def load() -> list[Skill]:
            await self._runtime()
            result = []
            for definition in definitions:
                logger.info(
                    "agentcore.template.skill.load.started template=%s skill=%s source=%s",
                    name,
                    definition.name,
                    definition.source,
                )
                if definition.source == "REFERENCE":
                    selected = await provider.resolve_skill_version(
                        definition.name, definition.selector_type, definition.selector_value
                    )
                    result.append(await self._skills.managed(definition.name, version=selected))
                else:
                    prefix = f"skills/{definition.name}/"
                    resources = {
                        path[len(prefix) :]: content
                        for path, content in files.items()
                        if path.startswith(prefix)
                    }
                    if "SKILL.md" not in resources:
                        raise ConfigError(f"Static Skill {definition.name!r} has no SKILL.md")
                    prepared = _PreparedSkill(artifact.version, resources, digest)

                    def materialize(
                        definition: _SkillDefinition = definition,
                        prepared: _PreparedSkill = prepared,
                    ) -> Skill:
                        root = _materialize_directory(
                            self._workspace
                            / workspace_id
                            / name
                            / artifact.version
                            / digest
                            / definition.name,
                            prepared,
                            {
                                "source": "template",
                                "workspaceId": workspace_id,
                                "name": definition.name,
                                "version": artifact.version,
                                "digest": digest,
                            },
                        )
                        return replace(
                            _load_skill_directory(root, "local", None, digest),
                            name=definition.name,
                            workspace_id=workspace_id,
                        )

                    result.append(await anyio.to_thread.run_sync(materialize))
            return result

        logger.info(
            "agentcore.template.get.succeeded name=%s version=%s skills=%s",
            name,
            artifact.version,
            len(definitions),
        )
        return AgentTemplate(name, artifact.version, instruction, load)


def _parse(content: bytes) -> tuple[dict[str, bytes], list[_SkillDefinition], str]:
    files: dict[str, bytes] = {}
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            for entry in archive.infolist():
                if entry.is_dir():
                    continue
                path = _safe_resource_path(entry.filename).as_posix()
                if path in files or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ConfigError("Agent template contains a duplicate path or symbolic link")
                total += entry.file_size
                if total > 50 * 1024 * 1024:
                    raise ConfigError("Expanded Agent template exceeds 50 MiB")
                files[path] = archive.read(entry)
        manifest = json.loads(files.get("manifest.json", b"{}"))
        if not isinstance(manifest, dict) or not isinstance(manifest.get("skills", []), list):
            raise ConfigError("Agent template manifest skills must be an array")
        definitions: dict[str, _SkillDefinition] = {}
        for value in manifest.get("skills", []):
            value = {"name": value} if isinstance(value, str) else value
            if not isinstance(value, dict) or not isinstance(value.get("name"), str):
                raise ConfigError("Agent template contains an invalid Skill definition")
            name = value["name"]
            _validate_directory_component(name, "Skill name")
            source = value.get("sourceType")
            if source is None:
                source = "STATIC"
            elif value.get("version") is not None:
                raise ConfigError(f"Template Skill {name!r} must use versionSelector, not version")
            selector = value.get("versionSelector")
            if source not in {"STATIC", "REFERENCE"} or source == "STATIC" and selector is not None:
                raise ConfigError(
                    f"Agent template Skill {name!r} has an invalid source or selector"
                )
            if selector is None:
                selector = {"type": "LABEL", "value": "latest"}
            if (
                not isinstance(selector, dict)
                or selector.get("type") not in {"LABEL", "VERSION"}
                or not isinstance(selector.get("value"), str)
                or not selector["value"].strip()
            ):
                raise ConfigError(f"Agent template Skill {name!r} has an invalid version selector")
            if name in definitions:
                raise ConfigError(f"Duplicate template Skill: {name}")
            definitions[name] = _SkillDefinition(name, source, selector["type"], selector["value"])
        for path in files:
            parts = path.split("/")
            if len(parts) == 3 and parts[0] == "skills" and parts[2] == "SKILL.md":
                name = parts[1]
                if name in definitions and definitions[name].source == "REFERENCE":
                    raise ConfigError(f"Template Skill is both a reference and a file: {name}")
                definitions.setdefault(name, _SkillDefinition(name, "STATIC"))
        agents = files.get("config/AGENTS.md", b"").decode("utf-8")
        soul = files.get("config/SOUL.md", b"").decode("utf-8")
        instruction = (
            f"## AGENTS.md\n\n{agents.strip()}\n\n## SOUL.md\n\n{soul.strip()}"
            if agents.strip() and soul.strip()
            else agents
            if agents.strip()
            else soul
        )
        return files, list(definitions.values()), instruction
    except (zipfile.BadZipFile, UnicodeError, ValueError) as exc:
        raise ConfigError("Agent template package is invalid") from exc
