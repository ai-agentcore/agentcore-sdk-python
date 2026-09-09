"""Built-in Worker Skills selected explicitly by user code."""

from pathlib import Path

from agentcore.skill import Skill

_ROOT = Path(__file__).with_name("worker_skills")
_SKILLS = (
    (
        "task-execution",
        "Execute an assigned collaboration Subtask through its Task Service lifecycle.",
    ),
    (
        "file-sharing",
        "Synchronize non-Task Team files; Task and Subtask files belong to task-execution.",
    ),
)


def worker_skills() -> list[Skill]:
    result: list[Skill] = []
    for name, description in _SKILLS:
        root = _ROOT / name
        instruction_path = root / "SKILL.md"
        result.append(
            Skill(
                name=name,
                description=description,
                version="1",
                instruction=instruction_path.read_text(encoding="utf-8"),
                root=root,
                files=("SKILL.md",),
                source="agentcore-collaboration",
            )
        )
    return result
