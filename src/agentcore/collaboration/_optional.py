"""Import the independently installed collaboration implementation on demand."""

from importlib import import_module
from types import ModuleType


def require_collaboration() -> ModuleType:
    try:
        return import_module("agentcore_collaboration")
    except ModuleNotFoundError as exc:
        if exc.name != "agentcore_collaboration":
            raise
        raise ImportError(
            'Collaboration requires: pip install "alibabacloud-agentcore-sdk[collaboration]"'
        ) from exc
