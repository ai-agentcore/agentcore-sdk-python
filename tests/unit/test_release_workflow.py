"""Validate the tag routing executed by the Python 3.11 release runner."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

tomllib = pytest.importorskip("tomllib", reason="Release workflow runs on Python 3.11")

ROOT = Path(__file__).parents[2]


@pytest.mark.parametrize(
    ("prefix", "directory", "package", "valid_version"),
    [
        ("sdk-v", ".", "alibabacloud-agentcore-sdk", True),
        ("collaboration-v", "packages/collaboration", "alibabacloud-agentcore-collaboration", True),
        ("sdk-v", ".", "alibabacloud-agentcore-sdk", False),
        (
            "collaboration-v",
            "packages/collaboration",
            "alibabacloud-agentcore-collaboration",
            False,
        ),
        ("v", ".", None, True),
    ],
)
def test_release_tag_routes_and_validates_version(
    tmp_path, prefix, directory, package, valid_version
):
    version = tomllib.loads((ROOT / directory / "pyproject.toml").read_text())["project"]["version"]
    tag = prefix + (version if valid_version else version + ".mismatch")
    workflow = yaml.safe_load((ROOT / ".github/workflows/publish-pypi.yml").read_text())
    step = next(s for s in workflow["jobs"]["build"]["steps"] if s.get("id") == "package")
    script = step["run"].split("\n", 1)[1].rsplit("\nPY", 1)[0]
    output = tmp_path / "output"
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env={**os.environ, "RELEASE_TAG": tag, "GITHUB_OUTPUT": str(output)},
        capture_output=True,
        text=True,
    )
    if not valid_version or package is None:
        assert result.returncode != 0
        assert not output.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert output.read_text() == f"directory={directory}\npackage={package}\n"
