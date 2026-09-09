from __future__ import annotations

from pathlib import Path

import pytest

from agentcore.errors import ConfigError
from agentcore.runtime.environment import RuntimeEnvironmentProvider


def test_runtime_environment_reads_controller_projection_without_executing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token = tmp_path / "agent token"
    env = tmp_path / "env"
    env.write_text(
        "export AGENTCORE_CONTROLLER_URL='https://controller.example.com/'\n"
        f"export AGENTCORE_AUTH_TOKEN_FILE='{token}'\n"
        "export AGENTCORE_CONTROL_ENDPOINT="
        "'https://agentcore-vpc.cn-hangzhou.aliyuncs.com'\n",
        encoding="utf-8",
    )

    runtime = RuntimeEnvironmentProvider(env).snapshot()

    assert runtime.controller_url == "https://controller.example.com"
    assert runtime.agent_sa_token_file == token
    assert runtime.control_plane_endpoint == ("https://agentcore-vpc.cn-hangzhou.aliyuncs.com")


def test_runtime_environment_falls_back_to_legacy_agentteams_fields(
    tmp_path: Path,
) -> None:
    env = tmp_path / "env"
    env.write_text(
        "export AGENTTEAMS_CONTROLLER_URL='https://legacy-controller.example.com'\n"
        "export AGENTTEAMS_AUTH_TOKEN_FILE='/var/run/legacy/token'\n",
        encoding="utf-8",
    )

    runtime = RuntimeEnvironmentProvider(env).snapshot()

    assert runtime.controller_url == "https://legacy-controller.example.com"
    assert runtime.agent_sa_token_file == Path("/var/run/legacy/token")


def test_runtime_environment_prefers_agentcore_fields_over_legacy_fields(
    tmp_path: Path,
) -> None:
    env = tmp_path / "env"
    env.write_text(
        "export AGENTTEAMS_CONTROLLER_URL='https://legacy-controller.example.com'\n"
        "export AGENTCORE_CONTROLLER_URL='https://controller.example.com'\n"
        "export AGENTTEAMS_AUTH_TOKEN_FILE='/var/run/legacy/token'\n"
        "export AGENTCORE_AUTH_TOKEN_FILE='/var/run/agentcore/token'\n",
        encoding="utf-8",
    )

    runtime = RuntimeEnvironmentProvider(env).snapshot()

    assert runtime.controller_url == "https://controller.example.com"
    assert runtime.agent_sa_token_file == Path("/var/run/agentcore/token")


def test_runtime_environment_uses_control_endpoint_environment_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = tmp_path / "env"
    env.write_text(
        "export AGENTTEAMS_CONTROLLER_URL='https://controller.example.com'\n"
        "export AGENTTEAMS_AUTH_TOKEN_FILE='/var/run/agentcore/agent/token'\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "AGENTCORE_CONTROL_ENDPOINT",
        "https://agentcore-vpc.cn-shanghai.aliyuncs.com",
    )

    runtime = RuntimeEnvironmentProvider(env).snapshot()
    assert runtime.control_plane_endpoint == ("https://agentcore-vpc.cn-shanghai.aliyuncs.com")


def test_runtime_environment_rejects_non_export_shell_content(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.write_text("curl https://attacker.example.com\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="invalid runtime env"):
        RuntimeEnvironmentProvider(env).snapshot()


def test_runtime_environment_requires_controller_and_token(
    tmp_path: Path,
) -> None:
    env = tmp_path / "env"
    env.write_text("export UNUSED='value'\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="AGENTCORE_CONTROLLER_URL"):
        RuntimeEnvironmentProvider(env).snapshot()

    env.write_text(
        "export AGENTCORE_CONTROLLER_URL='https://controller.example.com'\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="AGENTCORE_AUTH_TOKEN_FILE"):
        RuntimeEnvironmentProvider(env).snapshot()


def test_runtime_environment_rereads_named_collaboration_token(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.write_text("export AGENTTEAMS_WORKER_MATRIX_TOKEN='old-token'\n", encoding="utf-8")
    provider = RuntimeEnvironmentProvider(env)

    assert provider.value("AGENTTEAMS_WORKER_MATRIX_TOKEN") == "old-token"

    env.write_text("export AGENTTEAMS_WORKER_MATRIX_TOKEN='new-token'\n", encoding="utf-8")
    assert provider.value("AGENTTEAMS_WORKER_MATRIX_TOKEN") == "new-token"


def test_runtime_environment_prefers_agentcore_task_service_endpoint(
    tmp_path: Path,
) -> None:
    env = tmp_path / "env"
    env.write_text(
        "export AGENTCORE_TASK_SERVICE_ENDPOINT="
        "'https://task-service.example.com/agentteams-app/'\n"
        "export AGENTTEAMS_MATRIX_URL='https://gateway.example.com'\n",
        encoding="utf-8",
    )

    endpoint = RuntimeEnvironmentProvider(env).task_service_endpoint()

    assert endpoint == "https://task-service.example.com/agentteams-app"


def test_runtime_environment_derives_task_service_endpoint_from_matrix_url(
    tmp_path: Path,
) -> None:
    env = tmp_path / "env"
    env.write_text(
        "export AGENTTEAMS_MATRIX_URL='https://gateway.example.com/root/'\n",
        encoding="utf-8",
    )

    endpoint = RuntimeEnvironmentProvider(env).task_service_endpoint()

    assert endpoint == "https://gateway.example.com/root/agentteams-app"


def test_runtime_environment_does_not_duplicate_task_service_path(
    tmp_path: Path,
) -> None:
    env = tmp_path / "env"
    env.write_text(
        "export AGENTTEAMS_MATRIX_URL='https://gateway.example.com/agentteams-app/'\n",
        encoding="utf-8",
    )

    endpoint = RuntimeEnvironmentProvider(env).task_service_endpoint()

    assert endpoint == "https://gateway.example.com/agentteams-app"


def test_runtime_environment_rereads_task_service_endpoint(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.write_text(
        "export AGENTTEAMS_MATRIX_URL='https://old.example.com'\n",
        encoding="utf-8",
    )
    provider = RuntimeEnvironmentProvider(env)

    assert provider.task_service_endpoint() == "https://old.example.com/agentteams-app"

    env.write_text(
        "export AGENTTEAMS_MATRIX_URL='https://new.example.com'\n",
        encoding="utf-8",
    )
    assert provider.task_service_endpoint() == "https://new.example.com/agentteams-app"


def test_runtime_environment_requires_valid_task_service_endpoint(tmp_path: Path) -> None:
    env = tmp_path / "env"
    env.write_text("export UNUSED='value'\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="AGENTCORE_TASK_SERVICE_ENDPOINT"):
        RuntimeEnvironmentProvider(env).task_service_endpoint()

    env.write_text(
        "export AGENTCORE_TASK_SERVICE_ENDPOINT='file:///agentteams-app'\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="Task Service Endpoint"):
        RuntimeEnvironmentProvider(env).task_service_endpoint()
