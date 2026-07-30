from __future__ import annotations

import tarfile
from pathlib import Path
from types import SimpleNamespace

from adapters.kimi_code import KimiCode
from harbor.agents.installed.base import (
    AgentAuthenticationError,
    ApiConnectionClosedError,
    ApiOverloadedError,
    ApiUsageLimitError,
)


def test_auth_archive_contains_only_runtime_auth_files(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text('default_model = "kimi-code/k3"\n')
    (tmp_path / "tui.toml").write_text('theme = "dark"\n')
    credentials = tmp_path / "credentials"
    credentials.mkdir()
    (credentials / "kimi-code.json").write_text('{"token": "secret"}\n')
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "kimi-code.log").write_text("not copied\n")

    archive_path = KimiCode._build_auth_archive(tmp_path)
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            members = {member.name for member in archive.getmembers()}
    finally:
        archive_path.unlink(missing_ok=True)

    assert "config.toml" in members
    assert "tui.toml" in members
    assert "credentials/kimi-code.json" in members
    assert not any(member.startswith("logs") for member in members)


def test_solution_path_uses_last_requested_artifact() -> None:
    instruction = (
        "Inspect solution_1.py, then write the final artifact to /root/solution_12.py."
    )
    assert KimiCode._expected_solution_path(instruction) == "/root/solution_12.py"


def test_version_parser_accepts_cli_banner() -> None:
    assert KimiCode.parse_version(None, "kimi-code 0.29.2") == "0.29.2"


def test_k3_subscription_error_is_classified_as_authentication(tmp_path: Path) -> None:
    agent = KimiCode(logs_dir=tmp_path, model_name="kimi-code/k3")
    result = SimpleNamespace(
        return_code=1,
        stdout="401 Your current subscription does not have access to k3.",
        stderr="",
    )

    error = agent._classify_exec_error("kimi --model kimi-code/k3", result)

    assert isinstance(error, AgentAuthenticationError)


def test_k3_login_required_is_classified_as_authentication(tmp_path: Path) -> None:
    agent = KimiCode(logs_dir=tmp_path, model_name="kimi-code/k3")
    result = SimpleNamespace(
        return_code=1,
        stdout=(
            'auth.login_required: OAuth provider "managed:kimi-code" '
            "requires login before it can be used."
        ),
        stderr="",
    )

    error = agent._classify_exec_error("kimi --model kimi-code/k3", result)

    assert isinstance(error, AgentAuthenticationError)


def test_k3_overload_is_classified_for_retry(tmp_path: Path) -> None:
    agent = KimiCode(logs_dir=tmp_path, model_name="kimi-code/k3")
    result = SimpleNamespace(
        return_code=1,
        stdout=(
            "provider.rate_limit: 429 The engine is currently overloaded, "
            "please try again later"
        ),
        stderr="",
    )

    error = agent._classify_exec_error("kimi --model kimi-code/k3", result)

    assert isinstance(error, ApiOverloadedError)


def test_k3_usage_limit_is_classified_separately(tmp_path: Path) -> None:
    agent = KimiCode(logs_dir=tmp_path, model_name="kimi-code/k3")
    result = SimpleNamespace(
        return_code=1,
        stdout=(
            "provider.api_error: 403 You've reached your usage limit for "
            "this billing cycle."
        ),
        stderr="",
    )

    error = agent._classify_exec_error("kimi --model kimi-code/k3", result)

    assert isinstance(error, ApiUsageLimitError)


def test_k3_terminated_connection_is_classified_for_retry(tmp_path: Path) -> None:
    agent = KimiCode(logs_dir=tmp_path, model_name="kimi-code/k3")
    result = SimpleNamespace(
        return_code=1,
        stdout="provider.connection_error: terminated",
        stderr="",
    )

    error = agent._classify_exec_error("kimi --model kimi-code/k3", result)

    assert isinstance(error, ApiConnectionClosedError)


def test_refreshed_credentials_are_atomically_persisted(tmp_path: Path) -> None:
    source = tmp_path / "downloaded"
    source.mkdir()
    (source / "kimi-code.json").write_text('{"token": "fresh"}\n')
    ignored_link = source / "ignored-link"
    ignored_link.symlink_to(source / "kimi-code.json")

    home = tmp_path / "home"
    credentials = home / "credentials"
    credentials.mkdir(parents=True)
    target = credentials / "kimi-code.json"
    target.write_text('{"token": "stale"}\n')

    assert KimiCode._persist_refreshed_credentials(source, home) == 1
    assert target.read_text() == '{"token": "fresh"}\n'
    assert not (credentials / "ignored-link").exists()
