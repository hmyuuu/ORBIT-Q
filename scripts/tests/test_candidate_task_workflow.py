from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_candidate_task as materializer  # noqa: E402
import run_harbor_candidate as runner  # noqa: E402


def make_blueprint(
    tmp_path: Path,
    *,
    problem_id: int = 73,
    solution_id: int | None = None,
    candidate_id: str = "qsp-phase-recovery",
    metadata_name: str = "metadata.json",
    slug: str | None = None,
    solution_at_root: bool = False,
) -> Path:
    blueprint = tmp_path / "blueprint"
    blueprint.mkdir()
    (blueprint / "instruction.md").write_text(
        "# Recover QSP phases\n\nImplement the requested TensorCircuit computation.\n"
    )
    (blueprint / f"evaluate_{problem_id}.py").write_text(
        "def main():\n    print('Overall: PASS')\n\nif __name__ == '__main__':\n    main()\n"
    )
    actual_solution_id = problem_id if solution_id is None else solution_id
    solution_root = blueprint if solution_at_root else blueprint / "expert"
    solution_root.mkdir(exist_ok=True)
    (solution_root / f"solution_{actual_solution_id}.py").write_text(
        "def run_solution(config):\n    return {'phases': [0.0]}\n"
    )
    metadata = {
        "candidate_id": candidate_id,
        "title": "QSP phase recovery",
        "problem_id": problem_id,
        "difficulty_explanation": "Inverse synthesis with branch-sensitive checks.",
        "expert_time_estimate_hours": 6,
        "tags": ["inverse-problem", "qsp"],
    }
    if slug is not None:
        metadata["slug"] = slug
    (blueprint / metadata_name).write_text(json.dumps(metadata))
    return blueprint


def materialize(tmp_path: Path) -> Path:
    blueprint = make_blueprint(tmp_path)
    return materializer.materialize_candidate_task(blueprint, tmp_path / "staging")


def runner_args(task_dir: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--task-dir",
        str(task_dir),
        "--model",
        "gpt-5.6-sol",
        "--reasoning-effort",
        "high",
        "--jobs-dir",
        str(tmp_path / "candidate-jobs"),
        "--harbor-bin",
        sys.executable,
        *extra,
    ]


def test_materializes_flat_harbor_task_from_one_blueprint(tmp_path: Path) -> None:
    blueprint = make_blueprint(tmp_path)
    task_dir = materializer.materialize_candidate_task(blueprint, tmp_path / "staging")

    assert task_dir == (tmp_path / "staging" / "candidate-qsp-phase-recovery").resolve()
    expected = {
        "instruction.md",
        "task.toml",
        "candidate_metadata.json",
        "environment",
        "tests",
        "solution",
    }
    assert {path.name for path in task_dir.iterdir()} == expected
    assert (task_dir / "environment" / ".gitkeep").is_file()
    assert (task_dir / "tests" / "evaluate_73.py").is_file()
    assert (task_dir / "solution" / "solution_73.py").is_file()
    assert (task_dir / "tests" / "problem_id.txt").read_text() == "73\n"
    assert (task_dir / "solution" / "problem_id.txt").read_text() == "73\n"
    assert "/root/solution_73.py" in (task_dir / "instruction.md").read_text()

    shared_verifier_files = {
        path.name
        for path in (ROOT / "templates" / "challenge" / "tests").iterdir()
        if path.is_file()
    }
    assert shared_verifier_files <= {
        path.name for path in (task_dir / "tests").iterdir()
    }
    assert (task_dir / "tests" / "test.sh").stat().st_mode & 0o111
    assert (task_dir / "solution" / "solve.sh").stat().st_mode & 0o111

    task_config = tomllib.loads((task_dir / "task.toml").read_text())
    assert task_config["schema_version"] == "1.0"
    assert task_config["artifacts"][0] == "/root/solution_73.py"
    assert "candidate-problem" in task_config["metadata"]["tags"]
    assert (
        "orbit-q-candidate-qsp-phase-recovery-generated"
        in (task_dir / "task.toml").read_text()
    )


def test_materializer_rejects_mismatched_ids_and_existing_destination(
    tmp_path: Path,
) -> None:
    mismatch_root = tmp_path / "mismatch"
    mismatch_root.mkdir()
    mismatch = make_blueprint(mismatch_root, problem_id=7, solution_id=8)
    with pytest.raises(materializer.CandidateTaskError, match="numeric ids differ"):
        materializer.materialize_candidate_task(mismatch, tmp_path / "bad-stage")

    good_root = tmp_path / "good"
    good_root.mkdir()
    good = make_blueprint(good_root)
    output_root = tmp_path / "stage"
    materializer.materialize_candidate_task(good, output_root)
    with pytest.raises(materializer.CandidateTaskError, match="already exists"):
        materializer.materialize_candidate_task(good, output_root)


def test_materializer_rejects_root_solution_and_symlinked_expert_solution(
    tmp_path: Path,
) -> None:
    root_only_root = tmp_path / "root-only"
    root_only_root.mkdir()
    root_only = make_blueprint(root_only_root, solution_at_root=True)
    with pytest.raises(
        materializer.CandidateTaskError, match="only under blueprint expert"
    ):
        materializer.inspect_blueprint(root_only)

    linked_root = tmp_path / "linked"
    linked_root.mkdir()
    linked = make_blueprint(linked_root)
    reference = linked / "expert" / "solution_73.py"
    reference.unlink()
    reference.symlink_to(linked / "instruction.md")
    with pytest.raises(materializer.CandidateTaskError, match="regular file"):
        materializer.inspect_blueprint(linked)


def test_materializer_accepts_blueprint_json_metadata(tmp_path: Path) -> None:
    blueprint = make_blueprint(
        tmp_path,
        candidate_id="f09_qsp_phase_synthesis--benchmark--inverse",
        metadata_name="blueprint.json",
        slug="qsp-phase-synthesis",
    )
    task_dir = materializer.materialize_candidate_task(blueprint, tmp_path / "staging")
    assert task_dir.name == "candidate-qsp-phase-synthesis"
    copied = json.loads((task_dir / "candidate_metadata.json").read_text())
    assert copied["problem_id"] == 73
    assert copied["candidate_id"] == "f09_qsp_phase_synthesis--benchmark--inverse"


def test_materializer_rejects_scientifically_rejected_blueprint(tmp_path: Path) -> None:
    blueprint = make_blueprint(tmp_path, metadata_name="blueprint.json")
    metadata_path = blueprint / "blueprint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["status"] = "prototype_rejected"
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(
        materializer.CandidateTaskError, match="rejected candidate blueprints"
    ):
        materializer.inspect_blueprint(blueprint)


@pytest.mark.parametrize("name", ["tasks", "templates", "prompts", "adapters"])
def test_all_user_paths_reject_shared_or_canonical_trees(name: str) -> None:
    forbidden = ROOT / name / "candidate-anything"
    with pytest.raises(materializer.CandidateTaskError, match="must not be inside"):
        materializer.reject_forbidden_user_path(forbidden, label="test path")
    with pytest.raises(materializer.CandidateTaskError):
        materializer.validate_output_root(forbidden)
    with pytest.raises(materializer.CandidateTaskError):
        runner.validate_candidate_task_dir(forbidden)


def test_repository_local_output_is_candidate_area_only() -> None:
    assert materializer.validate_output_root(materializer.DEFAULT_OUTPUT_ROOT) == (
        materializer.DEFAULT_OUTPUT_ROOT.resolve()
    )
    with pytest.raises(materializer.CandidateTaskError, match="must be placed under"):
        materializer.validate_output_root(ROOT / "reports" / "candidate-task")
    with pytest.raises(materializer.CandidateTaskError, match="must be under"):
        runner.validate_jobs_root(ROOT / "jobs" / "candidate-run")


def test_runner_builds_exactly_one_tensorcircuit_codex_command(tmp_path: Path) -> None:
    task_dir = materialize(tmp_path)
    args = runner.parse_args(runner_args(task_dir, tmp_path))
    command, resolved_task = runner.build_harbor_command(args)

    assert resolved_task == task_dir
    assert command.count("-p") == 1
    assert command[command.index("-p") + 1] == str(task_dir)
    assert command[command.index("-n") + 1] == "1"
    assert command[command.index("--override-cpus") + 1] == "8"
    assert command[command.index("--override-memory-mb") + 1] == "8192"
    assert "adapters.framework_docker:FrameworkDockerEnvironment" in command
    assert "framework=tensorcircuit" in command
    assert "docker_image=challenge-benchmark-quantum-tensorcircuit:py311" in command
    assert "harbor.agents.installed.codex:Codex" in command
    assert "adapters.codex_para_verifier:CodexParaVerifier" in command
    assert "REQUIRED_QUANTUM_FRAMEWORK=tensorcircuit" in command
    assert "reasoning_effort=high" in command
    assert command[command.index("-m") + 1] == "gpt-5.6-sol"


def test_runner_is_dry_run_by_default(tmp_path: Path, monkeypatch, capsys) -> None:
    task_dir = materialize(tmp_path)

    def forbidden_run(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("dry run must not invoke Harbor")

    monkeypatch.setattr(runner.subprocess, "run", forbidden_run)
    assert runner.main(runner_args(task_dir, tmp_path)) == 0
    output = capsys.readouterr().out
    assert "Mode: DRY RUN" in output
    assert "Resource envelope: 8 CPUs, 8192 MiB memory" in output
    assert "No benchmark was started" in output


def test_runner_accepts_explicit_resource_envelope_and_rejects_nonpositive_values(
    tmp_path: Path,
) -> None:
    task_dir = materialize(tmp_path)
    args = runner.parse_args(
        runner_args(
            task_dir,
            tmp_path,
            "--override-cpus",
            "6",
            "--override-memory-mb",
            "12288",
        )
    )
    command, _ = runner.build_harbor_command(args)
    assert command[command.index("--override-cpus") + 1] == "6"
    assert command[command.index("--override-memory-mb") + 1] == "12288"

    for flag in ("--override-cpus", "--override-memory-mb"):
        invalid = runner.parse_args(runner_args(task_dir, tmp_path, flag, "0"))
        with pytest.raises(materializer.CandidateTaskError, match="positive integer"):
            runner.build_harbor_command(invalid)


def test_runner_bridges_credential_free_loopback_proxy_to_agent_and_verifier(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize(tmp_path)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:7890/")
    args = runner.parse_args(runner_args(task_dir, tmp_path, "--bridge-loopback-proxy"))
    command, _ = runner.build_harbor_command(args)

    assert "HTTP_PROXY=http://host.docker.internal:7890" in command
    assert "HTTPS_PROXY=http://host.docker.internal:7890" in command
    assert command.count("--agent-env") == 2
    assert command.count("--verifier-env") == 3


def test_runner_can_use_codex_para_with_auth_json_without_exposing_credentials(
    tmp_path: Path,
) -> None:
    task_dir = materialize(tmp_path)
    args = runner.parse_args(
        runner_args(
            task_dir,
            tmp_path,
            "--solver-agent",
            "codex-para",
            "--force-auth-json",
        )
    )
    command, _ = runner.build_harbor_command(args)

    assert "adapters.codex_para:CodexPara" in command
    assert command.count("force_auth_json=true") == 2
    assert not any("auth.json" in item for item in command)


def test_runner_rejects_force_auth_json_with_public_codex_agent(
    tmp_path: Path,
) -> None:
    task_dir = materialize(tmp_path)
    args = runner.parse_args(runner_args(task_dir, tmp_path, "--force-auth-json"))
    with pytest.raises(materializer.CandidateTaskError, match="codex-para"):
        runner.build_harbor_command(args)


@pytest.mark.parametrize(
    "proxy",
    [
        "socks5://127.0.0.1:7891",
        "http://proxy.example:7890",
        "http://user:secret@127.0.0.1:7890",
        "http://127.0.0.1:7890/path",
    ],
)
def test_runner_rejects_unsafe_or_nonloopback_proxy_bridge(
    tmp_path: Path, monkeypatch, proxy: str
) -> None:
    task_dir = materialize(tmp_path)
    monkeypatch.setenv("HTTP_PROXY", proxy)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    args = runner.parse_args(runner_args(task_dir, tmp_path, "--bridge-loopback-proxy"))
    with pytest.raises(materializer.CandidateTaskError):
        runner.build_harbor_command(args)


def test_runner_requires_execute_and_still_runs_only_explicit_task(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize(tmp_path)
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=17)

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.main(runner_args(task_dir, tmp_path, "--execute"))

    assert result == 17
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command.count("-p") == 1
    assert command[command.index("-p") + 1] == str(task_dir)
    assert kwargs["cwd"] == ROOT
    assert kwargs["check"] is False


def test_runner_rejects_non_tensorcircuit_image_and_uncanaried_task(
    tmp_path: Path,
) -> None:
    task_dir = materialize(tmp_path)
    bad_image_args = runner.parse_args(
        runner_args(task_dir, tmp_path, "--docker-image", "pennylane:latest")
    )
    with pytest.raises(materializer.CandidateTaskError, match="TensorCircuit image"):
        runner.build_harbor_command(bad_image_args)

    task_toml = task_dir / "task.toml"
    task_toml.write_text(task_toml.read_text().replace("harbor-canary GUID", "removed"))
    with pytest.raises(materializer.CandidateTaskError, match="staging canary"):
        runner.validate_candidate_task_dir(task_dir)


def test_task_dir_and_model_are_explicit_cli_requirements() -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--model", "gpt-5.6-sol"])
    with pytest.raises(SystemExit):
        runner.parse_args(["--task-dir", "/tmp/candidate"])


def test_execute_uses_subprocess_argument_vector_not_shell() -> None:
    source = (SCRIPTS / "run_harbor_candidate.py").read_text()
    assert "shell=True" not in source
    assert ".glob(" not in source
    assert ".rglob(" not in source
