from __future__ import annotations

import hashlib
import json
import shutil
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


def materialize_expert(tmp_path: Path) -> Path:
    workspace = tmp_path / "discovery"
    workspace.mkdir(exist_ok=True)
    discovery = ROOT / "reports" / "orbit_q_problem_discovery"
    for name in ("candidates.jsonl", "shortlist.json"):
        (workspace / name).write_bytes((discovery / name).read_bytes())
    blueprint = discovery / "blueprints" / "robust-leakage-grape"
    shutil.copytree(blueprint, workspace / "blueprints" / "robust-leakage-grape")
    return materializer.materialize_candidate_task(
        blueprint, tmp_path / "expert-staging", workspace
    )


def runner_args(task_dir: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--task-dir",
        str(task_dir),
        "--workspace",
        str(tmp_path / "discovery"),
        "--manifest",
        str(tmp_path / "authorization.json"),
        "--candidate-seed",
        "730019",
        "--jobs-dir",
        str(tmp_path / "candidate-jobs"),
        "--harbor-bin",
        sys.executable,
        *extra,
    ]


def expert_runner_args(task_dir: Path, tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--task-dir",
        str(task_dir),
        "--expert-only",
        "--workspace",
        str(tmp_path / "discovery"),
        "--candidate-seed",
        "1092026",
        "--container-image-digest",
        "sha256:" + "e" * 64,
        "--jobs-dir",
        str(tmp_path / "candidate-jobs"),
        "--harbor-bin",
        sys.executable,
        *extra,
    ]


def authorized_spec(**updates) -> dict:
    spec = {
        "candidate_id": "qsp-phase-recovery",
        "manifest_hash": "1" * 64,
        "protocol_seed": 730019,
        "solver_agent": "codex",
        "solver_agent_import_path": "harbor.agents.installed.codex:Codex",
        "model": "gpt-5.6-sol",
        "provider_snapshot": "provider-build-test",
        "reasoning_effort": "high",
        "token_budget": 5000000,
        "token_budget_metric": "input_plus_output_tokens",
        "token_budget_enforcement": "post_run_admission_only",
        "wall_time_sec": 1800,
        "solver_agent_version": "0.146.0",
        "harbor_version": "0.20.0",
        "audit_model": "gpt-5",
        "build_identity_policy": "require_runtime_observation_for_hardness",
        "tools_policy": "harbor_default_no_mcp",
        "network_policy": "public",
        "proxy_policy": "none",
        "cpus": 8,
        "memory_mb": 8192,
        "docker_image": "challenge-benchmark-quantum-tensorcircuit:py311",
        "container_image_digest": "sha256:" + "e" * 64,
        "task_bundle_sha256": "a" * 64,
        "harbor_task_digest": "b" * 64,
        "task_checksum": "c" * 64,
        "framework_prompt_sha256": "d" * 64,
        "force_auth_json": False,
    }
    spec.update(updates)
    return spec


@pytest.fixture(autouse=True)
def stub_authorized_runner(monkeypatch) -> None:
    monkeypatch.setattr(
        runner,
        "authorized_model_execution",
        lambda *args, **kwargs: authorized_spec(),
    )
    monkeypatch.setattr(
        runner,
        "reserve_model_trial_seed",
        lambda *args, **kwargs: {"reservation_hash": "f" * 64},
    )
    monkeypatch.setattr(runner, "verify_harbor_version", lambda *args: None)


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


def test_materializer_does_not_bake_host_candidate_seed_into_agent_environment(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv(runner.CANDIDATE_SEED_ENV, "730019")
    task_dir = materialize(tmp_path)
    task_text = (task_dir / "task.toml").read_text()
    task_config = tomllib.loads(task_text)

    assert runner.CANDIDATE_SEED_ENV not in task_text
    assert runner.CANDIDATE_SEED_ENV not in task_config["environment"]["env"]
    assert "env" not in task_config["agent"]


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
    assert (
        "docker_image=challenge-benchmark-quantum-tensorcircuit:py311@sha256:"
        + "e" * 64
    ) in command
    assert "harbor.agents.installed.codex:Codex" in command
    assert "adapters.codex_para_verifier:CodexParaVerifier" in command
    assert "REQUIRED_QUANTUM_FRAMEWORK=tensorcircuit" in command
    assert "reasoning_effort=high" in command
    assert "version=0.146.0" in command
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


def test_runner_derives_resource_envelope_and_rejects_cli_overrides(
    tmp_path: Path,
) -> None:
    task_dir = materialize(tmp_path)
    args = runner.parse_args(runner_args(task_dir, tmp_path))
    command, _ = runner.build_harbor_command(args)
    assert command[command.index("--override-cpus") + 1] == "8"
    assert command[command.index("--override-memory-mb") + 1] == "8192"
    assert command[command.index("--cpus") + 1] == "limit"
    assert command[command.index("--memory") + 1] == "limit"

    for flag in ("--override-cpus", "--override-memory-mb"):
        invalid = runner.parse_args(runner_args(task_dir, tmp_path, flag, "6"))
        with pytest.raises(materializer.CandidateTaskError, match="derived"):
            runner.build_harbor_command(invalid)


def test_runner_passes_candidate_seed_only_to_verifier(tmp_path: Path) -> None:
    task_dir = materialize(tmp_path)
    args = runner.parse_args(runner_args(task_dir, tmp_path))
    command, _ = runner.build_harbor_command(args)

    seed_binding = "ORBIT_Q_CANDIDATE_SEED=730019"
    assert seed_binding in command
    seed_index = command.index(seed_binding)
    assert command[seed_index - 1] == "--verifier-env"
    assert seed_binding not in [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "--agent-env"
    ]


def test_runner_normalizes_exported_candidate_seed_to_verifier_only(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize(tmp_path)
    monkeypatch.setenv(runner.CANDIDATE_SEED_ENV, "999999")
    argv = runner_args(task_dir, tmp_path)
    seed_index = argv.index("--candidate-seed")
    del argv[seed_index : seed_index + 2]
    args = runner.parse_args(argv)
    with pytest.raises(materializer.CandidateTaskError, match="explicit"):
        runner.build_harbor_command(args)


def test_execute_scrubs_host_candidate_seed_after_verifier_binding(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize(tmp_path)
    calls: list[tuple[list[str], dict]] = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv(runner.CANDIDATE_SEED_ENV, "999999")
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    result = runner.main(
        runner_args(
            task_dir,
            tmp_path,
            "--candidate-seed",
            "730019",
            "--execute",
        )
    )

    assert result == 0
    assert len(calls) == 1
    command, kwargs = calls[0]
    binding = f"{runner.CANDIDATE_SEED_ENV}=730019"
    assert command.count(binding) == 1
    assert command[command.index(binding) - 1] == "--verifier-env"
    assert runner.CANDIDATE_SEED_ENV not in kwargs["env"]


def test_runner_rejects_negative_candidate_seed(tmp_path: Path) -> None:
    task_dir = materialize(tmp_path)
    args = runner.parse_args(runner_args(task_dir, tmp_path, "--candidate-seed", "-1"))
    with pytest.raises(materializer.CandidateTaskError, match="non-negative integer"):
        runner.build_harbor_command(args)


def test_runner_bridges_credential_free_loopback_proxy_to_agent_and_verifier(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize(tmp_path)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://localhost:7890/")
    monkeypatch.setattr(
        runner,
        "authorized_model_execution",
        lambda *args, **kwargs: authorized_spec(proxy_policy="loopback_bridge"),
    )
    args = runner.parse_args(runner_args(task_dir, tmp_path))
    command, _ = runner.build_harbor_command(args)

    assert "HTTP_PROXY=http://host.docker.internal:7890" in command
    assert "HTTPS_PROXY=http://host.docker.internal:7890" in command
    assert command.count("--agent-env") == 2
    assert command.count("--verifier-env") == 13


def test_runner_can_use_codex_para_with_auth_json_without_exposing_credentials(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize(tmp_path)
    monkeypatch.setattr(
        runner,
        "authorized_model_execution",
        lambda *args, **kwargs: authorized_spec(
            solver_agent="codex-para",
            solver_agent_import_path="adapters.codex_para:CodexPara",
            force_auth_json=True,
        ),
    )
    args = runner.parse_args(runner_args(task_dir, tmp_path))
    command, _ = runner.build_harbor_command(args)

    assert "adapters.codex_para:CodexPara" in command
    assert command.count("force_auth_json=true") == 2
    assert not any("auth.json" in item for item in command)


def test_runner_expert_only_omits_solver_and_keeps_single_candidate_verifier(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize_expert(tmp_path)
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--force-auth-json",
            "--bridge-loopback-proxy",
        )
    )
    command, resolved_task = runner.build_harbor_command(args)

    assert resolved_task == task_dir
    assert command.count("-p") == 1
    assert command[command.index("-p") + 1] == str(task_dir)
    assert command[command.index("-n") + 1] == "1"
    assert "--agent-import-path" not in command
    assert "--agent-kwarg" not in command
    assert "--agent-env" not in command
    assert "-m" not in command
    assert command[command.index("--agent") + 1] == "oracle"
    assert "adapters.codex_para_verifier:CodexParaVerifier" in command
    assert "ORBIT_Q_NON_HARDNESS_RUN=expert_prequalification" in command
    assert command.count("force_auth_json=true") == 1
    assert "HTTP_PROXY=http://host.docker.internal:7890" in command
    assert "HTTPS_PROXY=http://host.docker.internal:7890" in command


def test_runner_expert_only_rejects_evaluator_without_admission_protocol(
    tmp_path: Path,
) -> None:
    task_dir = materialize(tmp_path)
    args = runner.parse_args(expert_runner_args(task_dir, tmp_path))

    with pytest.raises(
        materializer.CandidateTaskError,
        match="candidate-specific expert admission metrics",
    ):
        runner.build_harbor_command(args)


@pytest.mark.parametrize(
    "relative",
    (
        "tests/score_submission.py",
        "tests/test.sh",
        "tests/static_policy.py",
        "tests/audit_codex.py",
        "solution/solve.sh",
        "task.toml",
    ),
)
def test_expert_runner_rejects_tampered_trusted_materialization(
    tmp_path: Path, relative: str
) -> None:
    task_dir = materialize_expert(tmp_path)
    target = task_dir / relative
    target.write_bytes(target.read_bytes() + b"\n# staged substitution\n")
    args = runner.parse_args(expert_runner_args(task_dir, tmp_path))

    with pytest.raises(materializer.CandidateTaskError, match="materialized"):
        runner.build_harbor_command(args)


def test_expert_runner_rejects_unexpected_task_file_and_shell_mode_drift(
    tmp_path: Path,
) -> None:
    extra_root = tmp_path / "extra"
    extra_root.mkdir()
    extra_task = materialize_expert(extra_root)
    (extra_task / "tests" / "operator-summary.py").write_text("ADMITTED = True\n")
    extra_args = runner.parse_args(expert_runner_args(extra_task, extra_root))
    with pytest.raises(materializer.CandidateTaskError, match="file set"):
        runner.build_harbor_command(extra_args)

    mode_root = tmp_path / "mode"
    mode_root.mkdir()
    mode_task = materialize_expert(mode_root)
    (mode_task / "tests" / "test.sh").chmod(0o644)
    mode_args = runner.parse_args(expert_runner_args(mode_task, mode_root))
    with pytest.raises(materializer.CandidateTaskError, match="trusted template"):
        runner.build_harbor_command(mode_args)


def test_expert_runner_collects_one_real_harbor_oracle_layout(tmp_path: Path) -> None:
    """Exercise production command bindings and collector against Harbor-shaped files."""

    task_dir = materialize_expert(tmp_path)
    evidence_root = tmp_path / "expert-evidence"
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(evidence_root),
            "--job-name",
            "expert-109-seed-1092026",
        )
    )
    command, _ = runner.build_harbor_command(args)
    assert command[command.index("--agent") + 1] == "oracle"
    assert command[command.index("-n") + 1] == "1"
    assert command[command.index("--max-retries") + 1] == "0"

    resolved = args._resolved_run
    verifier_env = runner._required_expert_verifier_env(resolved)
    image_reference = f"{resolved['docker_image']}@{resolved['container_image_digest']}"
    environment = {
        "import_path": "adapters.framework_docker:FrameworkDockerEnvironment",
        "cpu_enforcement_policy": "limit",
        "memory_enforcement_policy": "limit",
        "override_cpus": 8,
        "override_memory_mb": 8192,
        "kwargs": {
            "framework": "tensorcircuit",
            "docker_image": image_reference,
        },
    }
    verifier = {
        "env": verifier_env,
        "import_path": "adapters.codex_para_verifier:CodexParaVerifier",
        "kwargs": {"audit_model": "gpt-5", "expert_only": True},
        "disable": False,
    }
    oracle = {
        "name": "oracle",
        "import_path": None,
        "model_name": None,
        "kwargs": {},
        "skills": [],
        "mcp_servers": [],
        "extra_allowed_hosts": [],
    }
    raw_config = {
        "task": {"path": str(task_dir)},
        "agent": oracle,
        "environment": environment,
        "verifier": verifier,
    }
    harbor_task_digest = runner._harbor_task_digest(task_dir)
    task_checksum = runner._harbor_legacy_task_checksum(task_dir)
    # Harbor --print-config and real trial sidecars omit the default Oracle block.
    sidecar_config = {
        "task": {"path": str(task_dir)},
        "environment": environment,
        "verifier": verifier,
    }
    trial_lock = {
        "schema_version": 1,
        "task": {
            "name": task_dir.name,
            "type": "local",
            "digest": f"sha256:{harbor_task_digest}",
            "path": str(task_dir),
        },
        "install_only": False,
        "skills": [],
        "timeout_multiplier": 1.0,
        "extra_instructions": [
            {
                "path": str(runner.TENSORCIRCUIT_PROMPT),
                "digest": f"sha256:{resolved['framework_prompt_sha256']}",
            }
        ],
        "agent": oracle,
        "environment": environment,
        "verifier": verifier,
    }
    job_lock = {
        "schema_version": 2,
        "created_at": "2026-08-10T00:00:00Z",
        "harbor": {"version": "0.20.0", "is_editable": False},
        "n_concurrent_trials": 1,
        "retry": {"max_retries": 0},
        "trials": [trial_lock],
    }
    case_digest = hashlib.sha256(b"real-layout-case").hexdigest()
    admission = {
        "status": "admitted",
        "minimum_margin": 0.25,
        "margins": [
            {
                "metric": "heldout_worst_infidelity",
                "direction": "at_most",
                "observed": 0.75,
                "threshold": 1.0,
                "absolute_margin": 0.25,
                "passed": True,
            }
        ],
    }
    policy = [
        {
            "metric": "heldout_worst_infidelity",
            "direction": "at_most",
            "threshold": 1.0,
        }
    ]
    policy_hash = hashlib.sha256(
        json.dumps(policy, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    admission_record = {
        "schema_version": 1,
        "producer": "orbit_q_score_submission_from_candidate_evaluator",
        "candidate_id": resolved["candidate_id"],
        "candidate_hash": resolved["candidate_hash"],
        "protocol_seed": resolved["protocol_seed"],
        "case_digest": case_digest,
        "expert_baseline_sha256": resolved["expert_baseline_sha256"],
        "evaluator_sha256": resolved["evaluator_sha256"],
        "task_bundle_sha256": resolved["task_bundle_sha256"],
        "framework_prompt_sha256": resolved["framework_prompt_sha256"],
        "verifier_harness_sha256": resolved["verifier_harness_sha256"],
        "container_image_digest": resolved["container_image_digest"],
        "expert_passed": True,
        "threshold_policy_sha256": policy_hash,
        "admission": admission,
    }
    serialized_admission = json.dumps(admission_record, indent=2, sort_keys=True) + "\n"
    admission_hash = hashlib.sha256(serialized_admission.encode()).hexdigest()
    rewards = {
        "reward": 1.0,
        "functional_score": 1.0,
        "static_policy_score": 1.0,
        "llm_audit_score": 1.0,
        "runtime_sec": 12.5,
        **{
            f"expert_admission_sha256_word_{index}": int(
                admission_hash[offset : offset + 8], 16
            )
            for index, offset in enumerate(range(0, 64, 8))
        },
    }
    result = {
        "id": "trial-1092026",
        "finished_at": "2026-08-10T00:01:00Z",
        "exception_info": None,
        "task_checksum": task_checksum,
        "config": raw_config,
        "agent_info": {"name": "oracle", "version": "1.0.0", "model_info": None},
        "agent_result": {
            "n_input_tokens": None,
            "n_output_tokens": None,
            "cost_usd": None,
        },
        "verifier_result": {"rewards": rewards},
    }

    job_dir = tmp_path / "candidate-jobs" / args.job_name
    trial_dir = job_dir / "candidate-robust-leakage-grape__fixture"
    verifier_dir = trial_dir / "verifier"
    artifact_dir = trial_dir / "artifacts" / "root"
    verifier_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    (job_dir / "lock.json").write_text(json.dumps(job_lock))
    (trial_dir / "lock.json").write_text(json.dumps(trial_lock))
    (trial_dir / "config.json").write_text(json.dumps(sidecar_config))
    (trial_dir / "result.json").write_text(json.dumps(result))
    (verifier_dir / "expert-admission.json").write_text(serialized_admission)
    (verifier_dir / "functional-stdout.txt").write_text(
        json.dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": resolved["protocol_seed"],
                    "case_digest": case_digest,
                }
            }
        )
        + "\n"
        + json.dumps(
            {
                "orbit_q_expert_admission_metrics": {
                    "schema_version": 1,
                    "protocol_seed": resolved["protocol_seed"],
                    "case_digest": case_digest,
                    "metrics": [],
                }
            }
        )
        + "\nOverall: PASS\n"
    )
    (artifact_dir / "solution_109.py").write_bytes(
        (task_dir / "solution" / "solution_109.py").read_bytes()
    )

    case_path = runner.collect_expert_prequalification_evidence(args, task_dir)
    case = json.loads(case_path.read_text())
    assert case["protocol_seed"] == 1092026
    assert case["case_digest"] == case_digest
    assert case["expert_passed"] is True
    assert set(case["raw_evidence"]) == {
        "raw_job",
        "harbor_config",
        "harbor_lock",
        "harbor_job_lock",
        "functional_output",
        "admission_record",
    }
    for item in case["raw_evidence"].values():
        assert (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )


@pytest.mark.parametrize(
    ("flag", "value"),
    (("--override-cpus", "0"), ("--override-memory-mb", "0")),
)
def test_runner_expert_only_rejects_explicit_zero_resources(
    tmp_path: Path, flag: str, value: str
) -> None:
    task_dir = materialize_expert(tmp_path)
    args = runner.parse_args(expert_runner_args(task_dir, tmp_path, flag, value))
    with pytest.raises(materializer.CandidateTaskError, match="positive integer"):
        runner.build_harbor_command(args)


def test_runner_rejects_force_auth_json_with_public_codex_agent(
    tmp_path: Path,
) -> None:
    task_dir = materialize(tmp_path)
    args = runner.parse_args(runner_args(task_dir, tmp_path, "--force-auth-json"))
    with pytest.raises(materializer.CandidateTaskError, match="derived"):
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
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setattr(
        runner,
        "authorized_model_execution",
        lambda *args, **kwargs: authorized_spec(proxy_policy="loopback_bridge"),
    )
    args = runner.parse_args(runner_args(task_dir, tmp_path))
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
    task_dir = materialize_expert(tmp_path)
    bad_image_args = runner.parse_args(
        expert_runner_args(task_dir, tmp_path, "--docker-image", "pennylane:latest")
    )
    with pytest.raises(materializer.CandidateTaskError, match="TensorCircuit image"):
        runner.build_harbor_command(bad_image_args)

    task_toml = task_dir / "task.toml"
    task_toml.write_text(task_toml.read_text().replace("harbor-canary GUID", "removed"))
    with pytest.raises(materializer.CandidateTaskError, match="staging canary"):
        runner.validate_candidate_task_dir(task_dir)


def test_task_dir_is_cli_required_and_model_run_requires_manifest(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        runner.parse_args(["--model", "gpt-5.6-sol"])
    task_dir = materialize(tmp_path)
    args = runner.parse_args(
        [
            "--task-dir",
            str(task_dir),
            "--candidate-seed",
            "730019",
            "--jobs-dir",
            str(tmp_path / "jobs"),
        ]
    )
    with pytest.raises(materializer.CandidateTaskError, match="authorization"):
        runner.build_harbor_command(args)


def test_execute_without_manifest_never_starts_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize(tmp_path)

    def forbidden_run(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("unauthorized execute must not invoke a subprocess")

    monkeypatch.setattr(runner.subprocess, "run", forbidden_run)
    result = runner.main(
        [
            "--task-dir",
            str(task_dir),
            "--candidate-seed",
            "730019",
            "--jobs-dir",
            str(tmp_path / "jobs"),
            "--execute",
        ]
    )
    assert result == 2


def test_execute_uses_subprocess_argument_vector_not_shell() -> None:
    source = (SCRIPTS / "run_harbor_candidate.py").read_text()
    assert "shell=True" not in source
    assert ".glob(" not in source
    assert ".rglob(" not in source
