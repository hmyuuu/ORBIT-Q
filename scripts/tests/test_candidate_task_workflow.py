from __future__ import annotations

import asyncio
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
from adapters import codex_para_verifier as verifier_adapter  # noqa: E402


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


def materialize_reviewed(tmp_path: Path, slug: str) -> Path:
    workspace = tmp_path / "discovery"
    workspace.mkdir(exist_ok=True)
    discovery = ROOT / "reports" / "orbit_q_problem_discovery"
    for name in ("candidates.jsonl", "shortlist.json"):
        (workspace / name).write_bytes((discovery / name).read_bytes())
    blueprint = discovery / "blueprints" / slug
    shutil.copytree(blueprint, workspace / "blueprints" / slug)
    return materializer.materialize_candidate_task(
        blueprint, tmp_path / "expert-staging", workspace
    )


def materialize_expert(tmp_path: Path) -> Path:
    return materialize_reviewed(tmp_path, "mixed_sld_qfim")


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
        "101021",
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


def make_expert_execution_approval(
    args,
    task_dir: Path,
    tmp_path: Path,
    *,
    create_reservation: bool = True,
    plan_cases: list[dict] | None = None,
) -> tuple[Path, Path]:
    """Create one test-only human batch approval for the already built command."""

    sealed = tmp_path / "sealed-expert-plan"
    sealed.mkdir(exist_ok=True)
    driver_path = (sealed / "execute_plan.py").resolve()
    driver_path.write_text("# frozen test execution driver\n")
    driver_path.chmod(0o700)
    plan_path = (sealed / "plan.json").resolve()
    if plan_cases is None:
        plan_cases = [
            {
                "ordinal": 1,
                "protocol_seed": args.candidate_seed,
                "job_name": args.job_name,
            }
        ]
    plan_path.write_text(json.dumps({"cases": plan_cases}, sort_keys=True) + "\n")
    plan_path.chmod(0o600)
    for private_root in (args.jobs_dir, args.expert_evidence_dir):
        private_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        private_root.chmod(0o700)
    bindings = runner._expert_execution_bindings(args, task_dir)
    approval = {
        "schema_version": 1,
        "approval_type": runner.EXPERT_EXECUTION_APPROVAL_TYPE,
        "approval_scope": runner.EXPERT_EXECUTION_APPROVAL_SCOPE,
        "decision": "approved",
        "approved": True,
        "reviewer": "human-reviewer@example.test",
        "reviewed_at": "2026-08-10T12:00:00+08:00",
        "notes": "Approved this sealed expert feasibility batch only.",
        "custody_acknowledgement": runner.EXPERT_CUSTODY_ACKNOWLEDGEMENT,
        "plan_path": str(plan_path),
        "plan_file_sha256": runner._file_sha256(plan_path),
        "execution_driver_path": str(driver_path),
        "execution_driver_sha256": runner._file_sha256(driver_path),
        "execution_bindings": bindings,
        "cases": [
            {
                **case,
                "expert_evidence_output": str(
                    Path(bindings["expert_evidence_root"])
                    / bindings["candidate_slug"]
                    / f"seed-{case['protocol_seed']}"
                ),
                "receipt_path": str(
                    Path(bindings["jobs_root"])
                    / runner.EXPERT_EXECUTION_RECEIPT_DIR
                    / f"{case['job_name']}.json"
                ),
            }
            for case in plan_cases
        ],
    }
    approval_path = (sealed / "human-expert-execution-approval.json").resolve()
    approval_path.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n")
    approval_path.chmod(0o600)
    reservation_path = sealed / runner.EXPERT_EXECUTION_START_RESERVATION
    if create_reservation:
        reservation = {
            "schema_version": 1,
            "status": "execution_started_terminal_schedule",
            "plan_path": str(plan_path),
            "plan_file_sha256": approval["plan_file_sha256"],
            "execution_driver_path": str(driver_path),
            "execution_driver_sha256": approval["execution_driver_sha256"],
            "human_approval_path": str(approval_path),
            "human_approval_sha256": runner._file_sha256(approval_path),
            "candidate_id": bindings["candidate_id"],
            "candidate_hash": bindings["candidate_hash"],
            "seed_count": len(plan_cases),
        }
        reservation_path.write_text(
            json.dumps(reservation, indent=2, sort_keys=True) + "\n"
        )
        reservation_path.chmod(0o600)
    args.expert_execution_approval = approval_path
    return approval_path, reservation_path


def write_successful_prior_expert_case(
    validation: dict,
    case: dict,
    *,
    case_digest: str | None = None,
) -> Path:
    bindings = validation["bindings"]
    jobs_root = Path(bindings["jobs_root"])
    (jobs_root / case["job_name"]).mkdir(parents=True)
    receipt_path = Path(case["receipt_path"])
    receipt_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    receipt_path.parent.chmod(0o700)
    receipt = {
        **runner._expected_expert_receipt(validation, case),
        "claimed_at": "2026-08-10T04:01:00+00:00",
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    receipt_path.chmod(0o600)

    evidence_path = Path(case["expert_evidence_output"])
    evidence_path.mkdir(parents=True)
    roles = {
        "raw_job",
        "harbor_config",
        "harbor_lock",
        "harbor_job_lock",
        "functional_output",
        "admission_record",
        "audit_details",
    }
    digest = (
        case_digest or hashlib.sha256(f"case-{case['ordinal']}".encode()).hexdigest()
    )
    raw_evidence = {}
    for role in roles:
        raw_path = (evidence_path / f"{role}.txt").resolve()
        if role == "raw_job":
            raw_path.write_text(
                json.dumps(
                    {
                        "id": f"trial-{case['ordinal']}",
                        "verifier_result": {
                            "rewards": {
                                "reward": 1.0,
                                "functional_score": 1.0,
                                "static_policy_score": 1.0,
                                "llm_audit_score": 1.0,
                                "llm_audit_skipped_score": 1.0,
                                "runtime_sec": 1.0,
                            }
                        },
                    }
                )
                + "\n"
            )
        elif role == "audit_details":
            raw_path.write_text(
                json.dumps(
                    {
                        "audit": {
                            "llm_audit_score": 1.0,
                            "llm_audit_skipped": True,
                        }
                    }
                )
                + "\n"
            )
        else:
            raw_path.write_text(f"trusted {role}\n")
        raw_evidence[role] = {
            "path": str(raw_path),
            "sha256": runner._file_sha256(raw_path),
        }
    run_evidence_sha256 = hashlib.sha256(
        json.dumps(
            {
                "job_id": f"trial-{case['ordinal']}",
                "artifact_sha256s": {
                    name: raw_evidence[name]["sha256"] for name in sorted(roles)
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    case_record = {
        "job_id": f"trial-{case['ordinal']}",
        "expert_passed": True,
        "runtime_sec": 1.0,
        "protocol_seed": case["protocol_seed"],
        "case_digest": digest,
        "admission": {"status": "admitted"},
        "threshold_policy_sha256": "a" * 64,
        "run_evidence_sha256": run_evidence_sha256,
        "raw_evidence": raw_evidence,
    }
    (evidence_path / "expert-prequalified-case.json").write_text(
        json.dumps(case_record, indent=2, sort_keys=True) + "\n"
    )
    (evidence_path / "expert-run-bindings.json").write_text(
        json.dumps(
            bindings,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return evidence_path


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
    monkeypatch.setattr(runner, "verify_harbor_version", lambda *args, **kwargs: None)


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


def test_mixed_qfim_materializes_with_exact_identity_and_admission_contract(
    tmp_path: Path,
) -> None:
    discovery = ROOT / "reports" / "orbit_q_problem_discovery"
    blueprint = discovery / "blueprints" / "mixed_sld_qfim"
    task_dir = materializer.materialize_candidate_task(
        blueprint, tmp_path / "qfim-staging", discovery
    )

    assert task_dir.name == "candidate-mixed-sld-qfim"
    metadata = json.loads((task_dir / "candidate_metadata.json").read_text())
    assert metadata["problem_id"] == 101
    assert metadata["discovery_binding"]["contract"] == "reviewed_shortlist_contract"
    assert metadata["seed_protocol"]["verifier_env"] == runner.CANDIDATE_SEED_ENV
    assert metadata["contract"]["effective_line_limit"] == 160

    protocol = json.loads(
        (task_dir / "tests" / "expert_admission_protocol.json").read_text()
    )
    assert protocol == {
        "schema_version": 1,
        "status": "candidate_specific_metrics_declared",
        "structured_output_key": "orbit_q_expert_admission_metrics",
    }
    assert (task_dir / "tests" / "evaluate_101.py").read_bytes() == (
        blueprint / "evaluate_101.py"
    ).read_bytes()
    contract = materializer.validate_materialized_task_contract(
        task_dir, discovery, require_expert_admission=True
    )
    assert all(
        len(contract[name]) == 64
        for name in (
            "verifier_harness_sha256",
            "instruction_sha256",
            "candidate_metadata_sha256",
            "task_policy_sha256",
            "solution_driver_sha256",
        )
    )

    task_config = tomllib.loads((task_dir / "task.toml").read_text())
    assert task_config["verifier"]["env"] == {
        "RUNTIME_FULL_SCORE_SEC": "180",
        "RUNTIME_ZERO_SCORE_SEC": "300",
        "MAX_EFFECTIVE_CODE_LINES": "160",
    }
    assert runner.CANDIDATE_SEED_ENV not in (task_dir / "task.toml").read_text()


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


def test_runner_expert_only_omits_solver_and_disables_provider_audit(
    tmp_path: Path,
) -> None:
    task_dir = materialize_expert(tmp_path)
    args = runner.parse_args(expert_runner_args(task_dir, tmp_path))
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
    assert "CODEX_AUDIT_ENABLED=0" in command
    assert "ORBIT_Q_AUDIT_POLICY=disabled_private_prequalification" in command
    assert "ORBIT_Q_PROVIDER_CALL_BUDGET=0" in command
    assert not any(item.startswith("audit_model=") for item in command)
    assert "force_auth_json=true" not in command
    assert not any("PROXY=" in item for item in command)


def test_expert_subprocess_env_is_an_exact_provider_free_allowlist() -> None:
    source = {
        "PATH": "/trusted/bin",
        "HOME": "/trusted/home",
        "LANG": "C.UTF-8",
        "PYTHONPATH": "/untrusted/pythonpath",
        "PYTHONDONTWRITEBYTECODE": "0",
        "AWS_ACCESS_KEY_ID": "aws-access",
        "AWS_SECRET_ACCESS_KEY": "aws-secret",
        "AWS_SESSION_TOKEN": "aws-session",
        "AWS_BEDROCK_RUNTIME_ENDPOINT": "https://bedrock.example.test",
        "BEDROCK_API_KEY": "bedrock-secret",
        "OPENAI_API_KEY": "openai-secret",
        "ANTHROPIC_API_KEY": "anthropic-secret",
        "ANTHROPIC_AUTH_TOKEN": "anthropic-token",
        "GOOGLE_APPLICATION_CREDENTIALS": "/private/google.json",
        "AZURE_CLIENT_SECRET": "azure-secret",
        "HF_TOKEN": "huggingface-secret",
        "HTTP_PROXY": "http://user:secret@proxy.example.test:8080",
        "HTTPS_PROXY": "http://user:secret@proxy.example.test:8080",
        "UNRELATED_HOST_VALUE": "also-not-allowlisted",
    }

    expert_env = runner.build_expert_subprocess_env(source)

    assert expert_env == {
        "PATH": "/trusted/bin",
        "HOME": "/trusted/home",
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(ROOT),
        "PYTHONDONTWRITEBYTECODE": "1",
    }


@pytest.mark.parametrize(
    "flag",
    (
        "--model",
        "--audit-model",
        "--solver-agent",
        "--reasoning-effort",
        "--force-auth-json",
        "--bridge-loopback-proxy",
    ),
)
def test_runner_expert_only_rejects_model_provider_options(
    tmp_path: Path, flag: str
) -> None:
    task_dir = materialize_expert(tmp_path)
    value = [] if flag in {"--force-auth-json", "--bridge-loopback-proxy"} else ["x"]
    if flag == "--solver-agent":
        value = ["codex"]
    args = runner.parse_args(expert_runner_args(task_dir, tmp_path, flag, *value))
    with pytest.raises(materializer.CandidateTaskError, match="disables all"):
        runner.build_harbor_command(args)


def test_expert_verifier_never_constructs_or_sets_up_codex(
    tmp_path: Path, monkeypatch
) -> None:
    observed = {}

    def fake_verifier_init(self, *args, **kwargs):
        self.trial_paths = SimpleNamespace(verifier_dir=tmp_path)
        self.logger = SimpleNamespace()
        self.override_env = {}

    def forbidden_codex(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("expert-only verifier must not construct CodexPara")

    async def fake_base_verify(self):
        observed.update(self.override_env)
        return "verified-without-provider"

    monkeypatch.setattr(verifier_adapter.Verifier, "__init__", fake_verifier_init)
    monkeypatch.setattr(verifier_adapter, "CodexPara", forbidden_codex)
    monkeypatch.setattr(verifier_adapter.Verifier, "verify", fake_base_verify)
    verifier = verifier_adapter.CodexParaVerifier(expert_only=True)

    result = asyncio.run(verifier.verify())

    assert result == "verified-without-provider"
    assert verifier._codex is None
    assert observed["CODEX_AUDIT_ENABLED"] == "0"
    assert observed["ORBIT_Q_AUDIT_POLICY"] == "disabled_private_prequalification"
    assert observed["ORBIT_Q_PROVIDER_CALL_BUDGET"] == "0"


def test_expert_execute_requires_human_approval_before_subprocess(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize_expert(tmp_path)

    def forbidden_run(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("unapproved expert execute must not start Harbor")

    monkeypatch.setattr(runner.subprocess, "run", forbidden_run)
    result = runner.main(expert_runner_args(task_dir, tmp_path, "--execute"))

    assert result == 2


def test_expert_execute_requires_driver_start_reservation(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize_expert(tmp_path)
    raw_args = expert_runner_args(
        task_dir,
        tmp_path,
        "--expert-evidence-dir",
        str(tmp_path / "expert-evidence"),
        "--job-name",
        "expert-101-seed-101021",
    )
    args = runner.parse_args(raw_args)
    runner.build_harbor_command(args)
    approval_path, reservation_path = make_expert_execution_approval(
        args, task_dir, tmp_path, create_reservation=False
    )
    assert not reservation_path.exists()

    def forbidden_run(*args, **kwargs):  # pragma: no cover - failure path
        raise AssertionError("driver bypass must not start Harbor")

    monkeypatch.setattr(runner.subprocess, "run", forbidden_run)
    result = runner.main(
        raw_args + ["--expert-execution-approval", str(approval_path), "--execute"]
    )

    assert result == 2


def test_expert_execute_rejects_insecure_reservation_and_symlinked_approval(
    tmp_path: Path,
) -> None:
    task_dir = materialize_expert(tmp_path)
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    approval_path, reservation_path = make_expert_execution_approval(
        args, task_dir, tmp_path
    )
    reservation_path.chmod(0o644)
    with pytest.raises(materializer.CandidateTaskError, match="mode 0600"):
        runner.validate_expert_execution_approval(args, task_dir)

    reservation_path.chmod(0o600)
    approval_link = tmp_path / "approval-link.json"
    approval_link.symlink_to(approval_path)
    args.expert_execution_approval = approval_link
    with pytest.raises(materializer.CandidateTaskError, match="nonsymlinked"):
        runner.validate_expert_execution_approval(args, task_dir)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("mode", "mode 0600"),
        ("binding", "execution_bindings"),
        ("plan", "plan.*SHA-256"),
    ),
)
def test_expert_approval_rejects_insecure_mode_or_binding_drift(
    tmp_path: Path, mutation: str, message: str
) -> None:
    task_dir = materialize_expert(tmp_path)
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    approval_path, _ = make_expert_execution_approval(args, task_dir, tmp_path)
    if mutation == "mode":
        approval_path.chmod(0o644)
    elif mutation == "binding":
        approval = json.loads(approval_path.read_text())
        approval["execution_bindings"]["cpus"] = True
        approval_path.write_text(json.dumps(approval, indent=2, sort_keys=True) + "\n")
        approval_path.chmod(0o600)
    else:
        plan_path = Path(json.loads(approval_path.read_text())["plan_path"])
        plan_path.chmod(0o600)
        plan_path.write_text(json.dumps({"cases": []}) + "\n")
        plan_path.chmod(0o600)

    with pytest.raises(materializer.CandidateTaskError, match=message):
        runner.validate_expert_execution_approval(args, task_dir)


@pytest.mark.parametrize(
    ("field", "mode"), (("plan_path", 0o644), ("execution_driver_path", 0o755))
)
def test_expert_approval_rejects_shared_read_access_to_private_plan_or_driver(
    tmp_path: Path, field: str, mode: int
) -> None:
    task_dir = materialize_expert(tmp_path)
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    approval_path, _ = make_expert_execution_approval(args, task_dir, tmp_path)
    protected_path = Path(json.loads(approval_path.read_text())[field])
    protected_path.chmod(mode)

    with pytest.raises(materializer.CandidateTaskError, match="owner-only mode"):
        runner.validate_expert_execution_approval(args, task_dir)


def test_expert_approval_binds_resolved_harbor_binary_bytes(tmp_path: Path) -> None:
    task_dir = materialize_expert(tmp_path)
    fake_harbor = tmp_path / "fake-harbor"
    fake_harbor.write_text("#!/bin/sh\nexit 0\n")
    fake_harbor.chmod(0o700)
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--harbor-bin",
            str(fake_harbor),
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    make_expert_execution_approval(args, task_dir, tmp_path)
    fake_harbor.write_text("#!/bin/sh\nexit 17\n")
    fake_harbor.chmod(0o700)

    with pytest.raises(materializer.CandidateTaskError, match="harbor_binary_sha256"):
        runner.validate_expert_execution_approval(args, task_dir)


def test_expert_bindings_cover_runner_python_launcher_target_and_venv_config(
    tmp_path: Path,
) -> None:
    task_dir = materialize_expert(tmp_path)
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)

    bindings = runner._expert_execution_bindings(args, task_dir)
    invocation = Path(runner.sys.executable).absolute()
    resolved = invocation.resolve(strict=True)
    venv_config = Path(runner.sys.prefix).absolute() / "pyvenv.cfg"

    assert bindings["runner_python_path"] == str(invocation)
    assert bindings["runner_python_resolved_path"] == str(resolved)
    assert bindings["runner_python_sha256"] == runner._file_sha256(resolved)
    assert bindings["runner_python_prefix_path"] == str(
        Path(runner.sys.prefix).absolute()
    )
    assert bindings["runner_python_venv_config_path"] == str(venv_config)
    assert bindings["runner_python_venv_config_sha256"] == runner._file_sha256(
        venv_config
    )


def test_expert_approval_rejects_runner_python_invocation_path_drift(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize_expert(tmp_path)
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    make_expert_execution_approval(args, task_dir, tmp_path)
    alternate_launcher = tmp_path / "alternate-python"
    alternate_launcher.symlink_to(Path(runner.sys.executable).resolve(strict=True))
    monkeypatch.setattr(runner.sys, "executable", str(alternate_launcher))

    with pytest.raises(materializer.CandidateTaskError, match="execution_bindings"):
        runner.validate_expert_execution_approval(args, task_dir)

    receipt = (
        args.jobs_dir / runner.EXPERT_EXECUTION_RECEIPT_DIR / f"{args.job_name}.json"
    )
    assert not receipt.exists()


def test_expert_approval_rejects_missing_runner_python_before_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize_expert(tmp_path)
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    make_expert_execution_approval(args, task_dir, tmp_path)
    monkeypatch.setattr(runner.sys, "executable", str(tmp_path / "missing-python"))

    with pytest.raises(
        materializer.CandidateTaskError, match="runner Python executable is missing"
    ):
        runner.validate_expert_execution_approval(args, task_dir)

    receipt = (
        args.jobs_dir / runner.EXPERT_EXECUTION_RECEIPT_DIR / f"{args.job_name}.json"
    )
    assert not receipt.exists()


def test_expert_approval_rejects_runner_python_symlink_retarget(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize_expert(tmp_path)
    launcher = tmp_path / "approved-python"
    launcher.symlink_to(Path(runner.sys.executable).resolve(strict=True))
    monkeypatch.setattr(runner.sys, "executable", str(launcher))
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    make_expert_execution_approval(args, task_dir, tmp_path)
    replacement = tmp_path / "replacement-python"
    replacement.write_bytes(b"not the approved interpreter\n")
    replacement.chmod(0o700)
    launcher.unlink()
    launcher.symlink_to(replacement)

    with pytest.raises(materializer.CandidateTaskError, match="execution_bindings"):
        runner.validate_expert_execution_approval(args, task_dir)

    receipt = (
        args.jobs_dir / runner.EXPERT_EXECUTION_RECEIPT_DIR / f"{args.job_name}.json"
    )
    assert not receipt.exists()


def test_expert_approval_rejects_shared_or_symlinked_private_output_roots(
    tmp_path: Path,
) -> None:
    task_dir = materialize_expert(tmp_path)
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    make_expert_execution_approval(args, task_dir, tmp_path)
    args.jobs_dir.chmod(0o755)
    with pytest.raises(materializer.CandidateTaskError, match="mode-0700"):
        runner.validate_expert_execution_approval(args, task_dir)

    args.jobs_dir.chmod(0o700)
    real_evidence = tmp_path / "real-evidence"
    real_evidence.mkdir(mode=0o700)
    args.expert_evidence_dir.chmod(0o700)
    args.expert_evidence_dir.rmdir()
    args.expert_evidence_dir.symlink_to(real_evidence, target_is_directory=True)
    with pytest.raises(materializer.CandidateTaskError, match="mode-0700"):
        runner.validate_expert_execution_approval(args, task_dir)


def test_expert_approval_claim_is_one_use_and_rejects_existing_outputs(
    tmp_path: Path,
) -> None:
    task_dir = materialize_expert(tmp_path)
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
            "--job-name",
            "expert-101-seed-101021",
        )
    )
    runner.build_harbor_command(args)
    make_expert_execution_approval(args, task_dir, tmp_path)
    validation = runner.validate_expert_execution_approval(args, task_dir)
    receipt_path = runner.claim_expert_execution_approval(validation)

    assert receipt_path.is_file()
    assert receipt_path.stat().st_mode & 0o777 == 0o600
    with pytest.raises(
        materializer.CandidateTaskError, match="current expert case has an existing"
    ):
        runner.claim_expert_execution_approval(validation)

    second_root = tmp_path / "existing-output"
    second_root.mkdir()
    task_dir_2 = materialize_expert(second_root)
    args_2 = runner.parse_args(
        expert_runner_args(
            task_dir_2,
            second_root,
            "--expert-evidence-dir",
            str(second_root / "expert-evidence"),
            "--job-name",
            "already-created",
        )
    )
    runner.build_harbor_command(args_2)
    make_expert_execution_approval(args_2, task_dir_2, second_root)
    validation_2 = runner.validate_expert_execution_approval(args_2, task_dir_2)
    (Path(validation_2["bindings"]["jobs_root"]) / args_2.job_name).mkdir(parents=True)
    with pytest.raises(
        materializer.CandidateTaskError, match="current expert case has an existing"
    ):
        runner.claim_expert_execution_approval(validation_2)


def test_approved_expert_execute_claims_before_single_subprocess_and_replay_fails(
    tmp_path: Path, monkeypatch
) -> None:
    task_dir = materialize_expert(tmp_path)
    raw_args = expert_runner_args(
        task_dir,
        tmp_path,
        "--expert-evidence-dir",
        str(tmp_path / "expert-evidence"),
        "--job-name",
        "expert-101-seed-101021",
    )
    setup_args = runner.parse_args(raw_args)
    runner.build_harbor_command(setup_args)
    approval_path, _ = make_expert_execution_approval(setup_args, task_dir, tmp_path)
    calls = []
    version_envs = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=17)

    def capture_version_env(harbor_bin, expected, *, env=None):
        version_envs.append(dict(env))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "verify_harbor_version", capture_version_env)
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-private-harbor")
    monkeypatch.setenv("OPENAI_BASE_URL", "must-not-reach-private-harbor")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "must-not-reach-private-harbor")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "must-not-reach-private-harbor")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "must-not-reach-private-harbor")
    monkeypatch.setenv("AWS_SESSION_TOKEN", "must-not-reach-private-harbor")
    monkeypatch.setenv("AWS_BEDROCK_RUNTIME_ENDPOINT", "must-not-reach-private-harbor")
    monkeypatch.setenv("BEDROCK_API_KEY", "must-not-reach-private-harbor")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/private/google.json")
    monkeypatch.setenv("AZURE_CLIENT_SECRET", "must-not-reach-private-harbor")
    monkeypatch.setenv("HF_TOKEN", "must-not-reach-private-harbor")
    monkeypatch.setenv("UNRELATED_HOST_VALUE", "not-allowlisted")
    expected_env = runner.build_expert_subprocess_env()
    execute_args = raw_args + [
        "--expert-execution-approval",
        str(approval_path),
        "--execute",
    ]
    assert runner.main(execute_args) == 17
    assert len(calls) == 1
    assert version_envs == [expected_env]
    assert calls[0][1]["umask"] == 0o077
    assert calls[0][1]["env"] == expected_env

    assert runner.main(execute_args) == 2
    assert len(calls) == 1
    assert version_envs == [expected_env, expected_env]


def test_expert_batch_rejects_later_case_first_and_failed_prior_claim(
    tmp_path: Path,
) -> None:
    task_dir = materialize_expert(tmp_path)
    cases = [
        {"ordinal": 1, "protocol_seed": 101021, "job_name": "expert-case-1"},
        {"ordinal": 2, "protocol_seed": 101022, "job_name": "expert-case-2"},
    ]
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--candidate-seed",
            "101022",
            "--job-name",
            "expert-case-2",
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    approval_path, _ = make_expert_execution_approval(
        args, task_dir, tmp_path, plan_cases=cases
    )
    validation = runner.validate_expert_execution_approval(args, task_dir)
    with pytest.raises(materializer.CandidateTaskError, match="terminally blocked"):
        runner.claim_expert_execution_approval(validation)

    first_args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--candidate-seed",
            "101021",
            "--job-name",
            "expert-case-1",
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
            "--expert-execution-approval",
            str(approval_path),
        )
    )
    runner.build_harbor_command(first_args)
    first_validation = runner.validate_expert_execution_approval(first_args, task_dir)
    runner.claim_expert_execution_approval(first_validation)
    with pytest.raises(materializer.CandidateTaskError, match="terminally blocked"):
        runner.claim_expert_execution_approval(validation)


def test_expert_batch_accepts_only_bound_successful_predecessor_and_no_later_artifact(
    tmp_path: Path,
) -> None:
    task_dir = materialize_expert(tmp_path)
    cases = [
        {"ordinal": 1, "protocol_seed": 101021, "job_name": "expert-case-1"},
        {"ordinal": 2, "protocol_seed": 101022, "job_name": "expert-case-2"},
    ]
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--candidate-seed",
            "101022",
            "--job-name",
            "expert-case-2",
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    make_expert_execution_approval(args, task_dir, tmp_path, plan_cases=cases)
    validation = runner.validate_expert_execution_approval(args, task_dir)
    write_successful_prior_expert_case(validation, validation["cases"][0])

    receipt_path = runner.claim_expert_execution_approval(validation)
    assert receipt_path.is_file()

    later_root = tmp_path / "later-artifact"
    later_root.mkdir()
    later_task = materialize_expert(later_root)
    first_args = runner.parse_args(
        expert_runner_args(
            later_task,
            later_root,
            "--candidate-seed",
            "101021",
            "--job-name",
            "expert-case-1",
            "--expert-evidence-dir",
            str(later_root / "expert-evidence"),
        )
    )
    runner.build_harbor_command(first_args)
    make_expert_execution_approval(first_args, later_task, later_root, plan_cases=cases)
    first_validation = runner.validate_expert_execution_approval(first_args, later_task)
    later_job = Path(first_validation["bindings"]["jobs_root"]) / cases[1]["job_name"]
    later_job.mkdir(parents=True)
    with pytest.raises(materializer.CandidateTaskError, match="later expert case"):
        runner.claim_expert_execution_approval(first_validation)


@pytest.mark.parametrize(
    "mutation", ("missing-role", "unexpected-role", "traversal", "symlink")
)
def test_expert_batch_rejects_prior_raw_custody_escape_or_role_drift(
    tmp_path: Path, mutation: str
) -> None:
    task_dir = materialize_expert(tmp_path)
    cases = [
        {"ordinal": 1, "protocol_seed": 101021, "job_name": "expert-case-1"},
        {"ordinal": 2, "protocol_seed": 101022, "job_name": "expert-case-2"},
    ]
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--candidate-seed",
            "101022",
            "--job-name",
            "expert-case-2",
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    make_expert_execution_approval(args, task_dir, tmp_path, plan_cases=cases)
    validation = runner.validate_expert_execution_approval(args, task_dir)
    evidence_path = write_successful_prior_expert_case(
        validation, validation["cases"][0]
    )
    case_path = evidence_path / "expert-prequalified-case.json"
    case_record = json.loads(case_path.read_text())
    raw = case_record["raw_evidence"]
    if mutation == "missing-role":
        raw.pop("audit_details")
    elif mutation == "unexpected-role":
        raw["operator_summary"] = dict(raw["functional_output"])
    elif mutation == "traversal":
        nested = evidence_path / "nested"
        nested.mkdir()
        raw_path = Path(raw["raw_job"]["path"])
        raw["raw_job"]["path"] = str(nested / ".." / raw_path.name)
    else:
        raw_path = Path(raw["raw_job"]["path"])
        target = evidence_path / "raw-job-target.json"
        target.write_bytes(raw_path.read_bytes())
        raw_path.unlink()
        raw_path.symlink_to(target.name)
        raw["raw_job"]["sha256"] = runner._file_sha256(target)
    case_path.write_text(json.dumps(case_record, indent=2, sort_keys=True) + "\n")

    with pytest.raises(materializer.CandidateTaskError):
        runner.claim_expert_execution_approval(validation)


def test_expert_batch_rejects_duplicate_prior_case_digests(tmp_path: Path) -> None:
    task_dir = materialize_expert(tmp_path)
    cases = [
        {"ordinal": 1, "protocol_seed": 101021, "job_name": "expert-case-1"},
        {"ordinal": 2, "protocol_seed": 101022, "job_name": "expert-case-2"},
        {"ordinal": 3, "protocol_seed": 101023, "job_name": "expert-case-3"},
    ]
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--candidate-seed",
            "101023",
            "--job-name",
            "expert-case-3",
            "--expert-evidence-dir",
            str(tmp_path / "expert-evidence"),
        )
    )
    runner.build_harbor_command(args)
    make_expert_execution_approval(args, task_dir, tmp_path, plan_cases=cases)
    validation = runner.validate_expert_execution_approval(args, task_dir)
    duplicate_digest = "d" * 64
    write_successful_prior_expert_case(
        validation, validation["cases"][0], case_digest=duplicate_digest
    )
    write_successful_prior_expert_case(
        validation, validation["cases"][1], case_digest=duplicate_digest
    )

    with pytest.raises(materializer.CandidateTaskError, match="duplicate case digest"):
        runner.claim_expert_execution_approval(validation)


@pytest.mark.parametrize("expert_only", (True, False))
def test_runner_rejects_hold_lifecycle_before_expert_or_model_command(
    tmp_path: Path, expert_only: bool
) -> None:
    task_dir = materialize_reviewed(tmp_path, "robust-leakage-grape")
    raw_args = (
        expert_runner_args(task_dir, tmp_path)
        if expert_only
        else runner_args(task_dir, tmp_path)
    )
    args = runner.parse_args(raw_args)

    with pytest.raises(
        materializer.CandidateTaskError, match="lifecycle is HOLD or rejected"
    ):
        runner.build_harbor_command(args)


@pytest.mark.parametrize(
    ("status", "blocked"),
    (
        ("reserve_hold_tc_runtime_unverified", True),
        ("prototype_rejected", True),
        ("all_thresholds_passed", False),
        ("direct_metrics_pass", False),
        (None, False),
    ),
)
def test_runner_lifecycle_guard_uses_tokens_not_substrings(
    status: object, blocked: bool
) -> None:
    assert runner._lifecycle_is_blocked(status) is blocked


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
    evidence_root.mkdir(mode=0o700)
    jobs_root = tmp_path / "candidate-jobs"
    jobs_root.mkdir(mode=0o700)
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(evidence_root),
            "--job-name",
            "expert-101-seed-101021",
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
        "kwargs": {"expert_only": True},
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
        "llm_audit_skipped_score": 1.0,
        "runtime_sec": 12.5,
        **{
            f"expert_admission_sha256_word_{index}": int(
                admission_hash[offset : offset + 8], 16
            )
            for index, offset in enumerate(range(0, 64, 8))
        },
    }
    result = {
        "id": "trial-101021",
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

    job_dir = jobs_root / args.job_name
    trial_dir = job_dir / "candidate-mixed-sld-qfim__fixture"
    verifier_dir = trial_dir / "verifier"
    artifact_dir = trial_dir / "artifacts" / "root"
    verifier_dir.mkdir(parents=True)
    artifact_dir.mkdir(parents=True)
    (job_dir / "lock.json").write_text(json.dumps(job_lock))
    (trial_dir / "lock.json").write_text(json.dumps(trial_lock))
    (trial_dir / "config.json").write_text(json.dumps(sidecar_config))
    (trial_dir / "result.json").write_text(json.dumps(result))
    (verifier_dir / "expert-admission.json").write_text(serialized_admission)
    (verifier_dir / "audit-details.json").write_text(
        json.dumps(
            {
                "audit": {
                    "llm_audit_score": 1.0,
                    "llm_audit_skipped": True,
                }
            }
        )
    )
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
    (artifact_dir / "solution_101.py").write_bytes(
        (task_dir / "solution" / "solution_101.py").read_bytes()
    )

    case_path = runner.collect_expert_prequalification_evidence(args, task_dir)
    case = json.loads(case_path.read_text())
    assert case["protocol_seed"] == 101021
    assert case["case_digest"] == case_digest
    assert case["expert_passed"] is True
    assert set(case["raw_evidence"]) == {
        "raw_job",
        "harbor_config",
        "harbor_lock",
        "harbor_job_lock",
        "functional_output",
        "admission_record",
        "audit_details",
    }
    for item in case["raw_evidence"].values():
        assert (
            hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
            == item["sha256"]
        )


def _prepare_stub_expert_collector_layout(tmp_path: Path, monkeypatch):
    task_dir = materialize_expert(tmp_path)
    evidence_root = tmp_path / "expert-evidence"
    evidence_root.mkdir(mode=0o700)
    jobs_root = tmp_path / "candidate-jobs"
    jobs_root.mkdir(mode=0o700)
    args = runner.parse_args(
        expert_runner_args(
            task_dir,
            tmp_path,
            "--expert-evidence-dir",
            str(evidence_root),
            "--job-name",
            "expert-101-seed-101021",
        )
    )
    runner.build_harbor_command(args)
    job_dir = jobs_root / args.job_name
    trial_dir = job_dir / "candidate-mixed-sld-qfim__fixture"
    verifier_dir = trial_dir / "verifier"
    verifier_dir.mkdir(parents=True)
    (trial_dir / "result.json").write_text('{"result": "trusted"}\n')
    (trial_dir / "config.json").write_text('{"config": "trusted"}\n')
    (trial_dir / "lock.json").write_text('{"trial_lock": "trusted"}\n')
    (job_dir / "lock.json").write_text('{"job_lock": "trusted"}\n')
    (verifier_dir / "functional-stdout.txt").write_text("Overall: PASS\n")
    (verifier_dir / runner.EXPERT_ADMISSION_FILENAME).write_text(
        '{"admission": "trusted"}\n'
    )
    (verifier_dir / "audit-details.json").write_text(
        '{"audit": {"llm_audit_score": 1.0, "llm_audit_skipped": true}}\n'
    )
    bindings = runner._expert_execution_bindings(args, task_dir)
    derived = {
        "result": {"id": "trial-101021"},
        "runtime_sec": 1.0,
        "protocol_seed": args._resolved_run["protocol_seed"],
        "case_digest": hashlib.sha256(b"stub-case").hexdigest(),
        "admission_record": {
            "admission": {"status": "admitted"},
            "threshold_policy_sha256": hashlib.sha256(b"policy").hexdigest(),
        },
        "harbor_task_digest": bindings["harbor_task_digest"],
        "task_checksum": bindings["task_checksum"],
    }
    monkeypatch.setattr(
        runner,
        "_validate_expert_harbor_artifacts",
        lambda **kwargs: (trial_dir, derived),
    )
    return args, task_dir, evidence_root


def test_expert_collector_rejects_precreated_candidate_parent_symlink(
    tmp_path: Path, monkeypatch
) -> None:
    args, task_dir, evidence_root = _prepare_stub_expert_collector_layout(
        tmp_path, monkeypatch
    )
    outside = tmp_path / "outside-evidence"
    outside.mkdir(mode=0o700)
    candidate_parent = evidence_root / args._resolved_run["candidate_slug"]
    candidate_parent.symlink_to(outside, target_is_directory=True)

    with pytest.raises(
        materializer.CandidateTaskError, match="not a regular private directory"
    ):
        runner.collect_expert_prequalification_evidence(args, task_dir)

    assert list(outside.iterdir()) == []


def test_expert_collector_parent_swap_after_recheck_cannot_escape(
    tmp_path: Path, monkeypatch
) -> None:
    args, task_dir, evidence_root = _prepare_stub_expert_collector_layout(
        tmp_path, monkeypatch
    )
    outside = tmp_path / "outside-evidence"
    outside.mkdir(mode=0o700)
    candidate_parent = evidence_root / args._resolved_run["candidate_slug"]
    parked_parent = evidence_root / "parked-candidate-parent"
    real_rename = runner.os.rename
    raced = False

    def race_parent_before_publish(
        source, destination, *, src_dir_fd=None, dst_dir_fd=None
    ):
        nonlocal raced
        if not raced:
            raced = True
            real_rename(candidate_parent, parked_parent)
            candidate_parent.symlink_to(outside, target_is_directory=True)
        return real_rename(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
        )

    monkeypatch.setattr(runner.os, "rename", race_parent_before_publish)

    with pytest.raises(
        materializer.CandidateTaskError, match="candidate evidence parent"
    ):
        runner.collect_expert_prequalification_evidence(args, task_dir)

    assert raced is True
    assert list(outside.iterdir()) == []
    assert list(parked_parent.iterdir()) == []


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
