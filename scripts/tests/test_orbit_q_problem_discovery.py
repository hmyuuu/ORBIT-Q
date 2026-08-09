"""Regression tests for the future-problem discovery and review gates."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from reports.orbit_q_problem_discovery.pipeline import (
    CONCEPT_REVIEW_ROLES,
    FAILURE_AUDIT_ROLES,
    PILOT_REVIEW_ROLES,
    audit_model_trial,
    authorize_model_test,
    build_workspace,
    generate_candidates,
    model_hardness_status,
    record_model_trial,
    record_prototype,
    record_review,
    select_shortlist,
    validate_catalog,
)


ROOT = Path(__file__).resolve().parents[2]
DISCOVERY = ROOT / "reports" / "orbit_q_problem_discovery"


def _catalog() -> dict:
    return json.loads((DISCOVERY / "catalog.json").read_text(encoding="utf-8"))


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "discovery"
    workspace.mkdir()
    shutil.copy2(DISCOVERY / "catalog.json", workspace / "catalog.json")
    build_workspace(workspace)
    return workspace


def _artifact_root(workspace: Path) -> Path:
    root = workspace.parent / ".artifacts" / "problem-discovery"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write_artifact(path: Path, content: str) -> dict[str, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }


def _prototype_bundle(workspace: Path, candidate_id: str) -> Path:
    state = json.loads((workspace / "review_state.json").read_text())
    candidate_hash = state["candidates"][candidate_id]["candidate_hash"]
    evidence_root = _artifact_root(workspace) / "prototype"
    expert_source = "".join(f"value_{index} = {index}\n" for index in range(140))
    artifacts = {
        "expert_baseline": {
            **_write_artifact(evidence_root / "expert.py", expert_source),
            "author": "expert-author",
        },
        "independent_oracle": {
            **_write_artifact(evidence_root / "oracle.py", "ORACLE = True\n"),
            "author": "oracle-author",
        },
        "evaluator": {
            **_write_artifact(evidence_root / "evaluator.py", "EVALUATOR = True\n"),
            "author": "evaluator-author",
        },
        "task_bundle": {
            **_write_artifact(evidence_root / "task.bundle", "frozen task bundle\n"),
            "author": "task-author",
        },
        "framework_prompt": {
            **_write_artifact(evidence_root / "prompt.md", "TensorCircuit prompt\n"),
            "author": "prompt-author",
        },
    }
    image_digest = "sha256:" + "e" * 64
    source_commit = "0123456789abcdef"
    canary = _write_artifact(
        evidence_root / "canary.json",
        json.dumps(
            {
                "passed": True,
                "container_image_digest": image_digest,
                "source_commit": source_commit,
            },
            indent=2,
        )
        + "\n",
    )
    verifier_runs = []
    for index in range(20):
        runtime = 40.0 + index / 10
        peak_memory = 2048.0 + index
        log = _write_artifact(
            evidence_root / f"verifier-{index}.json",
            json.dumps(
                {
                    "id": f"verifier-{index}",
                    "finished_at": "2026-08-09T00:00:00Z",
                    "verifier_result": {
                        "rewards": {
                            "reward": 1.0,
                            "functional_score": 1.0,
                            "static_policy_score": 1.0,
                            "llm_audit_score": 1.0,
                            "runtime_sec": runtime,
                        }
                    },
                    "orbit_q_attestation": {
                        "attestation_type": "operator_attested",
                        "attestor_id": "prototype-runner",
                        "container_image_digest": image_digest,
                        "peak_memory_mb": peak_memory,
                    },
                },
                indent=2,
            )
            + "\n",
        )
        verifier_runs.append(
            {
                "job_id": f"verifier-{index}",
                "passed": True,
                "runtime_sec": runtime,
                "peak_memory_mb": peak_memory,
                "container_image_digest": image_digest,
                "log": log,
            }
        )
    oracle_runs = [
        {
            "runtime_sec": 20.0 + index,
            "log": _write_artifact(
                evidence_root / f"oracle-{index}.json",
                json.dumps(
                    {
                        "passed": True,
                        "runtime_sec": 20.0 + index,
                        "oracle_method": "independent-density-or-covariance-oracle",
                    },
                    indent=2,
                )
                + "\n",
            ),
        }
        for index in range(3)
    ]
    alternatives = [
        {
            "accepted": True,
            "log": _write_artifact(
                evidence_root / f"alternative-{index}.json",
                json.dumps(
                    {
                        "accepted": True,
                        "implementation_style": f"valid-style-{index}",
                    },
                    indent=2,
                )
                + "\n",
            ),
        }
        for index in range(2)
    ]
    mutations = [
        {
            "name": f"mutation-{index}",
            "rejected": True,
            "log": _write_artifact(
                evidence_root / f"mutation-{index}.json",
                json.dumps(
                    {
                        "rejected": True,
                        "mutation_type": f"mutation-class-{index}",
                    },
                    indent=2,
                )
                + "\n",
            ),
        }
        for index in range(6)
    ]
    bundle = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "artifacts": artifacts,
        "environment": {
            "container_image_digest": image_digest,
            "source_commit": source_commit,
            "observed_cpu_count": 8,
            "observed_memory_mb": 8192,
        },
        "public_api_canary": {"passed": True, "log": canary},
        "verifier_runs": verifier_runs,
        "independent_oracle_runs": oracle_runs,
        "gold_effective_lines": 140,
        "valid_alternatives": alternatives,
        "mutation_tests": mutations,
    }
    bundle_path = evidence_root / "evidence.json"
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    return bundle_path


def _protocol_config(workspace: Path) -> Path:
    path = _artifact_root(workspace) / "protocol.json"
    config = {
        "model": "gpt-5.6-sol",
        "provider_snapshot": "provider-build-2026-08-09",
        "reasoning_effort": "max",
        "token_budget": 200000,
        "wall_time_sec": 1800,
        "solver_agent_version": "codex-cli-frozen",
        "harbor_version": "harbor-frozen",
        "audit_model": "gpt-5",
        "tools_policy": "orbit-q-default",
        "network_policy": "disabled",
        "hardware_class": "8cpu-8gib-test",
    }
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def _approved_workspace(tmp_path: Path) -> tuple[Path, str, Path]:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][
        0
    ]["id"]
    for role in CONCEPT_REVIEW_ROLES:
        record_review(
            workspace,
            candidate_id,
            "concept",
            role,
            "approve",
            f"reviewer-{role}",
            "reviewed evidence",
        )
    record_prototype(
        workspace, candidate_id, _prototype_bundle(workspace, candidate_id)
    )
    for role in PILOT_REVIEW_ROLES:
        record_review(
            workspace,
            candidate_id,
            "pilot",
            role,
            "approve",
            f"reviewer-{role}",
            "reviewed frozen pilot",
        )
    return workspace, candidate_id, _protocol_config(workspace)


def _record_failure_trials(
    workspace: Path, candidate_id: str, manifest_path: Path, seeds: list[int]
) -> None:
    for index, seed in enumerate(seeds):
        record_model_trial(
            workspace,
            candidate_id,
            manifest_path,
            f"trial-{index}",
            _trial_result(workspace, manifest_path, index, seed, passed=False),
        )
        for audit_index, role in enumerate(FAILURE_AUDIT_ROLES):
            audit_model_trial(
                workspace,
                candidate_id,
                f"trial-{index}",
                role,
                "algorithm_design",
                f"failure-auditor-{audit_index}",
                "trace inspected",
            )


def _trial_result(
    workspace: Path,
    manifest_path: Path,
    index: int,
    seed: int,
    *,
    passed: bool,
) -> Path:
    manifest = json.loads(manifest_path.read_text())
    result_root = _artifact_root(workspace) / "trials"
    score = 1.0 if passed else 0.0
    raw_result = {
        "id": f"job-{index}",
        "finished_at": "2026-08-09T00:00:00Z",
        "config": {
            "agent": {
                "model_name": manifest["model"],
                "kwargs": {
                    "reasoning_effort": manifest["protocol_config"]["reasoning_effort"]
                },
            },
            "verifier": {
                "kwargs": {"audit_model": manifest["protocol_config"]["audit_model"]}
            },
        },
        "verifier_result": {
            "rewards": {
                "reward": score,
                "functional_score": score,
                "static_policy_score": 1.0,
                "llm_audit_score": 1.0,
                "runtime_sec": 30.0 if passed else 60.0,
            }
        },
        "exception_info": None,
        "orbit_q_attestation": {
            "attestation_type": "operator_attested",
            "attestor_id": "test-runner",
            "attested_at": "2026-08-09T00:00:00Z",
            "manifest_hash": manifest["manifest_hash"],
            "candidate_hash": manifest["candidate_hash"],
            "protocol_config_sha256": manifest["protocol_config_sha256"],
            "container_image_digest": manifest["prototype"]["container_image_digest"],
            "framework_prompt_sha256": manifest["prototype"]["framework_prompt_sha256"],
            "task_bundle_sha256": manifest["prototype"]["task_bundle_sha256"],
            "protocol_seed": seed,
            "protocol_config": manifest["protocol_config"],
        },
    }
    raw_job = _write_artifact(
        result_root / f"raw-job-{index}.json",
        json.dumps(raw_result, indent=2) + "\n",
    )
    transcript = _write_artifact(
        result_root / f"transcript-{index}.log", f"solver transcript {index}\n"
    )
    result = {
        "schema_version": 1,
        "manifest_hash": manifest["manifest_hash"],
        "candidate_hash": manifest["candidate_hash"],
        "model": manifest["model"],
        "protocol_config_sha256": manifest["protocol_config_sha256"],
        "container_image_digest": manifest["prototype"]["container_image_digest"],
        "framework_prompt_sha256": manifest["prototype"]["framework_prompt_sha256"],
        "task_bundle_sha256": manifest["prototype"]["task_bundle_sha256"],
        "job_id": f"job-{index}",
        "protocol_seed": seed,
        "execution_status": "completed",
        "reward": score,
        "functional_score": score,
        "static_policy_score": 1.0,
        "llm_audit_score": 1.0,
        "runtime_sec": 30.0 if passed else 60.0,
        "raw_job": raw_job,
        "solver_transcript": transcript,
    }
    path = result_root / f"trial-result-{index}.json"
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path


def test_catalog_expands_to_auditable_thousand_regimes() -> None:
    catalog = _catalog()
    assert validate_catalog(catalog) == []
    candidates = generate_candidates(catalog)
    assert len(candidates) == 1000
    assert len({row["id"] for row in candidates}) == 1000
    assert len({row["family_id"] for row in candidates}) == 50
    assert {row["empirical_status"] for row in candidates} == {"untested"}
    assert all(row["empirical_pass_rate"] is None for row in candidates)
    assert all(len(row["candidate_hash"]) == 64 for row in candidates)


def test_shortlist_is_ten_diverse_human_review_candidates() -> None:
    catalog = _catalog()
    shortlist = select_shortlist(catalog, generate_candidates(catalog))
    assert [row["rank"] for row in shortlist] == list(range(1, 11))
    assert len({row["family_id"] for row in shortlist}) == 10
    assert len({row["domain"] for row in shortlist}) >= 8
    assert {row["situation"]["id"] for row in shortlist} == {
        "forward",
        "inverse",
        "finite_shot_noise",
        "adaptive",
        "robust_ensemble",
    }
    assert {row["screen_status"] for row in shortlist} == {"shortlist_eligible"}


def test_machine_readable_contracts_are_valid_json() -> None:
    for name in (
        "research_protocol.json",
        "prototype_evidence.schema.json",
        "trial_result.schema.json",
        "protocol_config.example.json",
    ):
        assert isinstance(json.loads((DISCOVERY / name).read_text()), dict)
    protocol = json.loads((DISCOVERY / "research_protocol.json").read_text())
    assert protocol["stage_order"] == [
        "search",
        "ingest",
        "normalize",
        "dedupe",
        "coverage-sample",
        "score",
        "human-review",
    ]
    assert len({row["id"] for row in protocol["query_strategies"]}) == 5
    assert protocol["semantic_deduplication"]["automatic_action"].startswith(
        "Create review clusters only"
    )
    assert protocol["promotion_contract"]["automatic_catalog_edit_allowed"] is False
    assert (
        protocol["quota_policy"]["coverage_sample"][
            "expected_candidate_contract_count"
        ]["value"]
        == 1000
    )


def test_fresh_build_is_deterministic_and_has_clean_ledger(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    first = {
        name: (workspace / name).read_bytes()
        for name in (
            "candidates.jsonl",
            "shortlist.json",
            "shortlist.md",
            "summary.json",
            "review_state.json",
        )
    }
    build_workspace(workspace)
    assert first == {name: (workspace / name).read_bytes() for name in first}
    assert b"stale_candidate_hash" not in first["review_state.json"]
    ledger = json.loads(first["review_state.json"])
    assert set(ledger["candidates"]) == set(ledger["active_shortlist_ids"])


def test_real_test_stays_blocked_until_verified_human_gates(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][
        0
    ]["id"]
    config = _protocol_config(workspace)
    output = _artifact_root(workspace) / "manifests" / "blocked.json"
    with pytest.raises(PermissionError):
        authorize_model_test(
            workspace, candidate_id, config, "pilot", list(range(5)), output
        )

    for role in CONCEPT_REVIEW_ROLES:
        record_review(
            workspace,
            candidate_id,
            "concept",
            role,
            "approve",
            f"reviewer-{role}",
            "reviewed",
        )
    with pytest.raises(FileNotFoundError):
        record_prototype(
            workspace,
            candidate_id,
            _artifact_root(workspace) / "missing-evidence.json",
        )
    bundle_path = _prototype_bundle(workspace, candidate_id)
    bundle = json.loads(bundle_path.read_text())
    bundle["valid_alternatives"][1]["log"] = bundle["valid_alternatives"][0]["log"]
    bundle_path.write_text(json.dumps(bundle, indent=2) + "\n")
    with pytest.raises(ValueError, match="valid-alternative logs and styles"):
        record_prototype(workspace, candidate_id, bundle_path)


def test_five_failures_are_only_a_pilot_signal(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "pilot.json"
    seeds = [11, 12, 13, 14, 15]
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", seeds, manifest_path
    )
    _record_failure_trials(workspace, candidate_id, manifest_path, seeds)
    status = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert status["conclusion"] == "pilot_hardness_signal_ready"
    assert status["schedule_complete"] is True
    assert status["pass_at_k"] == 0
    with pytest.raises(ValueError, match="precommitted"):
        record_model_trial(
            workspace,
            candidate_id,
            manifest_path,
            "optional-extra",
            _trial_result(workspace, manifest_path, 99, 99, passed=False),
        )


def test_any_raw_pass_blocks_hardness_before_audit(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "pass.json"
    manifest = authorize_model_test(
        workspace, candidate_id, config, "pilot", list(range(5)), manifest_path
    )
    record_model_trial(
        workspace,
        candidate_id,
        manifest_path,
        "trial-pass",
        _trial_result(workspace, manifest_path, 100, 0, passed=True),
    )
    status = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert status["conclusion"] == "solved_in_at_least_one_valid_trial"


def test_typed_summary_cannot_override_raw_harbor_result(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "tamper.json"
    authorize_model_test(
        workspace, candidate_id, config, "pilot", list(range(5)), manifest_path
    )
    result_path = _trial_result(workspace, manifest_path, 101, 0, passed=True)
    result = json.loads(result_path.read_text())
    result["reward"] = 0.0
    result_path.write_text(json.dumps(result, indent=2) + "\n")
    with pytest.raises(ValueError, match="disagrees with raw Harbor evidence"):
        record_model_trial(
            workspace, candidate_id, manifest_path, "tampered", result_path
        )


def test_confirmation_requires_twenty_audited_substantive_failures(
    tmp_path: Path,
) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    manifest_path = _artifact_root(workspace) / "manifests" / "confirmation.json"
    seeds = list(range(20))
    manifest = authorize_model_test(
        workspace, candidate_id, config, "confirmation", seeds, manifest_path
    )
    _record_failure_trials(workspace, candidate_id, manifest_path, seeds)
    status = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert status["conclusion"] == "protocol_scoped_model_hard_evidence_ready"
    assert status["valid_trials"] == 20
    state = json.loads((workspace / "review_state.json").read_text())
    expert_path = Path(
        state["candidates"][candidate_id]["prototype"]["expert_baseline_path"]
    )
    expert_path.write_text(expert_path.read_text() + "changed = True\n")
    drifted = model_hardness_status(workspace, candidate_id, manifest["manifest_hash"])
    assert drifted["conclusion"] == "evidence_drift_or_gate_revoked"


def test_authorization_recomputes_shortlist_content_hash(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    shortlist_path = workspace / "shortlist.json"
    shortlist = json.loads(shortlist_path.read_text())
    candidate = next(
        row for row in shortlist["candidates"] if row["id"] == candidate_id
    )
    candidate["objective"] += " tampered"
    shortlist_path.write_text(json.dumps(shortlist, indent=2) + "\n")
    with pytest.raises(ValueError, match="hash does not match"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "pilot",
            list(range(5)),
            _artifact_root(workspace) / "tampered-shortlist.json",
        )


def test_confirmation_and_safe_output_are_enforced(tmp_path: Path) -> None:
    workspace, candidate_id, config = _approved_workspace(tmp_path)
    with pytest.raises(ValueError, match="at least 20"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "confirmation",
            list(range(19)),
            _artifact_root(workspace) / "too-few.json",
        )
    with pytest.raises(ValueError, match="must stay under"):
        authorize_model_test(
            workspace,
            candidate_id,
            config,
            "pilot",
            list(range(5)),
            tmp_path / "unsafe-output.json",
        )


def test_reviewers_and_failure_audits_are_append_only(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][
        0
    ]["id"]
    first_role, second_role = CONCEPT_REVIEW_ROLES[:2]
    with pytest.raises(ValueError, match="nonempty identity"):
        record_review(
            workspace, candidate_id, "concept", first_role, "approve", " ", "ok"
        )
    record_review(
        workspace, candidate_id, "concept", first_role, "approve", "One Person", "ok"
    )
    with pytest.raises(ValueError, match="distinct reviewer"):
        record_review(
            workspace,
            candidate_id,
            "concept",
            second_role,
            "approve",
            "  one   person  ",
            "ok",
        )
    with pytest.raises(ValueError, match="append-only"):
        record_review(
            workspace,
            candidate_id,
            "concept",
            first_role,
            "reject",
            "different-person",
            "changed mind",
        )
