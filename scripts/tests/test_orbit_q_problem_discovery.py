"""Regression tests for the future-problem discovery and review gates."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from reports.orbit_q_problem_discovery.pipeline import (
    CONCEPT_REVIEW_ROLES,
    PILOT_REVIEW_ROLES,
    audit_model_trial,
    authorize_model_test,
    build_workspace,
    candidate_gate_status,
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


def test_catalog_expands_to_exactly_one_thousand_unique_candidates() -> None:
    catalog = _catalog()
    assert validate_catalog(catalog) == []
    candidates = generate_candidates(catalog)
    assert len(candidates) == 1000
    assert len({row["id"] for row in candidates}) == 1000
    assert {row["empirical_status"] for row in candidates} == {"untested"}
    assert all(row["empirical_pass_rate"] is None for row in candidates)
    assert all(20 <= row["design_score"] <= 100 for row in candidates)


def test_shortlist_is_ten_diverse_human_review_candidates() -> None:
    catalog = _catalog()
    shortlist = select_shortlist(catalog, generate_candidates(catalog))
    assert [row["rank"] for row in shortlist] == list(range(1, 11))
    assert len({row["family_id"] for row in shortlist}) == 10
    assert len({row["domain"] for row in shortlist}) >= 8
    assert {row["screen_status"] for row in shortlist} == {"shortlist_eligible"}


def test_real_test_manifest_is_blocked_until_every_human_gate(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    shortlist = json.loads((workspace / "shortlist.json").read_text())["candidates"]
    candidate_id = shortlist[0]["id"]
    state = json.loads((workspace / "review_state.json").read_text())
    assert candidate_gate_status(state, candidate_id)["ready_for_model_test"] is False
    with pytest.raises(PermissionError):
        authorize_model_test(
            workspace, candidate_id, "gpt-5.6-sol", 5, tmp_path / "blocked.json"
        )

    for role in CONCEPT_REVIEW_ROLES:
        record_review(
            workspace, candidate_id, "concept", role, "approve", "test-reviewer", "ok"
        )
    record_prototype(
        workspace,
        candidate_id,
        {
            "expert_baseline_path": "staging/expert.py",
            "expert_baseline_sha256": "a" * 64,
            "independent_oracle_path": "staging/oracle.py",
            "independent_oracle_sha256": "b" * 64,
            "evaluator_path": "staging/evaluate.py",
            "evaluator_sha256": "c" * 64,
            "verifier_job_id": "verifier-only-001",
            "container_image_digest": "sha256:" + "e" * 64,
            "framework_prompt_sha256": "d" * 64,
            "source_commit": "0123456789abcdef",
            "public_api_canary_passed": True,
            "expert_runtime_p95_sec": 42.0,
            "independent_oracle_runtime_sec": 21.0,
            "gold_effective_lines": 140,
            "reproducibility_runs": 20,
            "observed_cpu_count": 8,
            "observed_memory_mb": 8192,
            "expert_peak_memory_mb": 2048,
            "verifier_only_passed": True,
        },
    )
    for role in PILOT_REVIEW_ROLES:
        record_review(
            workspace, candidate_id, "pilot", role, "approve", "test-reviewer", "ok"
        )

    manifest = authorize_model_test(
        workspace, candidate_id, "gpt-5.6-sol", 5, tmp_path / "authorized.json"
    )
    assert manifest["candidate_id"] == candidate_id
    assert manifest["independent_trials"] == 5
    assert (tmp_path / "authorized.json").exists()

    for index in range(5):
        record_model_trial(
            workspace,
            candidate_id,
            tmp_path / "authorized.json",
            f"trial-{index}",
            {
                "job_id": f"job-{index}",
                "protocol_seed": index,
                "result_sha256": f"{index}" * 64,
                "execution_status": "completed",
                "reward": 0.0,
                "functional_score": 0.0,
                "static_policy_score": 1.0,
                "llm_audit_score": 1.0,
                "runtime_sec": 60.0,
            },
        )
        audit_model_trial(
            workspace,
            candidate_id,
            f"trial-{index}",
            "algorithm_design" if index < 4 else "infrastructure",
            "failure-auditor",
            "trace inspected",
        )

    state = json.loads((workspace / "review_state.json").read_text())
    status = model_hardness_status(state, candidate_id, manifest["manifest_hash"])
    assert status["conclusion"] == "insufficient_valid_trials"
    assert status["substantive_failures"] == 4
    assert status["excluded_failures"] == 1

    audit_model_trial(
        workspace,
        candidate_id,
        "trial-4",
        "algorithm_design",
        "failure-auditor",
        "infrastructure ruled out after trace review",
    )
    state = json.loads((workspace / "review_state.json").read_text())
    status = model_hardness_status(state, candidate_id, manifest["manifest_hash"])
    assert status["conclusion"] == "protocol_scoped_model_hard_evidence_ready"
    assert status["pass_at_k"] == 0


def test_authorization_requires_repeated_trials(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    candidate_id = json.loads((workspace / "shortlist.json").read_text())["candidates"][0][
        "id"
    ]
    with pytest.raises(ValueError, match="at least five"):
        authorize_model_test(
            workspace, candidate_id, "gpt-5.6-sol", 1, tmp_path / "one-shot.json"
        )
