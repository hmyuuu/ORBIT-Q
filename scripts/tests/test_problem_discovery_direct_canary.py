from __future__ import annotations

import hashlib
import json
from pathlib import Path

from reports.orbit_q_problem_discovery.pipeline import (
    _harbor_legacy_task_checksum,
    _harbor_task_digest,
    _path_sha256,
)
from scripts.materialize_candidate_task import validate_materialized_task_contract


ROOT = Path(__file__).resolve().parents[2]
DISCOVERY = ROOT / "reports" / "orbit_q_problem_discovery"
RECORD_PATH = DISCOVERY / "direct_canaries" / "mixed-sld-qfim-public.json"
BLUEPRINT = DISCOVERY / "blueprints" / "mixed_sld_qfim"
GRAPE_RECORD_PATH = DISCOVERY / "direct_canaries" / "robust-leakage-grape-public.json"
GRAPE_BLUEPRINT = DISCOVERY / "blueprints" / "robust-leakage-grape"
QFIM_TASK = (
    ROOT
    / ".artifacts"
    / "problem-discovery"
    / "candidate-tasks"
    / "candidate-mixed-sld-qfim"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def canonical_set_sha256(entries: list[dict[str, str]]) -> str:
    encoded = (
        json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_direct_canary_record_binds_current_public_sources() -> None:
    record = json.loads(RECORD_PATH.read_text())
    payload = dict(record)
    claimed_payload_sha = payload.pop("record_payload_sha256")
    assert canonical_sha256(payload) == claimed_payload_sha

    bindings = record["source_bindings"]
    assert bindings["blueprint_sha256"] == sha256(BLUEPRINT / "blueprint.json")
    assert bindings["evaluator_sha256"] == sha256(BLUEPRINT / "evaluate_101.py")
    assert bindings["expert_sha256"] == sha256(BLUEPRINT / "expert" / "solution_101.py")
    assert bindings["blueprint_instruction_sha256"] == sha256(
        BLUEPRINT / "instruction.md"
    )

    shortlist = json.loads((DISCOVERY / "shortlist.json").read_text())
    rows = [
        row
        for row in shortlist["candidates"]
        if row["id"] == record["candidate"]["candidate_id"]
    ]
    assert len(rows) == 1
    assert rows[0]["candidate_hash"] == record["candidate"]["discovery_candidate_hash"]

    assert (
        record["evidence_class"] == "direct_public_expert_canary_not_harbor_not_model"
    )
    assert record["execution"]["network"] == "none"
    for key in (
        "harbor_executed",
        "solver_agent_invoked",
        "model_invoked",
        "llm_audit_invoked",
        "canonical_tasks_executed",
    ):
        assert record["execution"][key] is False


def test_direct_canary_metrics_match_frozen_threshold_policy() -> None:
    record = json.loads(RECORD_PATH.read_text())
    blueprint = json.loads((BLUEPRINT / "blueprint.json").read_text())
    observed = record["result"]["admission_metrics"]
    thresholds = blueprint["expert_admission_protocol"]["strict_thresholds"]
    assert {row["metric"] for row in thresholds} == set(observed)
    for row in thresholds:
        value = observed[row["metric"]]
        if row["direction"] == "at_least":
            assert value > row["threshold"]
        else:
            assert row["direction"] == "at_most"
            assert value < row["threshold"]
    assert record["result"]["runtime_margin_seconds"] == (
        record["result"]["runtime_threshold_seconds"]
        - record["result"]["runtime_seconds"]
    )
    assert record["result"]["functional_overall"] == "PASS"
    assert record["result"]["strict_admission_metrics_passed"] is True
    assert record["result"]["static_policy_score"] == 1.0
    assert record["result"]["runtime_p95_established"] is False


def test_local_raw_canary_artifacts_match_when_present() -> None:
    record = json.loads(RECORD_PATH.read_text())
    raw = record["raw_artifacts"]
    assert not Path(raw["locator"]).is_absolute()
    artifact_root = ROOT / raw["locator"]
    if not artifact_root.is_dir():
        return

    for artifact in raw["files"]:
        path = artifact_root / artifact["name"]
        assert path.is_file()
        assert path.stat().st_size == artifact["size_bytes"]
        assert sha256(path) == artifact["sha256"]

    stdout_lines = (artifact_root / "stdout.txt").read_text().splitlines()
    identity = json.loads(stdout_lines[0])["orbit_q_case_identity"]
    assert identity == {
        key: record["case_identity"][key] for key in ("protocol_seed", "case_digest")
    }
    assert stdout_lines[-1] == "Overall: PASS"
    admission = json.loads(stdout_lines[-2])["orbit_q_expert_admission_metrics"]
    assert admission["case_digest"] == record["case_identity"]["case_digest"]
    assert admission["protocol_seed"] == record["case_identity"]["protocol_seed"]
    assert {row["metric"]: row["observed"] for row in admission["metrics"]} == (
        record["result"]["admission_metrics"]
    )

    static = json.loads((artifact_root / "static-policy.json").read_text())
    assert static["static_policy_score"] == record["result"]["static_policy_score"]
    assert static["line_count"] == record["result"]["effective_lines"]
    assert static["logical_statement_count"] == record["result"]["logical_statements"]


def test_qfim_record_binds_exact_materialized_task_when_present() -> None:
    if not QFIM_TASK.is_dir():
        return
    record = json.loads(RECORD_PATH.read_text())
    bindings = record["source_bindings"]
    contract = validate_materialized_task_contract(
        QFIM_TASK, DISCOVERY, require_expert_admission=True
    )
    assert contract["instruction_sha256"] == bindings["materialized_instruction_sha256"]
    assert (
        contract["candidate_metadata_sha256"]
        == bindings["materialized_candidate_metadata_sha256"]
    )
    assert contract["task_policy_sha256"] == bindings["materialized_task_policy_sha256"]
    assert contract["verifier_harness_sha256"] == bindings["verifier_harness_sha256"]
    assert _path_sha256(QFIM_TASK) == bindings["materialized_task_bundle_sha256"]
    assert _harbor_task_digest(QFIM_TASK) == bindings["harbor_packager_digest"]
    assert _harbor_legacy_task_checksum(QFIM_TASK) == bindings["legacy_task_checksum"]


def test_grape_public_canary_record_binds_current_tracked_sources() -> None:
    record = json.loads(GRAPE_RECORD_PATH.read_text())
    payload = dict(record)
    claimed_payload_sha = payload.pop("record_payload_sha256")
    assert canonical_sha256(payload) == claimed_payload_sha

    bindings = record["source_bindings"]
    assert bindings == {
        "blueprint_sha256": sha256(GRAPE_BLUEPRINT / "blueprint.json"),
        "evaluator_sha256": sha256(GRAPE_BLUEPRINT / "evaluate_109.py"),
        "expert_sha256": sha256(GRAPE_BLUEPRINT / "expert" / "solution_109.py"),
        "blueprint_instruction_sha256": sha256(GRAPE_BLUEPRINT / "instruction.md"),
    }
    blueprint = json.loads((GRAPE_BLUEPRINT / "blueprint.json").read_text())
    assert blueprint["direct_public_canary_record"] == str(
        GRAPE_RECORD_PATH.relative_to(ROOT)
    )
    shortlist = json.loads((DISCOVERY / "shortlist.json").read_text())
    rows = [
        row
        for row in shortlist["candidates"]
        if row["id"] == record["candidate"]["candidate_id"]
    ]
    assert len(rows) == 1
    assert rows[0]["candidate_hash"] == record["candidate"]["discovery_candidate_hash"]

    encoded = json.dumps(record, sort_keys=True)
    for prohibited in ('"protocol_seed"', '"case_digest"', '"command_argv"'):
        assert prohibited not in encoded
    assert "docker run" not in encoded
    assert record["raw_artifacts"]["raw_content_in_tracked_record"] is False
    assert record["execution"]["network"] == "none"
    audit = record["independent_read_only_audit"]
    assert audit["raw_admission_rows_and_threshold_comparisons_recomputed"] is True
    assert "did not independently recompute" in audit["physical_metric_provenance"]
    assert "admission_thresholds_and_metrics_recomputed" not in audit
    assert record["remaining_hold"]["independent_hidden_entropy_required"] is True
    assert record["remaining_hold"]["process_isolation_required"] is True


def test_grape_public_canary_aggregates_and_local_hashes_when_present() -> None:
    record = json.loads(GRAPE_RECORD_PATH.read_text())
    result = record["result"]
    thresholds = result["thresholds"]
    assert result["functional_overall"] == "PASS"
    assert result["tuning_role"]["pass_count"] == 4
    assert result["untouched_validation"]["pass_count"] == 8
    assert result["rerun_count"] == result["substitution_count"] == 0
    for phase in ("tuning_role", "untouched_validation"):
        metrics = result[phase]["metric_maxima"]
        assert metrics["heldout_p95_infidelity"] <= thresholds["maximum_p95_infidelity"]
        assert (
            metrics["heldout_worst_infidelity"]
            <= thresholds["maximum_worst_infidelity"]
        )
        assert metrics["heldout_worst_leakage"] <= thresholds["maximum_worst_leakage"]
        assert metrics["max_drive_amplitude"] <= thresholds["maximum_drive_amplitude"]
        assert metrics["max_slew_per_slice"] <= thresholds["maximum_slew_per_slice"]
        assert metrics["max_edge_amplitude"] <= thresholds["maximum_edge_amplitude"]

    raw = record["raw_artifacts"]
    artifact_root = ROOT / raw["locator"]
    if not artifact_root.is_dir():
        return
    protocol = record["protocol_bindings"]
    plan = artifact_root / "pre-run-plan.json"
    summary = artifact_root / "sanitized-summary.json"
    assert plan.stat().st_size == protocol["immutable_pre_run_plan_size_bytes"]
    assert sha256(plan) == protocol["immutable_pre_run_plan_sha256"]
    assert (
        sha256(artifact_root / "pre-run-plan.sha256")
        == protocol["pre_run_plan_sidecar_sha256"]
    )
    assert sha256(artifact_root / "run_case.py") == protocol["runner_sha256"]
    assert (
        sha256(artifact_root / "build_summary.py") == protocol["summary_builder_sha256"]
    )
    assert summary.stat().st_size == protocol["local_sanitized_summary_size_bytes"]
    assert sha256(summary) == protocol["local_sanitized_summary_sha256"]
    assert (
        sha256(artifact_root / "sanitized-summary.sha256")
        == protocol["local_sanitized_summary_sidecar_sha256"]
    )

    local_summary = json.loads(summary.read_text())
    file_hashes = local_summary["artifact_file_hashes"]
    assert len(file_hashes) == raw["artifact_file_count"]
    for item in file_hashes:
        assert sha256(artifact_root / item["path"]) == item["sha256"]
    assert canonical_set_sha256(file_hashes) == raw["artifact_set_sha256"]
    log_hashes = [
        item
        for item in file_hashes
        if item["path"].endswith(("/stdout.txt", "/stderr.txt"))
    ]
    assert len(log_hashes) == raw["raw_stream_file_count"]
    assert canonical_set_sha256(log_hashes) == raw["raw_log_set_sha256"]
