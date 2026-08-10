"""Regression tests for the sanitized exploratory Harbor evidence snapshot."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

import scripts.build_problem_discovery_evidence as evidence_builder
from scripts.build_problem_discovery_evidence import (
    PROTECTED_ARTIFACT_PATTERNS,
    _assert_sanitized,
    _derive_candidate,
    _evidence_payload_sha256,
    _load_source_bindings,
    _sanitize_metric,
    _scan_tool_calls,
    _source_artifact_set_sha256,
    _validate_json_schema,
    _validate_ledger_semantics,
    _validate_raw_task_source_bindings,
    build_ledger,
    render_markdown,
    verify_committed_snapshot,
)


ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "orbit_q_problem_discovery"
LEDGER_PATH = REPORT_DIR / "empirical_evidence.json"
REPORT_PATH = REPORT_DIR / "empirical_evidence.md"
SCHEMA_PATH = REPORT_DIR / "empirical_evidence.schema.json"
SOURCE_BINDINGS_PATH = REPORT_DIR / "empirical_evidence.sources.json"
JOBS_DIR = ROOT / ".artifacts" / "problem-discovery" / "candidate-jobs"


def _ledger() -> dict:
    return json.loads(LEDGER_PATH.read_text(encoding="utf-8"))


def _runs(candidate: dict) -> list[dict]:
    return [*candidate["expert_runs"], *candidate["model_pilots"]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_schema_accepts_checked_in_ledger() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    _validate_json_schema(_ledger(), schema)
    assert verify_committed_snapshot() == _ledger()


def test_claim_boundary_and_candidate_classifications_fail_closed() -> None:
    ledger = _ledger()
    assert (
        ledger["claim_boundary"]["evidence_tier"] == "exploratory_automated_scan_only"
    )
    assert ledger["claim_boundary"]["protocol_qualified_hardness_claim"] is False

    candidates = {row["problem_id"]: row for row in ledger["candidates"]}
    assert set(candidates) == {105, 106, 107, 108, 109, 111}
    for problem_id in (105, 106, 107, 108, 111):
        candidate = candidates[problem_id]
        assert candidate["classification"] == "solved_hardness_blocking"
        assert candidate["hardness_blocker"] is True
        assert len(candidate["model_pilots"]) == 1
    candidate_109 = candidates[109]
    assert candidate_109["classification"] == "pilot_hardness_signal"
    assert candidate_109["hardness_blocker"] is False
    assert len(candidate_109["expert_runs"]) == 1
    assert len(candidate_109["model_pilots"]) == 1

    for candidate in candidates.values():
        for run in _runs(candidate):
            assert run["operator_attested"] is False
            assert run["protocol_qualified_hardness_claim"] is False
            assert run["job"]["n_total_trials"] == 1
            assert run["job"]["n_completed_trials"] == 1
            assert run["job"]["n_errored_trials"] == 0
            assert run["job"]["n_retries"] == 0
            is_failed_109_pilot = (
                candidate["problem_id"] == 109 and run["evidence_role"] == "model_pilot"
            )
            expected_functional = 0.0 if is_failed_109_pilot else 1.0
            assert run["scores"]["compound_reward"] == expected_functional
            assert run["scores"]["functional_score"] == expected_functional
            assert run["scores"]["static_policy_score"] == 1.0
            assert run["scores"]["llm_audit_score"] == 1.0
            assert run["scores"]["functional_evaluator_runtime_sec"] >= 0.0
            assert run["functional_evidence"]["verifier_functional_passed"] is (
                not is_failed_109_pilot
            )
            assert run["functional_evidence"]["overall_pass_marker"] is (
                not is_failed_109_pilot
            )


def test_expert_model_pairing_strength_is_not_overstated() -> None:
    candidates = {row["problem_id"]: row for row in _ledger()["candidates"]}

    expert_105 = candidates[105]["expert_runs"][0]
    model_105 = candidates[105]["model_pilots"][0]
    assert expert_105["trial"]["task_checksum"] == model_105["trial"]["task_checksum"]
    assert expert_105["functional_evidence"]["seed"] is None
    assert model_105["functional_evidence"]["case_digest"] is None

    expert_106 = candidates[106]["expert_runs"][0]
    model_106 = candidates[106]["model_pilots"][0]
    assert expert_106["trial"]["task_checksum"] == model_106["trial"]["task_checksum"]
    assert expert_106["functional_evidence"]["seed"] == 1062026
    assert (
        expert_106["functional_evidence"]["case_digest"]
        == model_106["functional_evidence"]["case_digest"]
    )
    assert expert_106["functional_evidence"]["case_digest_scope"] == "full_sha256"

    expert_107 = next(
        run
        for run in candidates[107]["expert_runs"]
        if run["functional_evidence"]["seed"] == 1072027
    )
    model_107 = candidates[107]["model_pilots"][0]
    assert (
        expert_107["functional_evidence"]["seed"]
        == model_107["functional_evidence"]["seed"]
    )
    assert (
        expert_107["functional_evidence"]["case_digest"]
        == model_107["functional_evidence"]["case_digest"]
        == "428aab346b3b50c2"
    )
    assert (
        model_107["functional_evidence"]["case_digest_scope"]
        == "reported_16_hex_digest"
    )
    assert expert_107["trial"]["task_checksum"] != model_107["trial"]["task_checksum"]
    assert any(
        "different Harbor task checksums" in caveat
        for caveat in candidates[107]["evidence_caveats"]
    )
    assert candidates[107]["pairing_evidence"] == {
        "task_checksum_match": False,
        "seed_match": True,
        "case_digest_match": True,
        "task_matched_expert_trial_ids": [],
        "case_matched_expert_trial_ids": [expert_107["trial"]["id"]],
    }

    expert_108 = candidates[108]["expert_runs"][0]
    model_108 = candidates[108]["model_pilots"][0]
    assert expert_108["trial"]["task_checksum"] == model_108["trial"]["task_checksum"]
    assert (
        expert_108["functional_evidence"]["seed"]
        == model_108["functional_evidence"]["seed"]
        == 1082037
    )
    assert (
        expert_108["functional_evidence"]["case_digest"]
        == model_108["functional_evidence"]["case_digest"]
        == "1e932988c8113662"
    )
    assert (
        expert_108["functional_evidence"]["case_digest_scope"]
        == model_108["functional_evidence"]["case_digest_scope"]
        == "reported_16_hex_digest"
    )
    assert expert_108["static_policy"]["max_lines"] == 160
    assert model_108["static_policy"]["max_lines"] == 160
    assert expert_108["static_policy"]["line_count"] == 112
    assert model_108["static_policy"]["line_count"] == 90
    assert expert_108["functional_evidence"]["metrics"] == {
        "maximum_control_moment_error": 5.278e-15,
        "maximum_ritz_energy_error": 0.0,
    }
    assert model_108["functional_evidence"]["metrics"] == {
        "maximum_control_moment_error": 5.035e-15,
        "maximum_ritz_energy_error": 0.0,
    }

    expert_111 = candidates[111]["expert_runs"][0]
    model_111 = candidates[111]["model_pilots"][0]
    assert expert_111["trial"]["task_checksum"] == model_111["trial"]["task_checksum"]
    assert (
        expert_111["functional_evidence"]["seed"]
        == model_111["functional_evidence"]["seed"]
        == 1112037
    )
    assert (
        expert_111["functional_evidence"]["case_digest"]
        == model_111["functional_evidence"]["case_digest"]
        == "427074eb20fbd3b1"
    )
    assert (
        expert_111["functional_evidence"]["case_digest_scope"]
        == model_111["functional_evidence"]["case_digest_scope"]
        == "reported_16_hex_digest"
    )
    assert expert_111["static_policy"]["max_lines"] == 160
    assert model_111["static_policy"]["max_lines"] == 160
    assert expert_111["static_policy"]["line_count"] == 72
    assert model_111["static_policy"]["line_count"] == 100
    for run in (expert_111, model_111):
        assert run["scores"]["compound_reward"] == 1.0
        assert run["scores"]["functional_score"] == 1.0
        assert run["scores"]["static_policy_score"] == 1.0
        assert run["scores"]["llm_audit_score"] == 1.0
        assert run["llm_source_audit"]["faithfully_implements_problem"] is True
        assert run["llm_source_audit"]["obvious_implementation_error"] is False
        assert run["llm_source_audit"]["problem_alignment_issues"] == []
    assert expert_111["functional_evidence"]["metrics"] == {
        "active_cuts": 58,
        "maximum_energy_error": 4.996e-16,
        "maximum_gradient_error": 6.491e-06,
        "minimum_relative_cut_gap": 5.317e-06,
    }
    assert model_111["functional_evidence"]["metrics"] == {
        "active_cuts": 58,
        "maximum_energy_error": 5.551e-16,
        "maximum_gradient_error": 6.491e-06,
        "minimum_relative_cut_gap": 5.317e-06,
    }

    expert_109 = candidates[109]["expert_runs"][0]
    model_109 = candidates[109]["model_pilots"][0]
    assert expert_109["trial"]["task_checksum"] == model_109["trial"]["task_checksum"]
    assert (
        expert_109["functional_evidence"]["seed"]
        == model_109["functional_evidence"]["seed"]
        == 1092037
    )
    assert expert_109["functional_evidence"]["case_digest"] is None
    assert model_109["functional_evidence"]["case_digest"] is None


def test_109_pilot_is_a_substantive_failure_signal_not_a_hardness_claim() -> None:
    candidate = next(row for row in _ledger()["candidates"] if row["problem_id"] == 109)
    pilot = candidate["model_pilots"][0]
    metrics = pilot["functional_evidence"]["metrics"]
    checks = {
        check["metric"]: check
        for check in pilot["functional_evidence"]["evaluation_checks"]
    }

    assert pilot["scores"] == {
        "compound_reward": 0.0,
        "functional_score": 0.0,
        "functional_evaluator_runtime_sec": 62.622879,
        "llm_audit_score": 1.0,
        "runtime_score": 1.0,
        "static_policy_score": 1.0,
    }
    assert metrics["heldout_worst_infidelity"] == 0.0020573770597410723
    assert metrics["heldout_worst_infidelity"] > 0.00125
    assert metrics["heldout_p95_infidelity"] == 0.0014157664283936111
    assert metrics["heldout_p95_infidelity"] > 0.0007
    assert metrics["heldout_worst_leakage"] <= 0.00015
    assert metrics["max_drive_amplitude"] <= 3.15
    assert metrics["max_slew_per_slice"] <= 1.15
    assert metrics["max_edge_amplitude"] <= 0.035
    assert {metric for metric, check in checks.items() if not check["passed"]} == {
        "heldout_worst_infidelity",
        "heldout_p95_infidelity",
    }
    assert all(
        checks[metric]["passed"]
        for metric in (
            "heldout_worst_leakage",
            "max_drive_amplitude",
            "max_slew_per_slice",
            "max_edge_amplitude",
        )
    )
    assert pilot["functional_evidence"]["evaluation_check_source"] == (
        ".artifacts/problem-discovery/candidate-tasks/"
        "candidate-robust-leakage-grape.stale-before-admission-protocol-20260810/"
        "tests/evaluate_109.py; selected task content is bound by trial.task_checksum"
    )
    assert (
        "reports/orbit_q_problem_discovery/blueprints/robust-leakage-grape"
        not in pilot["functional_evidence"]["evaluation_check_source"]
    )
    assert pilot["functional_evidence"]["verifier_functional_return_code"] == 1
    assert pilot["llm_source_audit"]["faithfully_implements_problem"] is True
    assert pilot["llm_source_audit"]["obvious_implementation_error"] is False
    assert pilot["llm_source_audit"]["problem_alignment_issues"] == []
    assert pilot["static_policy"]["raw_simulator_hits"] == ["eigh"]
    assert pilot["protocol_qualified_hardness_claim"] is False
    assert candidate["hardness_blocker"] is False


def test_model_trace_audits_are_automated_and_human_review_is_explicitly_none() -> None:
    candidates = {row["problem_id"]: row for row in _ledger()["candidates"]}
    expected_counts = {
        105: (36, 29, {"exec": 24, "wait": 5}),
        106: (39, 32, {"exec": 28, "wait": 4}),
        107: (35, 28, {"exec": 28}),
        108: (18, 11, {"exec": 11}),
        109: (68, 61, {"exec": 43, "wait": 18}),
        111: (35, 28, {"exec": 26, "wait": 2}),
    }
    for problem_id, (step_count, call_count, by_name) in expected_counts.items():
        run = candidates[problem_id]["model_pilots"][0]
        audit = run["contamination_and_command_audit"]
        assert audit["status"] == "automated_scan_pass_human_review_none"
        assert audit["automated_scan_status"] == "pass"
        assert audit["trajectory_step_count"] == step_count
        assert audit["tool_call_count"] == call_count
        assert audit["tool_call_counts_by_name"] == by_name
        assert audit["allowed_tool_names"] == ["exec", "wait"]
        assert audit["disallowed_tool_name_scan"] == "no_matches_observed"
        assert audit["network_command_scan"] == "no_matches_observed"
        assert (
            audit["protected_benchmark_artifact_scan"]
            == "no_disallowed_access_observed"
        )
        assert audit["credential_or_environment_dump_scan"] == "no_matches_observed"
        assert audit["installed_package_test_search_observed"] is (
            problem_id in {105, 109}
        )
        assert len(audit["automated_scan_raw_rollout_sha256"]) == 64
        assert audit["human_review"] == {
            "status": "none",
            "reviewer_id": None,
            "reviewed_at": None,
            "reviewed_raw_rollout_sha256": None,
            "decision": None,
        }
        assert "pretraining contamination" in audit["unassessed"]
        assert audit["cryptographically_signed_command_attestation"] is False

    for candidate in candidates.values():
        for run in candidate["expert_runs"]:
            audit = run["contamination_and_command_audit"]
            assert audit["status"] == "not_applicable_oracle_copy"
            assert audit["automated_scan_raw_rollout_sha256"] is None


def test_installed_package_tests_are_not_confused_with_benchmark_root_tests() -> None:
    def protected(command: str) -> bool:
        return any(pattern.search(command) for pattern in PROTECTED_ARTIFACT_PATTERNS)

    assert not protected(
        "rg FGSSimulator /usr/local/lib/python3.11/site-packages/tensorcircuit*/tests"
    )
    assert not protected(
        "rg grape /usr/local/lib/python3.11/site-packages/tensorcircuit/tests"
    )
    assert protected("rg hidden /tests")
    assert protected("cat /logs/verifier/reward.json")
    assert protected("cd / && cat tests/test.sh")
    assert protected("cat ../tests/test.sh")
    assert protected("cat ../../logs/verifier/reward.json")
    assert protected(
        "python -c \"from pathlib import Path; Path('/', 'tests', 'test.sh').read_text()\""
    )
    assert protected(
        "python -c \"from pathlib import Path; (Path('/') / 'tests' / 'test.sh').read_text()\""
    )
    assert protected(
        "python -c \"from pathlib import Path; Path('/').joinpath('logs', 'verifier').read_text()\""
    )


@pytest.mark.parametrize(
    "calls",
    [
        [("web_search", '{"query":"benchmark answer"}')],
        [("exec", '{"cmd":"cd / && cat tests/test.sh"}')],
        [("exec", '{"cmd":"cat ../tests/test.sh"}')],
        [("exec", '{"cmd":"cat ../../logs/verifier/reward.json"}')],
        [
            (
                "exec",
                '{"cmd":"python -c \\"from pathlib import Path; '
                "Path('/', 'tests', 'test.sh').read_text()\\\"" + "}",
            )
        ],
        [
            (
                "exec",
                '{"cmd":"python -c \\"from pathlib import Path; '
                "(Path('/') / 'tests' / 'test.sh').read_text()\\\"" + "}",
            )
        ],
        [
            (
                "exec",
                '{"cmd":"python -c \\"from pathlib import Path; '
                "Path('/').joinpath('logs', 'verifier').read_text()\\\"" + "}",
            )
        ],
    ],
)
def test_trace_scan_rejects_disallowed_tools_and_protected_path_bypasses(
    calls: list[tuple[str, str]],
) -> None:
    assert any(_scan_tool_calls(calls).values())


def test_sanitized_ledger_contains_no_raw_config_or_secret_bearing_values() -> None:
    ledger = _ledger()
    serialized = json.dumps(ledger, sort_keys=True)
    forbidden_fragments = (
        "host.docker.internal",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        '"env":',
        '"kwargs":',
        "force_auth_json",
        "/Users/",
        "sk-",
    )
    for fragment in forbidden_fragments:
        assert fragment not in serialized

    assert ledger["sanitization"] == {
        "policy": "whitelisted_fields_and_sha256_only",
        "raw_auth_copied": False,
        "raw_environment_copied": False,
        "raw_config_copied": False,
        "raw_transcript_or_rollout_copied": False,
        "note": ledger["sanitization"]["note"],
    }
    forbidden_artifact_names = {"lock.json", "codex.txt"}
    for candidate in ledger["candidates"]:
        for run in _runs(candidate):
            assert "redacted_config" in run["artifacts"]
            if run["evidence_role"] == "expert_oracle":
                assert "oracle_transcript" in run["artifacts"]
            for artifact in run["artifacts"].values():
                path = Path(artifact["path"])
                assert artifact["path"].startswith(
                    ".artifacts/problem-discovery/candidate-jobs/"
                )
                assert path.name not in forbidden_artifact_names
                assert "/sessions/" not in artifact["path"]
                assert len(artifact["sha256"]) == 64

    source_bindings = json.loads(SOURCE_BINDINGS_PATH.read_text(encoding="utf-8"))
    _assert_sanitized(source_bindings, location="source_bindings")
    for binding in source_bindings["task_bindings"]:
        assert binding["source_snapshot"].startswith(
            ".artifacts/problem-discovery/candidate-tasks/"
        )


def test_markdown_is_a_deterministic_render_of_the_json() -> None:
    assert render_markdown(_ledger()) == REPORT_PATH.read_text(encoding="utf-8")


def test_committed_snapshot_check_works_without_ignored_raw_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(evidence_builder, "JOBS_DIR", tmp_path / "missing-jobs")
    monkeypatch.setattr(sys, "argv", ["build_problem_discovery_evidence.py", "--check"])
    assert evidence_builder.main() == 0
    assert verify_committed_snapshot() == _ledger()


def test_committed_check_never_falls_through_to_privileged_raw_rebuild(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    raw_dir = tmp_path / "present-but-untrusted-raw-jobs"
    raw_dir.mkdir()
    monkeypatch.setattr(evidence_builder, "JOBS_DIR", raw_dir)

    def reject_raw_rebuild(_: dict) -> None:
        raise AssertionError("--check must not read or rebuild ignored raw artifacts")

    monkeypatch.setattr(
        evidence_builder, "_assert_raw_rebuild_matches", reject_raw_rebuild
    )
    monkeypatch.setattr(sys, "argv", ["build_problem_discovery_evidence.py", "--check"])
    assert evidence_builder.main() == 0


def test_snapshot_binds_builder_configs_schema_source_manifest_and_payload() -> None:
    ledger = _ledger()
    assert ledger["integrity"]["builder"]["path"] == (
        "scripts/build_problem_discovery_evidence.py"
    )
    assert ledger["integrity"]["builder"]["sha256"] == _sha256(
        ROOT / "scripts" / "build_problem_discovery_evidence.py"
    )
    current = _source_artifact_set_sha256(
        ledger["candidates"], ledger["frozen_protocol_facts"], ledger["integrity"]
    )
    assert current == ledger["source_artifact_set_sha256"]
    assert ledger["evidence_payload_sha256"] == _evidence_payload_sha256(ledger)

    config_mutation = copy.deepcopy(ledger)
    config_mutation["candidates"][0]["model_pilots"][0]["artifacts"]["redacted_config"][
        "sha256"
    ] = "0" * 64
    assert (
        _source_artifact_set_sha256(
            config_mutation["candidates"],
            config_mutation["frozen_protocol_facts"],
            config_mutation["integrity"],
        )
        != current
    )

    manifest_mutation = copy.deepcopy(ledger)
    manifest_mutation["integrity"]["source_binding_manifest"]["sha256"] = "f" * 64
    assert (
        _source_artifact_set_sha256(
            manifest_mutation["candidates"],
            manifest_mutation["frozen_protocol_facts"],
            manifest_mutation["integrity"],
        )
        != current
    )

    payload_mutation = copy.deepcopy(ledger)
    payload_mutation["candidates"][0]["conclusion"] += " altered"
    assert payload_mutation["evidence_payload_sha256"] != _evidence_payload_sha256(
        payload_mutation
    )

    builder_mutation = copy.deepcopy(ledger)
    builder_mutation["integrity"]["builder"]["sha256"] = "e" * 64
    builder_mutation["source_artifact_set_sha256"] = _source_artifact_set_sha256(
        builder_mutation["candidates"],
        builder_mutation["frozen_protocol_facts"],
        builder_mutation["integrity"],
    )
    builder_mutation["snapshot_id"] = (
        "orbit-q-exploratory-harbor-sha256:"
        + builder_mutation["source_artifact_set_sha256"]
    )
    builder_mutation["evidence_payload_sha256"] = _evidence_payload_sha256(
        builder_mutation
    )
    with pytest.raises(ValueError, match="committed integrity artifact drift"):
        _validate_ledger_semantics(builder_mutation, _load_source_bindings())


def test_classification_model_and_task_drift_fail_closed() -> None:
    source_bindings = _load_source_bindings()

    outcome_candidate = copy.deepcopy(_ledger()["candidates"][0])
    outcome_run = outcome_candidate["model_pilots"][0]
    outcome_run["scores"]["compound_reward"] = 0.0
    outcome_run["scores"]["functional_score"] = 0.0
    outcome_run["functional_evidence"]["overall_pass_marker"] = False
    outcome_run["functional_evidence"]["verifier_functional_passed"] = False
    outcome_run["functional_evidence"]["verifier_functional_return_code"] = 1
    with pytest.raises(ValueError, match="threshold failure"):
        _derive_candidate(
            outcome_candidate["problem_id"],
            outcome_candidate["slug"],
            outcome_candidate["expert_runs"],
            outcome_candidate["model_pilots"],
        )

    classification_drift = copy.deepcopy(_ledger())
    classification_drift["candidates"][0]["classification"] = "pilot_hardness_signal"
    classification_drift["evidence_payload_sha256"] = _evidence_payload_sha256(
        classification_drift
    )
    with pytest.raises(ValueError, match="derived evidence drift"):
        _validate_ledger_semantics(classification_drift, source_bindings)

    conclusion_drift = copy.deepcopy(_ledger())
    conclusion_drift["candidates"][0]["conclusion"] = "self-authored conclusion"
    conclusion_drift["evidence_payload_sha256"] = _evidence_payload_sha256(
        conclusion_drift
    )
    with pytest.raises(ValueError, match="derived evidence drift"):
        _validate_ledger_semantics(conclusion_drift, source_bindings)

    model_drift = copy.deepcopy(_ledger())
    model_drift["candidates"][0]["model_pilots"][0]["agent"]["model"] = "other-model"
    model_drift["evidence_payload_sha256"] = _evidence_payload_sha256(model_drift)
    with pytest.raises(
        ValueError, match="derived evidence drift|model_pilot_model drift"
    ):
        _validate_ledger_semantics(model_drift, source_bindings)

    task_drift = copy.deepcopy(_ledger())
    task_drift["candidates"][0]["model_pilots"][0]["trial"]["task_checksum"] = (
        task_drift["candidates"][1]["model_pilots"][0]["trial"]["task_checksum"]
    )
    task_drift["evidence_payload_sha256"] = _evidence_payload_sha256(task_drift)
    with pytest.raises(ValueError, match="pairing|binding drift"):
        _validate_ledger_semantics(task_drift, source_bindings)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("training_rmse", "ghp_not-a-number"),
        ("training_rmse", True),
        ("training_rmse", -1.0),
        ("training_rmse", float("inf")),
        ("controls_shape", [44, 0]),
    ],
)
def test_metric_sanitizer_rejects_untyped_or_out_of_bounds_values(
    name: str, value: object
) -> None:
    with pytest.raises(ValueError):
        _sanitize_metric(name, value)


@pytest.mark.parametrize(
    "value",
    [
        {"nested": ["ghp_1234567890abcdef"]},
        {"OPENAI_API_KEY": "redacted"},
        {"password": "short-value-that-pattern-scans-would-miss"},
        {"path": ".artifacts/problem-discovery/../conf.local.toml"},
        {"message": "copied from /Users/operator/private/result.json"},
        {"message": "copied from /etc/passwd"},
        {"message": "copied from C:\\Users\\operator\\private\\result.json"},
        {"message": "open file:///root/private/result.json"},
        {"url": "https://operator:password@example.test/path"},
        {"token": "eyJabcdefgh.ijklmnop.qrstuvwx"},
    ],
)
def test_recursive_sanitizer_rejects_secret_and_path_payloads(value: object) -> None:
    with pytest.raises(ValueError):
        _assert_sanitized(value)


@pytest.mark.parametrize(
    ("reviewer_id", "reviewed_at", "message"),
    [
        ("", "2026-08-10T09:00:00+00:00", "reviewer identity"),
        ("reviewer-1", "2026-08-10T09:00:00", "lacks timezone"),
        ("reviewer-1", "not-a-date", "invalid human trace review timestamp"),
    ],
)
def test_human_trace_review_manifest_requires_identity_and_timezone(
    monkeypatch: pytest.MonkeyPatch,
    reviewer_id: str,
    reviewed_at: str,
    message: str,
) -> None:
    manifest = json.loads(SOURCE_BINDINGS_PATH.read_text(encoding="utf-8"))
    run = _ledger()["candidates"][0]["model_pilots"][0]
    manifest["human_trace_reviews"] = [
        {
            "job_name": run["job"]["name"],
            "reviewer_id": reviewer_id,
            "reviewed_at": reviewed_at,
            "reviewed_raw_rollout_sha256": run["contamination_and_command_audit"][
                "automated_scan_raw_rollout_sha256"
            ],
            "decision": "approved",
        }
    ]
    monkeypatch.setattr(evidence_builder, "_load_json", lambda _: manifest)
    with pytest.raises(ValueError, match=message):
        _load_source_bindings()


def test_human_trace_review_must_bind_the_exact_admitted_rollout() -> None:
    manifest = json.loads(SOURCE_BINDINGS_PATH.read_text(encoding="utf-8"))
    run = _ledger()["candidates"][0]["model_pilots"][0]
    manifest["human_trace_reviews"] = [
        {
            "job_name": run["job"]["name"],
            "reviewer_id": "reviewer-1",
            "reviewed_at": "2026-08-10T09:00:00+00:00",
            "reviewed_raw_rollout_sha256": "d" * 64,
            "decision": "approved",
        }
    ]
    with pytest.raises(ValueError, match="human trace review is not hash-bound"):
        _validate_ledger_semantics(_ledger(), manifest)


def test_committed_check_verifies_the_tracked_supporting_protocol_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ledger = _ledger()
    source = (
        ROOT / ledger["frozen_protocol_facts"]["image_digest_observation"]["source"]
    ).resolve()
    real_sha256 = evidence_builder._sha256

    def drift_supporting_source(path: Path) -> str:
        if path.resolve() == source:
            return "0" * 64
        return real_sha256(path)

    monkeypatch.setattr(evidence_builder, "_sha256", drift_supporting_source)
    with pytest.raises(ValueError, match="supporting protocol record hash drift"):
        _validate_ledger_semantics(ledger, _load_source_bindings())


def test_local_artifact_snapshot_rebuild_and_hashes_when_available() -> None:
    if not JOBS_DIR.is_dir():
        pytest.skip("ignored local Harbor artifacts are not available")
    checked_in = _ledger()
    _validate_raw_task_source_bindings(_load_source_bindings())
    assert build_ledger() == checked_in

    for candidate in checked_in["candidates"]:
        for run in _runs(candidate):
            for artifact in run["artifacts"].values():
                path = ROOT / artifact["path"]
                assert path.stat().st_size == artifact["size_bytes"]
                assert _sha256(path) == artifact["sha256"]


def test_raw_task_source_binding_rejects_a_mutated_file_digest() -> None:
    if not JOBS_DIR.is_dir():
        pytest.skip("ignored local Harbor artifacts are not available")
    manifest = _load_source_bindings()
    manifest["task_bindings"][0]["files"]["task.toml"] = "0" * 64
    with pytest.raises(ValueError, match="raw task source snapshot differs"):
        _validate_raw_task_source_bindings(manifest)


def test_raw_threshold_binding_rejects_self_authored_threshold_drift() -> None:
    if not JOBS_DIR.is_dir():
        pytest.skip("ignored local Harbor artifacts are not available")
    manifest = _load_source_bindings()
    binding = next(row for row in manifest["task_bindings"] if row["problem_id"] == 109)
    binding["functional_thresholds"]["heldout_p95_infidelity"] = 0.5
    with pytest.raises(ValueError, match="does not match frozen source"):
        _validate_raw_task_source_bindings(manifest)
