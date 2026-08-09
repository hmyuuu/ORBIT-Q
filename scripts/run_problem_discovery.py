#!/usr/bin/env python3
"""CLI for the human-gated future-problem discovery workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.orbit_q_problem_discovery.pipeline import (
    CONCEPT_REVIEW_ROLES,
    EXCLUDED_FAILURE_CLASSES,
    PILOT_REVIEW_ROLES,
    SUBSTANTIVE_FAILURE_CLASSES,
    TRIAL_EXECUTION_STATUSES,
    audit_model_trial,
    authorize_model_test,
    build_workspace,
    candidate_gate_status,
    model_hardness_status,
    record_model_trial,
    record_prototype,
    record_review,
)


DEFAULT_WORKSPACE = ROOT / "reports" / "orbit_q_problem_discovery"


def _print(value: object) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build, review, and authorize future ORBIT-Q problem candidates."
    )
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="rebuild the 1,000-candidate screen and shortlist")

    status = sub.add_parser("status", help="show human-gate status")
    status.add_argument("--candidate")

    review = sub.add_parser("review", help="record one named human review")
    review.add_argument("--candidate", required=True)
    review.add_argument("--gate", choices=("concept", "pilot"), required=True)
    review.add_argument(
        "--role", choices=CONCEPT_REVIEW_ROLES + PILOT_REVIEW_ROLES, required=True
    )
    review.add_argument("--decision", choices=("approve", "reject"), required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--note", required=True)

    prototype = sub.add_parser(
        "record-prototype", help="record expert baseline and verifier-only evidence"
    )
    prototype.add_argument("--candidate", required=True)
    prototype.add_argument("--expert-baseline-path", required=True)
    prototype.add_argument("--expert-baseline-sha256", required=True)
    prototype.add_argument("--independent-oracle-path", required=True)
    prototype.add_argument("--independent-oracle-sha256", required=True)
    prototype.add_argument("--evaluator-path", required=True)
    prototype.add_argument("--evaluator-sha256", required=True)
    prototype.add_argument("--verifier-job-id", required=True)
    prototype.add_argument("--container-image-digest", required=True)
    prototype.add_argument("--framework-prompt-sha256", required=True)
    prototype.add_argument("--source-commit", required=True)
    prototype.add_argument("--public-api-canary-passed", action="store_true")
    prototype.add_argument("--expert-runtime-p95-sec", type=float, required=True)
    prototype.add_argument("--independent-oracle-runtime-sec", type=float, required=True)
    prototype.add_argument("--gold-effective-lines", type=int, required=True)
    prototype.add_argument("--reproducibility-runs", type=int, required=True)
    prototype.add_argument("--observed-cpu-count", type=int, required=True)
    prototype.add_argument("--observed-memory-mb", type=float, required=True)
    prototype.add_argument("--expert-peak-memory-mb", type=float, required=True)
    prototype.add_argument("--verifier-only-passed", action="store_true")
    prototype.add_argument("--notes", default="")

    authorize = sub.add_parser(
        "authorize", help="emit a run manifest; never launches a model or Harbor"
    )
    authorize.add_argument("--candidate", required=True)
    authorize.add_argument("--model", default="gpt-5.6-sol")
    authorize.add_argument("--trials", type=int, default=5)
    authorize.add_argument("--output", type=Path, required=True)

    trial = sub.add_parser(
        "record-trial", help="record raw Harbor outcome; does not classify failures"
    )
    trial.add_argument("--candidate", required=True)
    trial.add_argument("--manifest", type=Path, required=True)
    trial.add_argument("--trial-id", required=True)
    trial.add_argument("--job-id", required=True)
    trial.add_argument("--protocol-seed", type=int, required=True)
    trial.add_argument("--result-sha256", required=True)
    trial.add_argument("--execution-status", choices=TRIAL_EXECUTION_STATUSES, required=True)
    trial.add_argument("--reward", type=float, required=True)
    trial.add_argument("--functional-score", type=float, required=True)
    trial.add_argument("--static-policy-score", type=float, required=True)
    trial.add_argument("--llm-audit-score", type=float, required=True)
    trial.add_argument("--runtime-sec", type=float, required=True)
    trial.add_argument("--notes", default="")

    audit = sub.add_parser(
        "audit-trial", help="human-classify one recorded pass or failure"
    )
    audit.add_argument("--candidate", required=True)
    audit.add_argument("--trial-id", required=True)
    audit.add_argument(
        "--failure-class",
        choices=("success",) + SUBSTANTIVE_FAILURE_CLASSES + EXCLUDED_FAILURE_CLASSES,
        required=True,
    )
    audit.add_argument("--reviewer", required=True)
    audit.add_argument("--note", required=True)

    hardness = sub.add_parser(
        "hardness-status", help="summarize audited evidence for one frozen manifest"
    )
    hardness.add_argument("--candidate", required=True)
    hardness.add_argument("--manifest-hash", required=True)

    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.command == "build":
        _print(build_workspace(workspace))
    elif args.command == "status":
        state = json.loads((workspace / "review_state.json").read_text())
        if args.candidate:
            _print(candidate_gate_status(state, args.candidate))
        else:
            _print(
                {
                    candidate_id: candidate_gate_status(state, candidate_id)
                    for candidate_id in state["active_shortlist_ids"]
                }
            )
    elif args.command == "review":
        _print(
            record_review(
                workspace,
                args.candidate,
                args.gate,
                args.role,
                args.decision,
                args.reviewer,
                args.note,
            )
        )
    elif args.command == "record-prototype":
        _print(
            record_prototype(
                workspace,
                args.candidate,
                {
                    "expert_baseline_path": args.expert_baseline_path,
                    "expert_baseline_sha256": args.expert_baseline_sha256,
                    "independent_oracle_path": args.independent_oracle_path,
                    "independent_oracle_sha256": args.independent_oracle_sha256,
                    "evaluator_path": args.evaluator_path,
                    "evaluator_sha256": args.evaluator_sha256,
                    "verifier_job_id": args.verifier_job_id,
                    "container_image_digest": args.container_image_digest,
                    "framework_prompt_sha256": args.framework_prompt_sha256,
                    "source_commit": args.source_commit,
                    "public_api_canary_passed": args.public_api_canary_passed,
                    "expert_runtime_p95_sec": args.expert_runtime_p95_sec,
                    "independent_oracle_runtime_sec": args.independent_oracle_runtime_sec,
                    "gold_effective_lines": args.gold_effective_lines,
                    "reproducibility_runs": args.reproducibility_runs,
                    "observed_cpu_count": args.observed_cpu_count,
                    "observed_memory_mb": args.observed_memory_mb,
                    "expert_peak_memory_mb": args.expert_peak_memory_mb,
                    "verifier_only_passed": args.verifier_only_passed,
                    "notes": args.notes,
                },
            )
        )
    elif args.command == "authorize":
        _print(
            authorize_model_test(
                workspace,
                args.candidate,
                args.model,
                args.trials,
                args.output.resolve(),
            )
        )
    elif args.command == "record-trial":
        _print(
            record_model_trial(
                workspace,
                args.candidate,
                args.manifest.resolve(),
                args.trial_id,
                {
                    "job_id": args.job_id,
                    "protocol_seed": args.protocol_seed,
                    "result_sha256": args.result_sha256,
                    "execution_status": args.execution_status,
                    "reward": args.reward,
                    "functional_score": args.functional_score,
                    "static_policy_score": args.static_policy_score,
                    "llm_audit_score": args.llm_audit_score,
                    "runtime_sec": args.runtime_sec,
                    "notes": args.notes,
                },
            )
        )
    elif args.command == "audit-trial":
        _print(
            audit_model_trial(
                workspace,
                args.candidate,
                args.trial_id,
                args.failure_class,
                args.reviewer,
                args.note,
            )
        )
    elif args.command == "hardness-status":
        state = json.loads((workspace / "review_state.json").read_text())
        _print(model_hardness_status(state, args.candidate, args.manifest_hash))


if __name__ == "__main__":
    main()
