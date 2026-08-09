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

from reports.orbit_q_problem_discovery.pipeline import (  # noqa: E402
    CONCEPT_REVIEW_ROLES,
    EXCLUDED_FAILURE_CLASSES,
    FAILURE_AUDIT_ROLES,
    PILOT_REVIEW_ROLES,
    SUBSTANTIVE_FAILURE_CLASSES,
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


def _seed_schedule(value: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated integers"
        ) from error
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _validated_workspace(value: Path) -> Path:
    workspace = value.resolve()
    if workspace.is_relative_to(ROOT):
        artifact_root = (ROOT / ".artifacts" / "problem-discovery").resolve()
        if workspace != DEFAULT_WORKSPACE.resolve() and not workspace.is_relative_to(
            artifact_root
        ):
            raise ValueError(
                "in-repository workspaces are restricted to the discovery report or "
                f"{artifact_root}"
            )
    return workspace


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
    prototype.add_argument("--evidence-bundle", type=Path, required=True)

    authorize = sub.add_parser(
        "authorize", help="emit a run manifest; never launches a model or Harbor"
    )
    authorize.add_argument("--candidate", required=True)
    authorize.add_argument("--protocol-config", type=Path, required=True)
    authorize.add_argument(
        "--stage", choices=("pilot", "confirmation"), default="pilot"
    )
    authorize.add_argument("--seeds", type=_seed_schedule, required=True)
    authorize.add_argument("--output", type=Path, required=True)

    trial = sub.add_parser(
        "record-trial", help="record raw Harbor outcome; does not classify failures"
    )
    trial.add_argument("--candidate", required=True)
    trial.add_argument("--manifest", type=Path, required=True)
    trial.add_argument("--trial-id", required=True)
    trial.add_argument("--result", type=Path, required=True)

    audit = sub.add_parser(
        "audit-trial", help="human-classify one recorded pass or failure"
    )
    audit.add_argument("--candidate", required=True)
    audit.add_argument("--trial-id", required=True)
    audit.add_argument("--role", choices=FAILURE_AUDIT_ROLES, required=True)
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
    workspace = _validated_workspace(args.workspace)
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
                args.evidence_bundle.resolve(),
            )
        )
    elif args.command == "authorize":
        _print(
            authorize_model_test(
                workspace,
                args.candidate,
                args.protocol_config.resolve(),
                args.stage,
                args.seeds,
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
                args.result.resolve(),
            )
        )
    elif args.command == "audit-trial":
        _print(
            audit_model_trial(
                workspace,
                args.candidate,
                args.trial_id,
                args.role,
                args.failure_class,
                args.reviewer,
                args.note,
            )
        )
    elif args.command == "hardness-status":
        _print(model_hardness_status(workspace, args.candidate, args.manifest_hash))


if __name__ == "__main__":
    main()
