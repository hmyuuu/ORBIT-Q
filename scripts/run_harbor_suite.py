#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SINGLE_RUNNER = ROOT / "scripts" / "run_harbor_challenge.py"


def parse_challenges(value: str) -> list[int]:
    selected: list[int] = []
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            raw_start, raw_end = part.split("-", 1)
            start, end = int(raw_start), int(raw_end)
            if start > end:
                raise ValueError(f"Invalid descending challenge range: {part}")
            selected.extend(range(start, end + 1))
        else:
            selected.append(int(part))

    unique = sorted(set(selected))
    if not unique or unique[0] < 1 or unique[-1] > 12:
        raise ValueError("Challenges must be in the inclusive range 1-12")
    return unique


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def summarize_result(result: dict[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {"status": "missing_result"}

    stats = result.get("stats") or {}
    n_running_trials = int(stats.get("n_running_trials") or 0)
    n_errored_trials = int(stats.get("n_errored_trials") or 0)
    if result.get("finished_at") is None:
        status = "running" if n_running_trials else "incomplete"
    elif n_errored_trials:
        status = "error"
    else:
        status = "completed"
    summary: dict[str, Any] = {
        "status": status,
        "finished_at": result.get("finished_at"),
        "n_completed_trials": int(stats.get("n_completed_trials") or 0),
        "n_errored_trials": n_errored_trials,
        "n_running_trials": n_running_trials,
        "n_retries": int(stats.get("n_retries") or 0),
        "exception_types": [],
        "metrics": [],
    }
    for evaluation in (stats.get("evals") or {}).values():
        summary["exception_types"].extend(
            (evaluation.get("exception_stats") or {}).keys()
        )
        for metrics in evaluation.get("metrics") or []:
            summary["metrics"].append(
                {
                    key: metrics.get(key)
                    for key in (
                        "reward",
                        "functional_score",
                        "static_policy_score",
                        "llm_audit_score",
                        "runtime_sec",
                    )
                }
            )
    summary["exception_types"] = sorted(set(summary["exception_types"]))
    return summary


def attempt_index(base_name: str, candidate_name: str) -> int | None:
    if candidate_name == base_name:
        return 1
    match = re.fullmatch(rf"{re.escape(base_name)}-retry-(\d+)", candidate_name)
    return int(match.group(1)) if match else None


def existing_attempts(
    jobs_dir: Path, base_name: str
) -> list[tuple[int, Path, dict[str, Any]]]:
    attempts: list[tuple[int, Path, dict[str, Any]]] = []
    for path in jobs_dir.glob(f"{base_name}*"):
        if not path.is_dir():
            continue
        index = attempt_index(base_name, path.name)
        if index is not None:
            attempts.append(
                (index, path, summarize_result(read_json(path / "result.json")))
            )
    return sorted(attempts, key=lambda item: item[0])


def write_summary(
    path: Path,
    *,
    job_prefix: str,
    challenges: list[int],
    runs: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "job_prefix": job_prefix,
        "challenges": challenges,
        "updated_at": datetime.now(UTC).isoformat(),
        "runs": runs,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Run a sequential, resumable Harbor challenge suite. Unrecognized "
            "arguments are forwarded to run_harbor_challenge.py."
        )
    )
    parser.add_argument(
        "--challenges",
        default="1-12",
        help="Comma-separated challenge ids or ranges (default: 1-12).",
    )
    parser.add_argument(
        "--job-prefix",
        default="benchmark-suite",
        help="Job-name prefix; each job ends in -challenge-NN.",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=ROOT / "jobs",
        help="Harbor output directory.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run challenges even when their job result already exists.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a Harbor command or trial error.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help="Suite summary JSON path (default: <jobs-dir>/<prefix>-summary.json).",
    )
    return parser.parse_known_args()


def main() -> int:
    args, forwarded = parse_args()
    challenges = parse_challenges(args.challenges)
    jobs_dir = args.jobs_dir.expanduser().resolve()
    summary_path = (
        args.summary_path.expanduser().resolve()
        if args.summary_path
        else jobs_dir / f"{args.job_prefix}-summary.json"
    )
    runs: list[dict[str, Any]] = []
    failed = False

    for challenge in challenges:
        challenge_name = f"challenge-{challenge:02d}"
        base_job_name = f"{args.job_prefix}-{challenge_name}"
        attempts = existing_attempts(jobs_dir, base_job_name)
        completed_attempts = [
            attempt for attempt in attempts if attempt[2]["status"] == "completed"
        ]
        active_attempts = [
            attempt
            for attempt in attempts
            if attempt[2]["status"] in {"running", "incomplete"}
        ]

        if completed_attempts and not args.rerun_completed:
            _, job_dir, existing = completed_attempts[-1]
            job_name = job_dir.name
            print(f"{challenge_name}: already completed in {job_name}; skipping")
            runs.append(
                {
                    "challenge": challenge_name,
                    "job_name": job_name,
                    "job_dir": str(job_dir),
                    **existing,
                }
            )
            write_summary(
                summary_path,
                job_prefix=args.job_prefix,
                challenges=challenges,
                runs=runs,
            )
            continue

        if active_attempts:
            _, job_dir, existing = active_attempts[-1]
            print(
                f"{challenge_name}: existing unfinished job requires inspection: "
                f"{job_dir}",
                file=sys.stderr,
            )
            runs.append(
                {
                    "challenge": challenge_name,
                    "job_name": job_dir.name,
                    "job_dir": str(job_dir),
                    **existing,
                }
            )
            failed = True
            write_summary(
                summary_path,
                job_prefix=args.job_prefix,
                challenges=challenges,
                runs=runs,
            )
            break

        next_attempt = max((attempt[0] for attempt in attempts), default=0) + 1
        job_name = (
            base_job_name
            if next_attempt == 1
            else f"{base_job_name}-retry-{next_attempt:02d}"
        )
        job_dir = jobs_dir / job_name

        cmd = [
            sys.executable,
            str(SINGLE_RUNNER),
            "--challenge",
            f"{challenge:02d}",
            "--jobs-dir",
            str(jobs_dir),
            "--job-name",
            job_name,
            *forwarded,
        ]
        print(f"{challenge_name}: starting {job_name}", flush=True)
        return_code = subprocess.run(cmd, cwd=ROOT, check=False).returncode
        outcome = summarize_result(read_json(job_dir / "result.json"))
        outcome.update(
            {
                "challenge": challenge_name,
                "job_name": job_name,
                "job_dir": str(job_dir),
                "runner_return_code": return_code,
                "prior_attempts": [
                    {
                        "job_name": path.name,
                        "status": prior["status"],
                        "finished_at": prior.get("finished_at"),
                    }
                    for _, path, prior in attempts
                ],
            }
        )
        runs.append(outcome)
        write_summary(
            summary_path,
            job_prefix=args.job_prefix,
            challenges=challenges,
            runs=runs,
        )

        if return_code != 0 or outcome["status"] != "completed":
            failed = True
            print(
                f"{challenge_name}: failed; see {job_dir}",
                file=sys.stderr,
                flush=True,
            )
            if not args.continue_on_error:
                break
        else:
            print(f"{challenge_name}: completed", flush=True)

    print(f"Suite summary: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
