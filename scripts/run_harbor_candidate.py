#!/usr/bin/env python3
"""Preview or execute Harbor for exactly one isolated candidate task."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Sequence
from urllib.parse import urlsplit, urlunsplit

from materialize_candidate_task import (
    CandidateTaskError,
    DEFAULT_OUTPUT_ROOT,
    ROOT,
    reject_forbidden_user_path,
)


DEFAULT_HARBOR = ROOT / ".conda" / "harbor-py312" / "bin" / "harbor"
DEFAULT_JOBS_ROOT = ROOT / ".artifacts" / "problem-discovery" / "candidate-jobs"
TENSORCIRCUIT_PROMPT = ROOT / "prompts" / "frameworks" / "tensorcircuit.md"
TENSORCIRCUIT_IMAGE = "challenge-benchmark-quantum-tensorcircuit:py311"
SOLVER_AGENTS = {
    "codex": "harbor.agents.installed.codex:Codex",
    "codex-para": "adapters.codex_para:CodexPara",
}
CANDIDATE_CANARY = "harbor-canary GUID orbit-q-candidate-"
CANDIDATE_CANARY_RE = re.compile(
    r"^# harbor-canary GUID orbit-q-candidate-[a-z0-9-]+-generated$", re.MULTILINE
)
JOB_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
LOOPBACK_PROXY_HOSTS = {"127.0.0.1", "::1", "localhost"}


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def validate_candidate_task_dir(path: Path) -> Path:
    """Validate one explicit task path without looking in canonical task roots."""

    task_dir = reject_forbidden_user_path(path, label="candidate task")
    repository_root = ROOT.resolve()
    candidate_root = DEFAULT_OUTPUT_ROOT.resolve()
    if _is_within(task_dir, repository_root) and not _is_within(
        task_dir, candidate_root
    ):
        raise CandidateTaskError(
            "repository-local candidate tasks must be under "
            f"{DEFAULT_OUTPUT_ROOT.relative_to(ROOT)}"
        )
    if not task_dir.is_dir():
        raise CandidateTaskError(f"candidate task directory does not exist: {task_dir}")

    required = (
        task_dir / "instruction.md",
        task_dir / "task.toml",
        task_dir / "environment",
        task_dir / "tests" / "test.sh",
        task_dir / "tests" / "problem_id.txt",
        task_dir / "solution" / "solve.sh",
        task_dir / "solution" / "problem_id.txt",
    )
    missing = [
        str(item.relative_to(task_dir)) for item in required if not item.exists()
    ]
    if missing:
        raise CandidateTaskError(
            "candidate task is incomplete; missing: " + ", ".join(missing)
        )

    linked = [str(item.relative_to(task_dir)) for item in required if item.is_symlink()]
    if linked:
        raise CandidateTaskError(
            "candidate task structural paths must not be symbolic links: "
            + ", ".join(linked)
        )

    task_toml = (task_dir / "task.toml").read_text(errors="replace")
    if CANDIDATE_CANARY not in task_toml or not CANDIDATE_CANARY_RE.search(task_toml):
        raise CandidateTaskError(
            "task.toml lacks the ORBIT-Q candidate staging canary; materialize the "
            "blueprint with scripts/materialize_candidate_task.py"
        )

    test_id = (task_dir / "tests" / "problem_id.txt").read_text().strip()
    solution_id = (task_dir / "solution" / "problem_id.txt").read_text().strip()
    if not test_id.isdigit() or test_id == "0" or test_id != solution_id:
        raise CandidateTaskError(
            "candidate task problem ids are invalid or inconsistent"
        )
    evaluator = task_dir / "tests" / f"evaluate_{test_id}.py"
    reference = task_dir / "solution" / f"solution_{test_id}.py"
    if not evaluator.is_file() or not reference.is_file():
        raise CandidateTaskError(
            f"candidate task requires evaluate_{test_id}.py and solution_{test_id}.py"
        )
    if evaluator.is_symlink() or reference.is_symlink():
        raise CandidateTaskError(
            "candidate evaluator and solution must not be symbolic links"
        )
    return task_dir


def validate_jobs_root(path: Path) -> Path:
    jobs_root = reject_forbidden_user_path(path, label="jobs output")
    repository_root = ROOT.resolve()
    default_root = DEFAULT_JOBS_ROOT.resolve()
    if _is_within(jobs_root, repository_root) and not _is_within(
        jobs_root, default_root
    ):
        raise CandidateTaskError(
            "repository-local candidate jobs must be under "
            f"{DEFAULT_JOBS_ROOT.relative_to(ROOT)}"
        )
    return jobs_root


def default_harbor_bin() -> Path:
    return DEFAULT_HARBOR if DEFAULT_HARBOR.is_file() else Path("harbor")


def bridged_loopback_proxy_env() -> dict[str, str]:
    """Map credential-free host loopback HTTP proxies into Docker Desktop."""

    bridged: dict[str, str] = {}
    for key in ("HTTP_PROXY", "HTTPS_PROXY"):
        value = os.environ.get(key)
        if not value:
            continue
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"}:
            raise CandidateTaskError(
                f"{key} must use http:// or https:// for Docker proxy bridging"
            )
        if parsed.hostname not in LOOPBACK_PROXY_HOSTS or parsed.port is None:
            raise CandidateTaskError(
                f"{key} must point to a loopback host with an explicit port"
            )
        if parsed.username is not None or parsed.password is not None:
            raise CandidateTaskError(
                f"{key} must not contain credentials; use a credential-free local proxy"
            )
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise CandidateTaskError(f"{key} must be a plain proxy origin URL")
        netloc = f"host.docker.internal:{parsed.port}"
        bridged[key] = urlunsplit((parsed.scheme, netloc, "", "", ""))
    if not bridged:
        raise CandidateTaskError(
            "--bridge-loopback-proxy requires HTTP_PROXY or HTTPS_PROXY in the host environment"
        )
    return bridged


def build_harbor_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    """Build a single-task TensorCircuit Harbor invocation after safety checks."""

    task_dir = validate_candidate_task_dir(args.task_dir)
    jobs_root = validate_jobs_root(args.jobs_dir)
    if not TENSORCIRCUIT_PROMPT.is_file():
        raise CandidateTaskError(
            f"trusted TensorCircuit framework prompt is missing: {TENSORCIRCUIT_PROMPT}"
        )
    if not isinstance(args.model, str) or not args.model.strip():
        raise CandidateTaskError("--model must be non-empty")
    if not isinstance(args.reasoning_effort, str) or not args.reasoning_effort.strip():
        raise CandidateTaskError("--reasoning-effort must be non-empty")
    if "tensorcircuit" not in args.docker_image.lower():
        raise CandidateTaskError("--docker-image must identify a TensorCircuit image")
    if isinstance(args.override_cpus, bool) or args.override_cpus <= 0:
        raise CandidateTaskError("--override-cpus must be a positive integer")
    if isinstance(args.override_memory_mb, bool) or args.override_memory_mb <= 0:
        raise CandidateTaskError("--override-memory-mb must be a positive integer")
    if args.candidate_seed is not None and (
        isinstance(args.candidate_seed, bool) or args.candidate_seed < 0
    ):
        raise CandidateTaskError("--candidate-seed must be a non-negative integer")
    if not JOB_NAME_RE.fullmatch(args.job_name):
        raise CandidateTaskError(
            "--job-name may contain only letters, digits, dots, underscores, and hyphens"
        )

    audit_model = args.audit_model or args.model
    cmd = [
        str(args.harbor_bin),
        "run",
        "-p",
        str(task_dir),
        "--extra-instruction-path",
        str(TENSORCIRCUIT_PROMPT),
        "--environment-import-path",
        "adapters.framework_docker:FrameworkDockerEnvironment",
        "--environment-kwarg",
        "framework=tensorcircuit",
        "--environment-kwarg",
        f"docker_image={args.docker_image}",
    ]
    if not args.expert_only:
        cmd.extend(
            [
                "--agent-import-path",
                SOLVER_AGENTS[args.solver_agent],
                "--agent-kwarg",
                f"reasoning_effort={args.reasoning_effort}",
            ]
        )
    cmd.extend(
        [
            "--verifier-import-path",
            "adapters.codex_para_verifier:CodexParaVerifier",
            "--verifier-kwarg",
            f"audit_model={audit_model}",
            "--verifier-env",
            "REQUIRED_QUANTUM_FRAMEWORK=tensorcircuit",
        ]
    )
    if args.candidate_seed is not None:
        cmd.extend(
            [
                "--verifier-env",
                f"ORBIT_Q_CANDIDATE_SEED={args.candidate_seed}",
            ]
        )
    if args.force_auth_json:
        if not args.expert_only and args.solver_agent != "codex-para":
            raise CandidateTaskError(
                "--force-auth-json requires --solver-agent codex-para"
            )
        if not args.expert_only:
            cmd.extend(["--agent-kwarg", "force_auth_json=true"])
        cmd.extend(["--verifier-kwarg", "force_auth_json=true"])
    if args.bridge_loopback_proxy:
        for key, value in bridged_loopback_proxy_env().items():
            if not args.expert_only:
                cmd.extend(["--agent-env", f"{key}={value}"])
            cmd.extend(["--verifier-env", f"{key}={value}"])
    if not args.expert_only:
        cmd.extend(["-m", args.model])
    cmd.extend(
        [
            "-n",
            "1",
            "--override-cpus",
            str(args.override_cpus),
            "--override-memory-mb",
            str(args.override_memory_mb),
            "-o",
            str(jobs_root),
            "--job-name",
            args.job_name,
            "--yes",
        ]
    )
    return cmd, task_dir


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preview one explicit TensorCircuit candidate benchmark run. The command "
            "is dry-run only unless --execute is supplied."
        )
    )
    parser.add_argument(
        "--task-dir",
        type=Path,
        required=True,
        help=(
            "Exactly one materialized candidate task directory. Canonical tasks/, "
            "templates/, prompts/, and adapters/ paths are rejected."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Codex solver model, for example gpt-5.6-sol.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="high",
        help="Codex solver reasoning effort (default: high).",
    )
    parser.add_argument(
        "--solver-agent",
        choices=tuple(SOLVER_AGENTS),
        default="codex",
        help=(
            "Harbor Codex adapter (default: codex). Use codex-para with "
            "--force-auth-json for desktop OAuth credentials."
        ),
    )
    parser.add_argument(
        "--force-auth-json",
        action="store_true",
        help=(
            "Use the host's normal Codex auth.json through the codex-para agent "
            "and verifier adapters; the credential is never placed in the command."
        ),
    )
    parser.add_argument(
        "--audit-model",
        default=None,
        help="Codex verifier audit model (default: the solver model).",
    )
    parser.add_argument(
        "--expert-only",
        action="store_true",
        help=(
            "Run the packaged expert solution through the verifier without importing "
            "or invoking a solver agent."
        ),
    )
    parser.add_argument(
        "--docker-image",
        default=TENSORCIRCUIT_IMAGE,
        help=f"TensorCircuit framework image (default: {TENSORCIRCUIT_IMAGE}).",
    )
    parser.add_argument(
        "--jobs-dir",
        type=Path,
        default=DEFAULT_JOBS_ROOT,
        help=(
            "Candidate jobs root. Repository-local output is restricted to "
            ".artifacts/problem-discovery/candidate-jobs."
        ),
    )
    parser.add_argument(
        "--override-cpus",
        type=int,
        default=8,
        help="Harbor task CPU override matching the execution host (default: 8).",
    )
    parser.add_argument(
        "--override-memory-mb",
        type=int,
        default=8192,
        help="Harbor task memory override in MiB (default: 8192).",
    )
    parser.add_argument(
        "--candidate-seed",
        type=int,
        default=None,
        help=(
            "Optional hidden candidate seed passed only to the verifier as "
            "ORBIT_Q_CANDIDATE_SEED; it is never placed in the solver environment."
        ),
    )
    parser.add_argument(
        "--bridge-loopback-proxy",
        action="store_true",
        help=(
            "Pass credential-free host HTTP_PROXY/HTTPS_PROXY loopback origins to "
            "the agent and verifier via host.docker.internal."
        ),
    )
    parser.add_argument(
        "--job-name",
        default="candidate-tensorcircuit-codex",
        help="Harbor job name (default: candidate-tensorcircuit-codex).",
    )
    parser.add_argument(
        "--harbor-bin",
        type=Path,
        default=default_harbor_bin(),
        help="Harbor executable path.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually start Harbor. Without this flag, only print the command.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cmd, task_dir = build_harbor_command(args)
    except CandidateTaskError as exc:
        print(f"error: {exc}")
        return 2

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Candidate task: {task_dir}")
    print("Framework: tensorcircuit")
    if args.expert_only:
        print("Solver: not invoked (packaged expert verification)")
    else:
        print(f"Solver model: {args.model}")
        print(f"Solver agent: {args.solver_agent}")
        print(f"Solver reasoning effort: {args.reasoning_effort}")
    print(f"Verifier audit model: {args.audit_model or args.model}")
    print(
        f"Codex auth source: {'auth.json' if args.force_auth_json else 'environment'}"
    )
    print(
        "Resource envelope: "
        f"{args.override_cpus} CPUs, {args.override_memory_mb} MiB memory"
    )
    print(
        "Candidate verifier seed: "
        + (str(args.candidate_seed) if args.candidate_seed is not None else "default")
    )
    print(
        "Loopback proxy bridge: "
        + ("enabled" if args.bridge_loopback_proxy else "disabled")
    )
    print(f"Command: {shlex.join(cmd)}")
    if not args.execute:
        print("No benchmark was started. Re-run with --execute after human review.")
        return 0

    harbor_text = str(args.harbor_bin)
    if "/" not in harbor_text and shutil.which(harbor_text) is None:
        print(f"error: Harbor executable not found on PATH: {harbor_text}")
        return 2
    if "/" in harbor_text and not args.harbor_bin.expanduser().is_file():
        print(f"error: Harbor executable not found: {args.harbor_bin}")
        return 2

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
