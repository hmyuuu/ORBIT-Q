#!/usr/bin/env python3
"""Preview or execute Harbor for exactly one isolated candidate task."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit

from materialize_candidate_task import (
    CandidateTaskError,
    DEFAULT_OUTPUT_ROOT,
    EXPERT_ADMISSION_METRICS_KEY,
    EXPERT_ADMISSION_PROTOCOL_FILE,
    ROOT,
    discovery_candidate_binding,
    registered_reserve_binding,
    reject_forbidden_user_path,
    validate_materialized_task_contract,
)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reports.orbit_q_problem_discovery.pipeline import (  # noqa: E402
    authorized_model_execution,
    reserve_model_trial_seed,
)


DEFAULT_HARBOR = ROOT / ".conda" / "harbor-py312" / "bin" / "harbor"
DEFAULT_JOBS_ROOT = ROOT / ".artifacts" / "problem-discovery" / "candidate-jobs"
DEFAULT_DISCOVERY_WORKSPACE = ROOT / "reports" / "orbit_q_problem_discovery"
DEFAULT_EXPERT_EVIDENCE_ROOT = (
    ROOT / ".artifacts" / "problem-discovery" / "expert-prequalification"
)
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
CANDIDATE_SEED_ENV = "ORBIT_Q_CANDIDATE_SEED"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
EXPERT_ADMISSION_FILENAME = "expert-admission.json"
EXPERT_ADMISSION_REWARD_PREFIX = "expert_admission_sha256_word_"
EXPERT_EXECUTION_APPROVAL_TYPE = "orbit_q_expert_execution"
EXPERT_EXECUTION_APPROVAL_SCOPE = "sealed_batch"
EXPERT_EXECUTION_RECEIPT_DIR = ".expert-execution-receipts"
EXPERT_EXECUTION_START_RESERVATION = "execution-start-reservation.json"
EXPERT_AUDIT_POLICY = "disabled_private_prequalification"
EXPERT_PROVIDER_CALL_BUDGET = 0
EXPERT_CUSTODY_POLICY = "local_operator_custody_private"
EXPERT_CUSTODY_ACKNOWLEDGEMENT = "I_ACCEPT_LOCAL_OPERATOR_CUSTODY_NOT_HOST_CONFIDENTIAL"
PROXY_ENV_NAMES = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)
EXPERT_SUBPROCESS_ENV_ALLOWLIST = (
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TZ",
    "TERM",
    "NO_COLOR",
    "PYTHONUNBUFFERED",
    "XDG_RUNTIME_DIR",
    "__CF_USER_TEXT_ENCODING",
)


def _lifecycle_is_blocked(status: object) -> bool:
    """Return whether a machine lifecycle contains an explicit HOLD/reject token."""

    if not isinstance(status, str):
        return False
    tokens = {token for token in re.split(r"[^a-z0-9]+", status.casefold()) if token}
    return bool(tokens & {"hold", "rejected"})


def build_expert_subprocess_env(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the exact provider-free environment for driver, version, and Harbor."""

    source_env = os.environ if source is None else source
    result = {
        name: source_env[name]
        for name in EXPERT_SUBPROCESS_ENV_ALLOWLIST
        if source_env.get(name)
    }
    result["PYTHONPATH"] = str(ROOT)
    result["PYTHONDONTWRITEBYTECODE"] = "1"
    return result


def _runner_python_identity() -> dict[str, str]:
    """Bind the exact virtual-environment launcher and interpreter bytes."""

    if not sys.executable:
        raise CandidateTaskError("runner Python executable is unavailable")
    invocation = Path(os.path.abspath(Path(sys.executable).expanduser()))
    try:
        invocation_stat = invocation.lstat()
        resolved = invocation.resolve(strict=True)
        resolved_stat = resolved.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise CandidateTaskError(
            f"runner Python executable is missing: {invocation}"
        ) from exc
    if (
        not (stat.S_ISREG(invocation_stat.st_mode) or invocation.is_symlink())
        or resolved.is_symlink()
        or not stat.S_ISREG(resolved_stat.st_mode)
    ):
        raise CandidateTaskError(
            "runner Python executable must resolve to a regular file"
        )

    prefix = Path(os.path.abspath(Path(sys.prefix).expanduser()))
    try:
        prefix_stat = prefix.lstat()
        prefix_resolved = prefix.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise CandidateTaskError("runner Python environment prefix is missing") from exc
    if (
        prefix.is_symlink()
        or prefix_resolved != prefix
        or not stat.S_ISDIR(prefix_stat.st_mode)
    ):
        raise CandidateTaskError(
            "runner Python environment prefix must be a resolved directory"
        )
    venv_config = prefix / "pyvenv.cfg"
    try:
        config_stat = venv_config.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise CandidateTaskError(
            f"runner Python virtual-environment config is missing: {venv_config}"
        ) from exc
    if venv_config.is_symlink() or not stat.S_ISREG(config_stat.st_mode):
        raise CandidateTaskError(
            "runner Python virtual-environment config must be a regular nonsymlink file"
        )
    return {
        "runner_python_path": str(invocation),
        "runner_python_resolved_path": str(resolved),
        "runner_python_sha256": _file_sha256(resolved),
        "runner_python_prefix_path": str(prefix),
        "runner_python_venv_config_path": str(venv_config),
        "runner_python_venv_config_sha256": _file_sha256(venv_config),
    }


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _installed_harbor_version() -> str:
    try:
        return distribution_version("harbor")
    except PackageNotFoundError as exc:
        raise CandidateTaskError("installed Harbor version cannot be bound") from exc


def _path_sha256(path: Path) -> str:
    """Hash a task bundle with the same path-sensitive contract as the pipeline."""

    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        raise CandidateTaskError(f"cannot hash missing candidate artifact: {path}")
    files = []
    for directory, directory_names, file_names in os.walk(path, followlinks=False):
        directory_path = Path(directory)
        if any((directory_path / name).is_symlink() for name in directory_names):
            raise CandidateTaskError(
                f"candidate artifact bundle contains a symbolic link: {directory_path}"
            )
        files.extend(directory_path / name for name in file_names)
    files.sort()
    if not files:
        raise CandidateTaskError(f"candidate artifact directory is empty: {path}")
    digest = hashlib.sha256()
    for file_path in files:
        if file_path.is_symlink():
            raise CandidateTaskError(
                f"candidate artifact bundle contains a symbolic link: {file_path}"
            )
        relative = file_path.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(file_path)))
    return digest.hexdigest()


def _harbor_task_digest(task_dir: Path) -> str:
    """Recompute Harbor Packager.compute_content_hash for the exact task layout."""

    files = [task_dir / "task.toml", task_dir / "instruction.md"]
    for directory_name in ("environment", "tests", "solution"):
        for directory, _, file_names in os.walk(
            task_dir / directory_name, followlinks=False
        ):
            files.extend(
                Path(directory) / name
                for name in file_names
                if not (Path(directory) / name).is_symlink()
            )
    files.sort(key=lambda path: path.relative_to(task_dir).as_posix())
    outer = hashlib.sha256()
    for path in files:
        relative = path.relative_to(task_dir).as_posix()
        outer.update(f"{relative}\0{_file_sha256(path)}\n".encode())
    return outer.hexdigest()


def _harbor_legacy_task_checksum(task_dir: Path) -> str:
    """Recompute the default dirhash used by Harbor's Task.checksum field."""

    def directory_hash(directory: Path) -> str | None:
        descriptors = []
        for path in directory.iterdir():
            if path.is_symlink():
                raise CandidateTaskError(
                    f"candidate task checksum rejects symbolic link: {path}"
                )
            if path.is_dir():
                value = directory_hash(path)
                if value is None:
                    continue
                properties = (f"dirhash:{value}", f"name:{path.name}")
            elif path.is_file():
                properties = (f"data:{_file_sha256(path)}", f"name:{path.name}")
            else:
                raise CandidateTaskError(
                    f"candidate task checksum rejects special file: {path}"
                )
            descriptors.append("\0".join(sorted(properties)))
        if not descriptors:
            return None
        payload = "\0\0".join(sorted(descriptors)).encode()
        return hashlib.sha256(payload).hexdigest()

    checksum = directory_hash(task_dir)
    if checksum is None:
        raise CandidateTaskError("candidate task checksum rejects an empty task")
    return checksum


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise CandidateTaskError(f"invalid or missing {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CandidateTaskError(f"{label} must contain a JSON object: {path}")
    return value


def _read_protected_file(
    path: Path,
    label: str,
    *,
    required_mode: int | None = None,
    reject_group_other_write: bool = False,
) -> tuple[bytes, os.stat_result]:
    """Read a regular, nonsymlink file while enforcing local custody."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except (FileNotFoundError, OSError) as exc:
        raise CandidateTaskError(f"invalid or missing {label}: {path}") from exc
    try:
        file_stat = os.fstat(descriptor)
        mode = stat.S_IMODE(file_stat.st_mode)
        if not stat.S_ISREG(file_stat.st_mode):
            raise CandidateTaskError(f"{label} must be a regular file: {path}")
        if file_stat.st_uid != os.geteuid():
            raise CandidateTaskError(f"{label} must be owned by the current user")
        if required_mode is not None and mode != required_mode:
            raise CandidateTaskError(
                f"{label} must have mode {required_mode:04o}, observed {mode:04o}"
            )
        if reject_group_other_write and mode & 0o022:
            raise CandidateTaskError(
                f"{label} must not be writable by group or other users"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read()
        after = os.fstat(descriptor)
        if (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) != (
            file_stat.st_dev,
            file_stat.st_ino,
            file_stat.st_size,
            file_stat.st_mtime_ns,
        ):
            raise CandidateTaskError(f"{label} changed while it was read")
        return payload, file_stat
    finally:
        os.close(descriptor)


def _protected_json_object(
    path: Path, label: str, *, required_mode: int
) -> tuple[dict[str, Any], str]:
    payload, _ = _read_protected_file(path, label, required_mode=required_mode)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateTaskError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise CandidateTaskError(f"{label} must contain a JSON object: {path}")
    return value, hashlib.sha256(payload).hexdigest()


def _bound_regular_file(
    raw_path: object,
    expected_sha256: object,
    label: str,
    *,
    allowed_modes: set[int],
) -> tuple[Path, bytes]:
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise CandidateTaskError(f"{label} path must be absolute")
    path = Path(raw_path)
    try:
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise CandidateTaskError(f"invalid or missing {label}: {path}") from exc
    if str(path) != str(resolved):
        raise CandidateTaskError(f"{label} path must be resolved and nonsymlinked")
    if not isinstance(expected_sha256, str) or not SHA256_RE.fullmatch(expected_sha256):
        raise CandidateTaskError(f"{label} SHA-256 is invalid")
    payload, file_stat = _read_protected_file(resolved, label)
    mode = stat.S_IMODE(file_stat.st_mode)
    if mode not in allowed_modes:
        formatted = ", ".join(f"{value:04o}" for value in sorted(allowed_modes))
        raise CandidateTaskError(
            f"{label} must have an owner-only mode in {{{formatted}}}, "
            f"observed {mode:04o}"
        )
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise CandidateTaskError(f"{label} SHA-256 disagrees with the approval")
    return resolved, payload


def _require_exact_typed_value(observed: object, expected: object, label: str) -> None:
    """Compare JSON values without bool/int or int/float equivalence."""

    if type(observed) is not type(expected):
        raise CandidateTaskError(f"{label} has an invalid type")
    if isinstance(expected, dict):
        if set(observed) != set(expected):
            raise CandidateTaskError(f"{label} has an invalid shape")
        for key, value in expected.items():
            _require_exact_typed_value(observed[key], value, f"{label}.{key}")
        return
    if isinstance(expected, list):
        if len(observed) != len(expected):
            raise CandidateTaskError(f"{label} has an invalid length")
        for index, value in enumerate(expected):
            _require_exact_typed_value(observed[index], value, f"{label}[{index}]")
        return
    if observed != expected:
        raise CandidateTaskError(f"{label} disagrees with current execution bindings")


def _candidate_expert_bindings(
    task_dir: Path,
    *,
    workspace: Path,
    seed: int,
    docker_image: str,
    container_image_digest: str,
    cpus: int,
    memory_mb: int,
) -> dict[str, Any]:
    """Derive immutable expert-run bindings from the exact staged task."""

    metadata = _read_json_object(
        task_dir / "candidate_metadata.json", "candidate metadata"
    )
    candidate_id = metadata.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise CandidateTaskError("candidate metadata requires a nonempty candidate_id")
    problem_id = (task_dir / "tests" / "problem_id.txt").read_text().strip()
    if not problem_id.isdigit() or problem_id == "0":
        raise CandidateTaskError("candidate problem_id.txt must be a positive integer")
    evaluator = task_dir / "tests" / f"evaluate_{problem_id}.py"
    expert = task_dir / "solution" / f"solution_{problem_id}.py"
    for path, label in ((evaluator, "evaluator"), (expert, "expert baseline")):
        if path.is_symlink() or not path.is_file():
            raise CandidateTaskError(
                f"candidate {label} must be a regular file: {path}"
            )
    admission_protocol_path = task_dir / "tests" / EXPERT_ADMISSION_PROTOCOL_FILE
    if not admission_protocol_path.is_file():
        raise CandidateTaskError(
            "candidate evaluator lacks candidate-specific expert admission metrics"
        )
    admission_protocol = _read_json_object(
        admission_protocol_path, "expert admission protocol marker"
    )
    if admission_protocol != {
        "schema_version": 1,
        "status": "candidate_specific_metrics_declared",
        "structured_output_key": EXPERT_ADMISSION_METRICS_KEY,
    } or EXPERT_ADMISSION_METRICS_KEY not in evaluator.read_text(encoding="utf-8"):
        raise CandidateTaskError(
            "candidate evaluator lacks candidate-specific expert admission metrics"
        )
    stored_discovery_binding = metadata.get("discovery_binding")
    if not isinstance(stored_discovery_binding, dict):
        raise CandidateTaskError(
            "expert prequalification requires a materialized discovery_binding"
        )
    current_discovery_binding = discovery_candidate_binding(metadata, workspace)
    if stored_discovery_binding != current_discovery_binding:
        raise CandidateTaskError(
            "materialized candidate discovery binding is stale; rematerialize the task"
        )
    trusted_contract = validate_materialized_task_contract(
        task_dir, workspace, require_expert_admission=True
    )
    candidate_hash = current_discovery_binding["candidate_hash"]
    if not IMAGE_DIGEST_RE.fullmatch(container_image_digest):
        raise CandidateTaskError(
            "--container-image-digest must be sha256:<64 lowercase hex>"
        )
    if "@" in docker_image or "tensorcircuit" not in docker_image.casefold():
        raise CandidateTaskError(
            "--docker-image must be a TensorCircuit image name:tag without a digest"
        )
    return {
        "candidate_id": candidate_id,
        "candidate_hash": candidate_hash,
        "candidate_slug": task_dir.name.removeprefix("candidate-"),
        "problem_id": int(problem_id),
        "protocol_seed": seed,
        "expert_baseline_sha256": _file_sha256(expert),
        "evaluator_sha256": _file_sha256(evaluator),
        "task_bundle_sha256": _path_sha256(task_dir),
        "harbor_task_digest": _harbor_task_digest(task_dir),
        "task_checksum": _harbor_legacy_task_checksum(task_dir),
        "harbor_version": _installed_harbor_version(),
        "framework_prompt_sha256": _file_sha256(TENSORCIRCUIT_PROMPT),
        **trusted_contract,
        "container_image_digest": container_image_digest,
        "docker_image": docker_image,
        "cpus": cpus,
        "memory_mb": memory_mb,
        "task_dir": str(task_dir),
    }


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


def validate_expert_evidence_root(path: Path) -> Path:
    evidence_root = reject_forbidden_user_path(path, label="expert evidence output")
    repository_root = ROOT.resolve()
    default_root = DEFAULT_EXPERT_EVIDENCE_ROOT.resolve()
    if _is_within(evidence_root, repository_root) and not _is_within(
        evidence_root, default_root
    ):
        raise CandidateTaskError(
            "repository-local expert evidence must be under "
            f"{DEFAULT_EXPERT_EVIDENCE_ROOT.relative_to(ROOT)}"
        )
    return evidence_root


def _expert_execution_bindings(
    args: argparse.Namespace, task_dir: Path
) -> dict[str, Any]:
    """Return every shared field covered by a human expert-run approval."""

    resolved = args._resolved_run
    jobs_root = validate_jobs_root(args.jobs_dir)
    evidence_root = validate_expert_evidence_root(args.expert_evidence_dir)
    harbor_text = str(args.harbor_bin.expanduser())
    harbor_candidate = (
        Path(shutil.which(harbor_text))
        if "/" not in harbor_text and shutil.which(harbor_text) is not None
        else Path(harbor_text)
    )
    try:
        harbor_binary = harbor_candidate.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise CandidateTaskError(
            f"Harbor executable cannot be bound: {harbor_candidate}"
        ) from exc
    harbor_stat = harbor_binary.lstat()
    if harbor_binary.is_symlink() or not stat.S_ISREG(harbor_stat.st_mode):
        raise CandidateTaskError(
            f"resolved Harbor executable must be a regular file: {harbor_binary}"
        )
    return (
        {
            key: resolved[key]
            for key in (
                "candidate_id",
                "candidate_hash",
                "candidate_slug",
                "problem_id",
                "blueprint_path",
                "task_bundle_sha256",
                "harbor_task_digest",
                "task_checksum",
                "evaluator_sha256",
                "expert_baseline_sha256",
                "instruction_sha256",
                "candidate_metadata_sha256",
                "task_policy_sha256",
                "solution_driver_sha256",
                "verifier_harness_sha256",
                "framework_prompt_sha256",
                "docker_image",
                "container_image_digest",
                "cpus",
                "memory_mb",
                "audit_model",
                "audit_policy",
                "provider_call_budget",
                "custody_policy",
                "host_adversary_confidentiality",
                "force_auth_json",
                "proxy_policy",
            )
        }
        | {
            "task_dir": str(task_dir.resolve()),
            "jobs_root": str(jobs_root),
            "expert_evidence_root": str(evidence_root),
            "harbor_binary_path": str(harbor_binary),
            "harbor_binary_sha256": _file_sha256(harbor_binary),
            "harbor_version": resolved["harbor_version"],
            "runner_sha256": _file_sha256(Path(__file__).resolve()),
        }
        | _runner_python_identity()
    )


def _validate_private_output_root(path: Path, label: str) -> Path:
    raw = path.expanduser()
    absolute = Path(os.path.abspath(raw))
    try:
        resolved = raw.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise CandidateTaskError(f"approved expert {label} must already exist") from exc
    root_stat = resolved.lstat()
    if (
        absolute != resolved
        or raw.is_symlink()
        or not stat.S_ISDIR(root_stat.st_mode)
        or root_stat.st_uid != os.geteuid()
        or stat.S_IMODE(root_stat.st_mode) != 0o700
    ):
        raise CandidateTaskError(
            f"approved expert {label} must be a resolved, current-owner mode-0700 "
            "directory"
        )
    return resolved


def _recheck_private_candidate_evidence_parent(
    evidence_root: Path,
    candidate_parent: Path,
    parent_fd: int | None = None,
) -> None:
    expected = evidence_root / candidate_parent.name
    try:
        resolved = candidate_parent.resolve(strict=True)
        parent_stat = candidate_parent.lstat()
    except (FileNotFoundError, OSError) as exc:
        raise CandidateTaskError(
            "candidate evidence parent changed or disappeared"
        ) from exc
    held_stat = os.fstat(parent_fd) if parent_fd is not None else parent_stat
    if (
        candidate_parent.is_symlink()
        or resolved != expected
        or not resolved.is_relative_to(evidence_root)
        or not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.geteuid()
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
        or not stat.S_ISDIR(held_stat.st_mode)
        or (held_stat.st_dev, held_stat.st_ino)
        != (parent_stat.st_dev, parent_stat.st_ino)
    ):
        raise CandidateTaskError(
            "candidate evidence parent must remain a resolved current-owner "
            "mode-0700 directory beneath the exact evidence root"
        )


def _open_private_candidate_evidence_parent(
    evidence_root: Path, candidate_slug: str
) -> tuple[Path, int]:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", candidate_slug):
        raise CandidateTaskError("candidate evidence slug is invalid")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        root_fd = os.open(evidence_root, directory_flags)
    except OSError as exc:
        raise CandidateTaskError(
            "expert evidence root changed or is not a private directory"
        ) from exc
    try:
        try:
            root_stat = os.fstat(root_fd)
            path_stat = evidence_root.lstat()
            root_resolved = evidence_root.resolve(strict=True)
        except (FileNotFoundError, OSError) as exc:
            raise CandidateTaskError(
                "expert evidence root changed or disappeared"
            ) from exc
        if (
            evidence_root.is_symlink()
            or root_resolved != evidence_root
            or not stat.S_ISDIR(root_stat.st_mode)
            or root_stat.st_uid != os.geteuid()
            or stat.S_IMODE(root_stat.st_mode) != 0o700
            or (root_stat.st_dev, root_stat.st_ino)
            != (path_stat.st_dev, path_stat.st_ino)
        ):
            raise CandidateTaskError(
                "expert evidence root changed or is not current-owner mode 0700"
            )
        try:
            os.mkdir(candidate_slug, mode=0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        try:
            parent_fd = os.open(candidate_slug, directory_flags, dir_fd=root_fd)
        except OSError as exc:
            raise CandidateTaskError(
                "candidate evidence parent is not a regular private directory"
            ) from exc
    finally:
        os.close(root_fd)
    parent_stat = os.fstat(parent_fd)
    if (
        not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.geteuid()
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
    ):
        os.close(parent_fd)
        raise CandidateTaskError(
            "candidate evidence parent must be current-owner mode 0700"
        )
    candidate_parent = evidence_root / candidate_slug
    try:
        _recheck_private_candidate_evidence_parent(
            evidence_root, candidate_parent, parent_fd
        )
    except BaseException:
        os.close(parent_fd)
        raise
    return candidate_parent, parent_fd


def _directory_entry_exists(directory_fd: int, name: str) -> bool:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise CandidateTaskError("private directory entry name is invalid")
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _write_private_directory_file(directory_fd: int, name: str, payload: bytes) -> str:
    if not name or name in {".", ".."} or Path(name).name != name:
        raise CandidateTaskError("private evidence filename is invalid")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    succeeded = False
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(descriptor)
        succeeded = True
    finally:
        os.close(descriptor)
        if not succeeded:
            try:
                os.unlink(name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
    return hashlib.sha256(payload).hexdigest()


def _validate_reviewed_at(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise CandidateTaskError("expert approval reviewed_at must be nonempty")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise CandidateTaskError(
            "expert approval reviewed_at must be ISO-8601"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CandidateTaskError(
            "expert approval reviewed_at must include an explicit timezone"
        )


def _validate_approval_cases(
    *,
    approval_cases: object,
    plan_payload: bytes,
    bindings: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    case_keys = {
        "ordinal",
        "protocol_seed",
        "job_name",
        "expert_evidence_output",
        "receipt_path",
    }
    if not isinstance(approval_cases, list) or not approval_cases:
        raise CandidateTaskError("expert approval cases must be a nonempty array")
    if len(approval_cases) > 1000:
        raise CandidateTaskError("expert approval contains too many cases")
    normalized = []
    seen_seeds: set[int] = set()
    seen_jobs: set[str] = set()
    for expected_ordinal, row in enumerate(approval_cases, start=1):
        if not isinstance(row, dict) or set(row) != case_keys:
            raise CandidateTaskError("expert approval case has an invalid shape")
        ordinal = row["ordinal"]
        seed = row["protocol_seed"]
        job_name = row["job_name"]
        if type(ordinal) is not int or ordinal != expected_ordinal:
            raise CandidateTaskError(
                "expert approval case ordinals must be contiguous from one"
            )
        if type(seed) is not int or seed < 0 or seed in seen_seeds:
            raise CandidateTaskError(
                "expert approval case seeds must be unique non-negative integers"
            )
        if (
            not isinstance(job_name, str)
            or not JOB_NAME_RE.fullmatch(job_name)
            or job_name in seen_jobs
        ):
            raise CandidateTaskError(
                "expert approval case job names must be unique and valid"
            )
        evidence_output = (
            Path(bindings["expert_evidence_root"])
            / bindings["candidate_slug"]
            / f"seed-{seed}"
        )
        receipt_path = (
            Path(bindings["jobs_root"])
            / EXPERT_EXECUTION_RECEIPT_DIR
            / f"{job_name}.json"
        )
        expected_row = {
            "ordinal": ordinal,
            "protocol_seed": seed,
            "job_name": job_name,
            "expert_evidence_output": str(evidence_output),
            "receipt_path": str(receipt_path),
        }
        _require_exact_typed_value(row, expected_row, f"approval case {ordinal}")
        normalized.append(expected_row)
        seen_seeds.add(seed)
        seen_jobs.add(job_name)

    try:
        plan = json.loads(plan_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateTaskError(
            "sealed expert execution plan is invalid JSON"
        ) from exc
    plan_cases = plan.get("cases") if isinstance(plan, dict) else None
    expected_plan_cases = [
        {
            "ordinal": row["ordinal"],
            "protocol_seed": row["protocol_seed"],
            "job_name": row["job_name"],
        }
        for row in normalized
    ]
    _require_exact_typed_value(
        plan_cases, expected_plan_cases, "sealed expert execution plan cases"
    )
    current = [
        row
        for row in normalized
        if row["protocol_seed"] == args.candidate_seed
        and row["job_name"] == args.job_name
    ]
    if len(current) != 1:
        raise CandidateTaskError(
            "expert execution seed/job is not one exact approved plan case"
        )
    return current[0]


def validate_expert_execution_approval(
    args: argparse.Namespace, task_dir: Path
) -> dict[str, Any]:
    """Validate a protected human batch approval and driver-start reservation."""

    if args.expert_execution_approval is None:
        raise CandidateTaskError(
            "--expert-only --execute requires --expert-execution-approval"
        )
    raw_approval_path = args.expert_execution_approval.expanduser()
    if not raw_approval_path.is_absolute():
        raise CandidateTaskError("expert execution approval path must be absolute")
    try:
        approval_path = raw_approval_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise CandidateTaskError("expert execution approval is missing") from exc
    if raw_approval_path.is_symlink() or str(raw_approval_path) != str(approval_path):
        raise CandidateTaskError(
            "expert execution approval path must be resolved and nonsymlinked"
        )
    approval, approval_sha256 = _protected_json_object(
        approval_path, "expert execution approval", required_mode=0o600
    )
    expected_top_level = {
        "schema_version",
        "approval_type",
        "approval_scope",
        "decision",
        "approved",
        "reviewer",
        "reviewed_at",
        "notes",
        "custody_acknowledgement",
        "plan_path",
        "plan_file_sha256",
        "execution_driver_path",
        "execution_driver_sha256",
        "execution_bindings",
        "cases",
    }
    if set(approval) != expected_top_level:
        raise CandidateTaskError("expert execution approval has an invalid shape")
    expected_constants = {
        "schema_version": 1,
        "approval_type": EXPERT_EXECUTION_APPROVAL_TYPE,
        "approval_scope": EXPERT_EXECUTION_APPROVAL_SCOPE,
        "decision": "approved",
        "approved": True,
        "custody_acknowledgement": EXPERT_CUSTODY_ACKNOWLEDGEMENT,
    }
    for key, value in expected_constants.items():
        _require_exact_typed_value(approval.get(key), value, f"expert approval {key}")
    for key in ("reviewer", "notes"):
        value = approval.get(key)
        if not isinstance(value, str) or not value.strip():
            raise CandidateTaskError(f"expert approval {key} must be nonempty")
    _validate_reviewed_at(approval.get("reviewed_at"))

    jobs_root = _validate_private_output_root(args.jobs_dir, "jobs root")
    evidence_root = _validate_private_output_root(
        args.expert_evidence_dir, "evidence root"
    )

    plan_path, plan_payload = _bound_regular_file(
        approval.get("plan_path"),
        approval.get("plan_file_sha256"),
        "sealed expert execution plan",
        allowed_modes={0o600},
    )
    driver_path, _ = _bound_regular_file(
        approval.get("execution_driver_path"),
        approval.get("execution_driver_sha256"),
        "sealed expert execution driver",
        allowed_modes={0o600, 0o700},
    )
    bindings = _expert_execution_bindings(args, task_dir)
    if bindings["jobs_root"] != str(jobs_root) or bindings[
        "expert_evidence_root"
    ] != str(evidence_root):
        raise CandidateTaskError("approved expert output roots resolved inconsistently")
    _require_exact_typed_value(
        approval.get("execution_bindings"),
        bindings,
        "expert approval execution_bindings",
    )
    current_case = _validate_approval_cases(
        approval_cases=approval.get("cases"),
        plan_payload=plan_payload,
        bindings=bindings,
        args=args,
    )

    reservation_path = driver_path.parent / EXPERT_EXECUTION_START_RESERVATION
    reservation, _ = _protected_json_object(
        reservation_path,
        "expert execution start reservation",
        required_mode=0o600,
    )
    expected_reservation = {
        "schema_version": 1,
        "status": "execution_started_terminal_schedule",
        "plan_path": str(plan_path),
        "plan_file_sha256": approval["plan_file_sha256"],
        "execution_driver_path": str(driver_path),
        "execution_driver_sha256": approval["execution_driver_sha256"],
        "human_approval_path": str(approval_path),
        "human_approval_sha256": approval_sha256,
        "candidate_id": bindings["candidate_id"],
        "candidate_hash": bindings["candidate_hash"],
        "seed_count": len(approval["cases"]),
    }
    _require_exact_typed_value(
        reservation, expected_reservation, "expert execution start reservation"
    )
    return {
        "approval_path": approval_path,
        "approval_sha256": approval_sha256,
        "plan_path": plan_path,
        "plan_file_sha256": approval["plan_file_sha256"],
        "execution_driver_path": driver_path,
        "execution_driver_sha256": approval["execution_driver_sha256"],
        "bindings": bindings,
        "cases": approval["cases"],
        "current_case": current_case,
    }


def _path_lexists(path: Path) -> bool:
    return os.path.lexists(path)


def _expected_expert_receipt(
    validation: dict[str, Any], case: dict[str, Any]
) -> dict[str, Any]:
    bindings = validation["bindings"]
    return {
        "schema_version": 1,
        "status": "expert_execution_claimed",
        "approval_path": str(validation["approval_path"]),
        "approval_sha256": validation["approval_sha256"],
        "plan_path": str(validation["plan_path"]),
        "plan_file_sha256": validation["plan_file_sha256"],
        "execution_driver_path": str(validation["execution_driver_path"]),
        "execution_driver_sha256": validation["execution_driver_sha256"],
        "candidate_id": bindings["candidate_id"],
        "candidate_hash": bindings["candidate_hash"],
        "protocol_seed": case["protocol_seed"],
        "job_name": case["job_name"],
        "task_bundle_sha256": bindings["task_bundle_sha256"],
        "container_image_digest": bindings["container_image_digest"],
    }


def _validate_prior_case_receipt(
    validation: dict[str, Any], case: dict[str, Any]
) -> None:
    receipt_path = Path(case["receipt_path"])
    receipt, _ = _protected_json_object(
        receipt_path, "prior expert execution receipt", required_mode=0o600
    )
    expected = _expected_expert_receipt(validation, case)
    if set(receipt) != set(expected) | {"claimed_at"}:
        raise CandidateTaskError("prior expert execution receipt has invalid shape")
    _validate_reviewed_at(receipt.get("claimed_at"))
    _require_exact_typed_value(
        {key: receipt[key] for key in expected},
        expected,
        "prior expert execution receipt",
    )


def _validate_prior_case_evidence(
    validation: dict[str, Any], case: dict[str, Any]
) -> str:
    evidence_path = Path(case["expert_evidence_output"])
    if evidence_path.is_symlink() or not evidence_path.is_dir():
        raise CandidateTaskError(
            "earlier expert case lacks a regular successful evidence directory"
        )
    try:
        evidence_root = evidence_path.resolve(strict=True)
    except (FileNotFoundError, OSError) as exc:
        raise CandidateTaskError("earlier expert evidence path is invalid") from exc
    if str(evidence_path) != str(evidence_root):
        raise CandidateTaskError(
            "earlier expert evidence path must be resolved and nonsymlinked"
        )

    def custodied_json(path: Path, label: str) -> dict[str, Any]:
        payload, _ = _read_protected_file(path, label, reject_group_other_write=True)
        try:
            value = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CandidateTaskError(f"invalid {label}: {path}") from exc
        if not isinstance(value, dict):
            raise CandidateTaskError(f"{label} must contain a JSON object")
        return value

    case_record = custodied_json(
        evidence_root / "expert-prequalified-case.json",
        "prior expert prequalification case",
    )
    if (
        case_record.get("expert_passed") is not True
        or type(case_record.get("protocol_seed")) is not int
        or case_record.get("protocol_seed") != case["protocol_seed"]
        or not isinstance(case_record.get("job_id"), str)
        or not case_record["job_id"].strip()
        or type(case_record.get("runtime_sec")) not in {int, float}
        or isinstance(case_record.get("runtime_sec"), bool)
        or case_record["runtime_sec"] <= 0
        or not isinstance(case_record.get("case_digest"), str)
        or not SHA256_RE.fullmatch(case_record["case_digest"])
        or not isinstance(case_record.get("admission"), dict)
        or case_record["admission"].get("status") != "admitted"
    ):
        raise CandidateTaskError(
            "earlier expert case evidence is not a successful bound admission"
        )
    raw_evidence = case_record.get("raw_evidence")
    expected_roles = {
        "raw_job",
        "harbor_config",
        "harbor_lock",
        "harbor_job_lock",
        "functional_output",
        "admission_record",
        "audit_details",
    }
    if not isinstance(raw_evidence, dict) or set(raw_evidence) != expected_roles:
        raise CandidateTaskError(
            "earlier expert case evidence lacks the exact seven raw custody roles"
        )
    for item in raw_evidence.values():
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not isinstance(item["path"], str)
            or not isinstance(item["sha256"], str)
            or not SHA256_RE.fullmatch(item["sha256"])
        ):
            raise CandidateTaskError("earlier expert raw evidence has invalid shape")
        raw_path = Path(item["path"])
        try:
            if raw_path.is_symlink() or not raw_path.is_file():
                raise CandidateTaskError(
                    "earlier expert raw evidence must be a regular nonsymlink file"
                )
            resolved_raw = raw_path.resolve(strict=True)
            resolved_raw.relative_to(evidence_root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            raise CandidateTaskError(
                "earlier expert raw evidence escapes its evidence directory"
            ) from exc
        if (
            str(raw_path) != str(resolved_raw)
            or _file_sha256(resolved_raw) != item["sha256"]
        ):
            raise CandidateTaskError("earlier expert raw evidence hash is invalid")

    audit_details = custodied_json(
        Path(raw_evidence["audit_details"]["path"]),
        "prior expert audit details",
    )
    if audit_details.get("audit") != {
        "llm_audit_score": 1.0,
        "llm_audit_skipped": True,
    }:
        raise CandidateTaskError(
            "earlier expert audit details do not prove a skipped LLM audit"
        )
    raw_job = custodied_json(
        Path(raw_evidence["raw_job"]["path"]), "prior raw expert Harbor result"
    )
    rewards = raw_job.get("verifier_result", {}).get("rewards")
    if (
        not isinstance(rewards, dict)
        or raw_job.get("id") != case_record["job_id"]
        or rewards.get("reward") != 1
        or rewards.get("functional_score") != 1
        or rewards.get("static_policy_score") != 1
        or rewards.get("llm_audit_score") != 1
        or isinstance(rewards.get("llm_audit_skipped_score"), bool)
        or rewards.get("llm_audit_skipped_score") != 1
        or rewards.get("runtime_sec") != case_record["runtime_sec"]
    ):
        raise CandidateTaskError(
            "earlier expert raw result does not prove a complete provider-free pass"
        )
    evidence_hashes = {
        name: raw_evidence[name]["sha256"] for name in sorted(expected_roles)
    }
    run_hash = hashlib.sha256(
        json.dumps(
            {
                "job_id": case_record["job_id"],
                "artifact_sha256s": evidence_hashes,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    if case_record.get("run_evidence_sha256") != run_hash:
        raise CandidateTaskError("earlier expert run evidence hash is inconsistent")

    prior_bindings = custodied_json(
        evidence_root / "expert-run-bindings.json", "prior expert run bindings"
    )
    expected_bindings = validation["bindings"]
    _require_exact_typed_value(
        prior_bindings, expected_bindings, "prior expert run bindings"
    )
    return case_record["case_digest"]


def _validate_expert_batch_state(validation: dict[str, Any]) -> None:
    """Enforce strict sequential, stop-on-first-failure batch execution."""

    current_ordinal = validation["current_case"]["ordinal"]
    jobs_root = Path(validation["bindings"]["jobs_root"])
    prior_case_digests: set[str] = set()
    for case in validation["cases"]:
        job_path = jobs_root / case["job_name"]
        evidence_path = Path(case["expert_evidence_output"])
        receipt_path = Path(case["receipt_path"])
        artifacts = (receipt_path, job_path, evidence_path)
        if case["ordinal"] < current_ordinal:
            if not all(_path_lexists(path) for path in artifacts):
                raise CandidateTaskError(
                    "earlier approved expert case lacks receipt, job, or successful "
                    "evidence; the batch is terminally blocked"
                )
            if job_path.is_symlink() or not job_path.is_dir():
                raise CandidateTaskError("earlier expert Harbor job is invalid")
            _validate_prior_case_receipt(validation, case)
            case_digest = _validate_prior_case_evidence(validation, case)
            if case_digest in prior_case_digests:
                raise CandidateTaskError(
                    "earlier expert cases contain a duplicate case digest"
                )
            prior_case_digests.add(case_digest)
        elif case["ordinal"] == current_ordinal:
            if any(_path_lexists(path) for path in artifacts):
                raise CandidateTaskError(
                    "current expert case has an existing receipt, job, or evidence output"
                )
        elif any(_path_lexists(path) for path in artifacts):
            raise CandidateTaskError(
                "later expert case artifacts exist before their approved ordinal"
            )


def claim_expert_execution_approval(validation: dict[str, Any]) -> Path:
    """Atomically consume one approved seed/job before Harbor can start."""

    _validate_expert_batch_state(validation)
    case = validation["current_case"]
    receipt_path = Path(case["receipt_path"])

    receipt_parent = receipt_path.parent
    receipt_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    parent_stat = receipt_parent.lstat()
    if (
        receipt_parent.is_symlink()
        or not stat.S_ISDIR(parent_stat.st_mode)
        or parent_stat.st_uid != os.geteuid()
        or stat.S_IMODE(parent_stat.st_mode) != 0o700
    ):
        raise CandidateTaskError(
            "expert execution receipt directory must be owner-only mode 0700"
        )
    _validate_expert_batch_state(validation)
    payload = {
        **_expected_expert_receipt(validation, case),
        "claimed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    serialized = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode()
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(receipt_path, flags, 0o600)
    except FileExistsError as exc:
        raise CandidateTaskError(
            f"expert execution approval case was already consumed: {receipt_path}"
        ) from exc
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return receipt_path


def _required_expert_verifier_env(resolved: dict[str, Any]) -> dict[str, str]:
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "CODEX_AUDIT_ENABLED": "0",
        "REQUIRED_QUANTUM_FRAMEWORK": "tensorcircuit",
        "ORBIT_Q_NON_HARDNESS_RUN": "expert_prequalification",
        "ORBIT_Q_EXPERT_PREQUALIFICATION_MODE": "expert_only_oracle",
        "ORBIT_Q_AUDIT_POLICY": EXPERT_AUDIT_POLICY,
        "ORBIT_Q_PROVIDER_CALL_BUDGET": str(EXPERT_PROVIDER_CALL_BUDGET),
        "ORBIT_Q_CANDIDATE_ID": resolved["candidate_id"],
        "ORBIT_Q_CANDIDATE_HASH": resolved["candidate_hash"],
        "ORBIT_Q_CANDIDATE_SEED": str(resolved["protocol_seed"]),
        "ORBIT_Q_CONTAINER_IMAGE_DIGEST": resolved["container_image_digest"],
        "ORBIT_Q_EXPERT_BASELINE_SHA256": resolved["expert_baseline_sha256"],
        "ORBIT_Q_EVALUATOR_SHA256": resolved["evaluator_sha256"],
        "ORBIT_Q_TASK_BUNDLE_SHA256": resolved["task_bundle_sha256"],
        "ORBIT_Q_FRAMEWORK_PROMPT_SHA256": resolved["framework_prompt_sha256"],
        "ORBIT_Q_VERIFIER_HARNESS_SHA256": resolved["verifier_harness_sha256"],
    }


def _admission_digest_from_rewards(rewards: Any) -> str:
    if not isinstance(rewards, dict):
        raise CandidateTaskError("expert Harbor result is missing verifier rewards")
    words = []
    for index in range(8):
        value = rewards.get(f"{EXPERT_ADMISSION_REWARD_PREFIX}{index}")
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 0xFFFFFFFF
        ):
            raise CandidateTaskError(
                "expert Harbor result does not bind the admission SHA-256"
            )
        words.append(f"{value:08x}")
    return "".join(words)


def _case_identity_from_output(path: Path) -> tuple[int, str]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "orbit_q_case_identity" in value:
            if set(value) != {"orbit_q_case_identity"} or not isinstance(
                value["orbit_q_case_identity"], dict
            ):
                raise CandidateTaskError("malformed expert case identity output")
            rows.append(value["orbit_q_case_identity"])
    if len(rows) != 1 or set(rows[0]) != {"protocol_seed", "case_digest"}:
        raise CandidateTaskError(
            "expert functional output needs exactly one complete case identity"
        )
    seed = rows[0]["protocol_seed"]
    digest = rows[0]["case_digest"]
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
        or not isinstance(digest, str)
        or not SHA256_RE.fullmatch(digest)
    ):
        raise CandidateTaskError("expert functional case identity is invalid")
    return seed, digest


def _validate_expert_admission(
    path: Path, resolved: dict[str, Any], seed: int, case_digest: str
) -> dict[str, Any]:
    record = _read_json_object(path, "expert admission record")
    expected = {
        "schema_version": 1,
        "producer": "orbit_q_score_submission_from_candidate_evaluator",
        "candidate_id": resolved["candidate_id"],
        "candidate_hash": resolved["candidate_hash"],
        "protocol_seed": seed,
        "case_digest": case_digest,
        "expert_baseline_sha256": resolved["expert_baseline_sha256"],
        "evaluator_sha256": resolved["evaluator_sha256"],
        "task_bundle_sha256": resolved["task_bundle_sha256"],
        "framework_prompt_sha256": resolved["framework_prompt_sha256"],
        "verifier_harness_sha256": resolved["verifier_harness_sha256"],
        "container_image_digest": resolved["container_image_digest"],
        "expert_passed": True,
    }
    if set(record) != set(expected) | {"threshold_policy_sha256", "admission"}:
        raise CandidateTaskError("expert admission record has an invalid shape")
    mismatched = [
        name
        for name, expected_value in expected.items()
        if record.get(name) != expected_value
    ]
    if mismatched:
        raise CandidateTaskError(
            "expert admission disagrees with frozen run bindings: "
            + ", ".join(mismatched)
        )
    admission = record.get("admission")
    if not isinstance(admission, dict) or admission.get("status") != "admitted":
        raise CandidateTaskError("expert admission record is not admitted")
    margins = admission.get("margins")
    if not isinstance(margins, list) or not margins:
        raise CandidateTaskError("expert admission record has no margins")
    derived = []
    policy = []
    for row in margins:
        if not isinstance(row, dict) or set(row) != {
            "metric",
            "direction",
            "observed",
            "threshold",
            "absolute_margin",
            "passed",
        }:
            raise CandidateTaskError("expert admission margin has an invalid shape")
        if row["direction"] not in {"at_least", "at_most"}:
            raise CandidateTaskError("expert admission margin has invalid direction")
        observed = float(row["observed"])
        threshold = float(row["threshold"])
        margin = (
            observed - threshold
            if row["direction"] == "at_least"
            else threshold - observed
        )
        if row["passed"] is not True or margin <= 0 or row["absolute_margin"] != margin:
            raise CandidateTaskError(
                "expert admission margin was not recomputed strictly"
            )
        derived.append(margin)
        policy.append(
            {
                "metric": row["metric"],
                "direction": row["direction"],
                "threshold": threshold,
            }
        )
    if admission.get("minimum_margin") != min(derived):
        raise CandidateTaskError("expert minimum admission margin is inconsistent")
    policy_hash = hashlib.sha256(
        json.dumps(
            policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    if record.get("threshold_policy_sha256") != policy_hash:
        raise CandidateTaskError(
            "expert admission threshold policy hash is inconsistent"
        )
    return record


def _validate_expert_harbor_artifacts(
    *,
    task_dir: Path,
    job_dir: Path,
    resolved: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    """Validate one real Harbor oracle trial before evidence is copied."""

    if not job_dir.is_dir():
        raise CandidateTaskError(f"Harbor job output is missing: {job_dir}")
    trials = sorted(path for path in job_dir.iterdir() if path.is_dir())
    if len(trials) != 1:
        raise CandidateTaskError("expert Harbor job must contain exactly one trial")
    trial_dir = trials[0]
    result = _read_json_object(trial_dir / "result.json", "Harbor trial result")
    config = _read_json_object(trial_dir / "config.json", "Harbor trial config")
    trial_lock = _read_json_object(trial_dir / "lock.json", "Harbor trial lock")
    job_lock = _read_json_object(job_dir / "lock.json", "Harbor job lock")

    if (
        result.get("finished_at") in (None, "")
        or result.get("exception_info") is not None
    ):
        raise CandidateTaskError("expert Harbor trial did not finish cleanly")
    if _path_sha256(task_dir) != resolved["task_bundle_sha256"]:
        raise CandidateTaskError(
            "candidate task bundle changed during expert execution"
        )
    current_contract = validate_materialized_task_contract(
        task_dir,
        Path(resolved["workspace"]),
        require_expert_admission=True,
    )
    if any(resolved.get(name) != value for name, value in current_contract.items()):
        raise CandidateTaskError(
            "candidate trusted materialization contract changed during expert execution"
        )
    expected_task_path = task_dir.resolve()
    task_paths = [
        result.get("config", {}).get("task", {}).get("path"),
        config.get("task", {}).get("path"),
        trial_lock.get("task", {}).get("path"),
    ]
    if any(
        not isinstance(value, str) or Path(value).resolve() != expected_task_path
        for value in task_paths
    ):
        raise CandidateTaskError(
            "Harbor result/config/lock do not refer to the exact candidate task bundle"
        )
    locked_task = trial_lock.get("task", {})
    expected_harbor_task_digest = _harbor_task_digest(task_dir)
    if (
        locked_task.get("name") != task_dir.name
        or locked_task.get("type") != "local"
        or locked_task.get("digest") != f"sha256:{expected_harbor_task_digest}"
    ):
        raise CandidateTaskError(
            "Harbor trial lock has an invalid candidate task binding"
        )
    expected_task_checksum = _harbor_legacy_task_checksum(task_dir)
    if result.get("task_checksum") != expected_task_checksum:
        raise CandidateTaskError(
            "Harbor result task checksum does not derive from the candidate task"
        )

    result_config = result.get("config")
    if not isinstance(result_config, dict):
        raise CandidateTaskError("Harbor result has no resolved trial config")
    agent = result_config.get("agent", {})
    if (
        agent.get("name") != "oracle"
        or agent.get("import_path") is not None
        or agent.get("model_name") is not None
        or agent.get("kwargs", {})
        or agent.get("skills", [])
        or agent.get("mcp_servers", [])
    ):
        raise CandidateTaskError("expert prequalification invoked a non-oracle agent")
    agent_info = result.get("agent_info")
    if (
        not isinstance(agent_info, dict)
        or agent_info.get("name") != "oracle"
        or agent_info.get("model_info") is not None
    ):
        raise CandidateTaskError(
            "expert Harbor result does not identify the Oracle agent"
        )
    agent_result = result.get("agent_result")
    if not isinstance(agent_result, dict) or any(
        agent_result.get(name) is not None
        for name in ("n_input_tokens", "n_output_tokens", "cost_usd")
    ):
        raise CandidateTaskError(
            "expert Oracle result unexpectedly contains model usage"
        )
    verifier_info = result.get("verifier_info")
    if verifier_info is not None and (
        not isinstance(verifier_info, dict)
        or verifier_info.get("model_info") is not None
    ):
        raise CandidateTaskError(
            "private expert verifier unexpectedly reports model usage"
        )

    image_reference = f"{resolved['docker_image']}@{resolved['container_image_digest']}"
    for source in (result_config, trial_lock):
        environment = source.get("environment", {})
        if (
            environment.get("import_path")
            != "adapters.framework_docker:FrameworkDockerEnvironment"
            or environment.get("override_cpus") != resolved["cpus"]
            or environment.get("override_memory_mb") != resolved["memory_mb"]
            or environment.get("cpu_enforcement_policy") != "limit"
            or environment.get("memory_enforcement_policy") != "limit"
            or environment.get("kwargs", {})
            != {"framework": "tensorcircuit", "docker_image": image_reference}
        ):
            raise CandidateTaskError(
                "Harbor resources/image differ from expert bindings"
            )
    expected_verifier_env = _required_expert_verifier_env(resolved)
    allowed_proxy = (
        {"HTTP_PROXY", "HTTPS_PROXY"}
        if resolved["proxy_policy"] == "loopback_bridge"
        else set()
    )
    for source in (result_config, trial_lock):
        verifier = source.get("verifier", {})
        verifier_env = verifier.get("env", {})
        if (
            verifier.get("import_path")
            != "adapters.codex_para_verifier:CodexParaVerifier"
            or verifier.get("kwargs", {}) != {"expert_only": True}
            or any(
                verifier_env.get(key) != value
                for key, value in expected_verifier_env.items()
            )
            or set(verifier_env) != set(expected_verifier_env) | allowed_proxy
        ):
            raise CandidateTaskError("Harbor verifier differs from expert bindings")
    instructions = trial_lock.get("extra_instructions")
    if (
        not isinstance(instructions, list)
        or len(instructions) != 1
        or Path(instructions[0].get("path", "")).resolve()
        != TENSORCIRCUIT_PROMPT.resolve()
        or instructions[0].get("digest")
        != f"sha256:{resolved['framework_prompt_sha256']}"
    ):
        raise CandidateTaskError("Harbor lock lacks the exact framework prompt binding")
    if (
        job_lock.get("schema_version") != 2
        or job_lock.get("n_concurrent_trials") != 1
        or job_lock.get("retry", {}).get("max_retries") != 0
        or job_lock.get("trials") != [trial_lock]
    ):
        raise CandidateTaskError("Harbor job lock is not a single no-retry trial")
    harbor_identity = job_lock.get("harbor")
    if (
        not isinstance(harbor_identity, dict)
        or harbor_identity.get("version") != resolved["harbor_version"]
        or harbor_identity.get("is_editable") is not False
    ):
        raise CandidateTaskError("Harbor job lock identity differs from approval")

    rewards = result.get("verifier_result", {}).get("rewards")
    if not isinstance(rewards, dict) or any(
        rewards.get(name) != 1
        for name in (
            "reward",
            "functional_score",
            "static_policy_score",
            "llm_audit_score",
        )
    ):
        raise CandidateTaskError(
            "expert Harbor verifier did not produce a compound pass"
        )
    if (
        isinstance(rewards.get("llm_audit_skipped_score"), bool)
        or rewards.get("llm_audit_skipped_score") != 1
    ):
        raise CandidateTaskError(
            "private expert Harbor run did not prove the LLM audit was skipped"
        )
    audit_details = _read_json_object(
        trial_dir / "verifier" / "audit-details.json",
        "expert verifier audit details",
    )
    if audit_details.get("audit") != {
        "llm_audit_score": 1.0,
        "llm_audit_skipped": True,
    }:
        raise CandidateTaskError(
            "private expert audit details do not prove the LLM audit was skipped"
        )
    runtime = rewards.get("runtime_sec")
    if (
        isinstance(runtime, bool)
        or not isinstance(runtime, (int, float))
        or runtime <= 0
    ):
        raise CandidateTaskError("expert Harbor result has no positive runtime")

    functional_path = trial_dir / "verifier" / "functional-stdout.txt"
    admission_path = trial_dir / "verifier" / EXPERT_ADMISSION_FILENAME
    seed, case_digest = _case_identity_from_output(functional_path)
    if seed != resolved["protocol_seed"]:
        raise CandidateTaskError("expert Harbor case seed differs from requested seed")
    admission = _validate_expert_admission(admission_path, resolved, seed, case_digest)
    admission_digest = _file_sha256(admission_path)
    if _admission_digest_from_rewards(rewards) != admission_digest:
        raise CandidateTaskError(
            "raw Harbor result does not bind admission record hash"
        )

    copied_solution = (
        trial_dir / "artifacts" / "root" / f"solution_{resolved['problem_id']}.py"
    )
    if (
        not copied_solution.is_file()
        or _file_sha256(copied_solution) != resolved["expert_baseline_sha256"]
    ):
        raise CandidateTaskError(
            "Harbor Oracle artifact does not equal the frozen expert baseline"
        )
    return trial_dir, {
        "result": result,
        "runtime_sec": float(runtime),
        "protocol_seed": seed,
        "case_digest": case_digest,
        "admission_record": admission,
        "harbor_task_digest": locked_task["digest"].removeprefix("sha256:"),
        "task_checksum": result["task_checksum"],
    }


def collect_expert_prequalification_evidence(
    args: argparse.Namespace, task_dir: Path
) -> Path:
    """Copy one successful real Harbor oracle trial into immutable evidence."""

    if not args.expert_only:
        raise CandidateTaskError("only expert-only runs can produce expert evidence")
    resolved = args._resolved_run
    jobs_root = validate_jobs_root(
        _validate_private_output_root(args.jobs_dir, "jobs root")
    )
    evidence_root = validate_expert_evidence_root(
        _validate_private_output_root(args.expert_evidence_dir, "evidence root")
    )
    job_dir = jobs_root / args.job_name
    trial_dir, derived = _validate_expert_harbor_artifacts(
        task_dir=task_dir, job_dir=job_dir, resolved=resolved
    )
    candidate_parent, parent_fd = _open_private_candidate_evidence_parent(
        evidence_root, resolved["candidate_slug"]
    )
    destination_name = f"seed-{resolved['protocol_seed']}"
    destination = candidate_parent / destination_name
    sources = {
        "raw_job": trial_dir / "result.json",
        "harbor_config": trial_dir / "config.json",
        "harbor_lock": trial_dir / "lock.json",
        "harbor_job_lock": job_dir / "lock.json",
        "functional_output": trial_dir / "verifier" / "functional-stdout.txt",
        "admission_record": trial_dir / "verifier" / EXPERT_ADMISSION_FILENAME,
        "audit_details": trial_dir / "verifier" / "audit-details.json",
    }
    filenames = {
        "raw_job": "raw-job.json",
        "harbor_config": "resolved-config.json",
        "harbor_lock": "trial-lock.json",
        "harbor_job_lock": "job-lock.json",
        "functional_output": "functional-stdout.txt",
        "admission_record": EXPERT_ADMISSION_FILENAME,
        "audit_details": "audit-details.json",
    }
    staging_name: str | None = None
    staging_fd: int | None = None
    cleanup_names: list[str] = []
    published = False
    completed = False
    try:
        if _directory_entry_exists(parent_fd, destination_name):
            raise CandidateTaskError(
                f"refusing to overwrite existing expert evidence: {destination}"
            )
        _recheck_private_candidate_evidence_parent(
            evidence_root, candidate_parent, parent_fd
        )
        for _ in range(32):
            candidate_name = f".{destination_name}.{secrets.token_hex(12)}"
            try:
                os.mkdir(candidate_name, mode=0o700, dir_fd=parent_fd)
            except FileExistsError:
                continue
            staging_name = candidate_name
            break
        if staging_name is None:
            raise CandidateTaskError("could not allocate private evidence staging")
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
        )
        staging_fd = os.open(staging_name, directory_flags, dir_fd=parent_fd)
        staging_stat = os.fstat(staging_fd)
        if (
            not stat.S_ISDIR(staging_stat.st_mode)
            or staging_stat.st_uid != os.geteuid()
            or stat.S_IMODE(staging_stat.st_mode) != 0o700
        ):
            raise CandidateTaskError(
                "private evidence staging must be current-owner mode 0700"
            )
        raw_evidence = {}
        for name, source in sources.items():
            if source.is_symlink() or not source.is_file():
                raise CandidateTaskError(f"expert evidence source is invalid: {source}")
            filename = filenames[name]
            payload = source.read_bytes()
            digest = _write_private_directory_file(staging_fd, filename, payload)
            cleanup_names.append(filename)
            raw_evidence[name] = {
                "path": str(destination / filename),
                "sha256": digest,
            }
        job_id = derived["result"].get("id")
        if not isinstance(job_id, str) or not job_id.strip():
            raise CandidateTaskError("expert Harbor trial has no job id")
        evidence_hashes = {
            name: raw_evidence[name]["sha256"]
            for name in (
                "raw_job",
                "harbor_config",
                "harbor_lock",
                "harbor_job_lock",
                "functional_output",
                "admission_record",
                "audit_details",
            )
        }
        run_hash = hashlib.sha256(
            json.dumps(
                {"job_id": job_id, "artifact_sha256s": evidence_hashes},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        admission_record = derived["admission_record"]
        case = {
            "job_id": job_id,
            "expert_passed": True,
            "runtime_sec": derived["runtime_sec"],
            "protocol_seed": derived["protocol_seed"],
            "case_digest": derived["case_digest"],
            "admission": admission_record["admission"],
            "threshold_policy_sha256": admission_record["threshold_policy_sha256"],
            "run_evidence_sha256": run_hash,
            "raw_evidence": raw_evidence,
        }
        case_filename = "expert-prequalified-case.json"
        _write_private_directory_file(
            staging_fd,
            case_filename,
            (
                json.dumps(case, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
            ).encode(),
        )
        cleanup_names.append(case_filename)
        bindings = _expert_execution_bindings(args, task_dir)
        if (
            bindings["harbor_task_digest"] != derived["harbor_task_digest"]
            or bindings["task_checksum"] != derived["task_checksum"]
        ):
            raise CandidateTaskError(
                "expert evidence task identity changed during collection"
            )
        bindings_filename = "expert-run-bindings.json"
        _write_private_directory_file(
            staging_fd,
            bindings_filename,
            (
                json.dumps(bindings, indent=2, ensure_ascii=False, sort_keys=True)
                + "\n"
            ).encode(),
        )
        cleanup_names.append(bindings_filename)
        os.fsync(staging_fd)
        _recheck_private_candidate_evidence_parent(
            evidence_root, candidate_parent, parent_fd
        )
        if _directory_entry_exists(parent_fd, destination_name):
            raise CandidateTaskError(
                f"refusing to overwrite existing expert evidence: {destination}"
            )
        os.rename(
            staging_name,
            destination_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        published = True
        os.fsync(parent_fd)
        _recheck_private_candidate_evidence_parent(
            evidence_root, candidate_parent, parent_fd
        )
        published_stat = os.stat(
            destination_name, dir_fd=parent_fd, follow_symlinks=False
        )
        if (
            not stat.S_ISDIR(published_stat.st_mode)
            or published_stat.st_uid != os.geteuid()
            or stat.S_IMODE(published_stat.st_mode) != 0o700
            or (published_stat.st_dev, published_stat.st_ino)
            != (staging_stat.st_dev, staging_stat.st_ino)
        ):
            raise CandidateTaskError(
                "published expert evidence directory changed during collection"
            )
        completed = True
    finally:
        if not completed and staging_fd is not None:
            for name in cleanup_names:
                try:
                    os.unlink(name, dir_fd=staging_fd)
                except OSError:
                    pass
            cleanup_directory = destination_name if published else staging_name
            if cleanup_directory is not None:
                try:
                    os.rmdir(cleanup_directory, dir_fd=parent_fd)
                except OSError:
                    pass
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)
    return destination / "expert-prequalified-case.json"


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
    if set(bridged) != {"HTTP_PROXY", "HTTPS_PROXY"}:
        raise CandidateTaskError(
            "loopback proxy policy requires both HTTP_PROXY and HTTPS_PROXY"
        )
    return bridged


def build_harbor_command(args: argparse.Namespace) -> tuple[list[str], Path]:
    """Build a single-task TensorCircuit Harbor invocation after safety checks."""

    task_dir = validate_candidate_task_dir(args.task_dir)
    task_metadata = _read_json_object(
        task_dir / "candidate_metadata.json", "candidate metadata"
    )
    stored_binding = task_metadata.get("discovery_binding")
    current_reserve_binding = registered_reserve_binding(
        task_metadata, args.workspace.resolve()
    )
    stored_design_only = (
        isinstance(stored_binding, dict)
        and stored_binding.get("contract") == "post_snapshot_reserve_design_contract"
    )
    if stored_design_only or current_reserve_binding is not None:
        if current_reserve_binding is None or stored_binding != current_reserve_binding:
            raise CandidateTaskError(
                "post-snapshot reserve materialization binding is missing or stale"
            )
        raise CandidateTaskError(
            "post-snapshot reserve is design_only: neither expert nor model "
            "execution is authorized"
        )
    lifecycle_status = task_metadata.get("status")
    if _lifecycle_is_blocked(lifecycle_status):
        raise CandidateTaskError(
            "candidate lifecycle is HOLD or rejected: neither expert nor model "
            "execution is authorized"
        )
    jobs_root = validate_jobs_root(args.jobs_dir)
    if not TENSORCIRCUIT_PROMPT.is_file():
        raise CandidateTaskError(
            f"trusted TensorCircuit framework prompt is missing: {TENSORCIRCUIT_PROMPT}"
        )
    if not JOB_NAME_RE.fullmatch(args.job_name):
        raise CandidateTaskError(
            "--job-name may contain only letters, digits, dots, underscores, and hyphens"
        )

    if args.expert_only:
        if args.manifest is not None:
            raise CandidateTaskError("--manifest is only for authorized model runs")
        forbidden_private_provider_options = {
            "--model": args.model,
            "--audit-model": args.audit_model,
            "--solver-agent": args.solver_agent,
            "--reasoning-effort": args.reasoning_effort,
            "--force-auth-json": args.force_auth_json or None,
            "--bridge-loopback-proxy": args.bridge_loopback_proxy or None,
        }
        supplied = [
            name
            for name, value in forbidden_private_provider_options.items()
            if value is not None
        ]
        if supplied:
            raise CandidateTaskError(
                "private expert prequalification disables all model/provider "
                "options; remove: " + ", ".join(supplied)
            )
        if args.candidate_seed is None:
            raise CandidateTaskError(
                "expert prequalification requires exactly one explicit --candidate-seed"
            )
        if args.container_image_digest is None:
            raise CandidateTaskError(
                "expert prequalification requires --container-image-digest"
            )
        docker_image_name = args.docker_image or TENSORCIRCUIT_IMAGE
        cpus = 8 if args.override_cpus is None else args.override_cpus
        memory_mb = 8192 if args.override_memory_mb is None else args.override_memory_mb
        for label, value in (
            ("--override-cpus", cpus),
            ("--override-memory-mb", memory_mb),
        ):
            if isinstance(value, bool) or value <= 0:
                raise CandidateTaskError(f"{label} must be a positive integer")
        if args.candidate_seed is not None and (
            isinstance(args.candidate_seed, bool) or args.candidate_seed < 0
        ):
            raise CandidateTaskError("--candidate-seed must be a non-negative integer")
        expert_bindings = _candidate_expert_bindings(
            task_dir,
            workspace=args.workspace.resolve(),
            seed=args.candidate_seed,
            docker_image=docker_image_name,
            container_image_digest=args.container_image_digest,
            cpus=cpus,
            memory_mb=memory_mb,
        )
        docker_image = f"{docker_image_name}@{args.container_image_digest}"
        resolved = {
            **expert_bindings,
            "workspace": str(args.workspace.resolve()),
            "model": None,
            "solver_agent": None,
            "reasoning_effort": None,
            "audit_model": None,
            "audit_policy": EXPERT_AUDIT_POLICY,
            "provider_call_budget": EXPERT_PROVIDER_CALL_BUDGET,
            "custody_policy": EXPERT_CUSTODY_POLICY,
            "host_adversary_confidentiality": False,
            "force_auth_json": False,
            "proxy_policy": "none",
        }
    else:
        if args.expert_execution_approval is not None:
            raise CandidateTaskError(
                "--expert-execution-approval is only for expert-only execution"
            )
        if args.manifest is None:
            raise CandidateTaskError(
                "model runs require an active human authorization --manifest"
            )
        if args.candidate_seed is None:
            raise CandidateTaskError(
                "model runs require exactly one explicit --candidate-seed"
            )
        if isinstance(args.candidate_seed, bool) or args.candidate_seed < 0:
            raise CandidateTaskError("--candidate-seed must be a non-negative integer")
        frozen_overrides = {
            "--model": args.model,
            "--reasoning-effort": args.reasoning_effort,
            "--solver-agent": args.solver_agent,
            "--force-auth-json": args.force_auth_json or None,
            "--audit-model": args.audit_model,
            "--docker-image": args.docker_image,
            "--container-image-digest": args.container_image_digest,
            "--override-cpus": args.override_cpus,
            "--override-memory-mb": args.override_memory_mb,
            "--bridge-loopback-proxy": args.bridge_loopback_proxy or None,
        }
        supplied = [
            name for name, value in frozen_overrides.items() if value is not None
        ]
        if supplied:
            raise CandidateTaskError(
                "model protocol fields are derived from the manifest; remove: "
                + ", ".join(supplied)
            )
        try:
            resolved = authorized_model_execution(
                args.workspace.resolve(),
                args.manifest.resolve(),
                task_dir,
                TENSORCIRCUIT_PROMPT,
                args.candidate_seed,
            )
        except (FileNotFoundError, KeyError, PermissionError, ValueError) as error:
            raise CandidateTaskError(f"authorization rejected: {error}") from error
        task_config = tomllib.loads((task_dir / "task.toml").read_text())
        if task_config.get("agent", {}).get("timeout_sec") != resolved["wall_time_sec"]:
            raise CandidateTaskError(
                "hashed task.toml agent timeout differs from the authorized wall limit"
            )
        if (
            task_config.get("environment", {}).get("network_mode")
            != resolved["network_policy"]
        ):
            raise CandidateTaskError(
                "hashed task.toml network mode differs from the authorized policy"
            )
        audit_model = resolved["audit_model"]
        docker_image = (
            f"{resolved['docker_image']}@{resolved['container_image_digest']}"
        )
        cpus = resolved["cpus"]
        memory_mb = resolved["memory_mb"]

    args._resolved_run = resolved
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
        f"docker_image={docker_image}",
        "--timeout-multiplier",
        "1.0",
        "--agent-timeout-multiplier",
        "1.0",
        "--max-retries",
        "0",
        "--cpus",
        "limit",
        "--memory",
        "limit",
    ]
    if not args.expert_only:
        cmd.extend(
            [
                "--agent-import-path",
                resolved["solver_agent_import_path"],
                "--agent-kwarg",
                f"reasoning_effort={resolved['reasoning_effort']}",
                "--agent-kwarg",
                f"version={resolved['solver_agent_version']}",
            ]
        )
    else:
        cmd.extend(["--agent", "oracle"])
    cmd.extend(
        [
            "--verifier-import-path",
            "adapters.codex_para_verifier:CodexParaVerifier",
        ]
    )
    if not args.expert_only:
        cmd.extend(["--verifier-kwarg", f"audit_model={audit_model}"])
    cmd.extend(["--verifier-env", "REQUIRED_QUANTUM_FRAMEWORK=tensorcircuit"])
    if args.expert_only:
        cmd.extend(
            [
                "--verifier-kwarg",
                "expert_only=true",
                "--verifier-env",
                "ORBIT_Q_NON_HARDNESS_RUN=expert_prequalification",
                "--verifier-env",
                "CODEX_AUDIT_ENABLED=0",
                "--verifier-env",
                f"ORBIT_Q_AUDIT_POLICY={EXPERT_AUDIT_POLICY}",
                "--verifier-env",
                f"ORBIT_Q_PROVIDER_CALL_BUDGET={EXPERT_PROVIDER_CALL_BUDGET}",
            ]
        )
    if resolved["protocol_seed"] is not None:
        cmd.extend(
            [
                "--verifier-env",
                f"{CANDIDATE_SEED_ENV}={resolved['protocol_seed']}",
            ]
        )
    if args.expert_only:
        verifier_bindings = {
            "PYTHONDONTWRITEBYTECODE": "1",
            "ORBIT_Q_EXPERT_PREQUALIFICATION_MODE": "expert_only_oracle",
            "ORBIT_Q_CANDIDATE_ID": resolved["candidate_id"],
            "ORBIT_Q_CANDIDATE_HASH": resolved["candidate_hash"],
            "ORBIT_Q_CONTAINER_IMAGE_DIGEST": resolved["container_image_digest"],
            "ORBIT_Q_EXPERT_BASELINE_SHA256": resolved["expert_baseline_sha256"],
            "ORBIT_Q_EVALUATOR_SHA256": resolved["evaluator_sha256"],
            "ORBIT_Q_TASK_BUNDLE_SHA256": resolved["task_bundle_sha256"],
            "ORBIT_Q_FRAMEWORK_PROMPT_SHA256": resolved["framework_prompt_sha256"],
            "ORBIT_Q_VERIFIER_HARNESS_SHA256": resolved["verifier_harness_sha256"],
        }
        for key, value in verifier_bindings.items():
            cmd.extend(["--verifier-env", f"{key}={value}"])
    else:
        verifier_bindings = {
            "ORBIT_Q_AUTHORIZATION_MANIFEST_HASH": resolved["manifest_hash"],
            "ORBIT_Q_CONTAINER_IMAGE_DIGEST": resolved["container_image_digest"],
            "ORBIT_Q_TASK_BUNDLE_SHA256": resolved["task_bundle_sha256"],
            "ORBIT_Q_PROVIDER_SNAPSHOT": resolved["provider_snapshot"],
            "ORBIT_Q_TOKEN_BUDGET": str(resolved["token_budget"]),
            "ORBIT_Q_TOKEN_BUDGET_ENFORCEMENT": resolved["token_budget_enforcement"],
            "ORBIT_Q_BUILD_IDENTITY_POLICY": resolved["build_identity_policy"],
            "ORBIT_Q_NETWORK_POLICY": resolved["network_policy"],
            "ORBIT_Q_PROXY_POLICY": resolved["proxy_policy"],
        }
        for key, value in verifier_bindings.items():
            cmd.extend(["--verifier-env", f"{key}={value}"])
    if resolved["force_auth_json"]:
        if not args.expert_only:
            cmd.extend(["--agent-kwarg", "force_auth_json=true"])
        cmd.extend(["--verifier-kwarg", "force_auth_json=true"])
    if resolved["proxy_policy"] == "loopback_bridge":
        for key, value in bridged_loopback_proxy_env().items():
            if not args.expert_only:
                cmd.extend(["--agent-env", f"{key}={value}"])
            cmd.extend(["--verifier-env", f"{key}={value}"])
    if not args.expert_only:
        cmd.extend(["-m", resolved["model"]])
    cmd.extend(
        [
            "-k",
            "1",
            "-n",
            "1",
            "--override-cpus",
            str(cpus),
            "--override-memory-mb",
            str(memory_mb),
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
        default=None,
        help="Expert-only audit fallback. Model runs derive this from --manifest.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=None,
        help="Expert-only option; model runs derive reasoning from --manifest.",
    )
    parser.add_argument(
        "--solver-agent",
        choices=tuple(SOLVER_AGENTS),
        default=None,
        help=(
            "Expert-only option; model runs derive the frozen adapter from --manifest."
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
        "--workspace",
        type=Path,
        default=DEFAULT_DISCOVERY_WORKSPACE,
        help="Problem-discovery review workspace containing the authorization ledger.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Active human authorization manifest required for every model run.",
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
        "--expert-execution-approval",
        type=Path,
        default=None,
        help=(
            "Mode-0600 sealed human batch approval required by every "
            "--expert-only --execute invocation. Dry-run previews do not require it."
        ),
    )
    parser.add_argument(
        "--docker-image",
        default=None,
        help=(
            "Expert-only image override. Model runs use the digest-addressed "
            "image frozen in --manifest."
        ),
    )
    parser.add_argument(
        "--container-image-digest",
        default=None,
        help=(
            "Exact sha256:<digest> required for expert prequalification. Model "
            "runs derive it from the authorization manifest."
        ),
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
        "--expert-evidence-dir",
        type=Path,
        default=DEFAULT_EXPERT_EVIDENCE_ROOT,
        help=(
            "Immutable per-seed expert evidence output. Used only after a successful "
            "--expert-only --execute run."
        ),
    )
    parser.add_argument(
        "--override-cpus",
        type=int,
        default=None,
        help="Expert-only CPU override; model runs derive this from --manifest.",
    )
    parser.add_argument(
        "--override-memory-mb",
        type=int,
        default=None,
        help="Expert-only memory override; model runs derive this from --manifest.",
    )
    parser.add_argument(
        "--candidate-seed",
        type=int,
        default=None,
        help=(
            "Exactly one explicit expert-prequalification seed or one unused "
            "authorized model-run seed. It is passed only to the verifier; "
            "inherited environment values are never accepted."
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


def verify_harbor_version(
    harbor_bin: Path,
    expected: str,
    *,
    env: Mapping[str, str] | None = None,
) -> None:
    completed = subprocess.run(
        [str(harbor_bin), "--version"],
        capture_output=True,
        text=True,
        check=False,
        env=None if env is None else dict(env),
    )
    observed = " ".join((completed.stdout or completed.stderr or "").split())
    accepted = {observed, observed.removeprefix("harbor ").strip()}
    if completed.returncode != 0 or expected not in accepted:
        raise CandidateTaskError(
            f"Harbor version mismatch: expected {expected!r}, observed {observed!r}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cmd, task_dir = build_harbor_command(args)
        approval_validation = (
            validate_expert_execution_approval(args, task_dir)
            if args.execute and args.expert_only
            else None
        )
    except CandidateTaskError as exc:
        print(f"error: {exc}")
        return 2

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Candidate task: {task_dir}")
    print("Framework: tensorcircuit")
    resolved = args._resolved_run
    expert_process_env = build_expert_subprocess_env() if args.expert_only else None
    if args.expert_only:
        print("Solver: not invoked (packaged expert verification)")
        print("Evidence class: expert prequalification; never model-hardness evidence")
        print(f"Candidate contract: {resolved['candidate_hash']}")
        print(f"Task bundle SHA-256: {resolved['task_bundle_sha256']}")
        print(f"Expert baseline SHA-256: {resolved['expert_baseline_sha256']}")
        print(f"Evaluator SHA-256: {resolved['evaluator_sha256']}")
        print(f"Container image digest: {resolved['container_image_digest']}")
    else:
        print(f"Authorization manifest: {resolved['manifest_hash']}")
        print(f"Solver model: {resolved['model']}")
        print(f"Solver agent: {resolved['solver_agent']}")
        print(f"Solver reasoning effort: {resolved['reasoning_effort']}")
        print(
            "Token admission budget: "
            f"{resolved['token_budget']} {resolved['token_budget_metric']}"
        )
        print(
            "Token enforcement: post-run admission only; final hardness is "
            "fail-closed without a hard runtime cap"
        )
        print(f"Agent wall limit: {resolved['wall_time_sec']}s")
    if args.expert_only:
        print("Verifier LLM audit: disabled (private prequalification)")
        print("Provider-call budget: 0")
        print(
            "Custody: local trusted operator only; not confidential from same-host "
            "process observers"
        )
    else:
        print(f"Verifier audit model: {resolved['audit_model']}")
    if args.expert_only:
        print("Provider credentials: not loaded")
    else:
        print(
            "Codex auth source: "
            + ("auth.json" if resolved["force_auth_json"] else "environment")
        )
    print(
        "Resource envelope: "
        f"{resolved['cpus']} CPUs, {resolved['memory_mb']} MiB memory"
    )
    print(
        "Candidate verifier seed: "
        + (
            str(resolved["protocol_seed"])
            if resolved["protocol_seed"] is not None
            else "default"
        )
    )
    print(f"Proxy policy: {resolved['proxy_policy']}")
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

    if args.expert_only:
        try:
            verify_harbor_version(
                args.harbor_bin,
                resolved["harbor_version"],
                env=expert_process_env,
            )
        except CandidateTaskError as error:
            print(f"error: expert Harbor identity check failed: {error}")
            return 2
        job_path = validate_jobs_root(args.jobs_dir) / args.job_name
        evidence_path = (
            validate_expert_evidence_root(args.expert_evidence_dir)
            / resolved["candidate_slug"]
            / f"seed-{resolved['protocol_seed']}"
        )
        if _path_lexists(job_path):
            print(f"error: refusing to reuse existing Harbor job output: {job_path}")
            return 2
        if _path_lexists(evidence_path):
            print(f"error: refusing to overwrite expert evidence: {evidence_path}")
            return 2
        try:
            original_resolved = args._resolved_run
            refreshed_cmd, refreshed_task = build_harbor_command(args)
            refreshed_resolved = args._resolved_run
            if (
                refreshed_cmd != cmd
                or refreshed_task != task_dir
                or refreshed_resolved != original_resolved
            ):
                raise CandidateTaskError(
                    "expert execution bindings changed before approval claim"
                )
            refreshed_approval = validate_expert_execution_approval(args, task_dir)
            if (
                approval_validation is None
                or refreshed_approval["approval_sha256"]
                != approval_validation["approval_sha256"]
            ):
                raise CandidateTaskError(
                    "expert execution approval changed before approval claim"
                )
            receipt_path = claim_expert_execution_approval(refreshed_approval)
        except CandidateTaskError as error:
            print(f"error: expert execution authorization failed: {error}")
            return 2
        print(f"Expert execution receipt: {receipt_path}")

    if not args.expert_only:
        try:
            verify_harbor_version(args.harbor_bin, resolved["harbor_version"])
            refreshed = authorized_model_execution(
                args.workspace.resolve(),
                args.manifest.resolve(),
                task_dir,
                TENSORCIRCUIT_PROMPT,
                resolved["protocol_seed"],
            )
            if refreshed != resolved:
                raise CandidateTaskError(
                    "authorization changed between command construction and execution"
                )
            reserve_model_trial_seed(
                args.workspace.resolve(),
                args.manifest.resolve(),
                resolved["protocol_seed"],
                f"{args.job_name}:{resolved['protocol_seed']}",
            )
        except (
            CandidateTaskError,
            FileNotFoundError,
            KeyError,
            PermissionError,
            ValueError,
        ) as error:
            print(f"error: execution authorization failed: {error}")
            return 2

    env = (
        dict(expert_process_env) if expert_process_env is not None else dict(os.environ)
    )
    # The trial seed crosses the Harbor boundary only through --verifier-env.
    # Scrubbing a coincident host export avoids accidental adapter inheritance.
    env.pop(CANDIDATE_SEED_ENV, None)
    for name in PROXY_ENV_NAMES:
        env.pop(name, None)
    if not args.expert_only:
        env["PYTHONPATH"] = str(ROOT) + (
            f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
        )
    run_options: dict[str, Any] = {
        "cwd": ROOT,
        "env": env,
        "check": False,
    }
    if args.expert_only:
        run_options["umask"] = 0o077
    completed = subprocess.run(cmd, **run_options)
    if completed.returncode != 0 or not args.expert_only:
        return completed.returncode
    try:
        evidence_path = collect_expert_prequalification_evidence(args, task_dir)
    except CandidateTaskError as error:
        print(f"error: expert run completed but evidence collection failed: {error}")
        return 2
    print(f"Expert prequalification evidence: {evidence_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
