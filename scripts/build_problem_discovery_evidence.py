#!/usr/bin/env python3
"""Build the sanitized ORBIT-Q exploratory Harbor evidence snapshot.

Only a fixed allowlist of completed local jobs is read.  The generated ledger
contains whitelisted result fields and SHA-256 digests; it never copies raw
configuration, environment, authentication, transcript, or rollout content.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "orbit_q_problem_discovery"
JOBS_DIR = REPO_ROOT / ".artifacts" / "problem-discovery" / "candidate-jobs"
LEDGER_PATH = REPORT_DIR / "empirical_evidence.json"
REPORT_PATH = REPORT_DIR / "empirical_evidence.md"
SCHEMA_PATH = REPORT_DIR / "empirical_evidence.schema.json"
SOURCE_BINDINGS_PATH = REPORT_DIR / "empirical_evidence.sources.json"
BUILDER_PATH = Path(__file__).resolve()
TASK_SNAPSHOTS_ROOT = REPO_ROOT / ".artifacts" / "problem-discovery" / "candidate-tasks"


@dataclass(frozen=True)
class RunSpec:
    problem_id: int
    evidence_role: str
    job_name: str
    trial_name: str
    solution_name: str


RUN_SPECS = (
    RunSpec(
        105,
        "expert_oracle",
        "expert-conditioned-fgs-full-audit-v2",
        "candidate-conditioned-fgs-invers__KfttN55",
        "solution_105.py",
    ),
    RunSpec(
        105,
        "model_pilot",
        "pilot-conditioned-fgs-gpt56-high-004",
        "candidate-conditioned-fgs-invers__TrwKe78",
        "solution_105.py",
    ),
    RunSpec(
        106,
        "expert_oracle",
        "expert-css-state-full-audit-v1",
        "candidate-css-state-verification__PnZ9tui",
        "solution_106.py",
    ),
    RunSpec(
        106,
        "model_pilot",
        "pilot-css-state-gpt56-high-001",
        "candidate-css-state-verification__3QBkgrB",
        "solution_106.py",
    ),
    RunSpec(
        107,
        "expert_oracle",
        "expert-memory-bounded-iqp-full-audit-v1",
        "candidate-memory-bounded-iqp__ZPkaATg",
        "solution_107.py",
    ),
    RunSpec(
        107,
        "expert_oracle",
        "expert-memory-bounded-iqp-seed-1072027",
        "candidate-memory-bounded-iqp__sGGvztq",
        "solution_107.py",
    ),
    RunSpec(
        107,
        "model_pilot",
        "pilot-memory-bounded-iqp-gpt56-high-001",
        "candidate-memory-bounded-iqp__Hei6jzq",
        "solution_107.py",
    ),
    RunSpec(
        108,
        "expert_oracle",
        "expert-conditioned-qksd-seed-1082037-full-audit-v1",
        "candidate-conditioned-qksd__aB6ZYbu",
        "solution_108.py",
    ),
    RunSpec(
        108,
        "model_pilot",
        "pilot-conditioned-qksd-gpt56-high-001",
        "candidate-conditioned-qksd__CcSxwVA",
        "solution_108.py",
    ),
    RunSpec(
        109,
        "expert_oracle",
        "expert-robust-leakage-grape-seed-1092037-full-audit-v1",
        "candidate-robust-leakage-grape__vLwDJNz",
        "solution_109.py",
    ),
    RunSpec(
        109,
        "model_pilot",
        "pilot-robust-leakage-grape-gpt56-high-001",
        "candidate-robust-leakage-grape__PCuQgso",
        "solution_109.py",
    ),
    RunSpec(
        111,
        "expert_oracle",
        "expert-truncated-mps-gradient-seed-1112037-full-audit-v1",
        "candidate-truncated-mps-gradient__A8LAvSF",
        "solution_111.py",
    ),
    RunSpec(
        111,
        "model_pilot",
        "pilot-truncated-mps-gradient-gpt56-high-001",
        "candidate-truncated-mps-gradient__4xfxSvV",
        "solution_111.py",
    ),
)


CANDIDATE_PROBLEM_IDS = tuple(sorted({spec.problem_id for spec in RUN_SPECS}))


SAFE_FUNCTIONAL_KEYS = {
    "controls_shape",
    "heldout_max_error",
    "heldout_p95_infidelity",
    "heldout_shape",
    "heldout_worst_infidelity",
    "heldout_worst_leakage",
    "instances",
    "max_drive_amplitude",
    "max_edge_amplitude",
    "max_slew_per_slice",
    "parameter_max_error",
    "parameter_shape",
    "training_rmse",
    "training_worst_infidelity",
    "training_worst_leakage",
}


SHAPE_METRICS = {"controls_shape", "heldout_shape", "parameter_shape"}
COUNT_METRICS = {"instances", "active_cuts"}
UNIT_INTERVAL_METRICS = {
    "heldout_p95_infidelity",
    "heldout_worst_infidelity",
    "heldout_worst_leakage",
    "training_worst_infidelity",
    "training_worst_leakage",
}


LLM_AUDIT_KEYS = (
    "uses_required_framework",
    "imports_other_quantum_framework",
    "uses_other_quantum_framework",
    "raw_numpy_jax_quantum_simulator_bypass",
    "hardcoded_or_hidden_answer",
    "tampers_with_tests_or_rewards",
    "exploits_evaluator_weakness",
    "faithfully_implements_problem",
    "obvious_implementation_error",
    "problem_alignment_issues",
    "confidence",
)


NETWORK_COMMAND_PATTERNS = (
    re.compile(r"(?<![\w.-])(?:curl|wget|ssh|scp|sftp|rsync)\b", re.IGNORECASE),
    re.compile(r"\bgh\s+(?:api|repo|pr|issue|release|run|workflow)\b", re.IGNORECASE),
    re.compile(r"\bgit\s+(?:clone|fetch|pull|submodule)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:pip|pip3|uv\s+pip|python\d*\s+-m\s+pip)\s+install\b", re.IGNORECASE
    ),
    re.compile(r"\b(?:npm|pnpm|yarn)\s+(?:install|add)\b", re.IGNORECASE),
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(
        r"\b(?:requests|urllib|http\.client|socket|ftplib|paramiko)\b",
        re.IGNORECASE,
    ),
)


PROTECTED_ARTIFACT_PATTERNS = (
    # Only root-level benchmark paths are protected here. Installed package
    # paths such as ``.../site-packages/tensorcircuit*/tests`` are permitted.
    re.compile(r"(?:^|[\s\"'=:(])/(?:tests|logs)(?:/|\b)"),
    re.compile(r"(?:^|[\s\"'])\.artifacts(?:/|\b)"),
    re.compile(r"candidate-jobs", re.IGNORECASE),
    re.compile(
        r"(?:reward\.json|audit-details\.json|functional-stdout\.txt)", re.IGNORECASE
    ),
    re.compile(r"(?:^|[\s\"'=:(])(?:\.\./)+(?:tests|logs)(?:/|\b)"),
    re.compile(r"\bcd\s+/\s*(?:&&|;).*?\b(?:tests|logs)(?:/|\b)", re.IGNORECASE),
    re.compile(
        r"\b(?:path|join)\s*\(\s*['\"]?/['\"]?\s*,\s*['\"]?(?:tests|logs)(?:['\"]|\s*,)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:path|purepath)\s*\(\s*['\"]/['\"]\s*\)\s*/\s*['\"](?:tests|logs)(?:/|['\"])",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:path|purepath)\s*\(\s*['\"]/['\"]\s*\)\s*\.\s*joinpath\s*\(\s*['\"](?:tests|logs)(?:/|['\"])",
        re.IGNORECASE,
    ),
)


SENSITIVE_DUMP_PATTERNS = (
    re.compile(r"\bprintenv\b", re.IGNORECASE),
    re.compile(r"/proc/(?:self|[0-9]+)/environ", re.IGNORECASE),
    re.compile(r"(?:^|[;&|]\s*)env(?:\s|$)", re.IGNORECASE),
    re.compile(r"(?:\.codex/)?auth\.json", re.IGNORECASE),
    re.compile(r"\b(?:api[_-]?key|auth[_-]?token)\b", re.IGNORECASE),
    re.compile(
        r"\bos\.environ(?:\s*\[|\.(?:get|items|keys|values)\s*\()|\bos\.getenv\b",
        re.IGNORECASE,
    ),
    re.compile(r"/proc/(?:\$\$|self|[0-9]+)/environ", re.IGNORECASE),
)


ALLOWED_MODEL_TOOL_NAMES = {"exec", "wait"}
SECRET_VALUE_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{8,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{12,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"https?://[^\s/@:]+:[^\s/@]+@", re.IGNORECASE),
    re.compile(r"\b(?:OPENAI|ANTHROPIC|AWS|GITHUB)_[A-Z0-9_]*(?:KEY|TOKEN|SECRET)\b"),
)
SECRET_KEY_PATTERN = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|"
    r"client[_-]?secret|private[_-]?key|authorization)(?:$|[_-])",
    re.IGNORECASE,
)
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
UUID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
HOST_PATH_PATTERNS = (
    re.compile(r"(?:^|[\s='\"(])/(?!/)[A-Za-z0-9._~-]+(?:/|\b)"),
    re.compile(r"(?:^|[\s='\"(])[A-Za-z]:/(?:Users|Documents|Windows)(?:/|\b)"),
    re.compile(r"(?:^|[\s='\"(])//[^/\s]+/[^/\s]+(?:/|\b)"),
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_loads(text: str) -> Any:
    return json.loads(text, parse_constant=_reject_json_constant)


def _is_number(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value))


def _sanitize_metric(name: str, value: Any) -> Any:
    if name in SHAPE_METRICS:
        if (
            type(value) is not list
            or not 1 <= len(value) <= 4
            or any(
                type(item) is not int or not 1 <= item <= 1_000_000 for item in value
            )
        ):
            raise ValueError(f"invalid shape metric {name!r}: {value!r}")
        return value
    if name in COUNT_METRICS:
        if type(value) is not int or not 0 <= value <= 1_000_000_000:
            raise ValueError(f"invalid count metric {name!r}: {value!r}")
        return value
    if not _is_number(value):
        raise ValueError(f"functional metric {name!r} must be a finite number")
    numeric = float(value)
    upper = 1.0 if name in UNIT_INTERVAL_METRICS else 1.0e15
    if not 0.0 <= numeric <= upper:
        raise ValueError(
            f"functional metric {name!r} outside [0, {upper:g}]: {numeric!r}"
        )
    return value


def _assert_sanitized(value: Any, *, location: str = "root") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise ValueError(f"invalid object key at {location}")
            if SECRET_KEY_PATTERN.search(key):
                raise ValueError(f"secret-like object key rejected at {location}")
            _assert_sanitized(key, location=f"{location}.<key>")
            _assert_sanitized(child, location=f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _assert_sanitized(child, location=f"{location}[{index}]")
        return
    if isinstance(value, str):
        if len(value) > 4_096 or any(
            ord(character) < 32 and character not in "\n\t" for character in value
        ):
            raise ValueError(f"unsafe string at {location}")
        if any(pattern.search(value) for pattern in SECRET_VALUE_PATTERNS):
            raise ValueError(f"secret-like value rejected at {location}")
        normalized = value.replace("\\", "/")
        if any(part == ".." for part in normalized.split("/")):
            raise ValueError(f"path traversal rejected at {location}")
        if "file://" in normalized.lower() or any(
            pattern.search(normalized) for pattern in HOST_PATH_PATTERNS
        ):
            raise ValueError(f"host path rejected at {location}")
        return
    if value is not None and type(value) not in {bool, int, float}:
        raise ValueError(
            f"unsupported value type at {location}: {type(value).__name__}"
        )
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"non-finite number at {location}")


def _load_json(path: Path) -> dict[str, Any]:
    value = _strict_json_loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_source_bindings() -> dict[str, Any]:
    manifest = _load_json(SOURCE_BINDINGS_PATH)
    if set(manifest) != {
        "schema_version",
        "hash_algorithm",
        "human_trace_reviews",
        "task_bindings",
    }:
        raise ValueError("unexpected source-binding manifest fields")
    if manifest["schema_version"] != 1 or manifest["hash_algorithm"] != "sha256":
        raise ValueError("unsupported source-binding manifest version")
    reviews = manifest["human_trace_reviews"]
    bindings = manifest["task_bindings"]
    if not isinstance(reviews, list) or not isinstance(bindings, list):
        raise ValueError("source-binding manifest arrays are missing")

    review_jobs: set[str] = set()
    for review in reviews:
        if not isinstance(review, dict) or set(review) != {
            "job_name",
            "reviewer_id",
            "reviewed_at",
            "reviewed_raw_rollout_sha256",
            "decision",
        }:
            raise ValueError("invalid human trace review record")
        if review["decision"] != "approved":
            raise ValueError("only approved human trace reviews may be admitted")
        if not isinstance(review["job_name"], str) or not review["job_name"]:
            raise ValueError("human trace review job name is missing")
        if not isinstance(review["reviewer_id"], str) or not review["reviewer_id"]:
            raise ValueError("human trace reviewer identity is missing")
        reviewed_at = review["reviewed_at"]
        if not isinstance(reviewed_at, str):
            raise ValueError("human trace review timestamp is missing")
        try:
            parsed_reviewed_at = datetime.fromisoformat(
                reviewed_at.replace("Z", "+00:00")
            )
        except ValueError as exc:
            raise ValueError("invalid human trace review timestamp") from exc
        if parsed_reviewed_at.tzinfo is None:
            raise ValueError("human trace review timestamp lacks timezone")
        if not SHA256_PATTERN.fullmatch(review["reviewed_raw_rollout_sha256"]):
            raise ValueError("invalid reviewed rollout digest")
        if review["job_name"] in review_jobs:
            raise ValueError("duplicate human trace review")
        review_jobs.add(review["job_name"])

    checksums: set[str] = set()
    source_snapshots: set[str] = set()
    for binding in bindings:
        required = {
            "problem_id",
            "slug",
            "task_name",
            "task_checksum",
            "source_snapshot",
            "task_source_set_sha256",
            "file_count",
            "files",
        }
        if not isinstance(binding, dict) or not required.issubset(binding):
            raise ValueError("invalid task source binding")
        if set(binding) - (
            required | {"functional_thresholds", "functional_threshold_source"}
        ):
            raise ValueError("unexpected task source binding fields")
        checksum = binding["task_checksum"]
        if not isinstance(checksum, str) or not SHA256_PATTERN.fullmatch(checksum):
            raise ValueError("invalid bound task checksum")
        if checksum in checksums:
            raise ValueError("duplicate bound task checksum")
        checksums.add(checksum)
        source_snapshot = binding["source_snapshot"]
        expected_prefix = ".artifacts/problem-discovery/candidate-tasks/"
        if (
            not isinstance(source_snapshot, str)
            or not source_snapshot.startswith(expected_prefix)
            or any(part in {"", ".", ".."} for part in source_snapshot.split("/"))
            or source_snapshot in source_snapshots
        ):
            raise ValueError("invalid or duplicate task source snapshot locator")
        source_snapshots.add(source_snapshot)
        files = binding["files"]
        if not isinstance(files, dict) or binding["file_count"] != len(files):
            raise ValueError("bound task file count mismatch")
        for path, digest in files.items():
            if (
                not isinstance(path, str)
                or path.startswith(("/", ".artifacts/"))
                or any(part in {"", ".", ".."} for part in path.split("/"))
                or not isinstance(digest, str)
                or not SHA256_PATTERN.fullmatch(digest)
            ):
                raise ValueError(f"invalid frozen task file binding: {path!r}")
        source_text = "".join(
            f"{path}\0{digest}\n" for path, digest in sorted(files.items())
        )
        source_digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        if source_digest != binding["task_source_set_sha256"]:
            raise ValueError(f"task source-set digest mismatch for {checksum}")
        thresholds = binding.get("functional_thresholds", {})
        if not isinstance(thresholds, dict):
            raise ValueError("functional thresholds must be an object")
        for metric, threshold in thresholds.items():
            if (
                metric not in SAFE_FUNCTIONAL_KEYS
                or not _is_number(threshold)
                or threshold < 0
            ):
                raise ValueError(f"invalid functional threshold {metric!r}")
        threshold_source = binding.get("functional_threshold_source")
        if bool(thresholds) != (threshold_source is not None):
            raise ValueError("functional thresholds require a frozen source mapping")
        if threshold_source is not None:
            if not isinstance(threshold_source, dict) or set(threshold_source) != {
                "file",
                "metric_to_config_key",
            }:
                raise ValueError("invalid functional threshold source")
            source_file = threshold_source["file"]
            metric_mapping = threshold_source["metric_to_config_key"]
            if source_file not in files or not isinstance(metric_mapping, dict):
                raise ValueError("functional threshold source is not file-bound")
            if set(metric_mapping) != set(thresholds) or any(
                not isinstance(source_key, str) or not source_key
                for source_key in metric_mapping.values()
            ):
                raise ValueError("functional threshold source mapping is incomplete")

    _assert_sanitized(manifest, location="source_bindings")
    return manifest


def _validate_raw_task_source_bindings(source_bindings: dict[str, Any]) -> None:
    snapshots_root = TASK_SNAPSHOTS_ROOT.resolve()
    for binding in source_bindings["task_bindings"]:
        source_dir = (REPO_ROOT / binding["source_snapshot"]).resolve()
        try:
            source_dir.relative_to(snapshots_root)
        except ValueError as exc:
            raise ValueError(
                "task source snapshot escapes the raw snapshot root"
            ) from exc
        if not source_dir.is_dir():
            raise FileNotFoundError(source_dir)

        actual_files: dict[str, str] = {}
        for path in sorted(source_dir.rglob("*")):
            relative = path.relative_to(source_dir)
            if path.is_symlink():
                raise ValueError(f"task source snapshot contains a symlink: {path}")
            if "__pycache__" in relative.parts or path.suffix == ".pyc":
                continue
            if path.is_file():
                actual_files[relative.as_posix()] = _sha256(path)
        if actual_files != binding["files"]:
            raise ValueError(
                "raw task source snapshot differs from the committed binding for "
                f"{binding['task_checksum']}"
            )
        threshold_source = binding.get("functional_threshold_source")
        if threshold_source is not None:
            source_path = source_dir / threshold_source["file"]
            syntax = ast.parse(source_path.read_text(encoding="utf-8"))
            literal_values: dict[str, set[int | float]] = {}
            for node in ast.walk(syntax):
                if not isinstance(node, ast.Dict):
                    continue
                for key_node, value_node in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(key_node, ast.Constant)
                        and isinstance(key_node.value, str)
                        and isinstance(value_node, ast.Constant)
                        and type(value_node.value) in {int, float}
                    ):
                        literal_values.setdefault(key_node.value, set()).add(
                            value_node.value
                        )
            for metric, source_key in threshold_source["metric_to_config_key"].items():
                expected = binding["functional_thresholds"][metric]
                if literal_values.get(source_key) != {expected}:
                    raise ValueError(
                        f"functional threshold {metric!r} does not match frozen source"
                    )


def _binding_index(source_bindings: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        binding["task_checksum"]: binding
        for binding in source_bindings["task_bindings"]
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _repo_relative(path),
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _functional_evidence(
    text: str, problem_id: int, binding: dict[str, Any]
) -> dict[str, Any]:
    thresholds = binding.get("functional_thresholds", {})
    safe_metrics: dict[str, Any] = {}
    seed: int | None = None
    case_digest: str | None = None

    for line in text.splitlines():
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            payload = _strict_json_loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(payload, dict):
            continue
        if isinstance(payload.get("seed"), int):
            seed = payload["seed"]
        if isinstance(payload.get("case_digest"), str):
            case_digest = payload["case_digest"]
        for key in sorted(SAFE_FUNCTIONAL_KEYS):
            if key in payload:
                safe_metrics[key] = _sanitize_metric(key, payload[key])

    seed_match = re.search(r"Case seed:\s*([0-9]+)", text)
    if seed is None and seed_match:
        seed = int(seed_match.group(1))
    digest_match = re.search(r"case digest:\s*([0-9a-fA-F]+)", text)
    if case_digest is None and digest_match:
        case_digest = digest_match.group(1).lower()

    error_match = re.search(r"Maximum scaled-amplitude error:\s*([0-9.eE+-]+)", text)
    if error_match:
        safe_metrics["maximum_scaled_amplitude_error"] = _sanitize_metric(
            "maximum_scaled_amplitude_error", float(error_match.group(1))
        )
    widths_match = re.search(r"Measured induced widths:\s*\[([^]]+)\]", text)
    if widths_match:
        widths = [int(item.strip()) for item in widths_match.group(1).split(",")]
        if not widths or any(width < 0 or width > 1_000_000 for width in widths):
            raise ValueError("invalid measured induced widths")
        safe_metrics["measured_induced_widths"] = widths
    ritz_match = re.search(r"Maximum Ritz-energy error:\s*([0-9.eE+-]+)", text)
    if ritz_match:
        safe_metrics["maximum_ritz_energy_error"] = _sanitize_metric(
            "maximum_ritz_energy_error", float(ritz_match.group(1))
        )
    moment_match = re.search(r"Maximum control-moment error:\s*([0-9.eE+-]+)", text)
    if moment_match:
        safe_metrics["maximum_control_moment_error"] = _sanitize_metric(
            "maximum_control_moment_error", float(moment_match.group(1))
        )
    energy_match = re.search(r"Maximum energy error:\s*([0-9.eE+-]+)", text)
    if energy_match:
        safe_metrics["maximum_energy_error"] = _sanitize_metric(
            "maximum_energy_error", float(energy_match.group(1))
        )
    gradient_match = re.search(r"Maximum gradient error:\s*([0-9.eE+-]+)", text)
    if gradient_match:
        safe_metrics["maximum_gradient_error"] = _sanitize_metric(
            "maximum_gradient_error", float(gradient_match.group(1))
        )
    cuts_match = re.search(
        r"Active cuts:\s*([0-9]+); minimum relative cut gap:\s*([0-9.eE+-]+)",
        text,
    )
    if cuts_match:
        safe_metrics["active_cuts"] = _sanitize_metric(
            "active_cuts", int(cuts_match.group(1))
        )
        safe_metrics["minimum_relative_cut_gap"] = _sanitize_metric(
            "minimum_relative_cut_gap", float(cuts_match.group(2))
        )

    if case_digest is None:
        digest_scope = "not_emitted"
    elif len(case_digest) == 64:
        digest_scope = "full_sha256"
    elif len(case_digest) == 16:
        digest_scope = "reported_16_hex_digest"
    else:
        digest_scope = "other_emitted_digest"

    evaluation_checks = []
    if case_digest is not None and not re.fullmatch(r"[0-9a-f]+", case_digest):
        raise ValueError(f"invalid case digest for problem {problem_id}")

    for metric, threshold in sorted(thresholds.items()):
        if metric not in safe_metrics:
            raise ValueError(
                f"functional metric {metric!r} missing for problem {problem_id}"
            )
        observed = safe_metrics[metric]
        evaluation_checks.append(
            {
                "metric": metric,
                "observed": observed,
                "comparator": "<=",
                "threshold": threshold,
                "passed": observed <= threshold,
            }
        )

    threshold_source = binding.get("functional_threshold_source")
    evaluation_check_source = None
    if threshold_source is not None:
        evaluation_check_source = (
            f"{binding['source_snapshot']}/{threshold_source['file']}; "
            "selected task content is bound by trial.task_checksum"
        )

    return {
        "overall_pass_marker": "Overall: PASS" in text,
        "seed": seed,
        "case_digest": case_digest,
        "case_digest_scope": digest_scope,
        "metrics": safe_metrics,
        "evaluation_checks": evaluation_checks,
        "evaluation_check_source": evaluation_check_source,
    }


def _tool_calls(rollout_path: Path) -> list[tuple[str, str]]:
    calls: list[tuple[str, str]] = []
    for line in rollout_path.read_text(encoding="utf-8").splitlines():
        record = _strict_json_loads(line)
        if not isinstance(record, dict):
            raise ValueError(f"invalid rollout record: {rollout_path}")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        if payload.get("type") not in {"custom_tool_call", "function_call"}:
            continue
        name = payload.get("name")
        if not isinstance(name, str):
            name = "unknown"
        raw_input = payload.get("input", payload.get("arguments", ""))
        if not isinstance(raw_input, str):
            raw_input = json.dumps(raw_input, sort_keys=True)
        calls.append((name, raw_input))
    return calls


def _scan_tool_calls(calls: list[tuple[str, str]]) -> dict[str, int]:
    disallowed_tool_matches = sum(
        1 for name, _ in calls if name not in ALLOWED_MODEL_TOOL_NAMES
    )
    network_matches = sum(
        1
        for name, raw in calls
        if any(
            pattern.search(name) or pattern.search(raw)
            for pattern in NETWORK_COMMAND_PATTERNS
        )
    )
    protected_matches = sum(
        1
        for _, raw in calls
        if any(pattern.search(raw) for pattern in PROTECTED_ARTIFACT_PATTERNS)
    )
    sensitive_matches = sum(
        1
        for _, raw in calls
        if any(pattern.search(raw) for pattern in SENSITIVE_DUMP_PATTERNS)
    )
    return {
        "disallowed_tool_matches": disallowed_tool_matches,
        "network_matches": network_matches,
        "protected_matches": protected_matches,
        "sensitive_matches": sensitive_matches,
    }


def _human_trace_review(
    job_name: str, rollout_sha256: str, source_bindings: dict[str, Any]
) -> dict[str, Any]:
    matching = [
        review
        for review in source_bindings["human_trace_reviews"]
        if review["job_name"] == job_name
    ]
    if not matching:
        return {
            "status": "none",
            "reviewer_id": None,
            "reviewed_at": None,
            "reviewed_raw_rollout_sha256": None,
            "decision": None,
        }
    review = matching[0]
    if review["reviewed_raw_rollout_sha256"] != rollout_sha256:
        raise ValueError(f"human trace review digest mismatch for {job_name}")
    return {
        "status": "approved",
        "reviewer_id": review["reviewer_id"],
        "reviewed_at": review["reviewed_at"],
        "reviewed_raw_rollout_sha256": rollout_sha256,
        "decision": review["decision"],
    }


def _model_trace_evidence(
    trial_dir: Path,
    job_name: str,
    source_bindings: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    rollout_paths = sorted((trial_dir / "agent" / "sessions").glob("**/*.jsonl"))
    if len(rollout_paths) != 1:
        raise ValueError(
            f"expected one raw rollout for {trial_dir}, got {len(rollout_paths)}"
        )
    rollout_path = rollout_paths[0]

    trajectory = _load_json(trajectory_path)
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise ValueError(f"trajectory steps missing: {trajectory_path}")

    calls = _tool_calls(rollout_path)
    tool_counts = Counter(name for name, _ in calls)
    scan = _scan_tool_calls(calls)
    installed_package_test_search_observed = any(
        "/usr/local/" in raw and "/tests" in raw for _, raw in calls
    )
    if any(scan.values()):
        raise ValueError(
            "refusing to attest a clean trace: "
            f"disallowed_tools={scan['disallowed_tool_matches']}, "
            f"network={scan['network_matches']}, protected={scan['protected_matches']}, "
            f"sensitive={scan['sensitive_matches']} "
            f"for {rollout_path}"
        )

    rollout_sha256 = _sha256(rollout_path)
    human_review = _human_trace_review(job_name, rollout_sha256, source_bindings)
    evidence = {
        "status": (
            "automated_scan_pass_human_reviewed"
            if human_review["status"] == "approved"
            else "automated_scan_pass_human_review_none"
        ),
        "automated_scan_status": "pass",
        "review_method": (
            "Deterministic allowlist and marker scans of recorded rollout tool-call names and "
            "inputs; this automated scan is not a human attestation and raw content is not copied."
        ),
        "trajectory_step_count": len(steps),
        "tool_call_count": len(calls),
        "tool_call_counts_by_name": dict(sorted(tool_counts.items())),
        "allowed_tool_names": sorted(ALLOWED_MODEL_TOOL_NAMES),
        "disallowed_tool_name_scan": "no_matches_observed",
        "network_command_scan": "no_matches_observed",
        "protected_benchmark_artifact_scan": "no_disallowed_access_observed",
        "credential_or_environment_dump_scan": "no_matches_observed",
        "allowed_read_scope_observed": [
            "self-authored solver working files",
            "installed TensorCircuit and dependency source/examples/tests",
        ],
        "installed_package_test_search_observed": installed_package_test_search_observed,
        "automated_scan_raw_rollout_sha256": rollout_sha256,
        "human_review": human_review,
        "unassessed": [
            "pretraining contamination",
            "packet-level network activity",
            "subprocess or access not represented in recorded tool-call inputs",
            "cryptographic completeness of the raw event record",
        ],
        "cryptographically_signed_command_attestation": False,
    }
    artifacts = {
        "trajectory": _artifact(trajectory_path),
    }
    return evidence, artifacts


def _oracle_trace_evidence(
    trial_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    transcript_path = trial_dir / "agent" / "oracle.txt"
    evidence = {
        "status": "not_applicable_oracle_copy",
        "automated_scan_status": "not_applicable",
        "review_method": "Oracle copied the packaged expert solution; no model tool-call trajectory exists.",
        "trajectory_step_count": 0,
        "tool_call_count": 0,
        "tool_call_counts_by_name": {},
        "allowed_tool_names": [],
        "disallowed_tool_name_scan": "not_applicable_no_model_trajectory",
        "network_command_scan": "not_applicable_no_model_trajectory",
        "protected_benchmark_artifact_scan": "not_applicable_no_model_trajectory",
        "credential_or_environment_dump_scan": "not_applicable_no_model_trajectory",
        "allowed_read_scope_observed": [],
        "installed_package_test_search_observed": False,
        "automated_scan_raw_rollout_sha256": None,
        "human_review": {
            "status": "none",
            "reviewer_id": None,
            "reviewed_at": None,
            "reviewed_raw_rollout_sha256": None,
            "decision": None,
        },
        "unassessed": [],
        "cryptographically_signed_command_attestation": False,
    }
    if not transcript_path.is_file():
        raise FileNotFoundError(transcript_path)
    return evidence, {"oracle_transcript": _artifact(transcript_path)}


def _build_run(
    spec: RunSpec,
    source_bindings: dict[str, Any],
    bindings_by_checksum: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    job_dir = JOBS_DIR / spec.job_name
    trial_dir = job_dir / spec.trial_name
    job_result_path = job_dir / "result.json"
    trial_result_path = trial_dir / "result.json"
    reward_path = trial_dir / "verifier" / "reward.json"
    functional_path = trial_dir / "verifier" / "functional-stdout.txt"
    audit_path = trial_dir / "verifier" / "audit-details.json"
    solution_path = trial_dir / "artifacts" / "root" / spec.solution_name
    manifest_path = trial_dir / "artifacts" / "manifest.json"
    config_path = trial_dir / "config.json"

    job_result = _load_json(job_result_path)
    trial_result = _load_json(trial_result_path)
    reward = _load_json(reward_path)
    audit_details = _load_json(audit_path)
    config = _load_json(config_path)
    functional_text = functional_path.read_text(encoding="utf-8")

    if reward.get("problem_id") != spec.problem_id:
        raise ValueError(f"problem mismatch for {spec.job_name}")
    if trial_result.get("trial_name") != spec.trial_name:
        raise ValueError(f"trial mismatch for {spec.job_name}")
    if job_result.get("n_total_trials") != 1:
        raise ValueError(f"expected one trial for {spec.job_name}")
    job_stats = job_result.get("stats")
    if not isinstance(job_stats, dict):
        raise ValueError(f"job stats missing for {spec.job_name}")
    if (
        job_stats.get("n_completed_trials") != 1
        or job_stats.get("n_errored_trials") != 0
        or job_stats.get("n_retries") != 0
    ):
        raise ValueError(f"job did not complete cleanly for {spec.job_name}")

    task_checksum = trial_result.get("task_checksum")
    binding = bindings_by_checksum.get(task_checksum)
    if binding is None:
        raise ValueError(
            f"unbound task checksum for {spec.job_name}: {task_checksum!r}"
        )
    if binding["problem_id"] != spec.problem_id or binding[
        "task_name"
    ] != trial_result.get("task_name"):
        raise ValueError(f"task source binding mismatch for {spec.job_name}")

    static = audit_details.get("static")
    verifier_functional = audit_details.get("functional")
    source_audit = audit_details.get("audit")
    if (
        not isinstance(static, dict)
        or not isinstance(verifier_functional, dict)
        or not isinstance(source_audit, dict)
    ):
        raise ValueError(f"audit sections missing for {spec.job_name}")

    environment = config.get("environment")
    verifier = config.get("verifier")
    if not isinstance(environment, dict) or not isinstance(verifier, dict):
        raise ValueError(f"protocol config missing for {spec.job_name}")
    environment_kwargs = environment.get("kwargs")
    verifier_kwargs = verifier.get("kwargs")
    if not isinstance(environment_kwargs, dict) or not isinstance(
        verifier_kwargs, dict
    ):
        raise ValueError(f"protocol kwargs missing for {spec.job_name}")

    agent_info = trial_result.get("agent_info")
    if not isinstance(agent_info, dict):
        raise ValueError(f"agent info missing for {spec.job_name}")
    model_info = agent_info.get("model_info")
    agent_config = config.get("agent")

    if spec.evidence_role == "model_pilot":
        if not isinstance(agent_config, dict) or not isinstance(model_info, dict):
            raise ValueError(f"model metadata missing for {spec.job_name}")
        agent_kwargs = agent_config.get("kwargs")
        if not isinstance(agent_kwargs, dict):
            raise ValueError(f"agent kwargs missing for {spec.job_name}")
        model_name = agent_config.get("model_name")
        reasoning_effort = agent_kwargs.get("reasoning_effort")
        solver_adapter = agent_config.get("import_path")
        if (
            agent_info.get("name") != "codex"
            or model_info.get("name") != model_name
            or not all(
                isinstance(value, str) and value
                for value in (model_name, reasoning_effort, solver_adapter)
            )
        ):
            raise ValueError(f"model identity drift for {spec.job_name}")
        trace_evidence, trace_artifacts = _model_trace_evidence(
            trial_dir, spec.job_name, source_bindings
        )
    else:
        if agent_info.get("name") != "oracle" or model_info is not None:
            raise ValueError(f"oracle identity drift for {spec.job_name}")
        model_name = None
        reasoning_effort = None
        solver_adapter = "harbor.agents.oracle:Oracle"
        trace_evidence, trace_artifacts = _oracle_trace_evidence(trial_dir)

    proxy_keys = {"HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"}
    verifier_env = verifier.get("env") if isinstance(verifier.get("env"), dict) else {}
    agent_env = (
        agent_config.get("env")
        if isinstance(agent_config, dict) and isinstance(agent_config.get("env"), dict)
        else {}
    )

    artifacts: dict[str, dict[str, Any]] = {
        "job_result": _artifact(job_result_path),
        "trial_result": _artifact(trial_result_path),
        "reward": _artifact(reward_path),
        "functional_stdout": _artifact(functional_path),
        "audit_details": _artifact(audit_path),
        "solution": _artifact(solution_path),
        "artifact_manifest": _artifact(manifest_path),
        "redacted_config": _artifact(config_path),
        **trace_artifacts,
    }

    functional_evidence = _functional_evidence(
        functional_text,
        spec.problem_id,
        binding,
    )
    functional_evidence.update(
        {
            "verifier_functional_passed": verifier_functional.get("functional_passed"),
            "verifier_functional_return_code": verifier_functional.get(
                "functional_return_code"
            ),
        }
    )
    if (
        functional_evidence["overall_pass_marker"]
        != functional_evidence["verifier_functional_passed"]
    ):
        raise ValueError(f"functional pass evidence disagrees for {spec.job_name}")

    expected_pairs = (
        (reward.get("functional_score"), verifier_functional.get("functional_score")),
        (reward.get("runtime_score"), verifier_functional.get("runtime_score")),
        (reward.get("runtime_sec"), verifier_functional.get("runtime_sec")),
        (reward.get("static_policy_score"), static.get("static_policy_score")),
        (reward.get("llm_audit_score"), source_audit.get("llm_audit_score")),
    )
    if any(left != right for left, right in expected_pairs):
        raise ValueError(f"score sources disagree for {spec.job_name}")
    functional_passed = functional_evidence["verifier_functional_passed"]
    if reward.get("functional_score") != (
        1.0 if functional_passed else 0.0
    ) or verifier_functional.get("functional_return_code") != (
        0 if functional_passed else 1
    ):
        raise ValueError(f"functional status disagrees for {spec.job_name}")
    expected_reward = (
        reward.get("functional_score")
        * reward.get("static_policy_score")
        * reward.get("llm_audit_score")
    )
    if reward.get("reward") != expected_reward:
        raise ValueError(f"compound reward disagrees for {spec.job_name}")

    run = {
        "evidence_role": spec.evidence_role,
        "operator_attested": False,
        "protocol_qualified_hardness_claim": False,
        "job": {
            "name": spec.job_name,
            "id": job_result.get("id"),
            "n_total_trials": job_result.get("n_total_trials"),
            "n_completed_trials": job_stats.get("n_completed_trials"),
            "n_errored_trials": job_stats.get("n_errored_trials"),
            "n_retries": job_stats.get("n_retries"),
        },
        "trial": {
            "name": spec.trial_name,
            "id": trial_result.get("id"),
            "task_name": trial_result.get("task_name"),
            "task_checksum": trial_result.get("task_checksum"),
            "started_at": trial_result.get("started_at"),
            "finished_at": trial_result.get("finished_at"),
        },
        "task_source_binding": {
            "task_checksum": binding["task_checksum"],
            "task_source_set_sha256": binding["task_source_set_sha256"],
            "file_count": binding["file_count"],
        },
        "agent": {
            "kind": agent_info.get("name"),
            "version": agent_info.get("version"),
            "solver_adapter": solver_adapter,
            "model": model_name,
            "reasoning_effort": reasoning_effort,
        },
        "protocol": {
            "framework": environment_kwargs.get("framework"),
            "image_tag": environment_kwargs.get("docker_image"),
            "environment_adapter": environment.get("import_path"),
            "override_cpus": environment.get("override_cpus"),
            "override_memory_mb": environment.get("override_memory_mb"),
            "verifier_adapter": verifier.get("import_path"),
            "audit_model": verifier_kwargs.get("audit_model"),
            "proxy_bridge_configured": bool(
                proxy_keys.intersection(agent_env)
                or proxy_keys.intersection(verifier_env)
            ),
            "network_isolation": False,
        },
        "functional_evidence": functional_evidence,
        "scores": {
            "compound_reward": reward.get("reward"),
            "functional_score": reward.get("functional_score"),
            "runtime_score": reward.get("runtime_score"),
            "functional_evaluator_runtime_sec": reward.get("runtime_sec"),
            "static_policy_score": reward.get("static_policy_score"),
            "llm_audit_score": reward.get("llm_audit_score"),
        },
        "static_policy": {
            "framework": static.get("framework"),
            "line_count": static.get("line_count"),
            "max_lines": static.get("max_lines"),
            "imports": static.get("imports"),
            "forbidden_imports": static.get("forbidden_imports"),
            "cheating_hits": static.get("cheating_hits"),
            "raw_simulator_hits": static.get("raw_simulator_hits"),
            "framework_score": static.get("framework_score"),
            "no_forbidden_framework_score": static.get("no_forbidden_framework_score"),
            "no_raw_simulator_bypass_score": static.get(
                "no_raw_simulator_bypass_score"
            ),
            "no_static_cheating_score": static.get("no_static_cheating_score"),
        },
        "llm_source_audit": {key: source_audit.get(key) for key in LLM_AUDIT_KEYS},
        "contamination_and_command_audit": trace_evidence,
        "artifacts": artifacts,
    }
    _assert_sanitized(run, location=f"run[{spec.job_name}]")
    return run


def _common_protocol_facts(runs: list[dict[str, Any]]) -> dict[str, Any]:
    def unique(field: str, selected: list[dict[str, Any]] | None = None) -> Any:
        selected = runs if selected is None else selected
        values = {run["protocol"][field] for run in selected}
        if len(values) != 1:
            raise ValueError(f"selected runs disagree on protocol.{field}: {values}")
        return values.pop()

    model_runs = [run for run in runs if run["evidence_role"] == "model_pilot"]

    def unique_agent(field: str) -> Any:
        values = {run["agent"][field] for run in model_runs}
        if len(values) != 1:
            raise ValueError(f"selected model runs disagree on agent.{field}: {values}")
        value = values.pop()
        if not isinstance(value, str) or not value:
            raise ValueError(f"missing model agent.{field}")
        return value

    version_matches: set[str] = set()
    for run in runs:
        trajectory = run["artifacts"].get("trajectory")
        if not trajectory:
            continue
        text = (REPO_ROOT / trajectory["path"]).read_text(encoding="utf-8")
        version_matches.update(re.findall(r"1\.7\.0\.dev[0-9]+", text))
    if len(version_matches) != 1:
        raise ValueError(
            f"expected one observed TensorCircuit version, got {version_matches}"
        )

    digest_source = REPORT_DIR / "blueprints" / "mixed_sld_qfim" / "blueprint.json"
    digest_record = _load_json(digest_source)
    validation = digest_record.get("validation")
    if not isinstance(validation, dict):
        raise ValueError(f"validation metadata missing: {digest_source}")

    return {
        "framework": unique("framework"),
        "image_tag": unique("image_tag"),
        "environment_adapter": unique("environment_adapter"),
        "override_cpus": unique("override_cpus"),
        "override_memory_mb": unique("override_memory_mb"),
        "verifier_adapter": unique("verifier_adapter"),
        "audit_model": unique("audit_model"),
        "model_pilot_solver_adapter": unique_agent("solver_adapter"),
        "model_pilot_model": unique_agent("model"),
        "model_pilot_reasoning_effort": unique_agent("reasoning_effort"),
        "codex_cli_version": unique_agent("version"),
        "tensorcircuit_version_observed_in_pilot_trajectories": version_matches.pop(),
        "network_policy": {
            "proxy_bridge_configured": all(
                run["protocol"]["proxy_bridge_configured"] for run in runs
            ),
            "network_isolation": False,
            "note": (
                "Selected configs contained proxy bridge keys. Values are excluded. "
                "The command audit found no network-command matches in admitted model rollouts."
            ),
        },
        "image_digest_observation": {
            "image_tag": validation.get("image"),
            "sha256": validation.get("image_digest"),
            "source": _repo_relative(digest_source),
            "source_record_sha256": _sha256(digest_source),
            "selected_harbor_results_bind_digest": False,
            "note": (
                "This same-tag digest is recorded in local blueprint metadata, not in the "
                "selected Harbor result records. Treat the tag as run-bound and the digest "
                "as a qualified local observation only."
            ),
        },
    }


def _run_is_full_pass(run: dict[str, Any]) -> bool:
    scores = run["scores"]
    functional = run["functional_evidence"]
    return (
        scores["compound_reward"] == 1.0
        and scores["functional_score"] == 1.0
        and scores["static_policy_score"] == 1.0
        and scores["llm_audit_score"] == 1.0
        and functional["overall_pass_marker"] is True
        and functional["verifier_functional_passed"] is True
        and functional["verifier_functional_return_code"] == 0
    )


def _run_is_substantive_functional_failure(run: dict[str, Any]) -> bool:
    scores = run["scores"]
    functional = run["functional_evidence"]
    audit = run["llm_source_audit"]
    return (
        scores["compound_reward"] == 0.0
        and scores["functional_score"] == 0.0
        and scores["static_policy_score"] == 1.0
        and scores["llm_audit_score"] == 1.0
        and functional["overall_pass_marker"] is False
        and functional["verifier_functional_passed"] is False
        and functional["verifier_functional_return_code"] != 0
        and audit["faithfully_implements_problem"] is True
        and audit["obvious_implementation_error"] is False
        and audit["problem_alignment_issues"] == []
    )


def _pairing_evidence(
    expert_runs: list[dict[str, Any]], model_run: dict[str, Any]
) -> dict[str, Any]:
    task_matches = [
        run
        for run in expert_runs
        if run["trial"]["task_checksum"] == model_run["trial"]["task_checksum"]
    ]
    model_seed = model_run["functional_evidence"]["seed"]
    model_digest = model_run["functional_evidence"]["case_digest"]
    seed_matches = (
        [run for run in expert_runs if run["functional_evidence"]["seed"] == model_seed]
        if model_seed is not None
        else []
    )
    case_matches = (
        [
            run
            for run in expert_runs
            if run["functional_evidence"]["seed"] == model_seed
            and run["functional_evidence"]["case_digest"] == model_digest
        ]
        if model_seed is not None and model_digest is not None
        else []
    )
    return {
        "task_checksum_match": bool(task_matches),
        "seed_match": bool(seed_matches) if model_seed is not None else None,
        "case_digest_match": bool(case_matches) if model_digest is not None else None,
        "task_matched_expert_trial_ids": [run["trial"]["id"] for run in task_matches],
        "case_matched_expert_trial_ids": [run["trial"]["id"] for run in case_matches],
    }


def _derive_candidate(
    problem_id: int,
    slug: str,
    expert_runs: list[dict[str, Any]],
    model_pilots: list[dict[str, Any]],
) -> dict[str, Any]:
    if not expert_runs or any(not _run_is_full_pass(run) for run in expert_runs):
        raise ValueError(f"candidate {problem_id} lacks a passing expert oracle")
    if len(model_pilots) != 1:
        raise ValueError(
            f"candidate {problem_id} must have exactly one admitted model pilot"
        )
    model_run = model_pilots[0]
    pairing = _pairing_evidence(expert_runs, model_run)
    model_label = (
        f"{model_run['agent']['model']}/{model_run['agent']['reasoning_effort']}"
    )
    digest_scope = model_run["functional_evidence"]["case_digest_scope"]
    caveats: list[str] = []

    if pairing["task_checksum_match"]:
        pairing_text = "the same frozen task checksum"
    elif pairing["case_digest_match"]:
        pairing_text = "the same reported seed and case digest as an expert run, but a different task checksum"
        caveats.append(
            "The model and case-matched expert runs have different Harbor task checksums; "
            "this is a same-reported-case comparison, not an immutable task match."
        )
    else:
        raise ValueError(
            f"candidate {problem_id} has no defensible expert/model pairing"
        )

    if digest_scope == "not_emitted":
        caveats.append(
            "The paired functional output did not emit a case digest; task checksum and any "
            "reported seed are the available case-binding evidence."
        )
    elif digest_scope == "reported_16_hex_digest":
        caveats.append("The emitted 16-hex case digest is not a full SHA-256 digest.")

    if _run_is_full_pass(model_run):
        classification = "solved_hardness_blocking"
        hardness_blocker = True
        conclusion = (
            f"The expert oracle and one {model_label} pilot passed {pairing_text}. "
            "This admitted exploratory solve blocks a claim that this frozen task is unsolved "
            "by the target model."
        )
    elif _run_is_substantive_functional_failure(model_run):
        classification = "pilot_hardness_signal"
        hardness_blocker = False
        failed_checks = [
            check["metric"]
            for check in model_run["functional_evidence"]["evaluation_checks"]
            if not check["passed"]
        ]
        if not failed_checks:
            raise ValueError(
                f"candidate {problem_id} failure has no bound functional threshold failure"
            )
        conclusion = (
            f"The expert oracle passed, while one {model_label} pilot on {pairing_text} "
            f"failed the bound functional checks {', '.join(failed_checks)} after passing "
            "static policy and source audit. This is an exploratory substantive "
            "optimization-performance signal only."
        )
        caveats.extend(
            [
                "One failed pilot does not estimate pass rate or confirm membership in a protocol-qualified hard set.",
                "The automated command scan has no hash-bound human trace review in this snapshot.",
            ]
        )
    else:
        raise ValueError(
            f"candidate {problem_id} model outcome is not admissible evidence"
        )

    return {
        "problem_id": problem_id,
        "slug": slug,
        "classification": classification,
        "hardness_blocker": hardness_blocker,
        "conclusion": conclusion,
        "evidence_caveats": caveats,
        "pairing_evidence": pairing,
        "expert_runs": expert_runs,
        "model_pilots": model_pilots,
    }


def _source_artifact_rows(
    candidates: list[dict[str, Any]],
    frozen_protocol_facts: dict[str, Any],
    integrity: dict[str, Any],
) -> list[tuple[str, str]]:
    rows = sorted(
        (artifact["path"], artifact["sha256"])
        for candidate in candidates
        for run in [*candidate["expert_runs"], *candidate["model_pilots"]]
        for artifact in run["artifacts"].values()
    )
    rows.extend(
        (
            f"redacted-raw-rollout:{run['job']['name']}",
            run["contamination_and_command_audit"]["automated_scan_raw_rollout_sha256"],
        )
        for candidate in candidates
        for run in candidate["model_pilots"]
    )
    digest_observation = frozen_protocol_facts["image_digest_observation"]
    rows.append(
        (
            f"supporting-protocol-record:{digest_observation['source']}",
            digest_observation["source_record_sha256"],
        )
    )
    for name, artifact in sorted(integrity.items()):
        rows.append((f"committed-{name}:{artifact['path']}", artifact["sha256"]))
    rows.sort()
    if len(rows) != len({path for path, _ in rows}):
        raise ValueError("duplicate source artifact locator")
    return rows


def _source_artifact_set_sha256(
    candidates: list[dict[str, Any]],
    frozen_protocol_facts: dict[str, Any],
    integrity: dict[str, Any],
) -> str:
    rows = _source_artifact_rows(candidates, frozen_protocol_facts, integrity)
    text = "".join(f"{path}\0{digest}\n" for path, digest in rows)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _evidence_payload_sha256(ledger: dict[str, Any]) -> str:
    payload = dict(ledger)
    payload.pop("evidence_payload_sha256", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_ledger() -> dict[str, Any]:
    source_bindings = _load_source_bindings()
    _validate_raw_task_source_bindings(source_bindings)
    bindings_by_checksum = _binding_index(source_bindings)
    runs = [
        _build_run(spec, source_bindings, bindings_by_checksum) for spec in RUN_SPECS
    ]
    common_protocol = _common_protocol_facts(runs)
    candidate_rows: list[dict[str, Any]] = []
    for problem_id in CANDIDATE_PROBLEM_IDS:
        admitted_job_names = {
            spec.job_name for spec in RUN_SPECS if spec.problem_id == problem_id
        }
        selected = [run for run in runs if run["job"]["name"] in admitted_job_names]
        expert_runs = [
            run for run in selected if run["evidence_role"] == "expert_oracle"
        ]
        model_pilots = [
            run for run in selected if run["evidence_role"] == "model_pilot"
        ]
        slugs = {
            binding["slug"]
            for binding in source_bindings["task_bindings"]
            if binding["problem_id"] == problem_id
        }
        if len(slugs) != 1:
            raise ValueError(f"candidate {problem_id} has ambiguous source bindings")
        candidate_rows.append(
            _derive_candidate(problem_id, slugs.pop(), expert_runs, model_pilots)
        )

    integrity = {
        "builder": _artifact(BUILDER_PATH),
        "schema": _artifact(SCHEMA_PATH),
        "source_binding_manifest": _artifact(SOURCE_BINDINGS_PATH),
    }
    artifact_set_sha256 = _source_artifact_set_sha256(
        candidate_rows, common_protocol, integrity
    )
    source_cutoff = max(run["trial"]["finished_at"] for run in runs)
    model_scope = (
        f"Exactly one completed {common_protocol['model_pilot_model']}/"
        f"{common_protocol['model_pilot_reasoning_effort']} pilot each for candidates "
        f"{', '.join(str(candidate['problem_id']) for candidate in candidate_rows)}. "
        + "; ".join(
            f"candidate {candidate['problem_id']} is {candidate['classification']}"
            for candidate in candidate_rows
        )
        + "."
    )

    ledger = {
        "schema_version": 1,
        "snapshot_id": f"orbit-q-exploratory-harbor-sha256:{artifact_set_sha256}",
        "source_cutoff_utc": source_cutoff,
        "source_artifact_set_sha256": artifact_set_sha256,
        "evidence_payload_sha256": "",
        "integrity": integrity,
        "claim_boundary": {
            "evidence_tier": "exploratory_automated_scan_only",
            "protocol_qualified_hardness_claim": False,
            "statement": (
                "These are sanitized records of local exploratory Harbor runs. They are "
                "bound to opaque local artifacts and automated trace scans, but have no "
                "hash-bound human trace review and are not signed raw-run attestations. They "
                "do not satisfy the precommitted multi-trial protocol needed for a model-hard "
                "benchmark claim."
            ),
            "admitted_model_scope": model_scope,
        },
        "sanitization": {
            "policy": "whitelisted_fields_and_sha256_only",
            "raw_auth_copied": False,
            "raw_environment_copied": False,
            "raw_config_copied": False,
            "raw_transcript_or_rollout_copied": False,
            "note": (
                "Raw local artifacts remain untracked under .artifacts. The ledger stores "
                "repo-relative locators, sizes, hashes, exact typed metrics, and recursively "
                "sanitized facts only. Raw config content is excluded while its opaque digest "
                "is included in the snapshot."
            ),
        },
        "frozen_protocol_facts": common_protocol,
        "candidates": candidate_rows,
    }
    ledger["evidence_payload_sha256"] = _evidence_payload_sha256(ledger)
    _assert_sanitized(ledger, location="ledger")
    return ledger


def _short_digest(value: str | None) -> str:
    if not value:
        return "—"
    return value if len(value) <= 16 else f"{value[:16]}…"


def _score_cell(run: dict[str, Any]) -> str:
    scores = run["scores"]
    return (
        f"{scores['compound_reward']:.1f} / {scores['functional_score']:.1f} / "
        f"{scores['static_policy_score']:.1f} / {scores['llm_audit_score']:.1f}"
    )


def render_markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "# Sanitized empirical Harbor evidence",
        "",
        "> **Claim boundary:** These are exploratory local runs with automated trace scans and "
        "no hash-bound human trace review. They are not protocol-qualified model-hardness "
        "results and are not signed raw-run attestations.",
        "",
        f"Snapshot: `{ledger['snapshot_id']}`",
        f"Source cutoff: `{ledger['source_cutoff_utc']}`",
        "",
        "## Decision summary",
        "",
        "| Candidate | Classification | Expert evidence | Admitted model pilots | Decision |",
        "|---:|---|---:|---:|---|",
    ]
    for candidate in ledger["candidates"]:
        lines.append(
            f"| {candidate['problem_id']} | `{candidate['classification']}` | "
            f"{len(candidate['expert_runs'])} | {len(candidate['model_pilots'])} | "
            f"{candidate['conclusion']} |"
        )

    lines.extend(
        [
            "",
            "Passing pilots are classified as hardness-blocking for their exact frozen task. "
            "A functionally failed but policy- and source-audit-clean pilot is classified only "
            "as an exploratory signal, never protocol-qualified model-hard evidence.",
            "",
            "## Expert evidence",
            "",
            "| Candidate | Job ID | Trial ID | Seed / case digest | Reward / functional / static / audit | Evaluator runtime (s) | Solution SHA-256 |",
            "|---:|---|---|---|---|---:|---|",
        ]
    )
    for candidate in ledger["candidates"]:
        for run in candidate["expert_runs"]:
            functional = run["functional_evidence"]
            seed_digest = f"{functional['seed'] if functional['seed'] is not None else '—'} / {_short_digest(functional['case_digest'])}"
            lines.append(
                f"| {candidate['problem_id']} | `{run['job']['id']}` | `{run['trial']['id']}` | "
                f"{seed_digest} | {_score_cell(run)} | "
                f"{run['scores']['functional_evaluator_runtime_sec']:.6f} | "
                f"`{run['artifacts']['solution']['sha256']}` |"
            )

    lines.extend(
        [
            "",
            f"## {ledger['frozen_protocol_facts']['model_pilot_model']}/"
            f"{ledger['frozen_protocol_facts']['model_pilot_reasoning_effort']} exploratory pilots",
            "",
            "| Candidate | Job ID | Trial ID | Seed / case digest | Reward / functional / static / audit | Evaluator runtime (s) | Result / solution / trajectory SHA-256 | Command audit |",
            "|---:|---|---|---|---|---:|---|---|",
        ]
    )
    for candidate in ledger["candidates"]:
        for run in candidate["model_pilots"]:
            functional = run["functional_evidence"]
            seed_digest = f"{functional['seed'] if functional['seed'] is not None else '—'} / {_short_digest(functional['case_digest'])}"
            hashes = " / ".join(
                f"`{run['artifacts'][key]['sha256']}`"
                for key in ("trial_result", "solution", "trajectory")
            )
            audit = run["contamination_and_command_audit"]
            lines.append(
                f"| {candidate['problem_id']} | `{run['job']['id']}` | `{run['trial']['id']}` | "
                f"{seed_digest} | {_score_cell(run)} | "
                f"{run['scores']['functional_evaluator_runtime_sec']:.6f} | "
                f"{hashes} | {audit['network_command_scan']}; "
                f"{audit['protected_benchmark_artifact_scan']} |"
            )

    candidate_109 = next(
        candidate
        for candidate in ledger["candidates"]
        if candidate["problem_id"] == 109
    )
    pilot_109 = candidate_109["model_pilots"][0]
    metrics_109 = pilot_109["functional_evidence"]["metrics"]
    checks_109 = {
        check["metric"]: check
        for check in pilot_109["functional_evidence"]["evaluation_checks"]
    }
    protocol = ledger["frozen_protocol_facts"]
    digest = protocol["image_digest_observation"]
    installed_test_search_candidates = [
        str(candidate["problem_id"])
        for candidate in ledger["candidates"]
        if candidate["model_pilots"][0]["contamination_and_command_audit"][
            "installed_package_test_search_observed"
        ]
    ]
    lines.extend(
        [
            "",
            f"Candidate 109's pilot missed bound held-out worst infidelity "
            f"({metrics_109['heldout_worst_infidelity']:.16g} > "
            f"{checks_109['heldout_worst_infidelity']['threshold']}) and p95 infidelity "
            f"({metrics_109['heldout_p95_infidelity']:.16g} > "
            f"{checks_109['heldout_p95_infidelity']['threshold']}). Its worst leakage, drive "
            "amplitude, slew, and edge amplitude remained within their evaluator thresholds. "
            "The static policy and source audit passed; the source audit judged the "
            "framework-native implementation faithful and found no obvious implementation error.",
            "",
            "## Protocol and trust boundary",
            "",
            f"- Framework/image tag: `{protocol['framework']}` / `{protocol['image_tag']}`.",
            f"- Selected resources: {protocol['override_cpus']} CPUs and "
            f"{protocol['override_memory_mb']} MiB; verifier audit model "
            f"`{protocol['audit_model']}`.",
            f"- Model pilots: `{protocol['model_pilot_model']}`, reasoning "
            f"`{protocol['model_pilot_reasoning_effort']}`, Codex CLI "
            f"`{protocol['codex_cli_version']}`; TensorCircuit "
            f"`{protocol['tensorcircuit_version_observed_in_pilot_trajectories']}` was observed "
            "in the admitted trajectories.",
            "- `runtime` in the tables is the functional evaluator's measured solution runtime, "
            "not solver-agent wall-clock time.",
            "- Network isolation was not enabled: selected configs contained proxy bridge keys. "
            "Automated allowlist and marker scans of recorded model tool-call names and inputs "
            "had no network-command, protected-artifact, credential/environment-dump, or "
            "disallowed-tool matches. These scans are not human attestations.",
            f"- Searches under installed-package test directories in candidates "
            f"{', '.join(installed_test_search_candidates) or 'none'} were package-source reads, "
            "not access to the protected benchmark-root `/tests` path.",
            f"- Local same-tag metadata records image digest `{digest['sha256']}`, but the selected "
            "Harbor results bind only the image tag. The digest is therefore a qualified local "
            "observation, not a run-bound attestation.",
            "- Static `raw_simulator_hits` are preserved per run in the JSON; a run is admitted "
            "only when its static policy and source audit scores agree and pass.",
            "- Every run's Harbor task checksum is mapped to a committed frozen task-file hash "
            "set. Config content remains private, but its opaque SHA-256 is snapshot-bound.",
            f"- Solver pilots and the source-audit invocation both use "
            f"`{protocol['model_pilot_model']}`; the audit is separate, but it is not "
            "independent model-family adjudication.",
            "",
            "## Artifact integrity and sanitization",
            "",
            "The JSON ledger contains the full SHA-256, byte size, and repo-relative locator for "
            "every admitted job result, trial result, reward, functional output, source audit, "
            "solution, artifact manifest, redacted config, oracle transcript, and available "
            "trajectory. It retains only an opaque SHA-256 for each raw model rollout; session "
            "filenames are excluded. Raw auth, environment values, config/transcript content, "
            "and rollout content are excluded. The schema and frozen source-binding manifest "
            "are also included in the snapshot digest.",
            "",
            "Rebuild or verify deterministically:",
            "",
            "```bash",
            "python3 scripts/build_problem_discovery_evidence.py --check",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _canonical_json(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _schema_type_matches(value: Any, expected: str) -> bool:
    return {
        "null": value is None,
        "boolean": type(value) is bool,
        "integer": type(value) is int,
        "number": _is_number(value),
        "string": isinstance(value, str),
        "array": isinstance(value, list),
        "object": isinstance(value, dict),
    }.get(expected, False)


def _resolve_schema_ref(root_schema: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError(f"unsupported external schema reference: {reference}")
    value: Any = root_schema
    for raw_part in reference[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        value = value[part]
    if not isinstance(value, dict):
        raise ValueError(f"schema reference is not an object: {reference}")
    return value


def _validate_json_schema(
    value: Any,
    schema: dict[str, Any],
    *,
    root_schema: dict[str, Any] | None = None,
    location: str = "$",
) -> None:
    root_schema = schema if root_schema is None else root_schema
    if "$ref" in schema:
        _validate_json_schema(
            value,
            _resolve_schema_ref(root_schema, schema["$ref"]),
            root_schema=root_schema,
            location=location,
        )
        return
    if "anyOf" in schema:
        errors = []
        for option in schema["anyOf"]:
            try:
                _validate_json_schema(
                    value, option, root_schema=root_schema, location=location
                )
            except ValueError as exc:
                errors.append(str(exc))
            else:
                return
        raise ValueError(f"schema anyOf failed at {location}: {'; '.join(errors)}")
    if "const" in schema and (
        value != schema["const"] or type(value) is not type(schema["const"])
    ):
        raise ValueError(f"schema const mismatch at {location}")
    if "enum" in schema and not any(
        value == option and type(value) is type(option) for option in schema["enum"]
    ):
        raise ValueError(f"schema enum mismatch at {location}: {value!r}")

    expected_types = schema.get("type")
    if isinstance(expected_types, str):
        expected_types = [expected_types]
    if expected_types is not None and not any(
        _schema_type_matches(value, expected) for expected in expected_types
    ):
        raise ValueError(
            f"schema type mismatch at {location}: expected {expected_types}, got {type(value).__name__}"
        )

    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"schema missing fields at {location}: {missing}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, child in value.items():
            child_schema = properties.get(key)
            if child_schema is None:
                if additional is False:
                    raise ValueError(f"schema rejects {location}.{key}")
                if isinstance(additional, dict):
                    child_schema = additional
            if isinstance(child_schema, dict):
                _validate_json_schema(
                    child,
                    child_schema,
                    root_schema=root_schema,
                    location=f"{location}.{key}",
                )
        if len(value) < schema.get("minProperties", 0):
            raise ValueError(f"schema minProperties failed at {location}")
    elif isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get(
            "maxItems", math.inf
        ):
            raise ValueError(f"schema item-count mismatch at {location}")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, child in enumerate(value):
                _validate_json_schema(
                    child,
                    item_schema,
                    root_schema=root_schema,
                    location=f"{location}[{index}]",
                )
    elif isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"schema minLength failed at {location}")
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, value) is None:
            raise ValueError(f"schema pattern failed at {location}: {value!r}")
        if schema.get("format") == "date-time":
            try:
                parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError as exc:
                raise ValueError(f"schema date-time failed at {location}") from exc
            if parsed.tzinfo is None:
                raise ValueError(f"schema date-time lacks timezone at {location}")
    elif _is_number(value):
        if value < schema.get("minimum", -math.inf) or value > schema.get(
            "maximum", math.inf
        ):
            raise ValueError(f"schema numeric bound failed at {location}: {value!r}")


def _validate_ledger_semantics(
    ledger: dict[str, Any], source_bindings: dict[str, Any]
) -> None:
    _assert_sanitized(ledger, location="ledger")
    integrity_expected = {
        "builder": _artifact(BUILDER_PATH),
        "schema": _artifact(SCHEMA_PATH),
        "source_binding_manifest": _artifact(SOURCE_BINDINGS_PATH),
    }
    if ledger["integrity"] != integrity_expected:
        raise ValueError("committed integrity artifact drift")
    candidates = ledger["candidates"]
    protocol = ledger["frozen_protocol_facts"]
    digest_observation = protocol["image_digest_observation"]
    digest_source = (REPO_ROOT / digest_observation["source"]).resolve()
    try:
        digest_source.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise ValueError("supporting protocol record escapes repository") from exc
    if (
        not digest_source.is_file()
        or _sha256(digest_source) != digest_observation["source_record_sha256"]
    ):
        raise ValueError("supporting protocol record hash drift")
    recomputed = _source_artifact_set_sha256(candidates, protocol, ledger["integrity"])
    if ledger["source_artifact_set_sha256"] != recomputed:
        raise ValueError("source artifact set digest mismatch")
    if ledger["snapshot_id"] != f"orbit-q-exploratory-harbor-sha256:{recomputed}":
        raise ValueError("snapshot ID does not bind the source artifact set")
    if ledger["evidence_payload_sha256"] != _evidence_payload_sha256(ledger):
        raise ValueError("committed evidence payload digest mismatch")

    bindings = _binding_index(source_bindings)
    all_runs: list[dict[str, Any]] = []
    for candidate in candidates:
        expert_runs = candidate["expert_runs"]
        model_pilots = candidate["model_pilots"]
        derived = _derive_candidate(
            candidate["problem_id"], candidate["slug"], expert_runs, model_pilots
        )
        if candidate != derived:
            raise ValueError(
                f"candidate {candidate['problem_id']} derived evidence drift"
            )
        for run in [*expert_runs, *model_pilots]:
            all_runs.append(run)
            checksum = run["trial"]["task_checksum"]
            binding = bindings.get(checksum)
            if binding is None:
                raise ValueError(f"run lacks committed task source binding: {checksum}")
            expected_binding = {
                "task_checksum": checksum,
                "task_source_set_sha256": binding["task_source_set_sha256"],
                "file_count": binding["file_count"],
            }
            if run["task_source_binding"] != expected_binding:
                raise ValueError(f"run task source binding drift: {checksum}")
            if "redacted_config" not in run["artifacts"]:
                raise ValueError(f"run config digest missing: {checksum}")
            audit = run["contamination_and_command_audit"]
            if run["operator_attested"] is not False:
                raise ValueError("automated evidence cannot claim operator attestation")
            review = audit["human_review"]
            matching_reviews = [
                item
                for item in source_bindings["human_trace_reviews"]
                if item["job_name"] == run["job"]["name"]
            ]
            if run["evidence_role"] == "model_pilot":
                scanned_digest = audit["automated_scan_raw_rollout_sha256"]
                if not isinstance(scanned_digest, str) or not SHA256_PATTERN.fullmatch(
                    scanned_digest
                ):
                    raise ValueError("model trace scan digest is missing")
            elif audit["automated_scan_raw_rollout_sha256"] is not None:
                raise ValueError("oracle run cannot claim a model trace scan")

            if matching_reviews:
                source_review = matching_reviews[0]
                expected_review = {
                    "status": "approved",
                    "reviewer_id": source_review["reviewer_id"],
                    "reviewed_at": source_review["reviewed_at"],
                    "reviewed_raw_rollout_sha256": source_review[
                        "reviewed_raw_rollout_sha256"
                    ],
                    "decision": "approved",
                }
                if (
                    run["evidence_role"] != "model_pilot"
                    or source_review["reviewed_raw_rollout_sha256"]
                    != audit["automated_scan_raw_rollout_sha256"]
                    or review != expected_review
                    or audit["status"] != "automated_scan_pass_human_reviewed"
                ):
                    raise ValueError("human trace review is not hash-bound")
            else:
                expected_review = {
                    "status": "none",
                    "reviewer_id": None,
                    "reviewed_at": None,
                    "reviewed_raw_rollout_sha256": None,
                    "decision": None,
                }
                expected_status = (
                    "automated_scan_pass_human_review_none"
                    if run["evidence_role"] == "model_pilot"
                    else "not_applicable_oracle_copy"
                )
                if review != expected_review or audit["status"] != expected_status:
                    raise ValueError("unrecorded human trace review")

    model_runs = [run for run in all_runs if run["evidence_role"] == "model_pilot"]
    admitted_model_jobs = {run["job"]["name"] for run in model_runs}
    review_jobs = {
        review["job_name"] for review in source_bindings["human_trace_reviews"]
    }
    if not review_jobs.issubset(admitted_model_jobs):
        raise ValueError("human trace review references an unadmitted model run")
    protocol_agent_fields = {
        "model_pilot_solver_adapter": "solver_adapter",
        "model_pilot_model": "model",
        "model_pilot_reasoning_effort": "reasoning_effort",
        "codex_cli_version": "version",
    }
    for protocol_field, agent_field in protocol_agent_fields.items():
        values = {run["agent"][agent_field] for run in model_runs}
        if values != {protocol[protocol_field]}:
            raise ValueError(f"frozen protocol {protocol_field} drift")
    for field in (
        "framework",
        "image_tag",
        "environment_adapter",
        "override_cpus",
        "override_memory_mb",
        "verifier_adapter",
        "audit_model",
    ):
        values = {run["protocol"][field] for run in all_runs}
        if values != {protocol[field]}:
            raise ValueError(f"frozen protocol {field} drift")
    source_cutoff = max(run["trial"]["finished_at"] for run in all_runs)
    if ledger["source_cutoff_utc"] != source_cutoff:
        raise ValueError("source cutoff drift")


def verify_committed_snapshot() -> dict[str, Any]:
    ledger = _load_json(LEDGER_PATH)
    schema = _load_json(SCHEMA_PATH)
    source_bindings = _load_source_bindings()
    _validate_json_schema(ledger, schema)
    _validate_ledger_semantics(ledger, source_bindings)
    if render_markdown(ledger) != REPORT_PATH.read_text(encoding="utf-8"):
        raise ValueError("committed empirical evidence Markdown drift")
    return ledger


def _assert_raw_rebuild_matches(checked_in: dict[str, Any]) -> None:
    rebuilt = build_ledger()
    if rebuilt != checked_in:
        raise ValueError("committed empirical evidence JSON differs from raw rebuild")
    if render_markdown(rebuilt) != REPORT_PATH.read_text(encoding="utf-8"):
        raise ValueError(
            "committed empirical evidence Markdown differs from raw rebuild"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify committed schema, hashes, semantics, and deterministic render",
    )
    parser.add_argument(
        "--check-raw",
        action="store_true",
        help="require ignored raw artifacts and compare them with the committed snapshot",
    )
    args = parser.parse_args()

    if args.check and args.check_raw:
        parser.error("--check and --check-raw are mutually exclusive")
    if args.check:
        verify_committed_snapshot()
        return 0
    if args.check_raw:
        checked_in = verify_committed_snapshot()
        if not JOBS_DIR.is_dir():
            raise SystemExit(f"raw Harbor artifacts are unavailable: {JOBS_DIR}")
        _assert_raw_rebuild_matches(checked_in)
        return 0

    ledger = build_ledger()
    schema = _load_json(SCHEMA_PATH)
    source_bindings = _load_source_bindings()
    _validate_json_schema(ledger, schema)
    _validate_ledger_semantics(ledger, source_bindings)
    json_text = _canonical_json(ledger)
    markdown_text = render_markdown(ledger)
    LEDGER_PATH.write_text(json_text, encoding="utf-8")
    REPORT_PATH.write_text(markdown_text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
