"""Build and gate a source-grounded pool of future ORBIT-Q task concepts.

The screening score produced here is a design prior.  It is deliberately kept
separate from empirical model results: no candidate can be called "unsolved"
until an expert baseline, verifier-only smoke test, human pilot approval, and
repeated frozen-model trials have all happened.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import fcntl


SCORE_NAMES = (
    "agent_difficulty_prior",
    "tensorcircuit_path_confidence",
    "scientific_value",
    "verifier_feasibility",
    "runtime_fit",
    "determinism",
    "novelty",
    "anti_shortcut",
)

SCORE_WEIGHTS = {
    "agent_difficulty_prior": 0.22,
    "tensorcircuit_path_confidence": 0.16,
    "scientific_value": 0.12,
    "verifier_feasibility": 0.16,
    "runtime_fit": 0.10,
    "determinism": 0.08,
    "novelty": 0.10,
    "anti_shortcut": 0.06,
}

REQUIRED_PROTOCOL_CONFIG_FIELDS = (
    "solver_agent",
    "model",
    "provider_snapshot",
    "reasoning_effort",
    "token_budget",
    "token_budget_metric",
    "token_budget_enforcement",
    "wall_time_sec",
    "solver_agent_version",
    "harbor_version",
    "audit_model",
    "build_identity_policy",
    "tools_policy",
    "network_policy",
    "proxy_policy",
    "hardware_class",
    "cpus",
    "memory_mb",
    "docker_image",
    "container_image_digest",
    "task_bundle_sha256",
    "harbor_task_digest",
    "task_checksum",
    "framework_prompt_sha256",
    "force_auth_json",
)

SOLVER_AGENT_IMPORT_PATHS = {
    "codex": "harbor.agents.installed.codex:Codex",
    "codex-para": "adapters.codex_para:CodexPara",
}
VERIFIER_IMPORT_PATH = "adapters.codex_para_verifier:CodexParaVerifier"
ENVIRONMENT_IMPORT_PATH = "adapters.framework_docker:FrameworkDockerEnvironment"
CASE_IDENTITY_KEY = "orbit_q_case_identity"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")

CONCEPT_REVIEW_ROLES = (
    "quantum_scientist",
    "tensorcircuit_expert",
    "verifier_engineer",
)
PILOT_REVIEW_ROLES = ("benchmark_owner", "independent_reviewer")
FAILURE_AUDIT_ROLES = ("failure_auditor_primary", "failure_auditor_secondary")
TRIAL_EXECUTION_STATUSES = (
    "completed",
    "agent_timeout",
    "verifier_timeout",
    "infrastructure_error",
    "authentication_error",
)
SUBSTANTIVE_FAILURE_CLASSES = (
    "quantum_reasoning",
    "algorithm_design",
    "tensorcircuit_implementation",
    "numerical_method",
    "resource_strategy",
)
EXCLUDED_FAILURE_CLASSES = (
    "infrastructure",
    "authentication",
    "policy_refusal",
    "ambiguous_specification",
    "framework_impossibility",
    "evaluator_defect",
    "indeterminate",
)
REQUIRED_PROTOTYPE_FIELDS = (
    "candidate_id",
    "candidate_hash",
    "evidence_bundle_path",
    "evidence_bundle_sha256",
    "expert_baseline_path",
    "expert_baseline_sha256",
    "independent_oracle_path",
    "independent_oracle_sha256",
    "evaluator_path",
    "evaluator_sha256",
    "instruction_path",
    "instruction_sha256",
    "candidate_metadata_path",
    "candidate_metadata_sha256",
    "task_bundle_path",
    "task_bundle_sha256",
    "blueprint_path",
    "verifier_harness_sha256",
    "task_policy_sha256",
    "solution_driver_sha256",
    "framework_prompt_path",
    "container_image_digest",
    "docker_image",
    "harbor_version",
    "harbor_task_digest",
    "task_checksum",
    "framework_prompt_sha256",
    "source_commit",
    "artifact_authors",
    "independent_artifact_authors",
    "public_api_canary_passed",
    "public_api_canary_log_sha256",
    "expert_runtime_p95_sec",
    "independent_oracle_runtime_sec",
    "gold_effective_lines",
    "reproducibility_runs",
    "observed_cpu_count",
    "observed_memory_mb",
    "expert_peak_memory_mb",
    "verifier_only_passed",
    "valid_alternatives_accepted",
    "mutation_tests_rejected",
    "expert_prequalified_cases",
    "expert_prequalified_case_count",
)

SHORTLIST_REVIEW_FIELDS = (
    "rank",
    "why_plausibly_hard",
    "falsification_check",
    "review_questions",
)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _canonical_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def _normalized_nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return " ".join(value.split())


def _normalized_identity(value: Any, label: str) -> str:
    try:
        return _normalized_nonempty_text(value, label)
    except ValueError as error:
        raise ValueError(f"{label} must be a nonempty identity") from error


def _identity_key(value: Any) -> str:
    return _normalized_identity(value, "identity").casefold()


def _verified_payload_hash(value: dict[str, Any], hash_field: str, label: str) -> str:
    recorded_hash = value.get(hash_field)
    payload = {key: item for key, item in value.items() if key != hash_field}
    expected_hash = _canonical_hash(payload, length=64)
    if recorded_hash != expected_hash:
        raise ValueError(f"{label} hash does not match its complete payload")
    return expected_hash


def _append_event(state: dict[str, Any], event: dict[str, Any]) -> None:
    event_log = state.setdefault("event_log", [])
    previous = event_log[-1].get("event_hash", "0" * 64) if event_log else "0" * 64
    chained = {**event, "previous_event_hash": previous}
    chained["event_hash"] = _canonical_hash(chained, length=64)
    event_log.append(chained)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_sha256(path: Path) -> str:
    if path.is_file():
        return _file_sha256(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    files = sorted(row for row in path.rglob("*") if row.is_file())
    if not files:
        raise ValueError(f"evidence directory is empty: {path}")
    for file_path in files:
        if file_path.is_symlink():
            raise ValueError(
                f"symlinks are not allowed in evidence bundles: {file_path}"
            )
        relative = file_path.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(_file_sha256(file_path)))
    return digest.hexdigest()


def _harbor_task_digest(task_dir: Path) -> str:
    """Recompute Harbor Packager.compute_content_hash for a local task."""

    files = [task_dir / "task.toml", task_dir / "instruction.md"]
    for directory_name in ("environment", "tests", "solution"):
        files.extend(
            path
            for path in (task_dir / directory_name).rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    files.sort(key=lambda path: path.relative_to(task_dir).as_posix())
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(task_dir).as_posix()
        digest.update(f"{relative}\0{_file_sha256(path)}\n".encode())
    return digest.hexdigest()


def _harbor_legacy_task_checksum(task_dir: Path) -> str:
    """Recompute the default dirhash used by Harbor's Task.checksum."""

    def directory_hash(directory: Path) -> str | None:
        descriptors = []
        for path in directory.iterdir():
            if path.is_symlink():
                raise ValueError(f"Harbor task checksum rejects symlink: {path}")
            if path.is_dir():
                value = directory_hash(path)
                if value is None:
                    continue
                properties = (f"dirhash:{value}", f"name:{path.name}")
            elif path.is_file():
                properties = (f"data:{_file_sha256(path)}", f"name:{path.name}")
            else:
                raise ValueError(f"Harbor task checksum rejects special file: {path}")
            descriptors.append("\0".join(sorted(properties)))
        if not descriptors:
            return None
        return hashlib.sha256("\0\0".join(sorted(descriptors)).encode()).hexdigest()

    checksum = directory_hash(task_dir)
    if checksum is None:
        raise ValueError("Harbor task checksum rejects an empty task")
    return checksum


def _repository_root(root: Path) -> Path:
    if root.parent.name == "reports":
        return root.parents[1]
    return root.parent


def _safe_manifest_output(root: Path, output: Path) -> Path:
    allowed_root = (
        _repository_root(root) / ".artifacts" / "problem-discovery"
    ).resolve()
    resolved = output.resolve()
    if not resolved.is_relative_to(allowed_root):
        raise ValueError(f"run manifests must stay under {allowed_root}")
    if resolved.exists():
        raise FileExistsError(f"refusing to overwrite an existing manifest: {resolved}")
    return resolved


def _safe_evidence_path(root: Path, base: Path, value: str) -> Path:
    allowed_root = (
        _repository_root(root) / ".artifacts" / "problem-discovery"
    ).resolve()
    path = Path(value)
    resolved = (base / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_relative_to(allowed_root):
        raise ValueError(f"evidence must stay under {allowed_root}: {resolved}")
    return resolved


def _verified_evidence_item(
    root: Path, base: Path, item: dict[str, Any], label: str
) -> tuple[Path, str]:
    if not isinstance(item, dict) or not item.get("path") or not item.get("sha256"):
        raise ValueError(f"{label} needs path and sha256")
    path = _safe_evidence_path(root, base, item["path"])
    actual = _path_sha256(path)
    if actual != item["sha256"]:
        raise ValueError(f"{label} SHA-256 does not match {path}")
    return path, actual


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _effective_python_lines(path: Path) -> int:
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    if not files or any(file_path.suffix != ".py" for file_path in files):
        raise ValueError(
            "expert baseline must be a Python file or Python-only directory"
        )
    return sum(
        1
        for file_path in files
        for line in file_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )


def _parse_prototype_verifier_result(path: Path, image_digest: str) -> dict[str, Any]:
    result = _read_json_object(path, "prototype verifier result")
    if result.get("finished_at") in (None, ""):
        raise ValueError("prototype verifier result is not finished")
    job_id = _normalized_identity(result.get("id"), "prototype verifier job id")
    rewards = result.get("verifier_result", {}).get("rewards")
    if not isinstance(rewards, dict):
        raise ValueError("prototype verifier result is missing rewards")
    scores: dict[str, float] = {}
    for name in (
        "reward",
        "functional_score",
        "static_policy_score",
        "llm_audit_score",
    ):
        value = rewards.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"prototype verifier {name} must be numeric in [0, 1]")
        scores[name] = float(value)
    expected = (
        scores["functional_score"]
        * scores["static_policy_score"]
        * scores["llm_audit_score"]
    )
    if abs(scores["reward"] - expected) > 1e-9 or any(
        scores[name] <= 0
        for name in (
            "reward",
            "functional_score",
            "static_policy_score",
            "llm_audit_score",
        )
    ):
        raise ValueError("prototype verifier result is not a compound pass")
    runtime = rewards.get("runtime_sec")
    if (
        not isinstance(runtime, (int, float))
        or isinstance(runtime, bool)
        or runtime <= 0
    ):
        raise ValueError("prototype verifier runtime must be positive")
    attestation = result.get("orbit_q_attestation")
    if not isinstance(attestation, dict):
        raise ValueError("prototype verifier result needs orbit_q_attestation")
    if attestation.get("attestation_type") != "operator_attested":
        raise ValueError("prototype verifier evidence must be operator-attested")
    _normalized_identity(attestation.get("attestor_id"), "prototype attestor")
    if attestation.get("container_image_digest") != image_digest:
        raise ValueError("prototype verifier image digest does not match")
    peak_memory = attestation.get("peak_memory_mb")
    if (
        not isinstance(peak_memory, (int, float))
        or isinstance(peak_memory, bool)
        or peak_memory <= 0
    ):
        raise ValueError("prototype verifier peak memory must be positive")
    return {
        "job_id": job_id,
        "runtime_sec": float(runtime),
        "peak_memory_mb": float(peak_memory),
    }


def _validated_admission_margin(admission: Any) -> dict[str, Any]:
    """Validate a positive, machine-checkable expert admission margin."""
    if not isinstance(admission, dict) or set(admission) != {
        "status",
        "minimum_margin",
        "margins",
    }:
        raise ValueError(
            "expert case admission needs exactly status, minimum_margin, and margins"
        )
    if admission["status"] != "admitted":
        raise ValueError("expert-prequalified case admission status must be admitted")
    margins = admission["margins"]
    if not isinstance(margins, list) or not margins:
        raise ValueError("expert case admission needs at least one numerical margin")
    derived_margins: list[float] = []
    metric_names: list[str] = []
    for index, margin in enumerate(margins):
        if not isinstance(margin, dict) or set(margin) != {
            "metric",
            "direction",
            "observed",
            "threshold",
            "absolute_margin",
            "passed",
        }:
            raise ValueError(f"expert admission margin {index} has an invalid shape")
        metric = _normalized_identity(
            margin["metric"], f"expert admission margin {index} metric"
        )
        if metric != margin["metric"]:
            raise ValueError(
                f"expert admission margin {index} metric is not normalized"
            )
        if margin["direction"] not in {"at_least", "at_most"}:
            raise ValueError(f"expert admission margin {index} has invalid direction")
        values = (margin["observed"], margin["threshold"], margin["absolute_margin"])
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(float(value))
            for value in values
        ):
            raise ValueError(f"expert admission margin {index} must be finite numeric")
        observed, threshold, declared = (float(value) for value in values)
        derived = (
            observed - threshold
            if margin["direction"] == "at_least"
            else threshold - observed
        )
        if (
            margin["passed"] is not True
            or derived <= 0
            or declared <= 0
            or not math.isclose(declared, derived, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(
                f"expert admission margin {index} must be a strict, correctly computed pass"
            )
        metric_names.append(metric.casefold())
        derived_margins.append(derived)
    if len(metric_names) != len(set(metric_names)):
        raise ValueError("expert admission margin metric names must be unique")
    minimum_margin = admission["minimum_margin"]
    if (
        not isinstance(minimum_margin, (int, float))
        or isinstance(minimum_margin, bool)
        or not math.isfinite(float(minimum_margin))
        or float(minimum_margin) <= 0
        or not math.isclose(
            float(minimum_margin),
            min(derived_margins),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(
            "expert case minimum_margin must equal the smallest positive margin"
        )
    return admission


EXPERT_RUN_EVIDENCE_NAMES = (
    "raw_job",
    "harbor_config",
    "harbor_lock",
    "harbor_job_lock",
    "functional_output",
    "admission_record",
    "audit_details",
)


def _expert_prequalification_bindings(prototype: dict[str, Any]) -> dict[str, Any]:
    """Return the frozen facts every expert-only run must machine-bind."""
    bindings = {
        "candidate_id": prototype.get("candidate_id"),
        "candidate_hash": prototype.get("candidate_hash"),
        "expert_baseline_sha256": prototype.get("expert_baseline_sha256"),
        "evaluator_sha256": prototype.get("evaluator_sha256"),
        "task_bundle_sha256": prototype.get("task_bundle_sha256"),
        "framework_prompt_sha256": prototype.get("framework_prompt_sha256"),
        "verifier_harness_sha256": prototype.get("verifier_harness_sha256"),
        "instruction_sha256": prototype.get("instruction_sha256"),
        "candidate_metadata_sha256": prototype.get("candidate_metadata_sha256"),
        "task_policy_sha256": prototype.get("task_policy_sha256"),
        "solution_driver_sha256": prototype.get("solution_driver_sha256"),
        "container_image_digest": prototype.get("container_image_digest"),
        "docker_image": prototype.get("docker_image"),
        "harbor_version": prototype.get("harbor_version"),
        "harbor_task_digest": prototype.get("harbor_task_digest"),
        "task_checksum": prototype.get("task_checksum"),
        "cpus": prototype.get("observed_cpu_count"),
        "memory_mb": prototype.get("observed_memory_mb"),
        "task_bundle_path": prototype.get("task_bundle_path"),
        "framework_prompt_path": prototype.get("framework_prompt_path"),
        "blueprint_path": prototype.get("blueprint_path"),
    }
    for name in (
        "candidate_id",
        "docker_image",
        "harbor_version",
    ):
        _normalized_identity(bindings[name], f"expert-run {name}")
    for name in (
        "candidate_hash",
        "expert_baseline_sha256",
        "evaluator_sha256",
        "task_bundle_sha256",
        "framework_prompt_sha256",
        "verifier_harness_sha256",
        "instruction_sha256",
        "candidate_metadata_sha256",
        "task_policy_sha256",
        "solution_driver_sha256",
        "harbor_task_digest",
        "task_checksum",
    ):
        _validated_sha256(bindings[name], f"expert-run {name}")
    if not IMAGE_DIGEST_RE.fullmatch(str(bindings["container_image_digest"])):
        raise ValueError("expert-run container image needs a full SHA-256 digest")
    for name in ("cpus", "memory_mb"):
        value = bindings[name]
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value <= 0
            or int(value) != value
        ):
            raise ValueError(f"expert-run {name} must be a positive integer")
        bindings[name] = int(value)
    if (
        "tensorcircuit" not in bindings["docker_image"].casefold()
        or "@" in bindings["docker_image"]
    ):
        raise ValueError("expert-run docker image must be a TensorCircuit name:tag")
    task_bundle_path = Path(str(bindings["task_bundle_path"])).resolve()
    if _path_sha256(task_bundle_path) != bindings["task_bundle_sha256"]:
        raise ValueError("expert-run task bundle hash changed")
    if _harbor_task_digest(task_bundle_path) != bindings["harbor_task_digest"]:
        raise ValueError("expert-run Harbor task digest is not derived from the task")
    if _harbor_legacy_task_checksum(task_bundle_path) != bindings["task_checksum"]:
        raise ValueError("expert-run Harbor task checksum is not derived from the task")
    metadata = _read_json_object(
        task_bundle_path / "candidate_metadata.json", "candidate task metadata"
    )
    if metadata.get("candidate_id") != bindings["candidate_id"]:
        raise ValueError("expert-run task metadata candidate_id differs")
    discovery_binding = metadata.get("discovery_binding")
    if (
        not isinstance(discovery_binding, dict)
        or discovery_binding.get("candidate_id") != bindings["candidate_id"]
        or discovery_binding.get("candidate_hash") != bindings["candidate_hash"]
    ):
        raise ValueError("expert-run task lacks the reviewed discovery binding")
    test_id = (task_bundle_path / "tests" / "problem_id.txt").read_text().strip()
    solution_id = (task_bundle_path / "solution" / "problem_id.txt").read_text().strip()
    if not test_id.isdigit() or test_id == "0" or test_id != solution_id:
        raise ValueError("expert-run task problem IDs are invalid")
    if (
        _file_sha256(task_bundle_path / "tests" / f"evaluate_{test_id}.py")
        != bindings["evaluator_sha256"]
        or _file_sha256(task_bundle_path / "solution" / f"solution_{test_id}.py")
        != bindings["expert_baseline_sha256"]
    ):
        raise ValueError("expert-run task evaluator/expert binding changed")
    from scripts.materialize_candidate_task import (  # noqa: PLC0415
        CandidateTaskError,
        validate_materialized_task_contract,
    )

    blueprint_path = Path(str(bindings["blueprint_path"])).resolve()
    try:
        current_contract = validate_materialized_task_contract(
            task_bundle_path,
            blueprint_path.parent.parent,
            require_expert_admission=True,
        )
    except CandidateTaskError as exc:
        raise ValueError(
            f"expert-run trusted materialization contract changed: {exc}"
        ) from exc
    if any(prototype.get(name) != value for name, value in current_contract.items()):
        raise ValueError("expert-run materialization contract binding changed")
    bindings["blueprint_path"] = str(blueprint_path)
    component_paths = {
        "instruction_sha256": task_bundle_path / "instruction.md",
        "candidate_metadata_sha256": task_bundle_path / "candidate_metadata.json",
        "task_policy_sha256": task_bundle_path / "task.toml",
        "solution_driver_sha256": task_bundle_path / "solution" / "solve.sh",
    }
    if any(
        _file_sha256(path) != bindings[name] for name, path in component_paths.items()
    ):
        raise ValueError("expert-run trusted materialization component changed")
    bindings["task_bundle_path"] = str(task_bundle_path)
    framework_prompt_path = Path(str(bindings["framework_prompt_path"])).resolve()
    if (
        not framework_prompt_path.is_file()
        or _file_sha256(framework_prompt_path) != bindings["framework_prompt_sha256"]
    ):
        raise ValueError("expert-run framework prompt path/hash differs")
    bindings["framework_prompt_path"] = str(framework_prompt_path)
    bindings["task_name"] = task_bundle_path.name
    bindings["problem_id"] = int(test_id)
    return bindings


def _verified_expert_run_item(
    root: Path | None,
    base: Path,
    item: Any,
    label: str,
) -> tuple[Path, str]:
    """Rehash one expert-run file, including when only ledger state is available."""
    if root is not None:
        return _verified_evidence_item(root, base, item, label)
    if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
        raise ValueError(f"{label} needs exactly path and sha256")
    if not isinstance(item["sha256"], str) or not SHA256_RE.fullmatch(item["sha256"]):
        raise ValueError(f"{label} needs a full SHA-256")
    candidate = Path(item["path"])
    path = (
        (base / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    resolved_base = base.resolve()
    allowed = next(
        (
            parent
            for parent in (resolved_base, *resolved_base.parents)
            if parent.name == "problem-discovery" and parent.parent.name == ".artifacts"
        ),
        resolved_base,
    )
    if not path.is_relative_to(allowed):
        raise ValueError(f"{label} escaped the problem-discovery artifact root")
    if not path.is_file() or _file_sha256(path) != item["sha256"]:
        raise ValueError(f"{label} SHA-256 changed")
    return path, item["sha256"]


def _expert_expected_config(bindings: dict[str, Any], seed: int) -> dict[str, Any]:
    verifier_env = _required_expert_verifier_env(bindings, seed)
    return {
        "schema_version": 1,
        "mode": "expert_only_oracle",
        "candidate_id": bindings["candidate_id"],
        "candidate_hash": bindings["candidate_hash"],
        "timeout_multiplier": 1.0,
        "install_only": False,
        "skills": [],
        "agent": None,
        "oracle": {
            "mode": "copy_frozen_expert_solution",
            "expert_baseline_sha256": bindings["expert_baseline_sha256"],
        },
        "environment": {
            "import_path": ENVIRONMENT_IMPORT_PATH,
            "cpu_enforcement_policy": "limit",
            "memory_enforcement_policy": "limit",
            "override_cpus": bindings["cpus"],
            "override_memory_mb": bindings["memory_mb"],
            "kwargs": {
                "framework": "tensorcircuit",
                "docker_image": (
                    f"{bindings['docker_image']}@{bindings['container_image_digest']}"
                ),
            },
        },
        "verifier": {
            "import_path": VERIFIER_IMPORT_PATH,
            "kwargs": {"expert_only": True},
            "env": verifier_env,
        },
    }


def _parse_expert_functional_output(path: Path) -> tuple[int, str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    nonempty = [
        (index, line.strip()) for index, line in enumerate(lines) if line.strip()
    ]
    if not nonempty:
        raise ValueError("expert functional output is empty")
    identity_rows: list[tuple[int, dict[str, Any]]] = []
    for index, line in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get(CASE_IDENTITY_KEY), dict):
            if set(value) != {CASE_IDENTITY_KEY}:
                raise ValueError("expert case identity line contains extra fields")
            identity_rows.append((index, value[CASE_IDENTITY_KEY]))
    if len(identity_rows) != 1:
        raise ValueError(
            "expert functional output must contain exactly one case identity"
        )
    identity_index, identity = identity_rows[0]
    if set(identity) != {"protocol_seed", "case_digest"}:
        raise ValueError("expert case identity payload contains extra fields")
    if identity_index != nonempty[0][0]:
        raise ValueError("expert case identity must be the first nonempty output line")
    verdicts = [
        line for _, line in nonempty if re.fullmatch(r"Overall: (?:PASS|FAIL)", line)
    ]
    if verdicts != ["Overall: PASS"] or nonempty[-1][1] != "Overall: PASS":
        raise ValueError("expert functional output must end in one final Overall: PASS")
    seed = identity.get("protocol_seed")
    digest = identity.get("case_digest")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("expert functional case seed must be nonnegative")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValueError("expert functional case identity needs a full digest")
    return seed, digest


def _parse_expert_admission_record(
    path: Path,
    bindings: dict[str, Any],
    seed: int,
    case_digest: str,
) -> tuple[dict[str, Any], str]:
    result = _read_json_object(path, "expert admission record")
    expected_keys = {
        "schema_version",
        "producer",
        "candidate_id",
        "candidate_hash",
        "protocol_seed",
        "case_digest",
        "expert_baseline_sha256",
        "evaluator_sha256",
        "task_bundle_sha256",
        "framework_prompt_sha256",
        "verifier_harness_sha256",
        "container_image_digest",
        "expert_passed",
        "threshold_policy_sha256",
        "admission",
    }
    if set(result) != expected_keys:
        raise ValueError("expert admission record has an invalid shape")
    expected = {
        "schema_version": 1,
        "producer": "orbit_q_score_submission_from_candidate_evaluator",
        "candidate_id": bindings["candidate_id"],
        "candidate_hash": bindings["candidate_hash"],
        "protocol_seed": seed,
        "case_digest": case_digest,
        "expert_baseline_sha256": bindings["expert_baseline_sha256"],
        "evaluator_sha256": bindings["evaluator_sha256"],
        "task_bundle_sha256": bindings["task_bundle_sha256"],
        "framework_prompt_sha256": bindings["framework_prompt_sha256"],
        "verifier_harness_sha256": bindings["verifier_harness_sha256"],
        "container_image_digest": bindings["container_image_digest"],
        "expert_passed": True,
    }
    mismatched = [name for name, value in expected.items() if result.get(name) != value]
    if mismatched:
        raise ValueError(
            f"expert admission record disagrees with frozen run bindings: {mismatched}"
        )
    admission = result["admission"]
    _validated_admission_margin(admission)
    threshold_policy = [
        {
            "metric": margin["metric"],
            "direction": margin["direction"],
            "threshold": float(margin["threshold"]),
        }
        for margin in admission["margins"]
    ]
    policy_hash = _canonical_hash(threshold_policy, length=64)
    if result["threshold_policy_sha256"] != policy_hash:
        raise ValueError("expert admission threshold policy hash is not derived")
    return admission, policy_hash


def _expert_admission_hash_from_rewards(rewards: Any) -> str:
    """Reassemble the verifier-bound admission SHA-256 from reward integers."""
    if not isinstance(rewards, dict):
        raise ValueError("raw expert Harbor result is missing verifier rewards")
    words: list[str] = []
    for index in range(8):
        value = rewards.get(f"expert_admission_sha256_word_{index}")
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value <= 0xFFFFFFFF
        ):
            raise ValueError("raw expert Harbor result does not bind admission SHA-256")
        words.append(f"{value:08x}")
    return "".join(words)


def _required_expert_verifier_env(
    bindings: dict[str, Any], seed: int
) -> dict[str, str]:
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "CODEX_AUDIT_ENABLED": "0",
        "REQUIRED_QUANTUM_FRAMEWORK": "tensorcircuit",
        "ORBIT_Q_NON_HARDNESS_RUN": "expert_prequalification",
        "ORBIT_Q_EXPERT_PREQUALIFICATION_MODE": "expert_only_oracle",
        "ORBIT_Q_AUDIT_POLICY": "disabled_private_prequalification",
        "ORBIT_Q_PROVIDER_CALL_BUDGET": "0",
        "ORBIT_Q_CANDIDATE_ID": bindings["candidate_id"],
        "ORBIT_Q_CANDIDATE_HASH": bindings["candidate_hash"],
        "ORBIT_Q_CANDIDATE_SEED": str(seed),
        "ORBIT_Q_CONTAINER_IMAGE_DIGEST": bindings["container_image_digest"],
        "ORBIT_Q_EXPERT_BASELINE_SHA256": bindings["expert_baseline_sha256"],
        "ORBIT_Q_EVALUATOR_SHA256": bindings["evaluator_sha256"],
        "ORBIT_Q_TASK_BUNDLE_SHA256": bindings["task_bundle_sha256"],
        "ORBIT_Q_FRAMEWORK_PROMPT_SHA256": bindings["framework_prompt_sha256"],
        "ORBIT_Q_VERIFIER_HARNESS_SHA256": bindings["verifier_harness_sha256"],
    }


def _validate_real_expert_config(
    config: Any,
    bindings: dict[str, Any],
    seed: int,
    *,
    require_full_agent: bool,
) -> None:
    """Validate Harbor's real resolved Oracle config, not a synthetic projection."""
    if not isinstance(config, dict):
        raise ValueError("expert Harbor config must be a JSON object")
    task = config.get("task")
    if not isinstance(task, dict) or Path(str(task.get("path", ""))).resolve() != Path(
        bindings["task_bundle_path"]
    ):
        raise ValueError("expert Harbor config points at a different task bundle")
    agent = config.get("agent")
    if require_full_agent:
        if not isinstance(agent, dict):
            raise ValueError("expert Harbor result lacks its Oracle agent config")
        if (
            agent.get("name") != "oracle"
            or agent.get("import_path") is not None
            or agent.get("model_name") is not None
            or agent.get("kwargs", {})
            or agent.get("skills", [])
            or agent.get("mcp_servers", [])
            or agent.get("extra_allowed_hosts", [])
        ):
            raise ValueError("expert prequalification config contains a solver agent")
    elif agent is not None and (
        not isinstance(agent, dict) or agent.get("name") not in (None, "oracle")
    ):
        raise ValueError("expert Harbor sidecar contradicts Oracle mode")
    environment = config.get("environment")
    image_reference = f"{bindings['docker_image']}@{bindings['container_image_digest']}"
    if not isinstance(environment, dict) or any(
        (
            environment.get("import_path") != ENVIRONMENT_IMPORT_PATH,
            environment.get("override_cpus") != bindings["cpus"],
            environment.get("override_memory_mb") != bindings["memory_mb"],
            environment.get("cpu_enforcement_policy", "limit") != "limit",
            environment.get("memory_enforcement_policy", "limit") != "limit",
            environment.get("kwargs", {})
            != {"framework": "tensorcircuit", "docker_image": image_reference},
            environment.get("mounts") not in (None, []),
            bool(environment.get("extra_docker_compose", [])),
            bool(environment.get("extra_allowed_hosts", [])),
        )
    ):
        raise ValueError("expert Harbor config changed image/resources/task isolation")
    verifier = config.get("verifier")
    if not isinstance(verifier, dict) or (
        verifier.get("import_path") != VERIFIER_IMPORT_PATH
        or verifier.get("disable", False) is not False
        or verifier.get("kwargs", {}) != {"expert_only": True}
    ):
        raise ValueError("expert Harbor config lacks the expert-only verifier")
    verifier_env = verifier.get("env")
    expected_env = _required_expert_verifier_env(bindings, seed)
    if not isinstance(verifier_env, dict) or any(
        verifier_env.get(name) != value for name, value in expected_env.items()
    ):
        raise ValueError("expert Harbor verifier bindings differ from the task")
    optional_env = set(verifier_env) - set(expected_env)
    if optional_env not in (set(), {"HTTP_PROXY", "HTTPS_PROXY"}):
        raise ValueError("expert Harbor verifier env contains unapproved fields")


def _parse_expert_prequalification_run(
    run: Any,
    bindings: dict[str, Any],
    base: Path,
    root: Path | None,
) -> dict[str, Any]:
    """Derive one seed admission solely from a finished expert Harbor run."""
    expected_run_keys = {
        "job_id",
        "expert_passed",
        "runtime_sec",
        "protocol_seed",
        "case_digest",
        "admission",
        "threshold_policy_sha256",
        "run_evidence_sha256",
        "raw_evidence",
    }
    if not isinstance(run, dict) or set(run) != expected_run_keys:
        raise ValueError("expert prequalification run has an invalid shape")
    raw_evidence = run.get("raw_evidence")
    if not isinstance(raw_evidence, dict) or set(raw_evidence) != set(
        EXPERT_RUN_EVIDENCE_NAMES
    ):
        raise ValueError("expert prequalification raw evidence is incomplete")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    canonical_items: dict[str, dict[str, str]] = {}
    for name in EXPERT_RUN_EVIDENCE_NAMES:
        path, digest = _verified_expert_run_item(
            root, base, raw_evidence[name], f"expert run {name}"
        )
        paths[name] = path
        hashes[name] = digest
        canonical_items[name] = {"path": str(path), "sha256": digest}

    seed, case_digest = _parse_expert_functional_output(paths["functional_output"])
    admission, policy_hash = _parse_expert_admission_record(
        paths["admission_record"], bindings, seed, case_digest
    )
    config = _read_json_object(paths["harbor_config"], "expert Harbor config")
    trial_lock = _read_json_object(paths["harbor_lock"], "expert Harbor trial lock")
    job_lock = _read_json_object(paths["harbor_job_lock"], "expert Harbor job lock")
    result = _read_json_object(paths["raw_job"], "raw expert Harbor result")
    audit_details = _read_json_object(
        paths["audit_details"], "expert verifier audit details"
    )
    if audit_details.get("audit") != {
        "llm_audit_score": 1.0,
        "llm_audit_skipped": True,
    }:
        raise ValueError(
            "private expert audit details do not prove a skipped LLM audit"
        )
    if (
        result.get("finished_at") in (None, "")
        or result.get("exception_info") is not None
        or result.get("status") not in (None, "finished", "completed")
    ):
        raise ValueError("expert Harbor result is not a finished clean run")
    result_config = result.get("config")
    _validate_real_expert_config(result_config, bindings, seed, require_full_agent=True)
    _validate_real_expert_config(config, bindings, seed, require_full_agent=False)
    _validate_real_expert_config(trial_lock, bindings, seed, require_full_agent=True)
    if trial_lock.get("schema_version") != 1:
        raise ValueError("expert Harbor trial lock must use schema_version=1")
    locked_task = trial_lock.get("task", {})
    if (
        locked_task.get("name") != bindings["task_name"]
        or locked_task.get("type") != "local"
        or locked_task.get("digest") != f"sha256:{bindings['harbor_task_digest']}"
    ):
        raise ValueError("expert Harbor trial lock task binding differs")
    instructions = trial_lock.get("extra_instructions")
    if (
        not isinstance(instructions, list)
        or len(instructions) != 1
        or instructions[0].get("digest")
        != f"sha256:{bindings['framework_prompt_sha256']}"
        or Path(str(instructions[0].get("path", ""))).resolve()
        != Path(bindings["framework_prompt_path"]).resolve()
    ):
        raise ValueError("expert Harbor trial lock framework prompt differs")
    if (
        trial_lock.get("install_only", False) is not False
        or trial_lock.get("skills", [])
        or trial_lock.get("timeout_multiplier", 1.0) != 1.0
    ):
        raise ValueError("expert Harbor trial lock contains expanded controls")
    if (
        job_lock.get("schema_version") != 2
        or job_lock.get("n_concurrent_trials") != 1
        or job_lock.get("retry", {}).get("max_retries") != 0
        or job_lock.get("trials") != [trial_lock]
    ):
        raise ValueError("expert Harbor job lock is not one no-retry trial")
    harbor_identity = job_lock.get("harbor")
    if (
        not isinstance(harbor_identity, dict)
        or harbor_identity.get("version") != bindings["harbor_version"]
        or harbor_identity.get("is_editable") is not False
    ):
        raise ValueError("expert Harbor job lock identity differs")
    if result.get("task_checksum") != bindings["task_checksum"]:
        raise ValueError("raw expert Harbor task checksum differs from the task")
    agent_info = result.get("agent_info")
    if (
        not isinstance(agent_info, dict)
        or agent_info.get("name") != "oracle"
        or agent_info.get("model_info") is not None
    ):
        raise ValueError("expert prequalification did not use Harbor Oracle")
    agent_result = result.get("agent_result")
    if not isinstance(agent_result, dict) or any(
        agent_result.get(name) is not None
        for name in ("n_input_tokens", "n_output_tokens", "cost_usd")
    ):
        raise ValueError("expert Oracle result contains solver-model usage")
    verifier_info = result.get("verifier_info")
    if verifier_info is not None and (
        not isinstance(verifier_info, dict)
        or verifier_info.get("model_info") is not None
    ):
        raise ValueError("private expert verifier reports model usage")
    rewards = result.get("verifier_result", {}).get("rewards")
    if not isinstance(rewards, dict):
        raise ValueError("raw expert Harbor result is missing verifier rewards")
    for name in (
        "reward",
        "functional_score",
        "static_policy_score",
        "llm_audit_score",
    ):
        value = rewards.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value != 1:
            raise ValueError("expert Harbor verifier did not produce a complete pass")
    if (
        isinstance(rewards.get("llm_audit_skipped_score"), bool)
        or rewards.get("llm_audit_skipped_score") != 1
    ):
        raise ValueError("private expert rewards do not prove a skipped LLM audit")
    runtime = rewards.get("runtime_sec")
    if (
        not isinstance(runtime, (int, float))
        or isinstance(runtime, bool)
        or not math.isfinite(float(runtime))
        or runtime <= 0
    ):
        raise ValueError("expert Harbor runtime must be positive")
    if _expert_admission_hash_from_rewards(rewards) != hashes["admission_record"]:
        raise ValueError("raw expert Harbor result does not bind the admission record")
    job_id = _normalized_identity(result.get("id"), "expert Harbor job id")
    run_hash = _canonical_hash(
        {"job_id": job_id, "artifact_sha256s": hashes}, length=64
    )
    derived = {
        "job_id": job_id,
        "expert_passed": True,
        "runtime_sec": float(runtime),
        "protocol_seed": seed,
        "case_digest": case_digest,
        "admission": admission,
        "threshold_policy_sha256": policy_hash,
        "run_evidence_sha256": run_hash,
        "raw_evidence": canonical_items,
    }
    normalized_declared = {**run, "raw_evidence": canonical_items}
    if normalized_declared != derived:
        raise ValueError("declared expert run summary disagrees with raw evidence")
    return derived


def _prequalified_case_index(
    prototype: dict[str, Any], root: Path | None = None
) -> dict[int, dict[str, Any]]:
    """Rehash and reparse every canonical expert-only prequalification run."""
    cases = prototype.get("expert_prequalified_cases")
    if not isinstance(cases, list) or len(cases) < 25:
        raise ValueError("prototype needs at least 25 expert-prequalified cases")
    if prototype.get("expert_prequalified_case_count") != len(cases):
        raise ValueError("expert-prequalified case count does not match its records")
    bindings = _expert_prequalification_bindings(prototype)
    bundle_path = Path(prototype.get("evidence_bundle_path", ""))
    base = bundle_path.parent
    index: dict[int, dict[str, Any]] = {}
    digests: set[str] = set()
    job_ids: set[str] = set()
    run_hashes: set[str] = set()
    threshold_policies: set[str] = set()
    for position, case in enumerate(cases):
        try:
            derived = _parse_expert_prequalification_run(case, bindings, base, root)
        except (
            KeyError,
            TypeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ) as error:
            raise ValueError(
                f"stored expert-prequalified run {position} is unverifiable"
            ) from error
        seed = derived["protocol_seed"]
        digest = derived["case_digest"]
        job_id = derived["job_id"]
        run_hash = derived["run_evidence_sha256"]
        if (
            seed in index
            or digest in digests
            or job_id in job_ids
            or run_hash in run_hashes
        ):
            raise ValueError(
                "expert-prequalified seeds, case digests, jobs, and run bundles must be unique"
            )
        index[seed] = derived
        digests.add(digest)
        job_ids.add(job_id)
        run_hashes.add(run_hash)
        threshold_policies.add(derived["threshold_policy_sha256"])
    if len(threshold_policies) != 1:
        raise ValueError("expert prequalification must use one fixed threshold policy")
    return index


def _selected_prequalified_cases(
    prototype: dict[str, Any], seed_schedule: list[int], root: Path
) -> list[dict[str, Any]]:
    index = _prequalified_case_index(prototype, root)
    missing = [seed for seed in seed_schedule if seed not in index]
    if missing:
        raise ValueError(
            "seed schedule contains cases without exact expert prequalification: "
            f"{missing}"
        )
    return [index[seed] for seed in seed_schedule]


def _revalidate_manifest_prequalified_cases(
    root: Path, manifest: dict[str, Any], prototype: dict[str, Any]
) -> dict[int, dict[str, Any]]:
    selected = manifest.get("expert_prequalified_cases")
    schedule = manifest.get("seed_schedule")
    if not isinstance(schedule, list) or not isinstance(selected, list):
        raise ValueError("manifest is missing its expert-prequalified case binding")
    current = _selected_prequalified_cases(prototype, schedule, root)
    if selected != current:
        raise ValueError("manifest expert-prequalified case binding changed")
    if [case.get("protocol_seed") for case in selected] != schedule:
        raise ValueError(
            "manifest case records do not exactly match seed schedule order"
        )
    return {case["protocol_seed"]: case for case in selected}


def _clamp_score(value: int) -> int:
    return max(1, min(5, value))


def validate_catalog(catalog: dict[str, Any]) -> list[str]:
    """Return catalog invariant violations without mutating the input."""
    errors: list[str] = []
    sources = catalog.get("sources", [])
    source_ids = [row.get("id") for row in sources]
    if len(source_ids) != len(set(source_ids)):
        errors.append("source ids must be unique")
    for source in sources:
        if not source.get("url") or not source.get("title"):
            errors.append(f"source {source.get('id')} needs title and url")
        if source.get("evidence_class") not in {
            "official",
            "literature",
            "survey_inference",
        }:
            errors.append(f"source {source.get('id')} has invalid evidence class")

    families = catalog.get("families", [])
    family_ids = [row.get("id") for row in families]
    if len(families) != 50:
        errors.append("catalog must contain exactly 50 source-grounded families")
    if len(family_ids) != len(set(family_ids)):
        errors.append("family ids must be unique")
    source_id_set = set(source_ids)
    priorities: list[int] = []
    for family in families:
        missing = [name for name in SCORE_NAMES if name not in family.get("scores", {})]
        if missing:
            errors.append(f"family {family.get('id')} missing scores: {missing}")
        for name, value in family.get("scores", {}).items():
            if name in SCORE_NAMES and (
                not isinstance(value, int) or not 1 <= value <= 5
            ):
                errors.append(f"family {family.get('id')} invalid {name}: {value}")
        unknown_sources = set(family.get("source_ids", [])) - source_id_set
        if unknown_sources:
            errors.append(
                f"family {family.get('id')} references unknown sources: "
                f"{sorted(unknown_sources)}"
            )
        role_sources = {
            source_id
            for values in family.get("source_roles", {}).values()
            for source_id in values
        }
        if role_sources - set(family.get("source_ids", [])):
            errors.append(
                f"family {family.get('id')} source roles must reference source_ids"
            )
        priority = family.get("shortlist_priority")
        if priority is not None:
            priorities.append(priority)
            if not family.get("recommended_scale") or not family.get(
                "recommended_situation"
            ):
                errors.append(
                    f"shortlisted family {family.get('id')} needs a recommended regime"
                )

    if sorted(priorities) != list(range(1, 11)):
        errors.append("shortlist priorities must be exactly 1 through 10")

    scales = catalog.get("scale_regimes", [])
    situations = catalog.get("situations", [])
    if len(scales) != 4 or len(situations) != 5:
        errors.append("catalog needs four scales and five situations")
    if len({row.get("id") for row in scales}) != len(scales):
        errors.append("scale ids must be unique")
    if len({row.get("id") for row in situations}) != len(situations):
        errors.append("situation ids must be unique")
    for axis in (*scales, *situations):
        unknown = set(axis.get("score_deltas", {})) - set(SCORE_NAMES)
        if unknown:
            errors.append(f"axis {axis.get('id')} has unknown score deltas: {unknown}")

    if len(families) * len(scales) * len(situations) != 1000:
        errors.append("family x scale x situation matrix must contain 1,000 candidates")
    return list(dict.fromkeys(errors))


def _candidate_status(scores: dict[str, int], design_score: float) -> str:
    if scores["tensorcircuit_path_confidence"] < 3:
        return "hold_framework_obstruction"
    if (
        min(
            scores["verifier_feasibility"],
            scores["runtime_fit"],
            scores["determinism"],
        )
        < 3
    ):
        return "hold_feasibility"
    if scores["agent_difficulty_prior"] >= 4 and design_score >= 76:
        return "shortlist_eligible"
    return "sampled"


def generate_candidates(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand 50 families across four scales and five situations."""
    errors = validate_catalog(catalog)
    if errors:
        raise ValueError("invalid discovery catalog: " + "; ".join(errors))

    candidates: list[dict[str, Any]] = []
    source_index = {row["id"]: row for row in catalog["sources"]}
    for family in catalog["families"]:
        source_snapshot_hash = _canonical_hash(
            [source_index[source_id] for source_id in family["source_ids"]],
            length=64,
        )
        for scale in catalog["scale_regimes"]:
            for situation in catalog["situations"]:
                scores = dict(family["scores"])
                for axis in (scale, situation):
                    for name, delta in axis.get("score_deltas", {}).items():
                        scores[name] = _clamp_score(scores[name] + delta)
                design_score = round(
                    sum(scores[name] * SCORE_WEIGHTS[name] for name in SCORE_NAMES)
                    * 20,
                    1,
                )
                candidate_id = f"{family['id']}--{scale['id']}--{situation['id']}"
                core = {
                    "id": candidate_id,
                    "family_id": family["id"],
                    "title": family["title"],
                    "domain": family["domain"],
                    "mechanism": family["mechanism"],
                    "objective": family["objective"],
                    "scale": scale,
                    "situation": situation,
                    "source_ids": family["source_ids"],
                    "source_roles": family.get(
                        "source_roles", {"scientific_basis": family["source_ids"]}
                    ),
                    "source_snapshot_hash": source_snapshot_hash,
                    "tensorcircuit_path": family["tensorcircuit_path"],
                    "oracle_design": family["oracle_design"],
                    "novelty_note": family["novelty_note"],
                    "current_task_overlap": family.get("current_task_overlap", []),
                    "scores": scores,
                    "design_score": design_score,
                }
                candidate_payload = {
                    **core,
                    "screen_status": _candidate_status(scores, design_score),
                    "empirical_status": "untested",
                    "empirical_pass_rate": None,
                    "risk_flags": {
                        "framework_obstruction": 6
                        - scores["tensorcircuit_path_confidence"],
                        "oracle_weakness": 6 - scores["verifier_feasibility"],
                        "runtime_overflow": 6 - scores["runtime_fit"],
                        "stochastic_flake": 6 - scores["determinism"],
                    },
                    "evidence_class": "survey_inference",
                }
                candidate = {
                    **candidate_payload,
                    "candidate_hash": _canonical_hash(candidate_payload, length=64),
                }
                candidates.append(candidate)
    if len(candidates) != 1000 or len({row["id"] for row in candidates}) != 1000:
        raise AssertionError("candidate generation must yield 1,000 unique rows")
    return candidates


def select_shortlist(
    catalog: dict[str, Any], candidates: Iterable[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Choose one human-designed regime for each of ten diverse top families."""
    candidate_rows = list(candidates)
    for row in candidate_rows:
        _verified_payload_hash(row, "candidate_hash", "screen candidate")
    if len({row.get("id") for row in candidate_rows}) != len(candidate_rows):
        raise ValueError("screen candidate ids must be unique")
    index = {row["id"]: row for row in candidate_rows}
    selected: list[dict[str, Any]] = []
    priority_families = sorted(
        (row for row in catalog["families"] if row.get("shortlist_priority")),
        key=lambda row: row["shortlist_priority"],
    )
    for family in priority_families:
        candidate_id = (
            f"{family['id']}--{family['recommended_scale']}--"
            f"{family['recommended_situation']}"
        )
        candidate = dict(index[candidate_id])
        if candidate["screen_status"] != "shortlist_eligible":
            raise ValueError(f"preferred candidate is not eligible: {candidate_id}")
        candidate.update(
            {
                "rank": family["shortlist_priority"],
                "why_plausibly_hard": family["why_plausibly_hard"],
                "falsification_check": family["falsification_check"],
                "review_questions": family["review_questions"],
            }
        )
        candidate["screen_candidate_hash"] = candidate.pop("candidate_hash")
        candidate["candidate_hash"] = _canonical_hash(candidate, length=64)
        selected.append(candidate)
    if len(selected) != 10:
        raise AssertionError("shortlist must contain exactly ten candidates")
    return selected


def _verified_shortlist_candidate(root: Path, candidate_id: str) -> dict[str, Any]:
    shortlist = _read_json_object(root / "shortlist.json", "shortlist")
    candidates = shortlist.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("shortlist candidates must be a list")
    if any(not isinstance(row, dict) for row in candidates):
        raise ValueError("each shortlist candidate must be a JSON object")
    candidate = next((row for row in candidates if row.get("id") == candidate_id), None)
    if candidate is None:
        raise KeyError(f"candidate is not in the current shortlist: {candidate_id}")
    if sum(row.get("id") == candidate_id for row in candidates) != 1:
        raise ValueError(f"shortlist candidate id is not unique: {candidate_id}")
    _verified_payload_hash(candidate, "candidate_hash", "shortlist candidate")

    source_rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        (root / "candidates.jsonl").read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"candidate row {line_number} must be a JSON object")
        if row.get("id") == candidate_id:
            source_rows.append(row)
    if len(source_rows) != 1:
        raise ValueError(
            f"candidate source row must occur exactly once: {candidate_id}"
        )
    source = source_rows[0]
    _verified_payload_hash(source, "candidate_hash", "candidate row")
    if candidate.get("screen_candidate_hash") != source["candidate_hash"]:
        raise ValueError("shortlist candidate is not bound to its source row hash")
    shortlist_screen_payload = {
        key: value
        for key, value in candidate.items()
        if key
        not in {
            "candidate_hash",
            "screen_candidate_hash",
            *SHORTLIST_REVIEW_FIELDS,
        }
    }
    source_payload = {
        key: value for key, value in source.items() if key != "candidate_hash"
    }
    if shortlist_screen_payload != source_payload:
        raise ValueError("shortlist candidate payload differs from its source row")
    return candidate


def _summary(
    catalog: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    shortlist = select_shortlist(catalog, candidates)
    return {
        "schema_version": 1,
        "candidate_count": len(candidates),
        "family_count": len(catalog["families"]),
        "source_count": len(catalog["sources"]),
        "shortlist_count": len(shortlist),
        "empirically_tested_count": 0,
        "evidence_classes": ["official", "literature", "survey_inference"],
        "domains": dict(sorted(Counter(row["domain"] for row in candidates).items())),
        "screen_statuses": dict(
            sorted(Counter(row["screen_status"] for row in candidates).items())
        ),
        "score_weights": SCORE_WEIGHTS,
        "catalog_source_snapshot_hash": _canonical_hash(catalog["sources"], length=64),
        "shortlist_snapshot": _canonical_hash(
            [(row["id"], row["candidate_hash"]) for row in shortlist], length=64
        ),
        "claim_boundary": (
            "Scores are design priors, not measured GPT-5.6-sol outcomes. "
            "No real solver trials were run during discovery."
        ),
    }


def _shortlist_markdown(
    catalog: dict[str, Any], shortlist: list[dict[str, Any]], summary: dict[str, Any]
) -> str:
    source_index = {row["id"]: row for row in catalog["sources"]}
    lines = [
        "# ORBIT-Q future-problem shortlist",
        "",
        "> **Claim boundary:** This is a source-grounded design screen. The scores are",
        "> priors, not evidence that GPT-5.6-sol failed. Real model tests remain blocked",
        "> until the human and expert-baseline gates pass.",
        "",
        "## Screening result",
        "",
        f"The deterministic matrix contains **{summary['candidate_count']:,} candidates**",
        f"from **{summary['family_count']} families**, four scale regimes, and five",
        "situations. Ten candidates were selected for design review; none was",
        "empirically tested in this frozen discovery snapshot; current expert/model",
        "evidence is maintained separately.",
        "",
        "| Rank | Candidate | Domain | Design prior | TC path | Verifier |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in shortlist:
        scores = row["scores"]
        lines.append(
            f"| {row['rank']} | `{row['id']}` — {row['title']} | {row['domain']} | "
            f"{row['design_score']:.1f}/100 | "
            f"{scores['tensorcircuit_path_confidence']}/5 | "
            f"{scores['verifier_feasibility']}/5 |"
        )

    for row in shortlist:
        lines.extend(
            [
                "",
                f"## {row['rank']}. {row['title']}",
                "",
                f"- Candidate: `{row['id']}` (`{row['candidate_hash']}`)",
                f"- Contract: {row['objective']} {row['scale']['description']} "
                f"{row['situation']['description']}",
                f"- TensorCircuit path: {row['tensorcircuit_path']}",
                f"- Oracle: {row['oracle_design']}",
                f"- Why plausibly hard: {row['why_plausibly_hard']}",
                f"- Falsification check: {row['falsification_check']}",
                f"- Current-suite relation: {row['novelty_note']}",
                "- Human review questions: " + "; ".join(row["review_questions"]),
                "- Evidence roles: "
                + "; ".join(
                    f"{role}="
                    + ", ".join(source_index[source_id]["title"] for source_id in ids)
                    for role, ids in row["source_roles"].items()
                ),
                "- Sources: "
                + ", ".join(
                    f"[{source_index[source_id]['title']}]"
                    f"({source_index[source_id]['url']})"
                    for source_id in row["source_ids"]
                ),
            ]
        )

    lines.extend(
        [
            "",
            "## Human-gated path to a real test",
            "",
            "1. Three concept roles approve: quantum scientist, TensorCircuit expert, and verifier engineer.",
            "2. An expert implementation and independent evaluator are built outside `tasks/`.",
            "3. Public-API canary and 20 verifier-only reproductions pass in the pinned image.",
            "4. Gold code is at most 160 effective lines; expert p95 is at most 180 seconds; the independent oracle is at most 240 seconds.",
            "5. Benchmark owner and independent reviewer approve the frozen pilot package.",
            "6. The CLI emits a hash-bound run manifest. It never launches Harbor itself.",
            "7. Run a precommitted five-trial GPT-5.6-sol pilot under one frozen protocol.",
            "8. Two independent auditors classify every failure and exclude infrastructure, auth, policy refusal, ambiguous spec, and framework impossibility.",
            "9. A model-hard evidence label requires a fresh precommitted confirmation with at least 20 valid trials and zero passes.",
            "",
            "Five failures are only a pilot signal. Only failures that remain after step 9 count toward a protocol-scoped model-hard evidence label.",
            "",
        ]
    )
    return "\n".join(lines)


def _initial_review_state(
    shortlist: list[dict[str, Any]], existing: dict[str, Any] | None
) -> dict[str, Any]:
    state = existing or {
        "schema_version": 1,
        "candidates": {},
        "event_log": [],
    }
    state.pop("source_snapshot_hash", None)
    active_ids = {row["id"] for row in shortlist}
    for row in shortlist:
        fresh_record = {
            "candidate_hash": row["candidate_hash"],
            "concept_reviews": {},
            "prototype": {},
            "pilot_reviews": {},
            "authorizations": [],
            "model_trials": {},
        }
        record = state["candidates"].setdefault(row["id"], fresh_record)
        record.setdefault("authorizations", [])
        record.setdefault("model_trials", {})
        if record.get("candidate_hash") != row["candidate_hash"]:
            has_evidence = any(
                record.get(name)
                for name in (
                    "concept_reviews",
                    "prototype",
                    "pilot_reviews",
                    "authorizations",
                    "model_trials",
                )
            )
            if has_evidence:
                archived = {
                    **record,
                    "invalidated_at": _utc_now(),
                    "superseded_by_candidate_hash": row["candidate_hash"],
                }
                state.setdefault("archived_candidate_versions", {}).setdefault(
                    row["id"], []
                ).append(archived)
                _append_event(
                    state,
                    {
                        "event": "candidate_version_invalidated",
                        "candidate_id": row["id"],
                        "prior_candidate_hash": record.get("candidate_hash"),
                        "candidate_hash": row["candidate_hash"],
                        "timestamp": archived["invalidated_at"],
                    },
                )
            state["candidates"][row["id"]] = fresh_record
    for candidate_id in list(state["candidates"]):
        if candidate_id in active_ids:
            continue
        record = state["candidates"][candidate_id]
        has_evidence = any(
            record.get(name)
            for name in (
                "concept_reviews",
                "prototype",
                "pilot_reviews",
                "authorizations",
                "model_trials",
            )
        )
        if has_evidence:
            archived_at = _utc_now()
            state.setdefault("archived_candidate_versions", {}).setdefault(
                candidate_id, []
            ).append({**record, "invalidated_at": archived_at})
            _append_event(
                state,
                {
                    "event": "candidate_removed_from_shortlist",
                    "candidate_id": candidate_id,
                    "candidate_hash": record.get("candidate_hash"),
                    "timestamp": archived_at,
                },
            )
        del state["candidates"][candidate_id]
    state["active_shortlist_ids"] = sorted(active_ids)
    state["shortlist_source_snapshot_hash"] = _canonical_hash(
        sorted({row["source_snapshot_hash"] for row in shortlist}), length=64
    )
    state["shortlist_snapshot"] = _canonical_hash(
        [(row["id"], row["candidate_hash"]) for row in shortlist], length=64
    )
    return state


def build_workspace(root: Path) -> dict[str, Any]:
    """Build deterministic artifacts and preserve any existing human reviews."""
    with _review_state_lock(root):
        return _build_workspace_locked(root)


def _build_workspace_locked(root: Path) -> dict[str, Any]:
    """Rebuild generated artifacts inside the review-ledger transaction lock."""
    catalog = _read_json(root / "catalog.json")
    candidates = generate_candidates(catalog)
    shortlist = select_shortlist(catalog, candidates)
    summary = _summary(catalog, candidates)

    (root / "candidates.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in candidates
        ),
        encoding="utf-8",
    )
    _write_json(
        root / "shortlist.json",
        {
            "schema_version": 1,
            "claim_boundary": summary["claim_boundary"],
            "candidates": shortlist,
        },
    )
    _write_json(root / "summary.json", summary)
    (root / "shortlist.md").write_text(
        _shortlist_markdown(catalog, shortlist, summary), encoding="utf-8"
    )

    review_path = root / "review_state.json"
    existing = _read_json(review_path) if review_path.exists() else None
    _write_json(review_path, _initial_review_state(shortlist, existing))
    return summary


def _decision_status(reviews: dict[str, Any], roles: Iterable[str]) -> dict[str, Any]:
    roles = tuple(roles)
    if not isinstance(reviews, dict):
        reviews = {}
    missing = [role for role in roles if role not in reviews]
    unexpected = sorted(set(reviews) - set(roles))
    invalid: list[str] = []
    reviewer_keys: list[str] = []
    for role in roles:
        review = reviews.get(role)
        if review is None:
            continue
        if not isinstance(review, dict):
            invalid.append(role)
            continue
        try:
            reviewer = _normalized_identity(review.get("reviewer"), "reviewer")
            note = _normalized_nonempty_text(review.get("note"), "review note")
            if reviewer != review.get("reviewer") or note != review.get("note"):
                invalid.append(role)
            reviewer_keys.append(_identity_key(reviewer))
        except ValueError:
            invalid.append(role)
    if len(reviewer_keys) != len(set(reviewer_keys)):
        invalid.append("duplicate_reviewer_identity")
    rejected = [
        role
        for role in roles
        if isinstance(reviews.get(role), dict)
        and reviews[role].get("decision") == "reject"
    ]
    approved = [
        role
        for role in roles
        if isinstance(reviews.get(role), dict)
        and reviews[role].get("decision") == "approve"
        and role not in invalid
    ]
    return {
        "passed": not missing
        and not unexpected
        and not invalid
        and not rejected
        and len(approved) == len(roles),
        "missing_roles": missing,
        "unexpected_roles": unexpected,
        "invalid_roles": sorted(set(invalid)),
        "rejected_roles": rejected,
        "approved_roles": approved,
    }


def _gate_fingerprint(record: dict[str, Any]) -> str:
    return _canonical_hash(
        {
            "candidate_hash": record["candidate_hash"],
            "concept_reviews": record.get("concept_reviews", {}),
            "prototype": record.get("prototype", {}),
            "pilot_reviews": record.get("pilot_reviews", {}),
        },
        length=64,
    )


def _revoke_active_authorization(
    state: dict[str, Any],
    record: dict[str, Any],
    candidate_id: str,
    reason: str,
    timestamp: str,
) -> None:
    manifest_hash = record.pop("active_authorization_hash", None)
    if not manifest_hash:
        return
    authorization = next(
        (
            row
            for row in record.get("authorizations", [])
            if row.get("manifest_hash") == manifest_hash
        ),
        None,
    )
    if authorization is not None:
        authorization["revoked_at"] = timestamp
        authorization["revocation_reason"] = reason
    _append_event(
        state,
        {
            "event": "model_test_authorization_revoked",
            "candidate_id": candidate_id,
            "manifest_hash": manifest_hash,
            "reason": reason,
            "timestamp": timestamp,
        },
    )


def _reviewer_identities(record: dict[str, Any]) -> set[str]:
    identities = [
        _normalized_identity(review.get("reviewer"), "reviewer")
        for group in ("concept_reviews", "pilot_reviews")
        for review in record.get(group, {}).values()
    ]
    identities.extend(
        _normalized_identity(author, "prototype artifact author")
        for author in record.get("prototype", {}).get("artifact_authors", [])
    )
    return {_identity_key(identity) for identity in identities}


def candidate_gate_status(state: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    record = state.get("candidates", {}).get(candidate_id)
    if record is None:
        raise KeyError(f"candidate is not in review state: {candidate_id}")
    concept = _decision_status(record.get("concept_reviews", {}), CONCEPT_REVIEW_ROLES)
    try:
        review_identities = [
            _identity_key(review.get("reviewer"))
            for group in ("concept_reviews", "pilot_reviews")
            for review in record.get(group, {}).values()
        ]
        reviewer_identity_integrity_ok = len(review_identities) == len(
            set(review_identities)
        )
    except (AttributeError, TypeError, ValueError):
        reviewer_identity_integrity_ok = False
    prototype = record.get("prototype", {})
    prototype_missing = [
        name
        for name in REQUIRED_PROTOTYPE_FIELDS
        if name not in prototype or prototype[name] in (None, "")
    ]

    def in_range(name: str, lower: float, upper: float) -> bool:
        value = prototype.get(name)
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and lower <= value <= upper
        )

    def is_hex(value: Any, lengths: tuple[int, ...]) -> bool:
        return (
            isinstance(value, str)
            and len(value) in lengths
            and all(character in "0123456789abcdef" for character in value.lower())
        )

    expert_runtime_ok = in_range("expert_runtime_p95_sec", 0.000001, 180)
    oracle_runtime_ok = in_range("independent_oracle_runtime_sec", 0.000001, 240)
    line_count_ok = in_range("gold_effective_lines", 1, 160)
    reproducibility_ok = in_range("reproducibility_runs", 20, 10_000)
    resource_envelope_ok = (
        in_range("observed_cpu_count", 1, 10_000)
        and in_range("observed_memory_mb", 1, 10_000_000)
        and in_range("expert_peak_memory_mb", 0.000001, 10_000_000)
        and prototype.get("expert_peak_memory_mb", float("inf"))
        <= 0.70 * prototype.get("observed_memory_mb", 0)
    )
    public_api_ok = prototype.get("public_api_canary_passed") is True
    verifier_ok = prototype.get("verifier_only_passed") is True
    artifact_hashes_ok = all(
        is_hex(prototype.get(name), (64,))
        for name in (
            "expert_baseline_sha256",
            "independent_oracle_sha256",
            "evaluator_sha256",
            "instruction_sha256",
            "candidate_metadata_sha256",
            "verifier_harness_sha256",
            "task_policy_sha256",
            "solution_driver_sha256",
            "task_bundle_sha256",
            "framework_prompt_sha256",
            "evidence_bundle_sha256",
        )
    )
    artifacts_still_match = True
    for path_name, hash_name in (
        ("expert_baseline_path", "expert_baseline_sha256"),
        ("independent_oracle_path", "independent_oracle_sha256"),
        ("evaluator_path", "evaluator_sha256"),
        ("instruction_path", "instruction_sha256"),
        ("candidate_metadata_path", "candidate_metadata_sha256"),
        ("task_bundle_path", "task_bundle_sha256"),
        ("framework_prompt_path", "framework_prompt_sha256"),
        ("evidence_bundle_path", "evidence_bundle_sha256"),
    ):
        try:
            artifacts_still_match = artifacts_still_match and (
                _path_sha256(Path(prototype[path_name])) == prototype[hash_name]
            )
        except (KeyError, FileNotFoundError, ValueError, OSError):
            artifacts_still_match = False
    supporting_evidence_still_matches = False
    try:
        evidence_bundle_path = Path(prototype["evidence_bundle_path"])
        evidence_bundle = _read_json(evidence_bundle_path)
        evidence_base = evidence_bundle_path.parent

        def bundle_item_matches(item: dict[str, Any]) -> bool:
            item_path = Path(item["path"])
            if not item_path.is_absolute():
                item_path = evidence_base / item_path
            return _path_sha256(item_path) == item["sha256"]

        supporting_items = [evidence_bundle["public_api_canary"]["log"]]
        supporting_items.extend(run["log"] for run in evidence_bundle["verifier_runs"])
        supporting_items.extend(
            run["log"] for run in evidence_bundle["independent_oracle_runs"]
        )
        for case in evidence_bundle["expert_prequalified_cases"]:
            supporting_items.extend(case["raw_evidence"].values())
        supporting_items.extend(
            run["log"] for run in evidence_bundle["valid_alternatives"]
        )
        supporting_items.extend(run["log"] for run in evidence_bundle["mutation_tests"])
        supporting_evidence_still_matches = all(
            bundle_item_matches(item) for item in supporting_items
        )
    except (
        KeyError,
        TypeError,
        FileNotFoundError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ):
        supporting_evidence_still_matches = False
    try:
        independent_authors = prototype.get("independent_artifact_authors", [])
        artifact_authors = prototype.get("artifact_authors", [])
        independent_artifacts_ok = (
            isinstance(independent_authors, list)
            and len(independent_authors) == 3
            and all(
                author == _normalized_identity(author, "independent artifact author")
                for author in independent_authors
            )
            and len({_identity_key(author) for author in independent_authors}) == 3
            and isinstance(artifact_authors, list)
            and all(
                author == _normalized_identity(author, "artifact author")
                for author in artifact_authors
            )
            and len(
                {
                    prototype.get("expert_baseline_sha256"),
                    prototype.get("independent_oracle_sha256"),
                    prototype.get("evaluator_sha256"),
                }
            )
            == 3
        )
    except ValueError:
        independent_artifacts_ok = False
    try:
        expert_prequalified_cases_ok = len(_prequalified_case_index(prototype)) >= 25
    except (
        KeyError,
        TypeError,
        FileNotFoundError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ):
        expert_prequalified_cases_ok = False
    verifier_challenge_ok = in_range(
        "valid_alternatives_accepted", 2, 10_000
    ) and in_range("mutation_tests_rejected", 6, 10_000)
    image_digest = prototype.get("container_image_digest", "")
    image_digest_ok = (
        isinstance(image_digest, str)
        and image_digest.startswith("sha256:")
        and is_hex(image_digest.removeprefix("sha256:"), (64,))
    )
    source_commit = prototype.get("source_commit")
    source_commit_ok = (
        isinstance(source_commit, str)
        and 7 <= len(source_commit) <= 64
        and all(character in "0123456789abcdef" for character in source_commit.lower())
    )
    prototype_passed = not prototype_missing and all(
        (
            public_api_ok,
            verifier_ok,
            artifact_hashes_ok,
            artifacts_still_match,
            supporting_evidence_still_matches,
            independent_artifacts_ok,
            expert_prequalified_cases_ok,
            verifier_challenge_ok,
            image_digest_ok,
            source_commit_ok,
            expert_runtime_ok,
            oracle_runtime_ok,
            line_count_ok,
            reproducibility_ok,
            resource_envelope_ok,
        )
    )
    pilot = _decision_status(record.get("pilot_reviews", {}), PILOT_REVIEW_ROLES)
    return {
        "candidate_id": candidate_id,
        "concept_gate": concept,
        "prototype_gate": {
            "passed": prototype_passed,
            "missing_fields": prototype_missing,
            "public_api_canary_passed": public_api_ok,
            "artifact_hashes_valid": artifact_hashes_ok,
            "artifacts_still_match": artifacts_still_match,
            "supporting_evidence_still_matches": supporting_evidence_still_matches,
            "independent_artifacts_valid": independent_artifacts_ok,
            "expert_prequalified_cases_valid": expert_prequalified_cases_ok,
            "verifier_challenge_passed": verifier_challenge_ok,
            "container_image_digest_valid": image_digest_ok,
            "source_commit_valid": source_commit_ok,
            "expert_p95_within_180s": expert_runtime_ok,
            "independent_oracle_within_240s": oracle_runtime_ok,
            "gold_within_160_effective_lines": line_count_ok,
            "at_least_20_reproducibility_runs": reproducibility_ok,
            "resource_envelope_valid": resource_envelope_ok,
            "verifier_only_passed": verifier_ok,
        },
        "pilot_gate": pilot,
        "reviewer_identity_integrity_passed": reviewer_identity_integrity_ok,
        "ready_for_model_test": concept["passed"]
        and prototype_passed
        and pilot["passed"]
        and reviewer_identity_integrity_ok,
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def record_review(
    root: Path,
    candidate_id: str,
    gate: str,
    role: str,
    decision: str,
    reviewer: str,
    note: str,
) -> dict[str, Any]:
    """Atomically append one human review and any resulting revocation."""
    with _review_state_lock(root):
        return _record_review_locked(
            root, candidate_id, gate, role, decision, reviewer, note
        )


def _record_review_locked(
    root: Path,
    candidate_id: str,
    gate: str,
    role: str,
    decision: str,
    reviewer: str,
    note: str,
) -> dict[str, Any]:
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject")
    roles = CONCEPT_REVIEW_ROLES if gate == "concept" else PILOT_REVIEW_ROLES
    if gate not in {"concept", "pilot"} or role not in roles:
        raise ValueError(f"invalid {gate} review role: {role}")
    reviewer = _normalized_identity(reviewer, "reviewer")
    note = _normalized_nonempty_text(note, "review note")
    path = root / "review_state.json"
    state = _read_json(path)
    record = state["candidates"][candidate_id]
    key = "concept_reviews" if gate == "concept" else "pilot_reviews"
    if role in record[key]:
        raise ValueError(
            "reviews are append-only; revise the candidate to review again"
        )
    if _identity_key(reviewer) in _reviewer_identities(record):
        raise ValueError("each review role requires a distinct reviewer identity")
    if role == "independent_reviewer" and _identity_key(reviewer) in {
        _identity_key(author)
        for author in record.get("prototype", {}).get("artifact_authors", [])
    }:
        raise ValueError("the independent reviewer cannot author a prototype artifact")
    timestamp = _utc_now()
    _revoke_active_authorization(
        state,
        record,
        candidate_id,
        f"{gate}_review_mutated",
        timestamp,
    )
    review = {
        "decision": decision,
        "reviewer": reviewer,
        "note": note,
        "timestamp": timestamp,
    }
    record[key][role] = review
    _append_event(
        state,
        {
            "event": "review",
            "candidate_id": candidate_id,
            "gate": gate,
            "role": role,
            **review,
        },
    )
    _write_json(path, state)
    return candidate_gate_status(state, candidate_id)


def record_prototype(
    root: Path, candidate_id: str, evidence_bundle_path: Path
) -> dict[str, Any]:
    """Atomically verify and append one immutable prototype evidence bundle."""
    with _review_state_lock(root):
        return _record_prototype_locked(root, candidate_id, evidence_bundle_path)


def _record_prototype_locked(
    root: Path, candidate_id: str, evidence_bundle_path: Path
) -> dict[str, Any]:
    """Verify a raw prototype bundle and record only derived gate evidence."""
    state_path = root / "review_state.json"
    state = _read_json(state_path)
    record = state["candidates"][candidate_id]
    if record.get("prototype"):
        raise ValueError(
            "prototype evidence is append-only; revise the candidate to replace it"
        )
    concept_status = _decision_status(
        record.get("concept_reviews", {}), CONCEPT_REVIEW_ROLES
    )
    if not concept_status["passed"]:
        raise PermissionError("concept reviews must pass before prototype evidence")

    bundle_path = _safe_evidence_path(
        root, evidence_bundle_path.parent, str(evidence_bundle_path)
    )
    if not bundle_path.is_file():
        raise FileNotFoundError(bundle_path)
    bundle = _read_json(bundle_path)
    if bundle.get("schema_version") != 2:
        raise ValueError("prototype evidence bundle must use schema_version=2")
    if bundle.get("candidate_id") != candidate_id:
        raise ValueError("prototype bundle candidate_id does not match")
    if bundle.get("candidate_hash") != record["candidate_hash"]:
        raise ValueError("prototype bundle candidate hash is stale")
    base = bundle_path.parent

    artifact_records = bundle.get("artifacts", {})
    artifact_names = (
        "expert_baseline",
        "independent_oracle",
        "evaluator",
        "instruction",
        "candidate_metadata",
        "task_bundle",
        "framework_prompt",
    )
    artifacts: dict[str, dict[str, str]] = {}
    for name in artifact_names:
        item = artifact_records.get(name, {})
        artifact_path, artifact_hash = _verified_evidence_item(root, base, item, name)
        author = _normalized_identity(item.get("author"), f"{name} author")
        artifacts[name] = {
            "path": str(artifact_path),
            "sha256": artifact_hash,
            "author": author,
        }
    independent_names = ("expert_baseline", "independent_oracle", "evaluator")
    independent_authors = [artifacts[name]["author"] for name in independent_names]
    independent_hashes = [artifacts[name]["sha256"] for name in independent_names]
    if len({_identity_key(author) for author in independent_authors}) != len(
        independent_authors
    ):
        raise ValueError("baseline, oracle, and evaluator require distinct authors")
    if len(set(independent_hashes)) != len(independent_hashes):
        raise ValueError("baseline, oracle, and evaluator must be distinct artifacts")

    task_bundle_path = Path(artifacts["task_bundle"]["path"]).resolve()
    if not task_bundle_path.is_dir():
        raise ValueError("task_bundle artifact must be the materialized task directory")
    from scripts.materialize_candidate_task import (  # noqa: PLC0415
        CandidateTaskError,
        validate_materialized_task_contract,
    )

    try:
        materialization_contract = validate_materialized_task_contract(
            task_bundle_path, root, require_expert_admission=True
        )
    except CandidateTaskError as exc:
        raise ValueError(
            f"materialized task fails the trusted source contract: {exc}"
        ) from exc
    task_metadata = _read_json_object(
        task_bundle_path / "candidate_metadata.json", "materialized candidate metadata"
    )
    discovery_binding = task_metadata.get("discovery_binding")
    if (
        task_metadata.get("candidate_id") != candidate_id
        or not isinstance(discovery_binding, dict)
        or discovery_binding.get("candidate_id") != candidate_id
        or discovery_binding.get("candidate_hash") != record["candidate_hash"]
        or discovery_binding.get("candidates_jsonl_sha256")
        != _file_sha256(root / "candidates.jsonl")
        or discovery_binding.get("shortlist_json_sha256")
        != _file_sha256(root / "shortlist.json")
    ):
        raise ValueError(
            "materialized task is not bound to the current reviewed candidate"
        )
    task_test_id = (task_bundle_path / "tests" / "problem_id.txt").read_text().strip()
    task_solution_id = (
        (task_bundle_path / "solution" / "problem_id.txt").read_text().strip()
    )
    if (
        not task_test_id.isdigit()
        or task_test_id == "0"
        or task_test_id != task_solution_id
    ):
        raise ValueError("materialized task problem IDs are invalid")
    task_evaluator = task_bundle_path / "tests" / f"evaluate_{task_test_id}.py"
    task_expert = task_bundle_path / "solution" / f"solution_{task_test_id}.py"
    if (
        not task_evaluator.is_file()
        or task_evaluator.is_symlink()
        or _file_sha256(task_evaluator) != artifacts["evaluator"]["sha256"]
        or not task_expert.is_file()
        or task_expert.is_symlink()
        or _file_sha256(task_expert) != artifacts["expert_baseline"]["sha256"]
    ):
        raise ValueError(
            "materialized task evaluator/expert differ from reviewed artifacts"
        )
    if (
        _file_sha256(task_bundle_path / "instruction.md")
        != artifacts["instruction"]["sha256"]
        or _file_sha256(task_bundle_path / "candidate_metadata.json")
        != artifacts["candidate_metadata"]["sha256"]
        or materialization_contract["instruction_sha256"]
        != artifacts["instruction"]["sha256"]
        or materialization_contract["candidate_metadata_sha256"]
        != artifacts["candidate_metadata"]["sha256"]
    ):
        raise ValueError(
            "materialized instruction/metadata differ from reviewed artifacts"
        )
    admission_marker = _read_json_object(
        task_bundle_path / "tests" / "expert_admission_protocol.json",
        "materialized expert admission protocol",
    )
    if admission_marker != {
        "schema_version": 1,
        "status": "candidate_specific_metrics_declared",
        "structured_output_key": "orbit_q_expert_admission_metrics",
    }:
        raise ValueError("materialized task lacks candidate-specific admission metrics")

    environment = bundle.get("environment", {})
    image_digest = environment.get("container_image_digest", "")
    if not (
        isinstance(image_digest, str)
        and image_digest.startswith("sha256:")
        and len(image_digest.removeprefix("sha256:")) == 64
        and all(
            character in "0123456789abcdef"
            for character in image_digest.removeprefix("sha256:").lower()
        )
    ):
        raise ValueError("environment needs a full container image SHA-256 digest")
    source_commit = environment.get("source_commit")
    if not (
        isinstance(source_commit, str)
        and 7 <= len(source_commit) <= 64
        and all(character in "0123456789abcdef" for character in source_commit.lower())
    ):
        raise ValueError("environment source_commit must be hexadecimal")
    docker_image = _normalized_identity(
        environment.get("docker_image"), "environment docker image"
    )
    if (
        "tensorcircuit" not in docker_image.casefold()
        or "@" in docker_image
        or any(character.isspace() for character in docker_image)
    ):
        raise ValueError("environment docker_image must be a TensorCircuit name:tag")
    harbor_version = _normalized_identity(
        environment.get("harbor_version"), "environment Harbor version"
    )
    harbor_task_digest = _validated_sha256(
        environment.get("harbor_task_digest"), "environment Harbor task digest"
    )
    task_checksum = _validated_sha256(
        environment.get("task_checksum"), "environment task checksum"
    )
    if harbor_task_digest != _harbor_task_digest(task_bundle_path):
        raise ValueError(
            "environment Harbor task digest is not derived from task_bundle"
        )
    if task_checksum != _harbor_legacy_task_checksum(task_bundle_path):
        raise ValueError("environment task checksum is not derived from task_bundle")
    observed_cpu_count = environment.get("observed_cpu_count")
    observed_memory_mb = environment.get("observed_memory_mb")
    if (
        not isinstance(observed_cpu_count, int)
        or isinstance(observed_cpu_count, bool)
        or observed_cpu_count < 1
    ):
        raise ValueError("observed_cpu_count must be a positive integer")
    if (
        not isinstance(observed_memory_mb, int)
        or isinstance(observed_memory_mb, bool)
        or observed_memory_mb <= 0
    ):
        raise ValueError("observed_memory_mb must be a positive integer")

    canary = bundle.get("public_api_canary", {})
    if canary.get("passed") is not True:
        raise ValueError("public API canary did not pass")
    canary_path, canary_log_hash = _verified_evidence_item(
        root, base, canary.get("log", {}), "public API canary log"
    )
    canary_result = _read_json_object(canary_path, "public API canary result")
    if (
        canary_result.get("passed") is not True
        or canary_result.get("container_image_digest") != image_digest
        or canary_result.get("source_commit") != source_commit
    ):
        raise ValueError(
            "public API canary result does not match the frozen environment"
        )

    verifier_runs = bundle.get("verifier_runs", [])
    if not isinstance(verifier_runs, list) or len(verifier_runs) < 20:
        raise ValueError("prototype bundle needs at least 20 verifier-only runs")
    verifier_job_ids: list[str] = []
    verifier_log_hashes: list[str] = []
    runtimes: list[float] = []
    peak_memories: list[float] = []
    verifier_records: list[dict[str, Any]] = []
    for index, run in enumerate(verifier_runs):
        log_path, log_hash = _verified_evidence_item(
            root, base, run.get("log", {}), f"verifier run {index} log"
        )
        derived = _parse_prototype_verifier_result(log_path, image_digest)
        declared = {
            "job_id": run.get("job_id"),
            "runtime_sec": float(run["runtime_sec"])
            if isinstance(run.get("runtime_sec"), (int, float))
            and not isinstance(run.get("runtime_sec"), bool)
            else None,
            "peak_memory_mb": float(run["peak_memory_mb"])
            if isinstance(run.get("peak_memory_mb"), (int, float))
            and not isinstance(run.get("peak_memory_mb"), bool)
            else None,
        }
        if (
            run.get("passed") is not True
            or run.get("container_image_digest") != image_digest
            or declared != derived
        ):
            raise ValueError(
                f"verifier run {index} disagrees with its structured result"
            )
        job_id = derived["job_id"]
        runtime = derived["runtime_sec"]
        peak_memory = derived["peak_memory_mb"]
        verifier_job_ids.append(job_id)
        verifier_log_hashes.append(log_hash)
        runtimes.append(float(runtime))
        peak_memories.append(float(peak_memory))
        verifier_records.append(
            {
                "job_id": job_id,
                "runtime_sec": float(runtime),
                "peak_memory_mb": float(peak_memory),
                "log_sha256": log_hash,
            }
        )
    if len(set(verifier_job_ids)) != len(verifier_job_ids):
        raise ValueError("verifier run job IDs must be unique")
    if len(set(verifier_log_hashes)) != len(verifier_log_hashes):
        raise ValueError("verifier run logs must be unique")

    prequalified_cases = bundle.get("expert_prequalified_cases", [])
    if not isinstance(prequalified_cases, list) or len(prequalified_cases) < 25:
        raise ValueError("prototype bundle needs at least 25 expert-prequalified cases")
    prequalification_bindings = {
        "candidate_id": candidate_id,
        "candidate_hash": record["candidate_hash"],
        "expert_baseline_sha256": artifacts["expert_baseline"]["sha256"],
        "evaluator_sha256": artifacts["evaluator"]["sha256"],
        "task_bundle_sha256": artifacts["task_bundle"]["sha256"],
        "framework_prompt_sha256": artifacts["framework_prompt"]["sha256"],
        "verifier_harness_sha256": materialization_contract["verifier_harness_sha256"],
        "container_image_digest": image_digest,
        "docker_image": docker_image,
        "harbor_version": harbor_version,
        "harbor_task_digest": harbor_task_digest,
        "task_checksum": task_checksum,
        "cpus": observed_cpu_count,
        "memory_mb": observed_memory_mb,
        "task_bundle_path": artifacts["task_bundle"]["path"],
        "framework_prompt_path": artifacts["framework_prompt"]["path"],
        "task_name": task_bundle_path.name,
        "problem_id": int(task_test_id),
    }
    stored_prequalified_cases: list[dict[str, Any]] = []
    prequalified_seeds: set[int] = set()
    prequalified_digests: set[str] = set()
    prequalified_job_ids: set[str] = set()
    prequalified_run_hashes: set[str] = set()
    threshold_policy_hashes: set[str] = set()
    prequalification_runtimes: list[float] = []
    for index, case in enumerate(prequalified_cases):
        derived = _parse_expert_prequalification_run(
            case, prequalification_bindings, base, root
        )
        seed = derived["protocol_seed"]
        digest = derived["case_digest"]
        job_id = derived["job_id"]
        run_hash = derived["run_evidence_sha256"]
        if (
            seed in prequalified_seeds
            or digest in prequalified_digests
            or job_id in prequalified_job_ids
            or run_hash in prequalified_run_hashes
        ):
            raise ValueError(
                "expert-prequalified seeds, case digests, jobs, and run bundles must be unique"
            )
        prequalified_seeds.add(seed)
        prequalified_digests.add(digest)
        prequalified_job_ids.add(job_id)
        prequalified_run_hashes.add(run_hash)
        threshold_policy_hashes.add(derived["threshold_policy_sha256"])
        prequalification_runtimes.append(derived["runtime_sec"])
        stored_prequalified_cases.append(derived)
    if len(threshold_policy_hashes) != 1:
        raise ValueError("expert prequalification must use one fixed threshold policy")

    oracle_runs = bundle.get("independent_oracle_runs", [])
    if not isinstance(oracle_runs, list) or len(oracle_runs) < 3:
        raise ValueError(
            "prototype bundle needs at least three independent-oracle runs"
        )
    oracle_runtimes: list[float] = []
    oracle_log_hashes: list[str] = []
    for index, run in enumerate(oracle_runs):
        runtime = run.get("runtime_sec")
        if (
            not isinstance(runtime, (int, float))
            or isinstance(runtime, bool)
            or runtime <= 0
        ):
            raise ValueError(f"oracle run {index} has invalid runtime")
        log_path, log_hash = _verified_evidence_item(
            root, base, run.get("log", {}), f"oracle run {index} log"
        )
        oracle_result = _read_json_object(log_path, f"oracle run {index} result")
        if (
            oracle_result.get("passed") is not True
            or oracle_result.get("runtime_sec") != runtime
            or not isinstance(oracle_result.get("oracle_method"), str)
            or not oracle_result["oracle_method"].strip()
        ):
            raise ValueError(f"oracle run {index} disagrees with its structured result")
        oracle_runtimes.append(float(runtime))
        oracle_log_hashes.append(log_hash)
    if len(set(oracle_log_hashes)) != len(oracle_log_hashes):
        raise ValueError("independent-oracle run logs must be unique")

    gold_effective_lines = bundle.get("gold_effective_lines")
    if not isinstance(gold_effective_lines, int) or isinstance(
        gold_effective_lines, bool
    ):
        raise ValueError("gold_effective_lines must be an integer")
    measured_effective_lines = _effective_python_lines(
        Path(artifacts["expert_baseline"]["path"])
    )
    if measured_effective_lines != gold_effective_lines:
        raise ValueError("gold_effective_lines does not match the expert baseline")
    alternatives = bundle.get("valid_alternatives", [])
    if not isinstance(alternatives, list) or len(alternatives) < 2:
        raise ValueError(
            "at least two independent valid implementation styles are required"
        )
    alternative_hashes: list[str] = []
    alternative_styles: list[str] = []
    for index, alternative in enumerate(alternatives):
        if alternative.get("accepted") is not True:
            raise ValueError(f"valid alternative {index} was not accepted")
        log_path, log_hash = _verified_evidence_item(
            root, base, alternative.get("log", {}), f"valid alternative {index} log"
        )
        result = _read_json_object(log_path, f"valid alternative {index} result")
        style = _normalized_identity(
            result.get("implementation_style"), f"valid alternative {index} style"
        )
        if result.get("accepted") is not True:
            raise ValueError(f"valid alternative {index} result was not accepted")
        alternative_hashes.append(log_hash)
        alternative_styles.append(style.casefold())
    if len(set(alternative_hashes)) != len(alternative_hashes) or len(
        set(alternative_styles)
    ) != len(alternative_styles):
        raise ValueError("valid-alternative logs and styles must be unique")

    mutation_tests = bundle.get("mutation_tests", [])
    if not isinstance(mutation_tests, list) or len(mutation_tests) < 6:
        raise ValueError("at least six verifier mutation tests are required")
    mutation_hashes: list[str] = []
    mutation_types: list[str] = []
    for index, mutation in enumerate(mutation_tests):
        if mutation.get("rejected") is not True:
            raise ValueError(f"verifier mutation {index} escaped rejection")
        log_path, log_hash = _verified_evidence_item(
            root, base, mutation.get("log", {}), f"mutation test {index} log"
        )
        result = _read_json_object(log_path, f"mutation test {index} result")
        mutation_type = _normalized_identity(
            result.get("mutation_type"), f"mutation test {index} type"
        )
        if result.get("rejected") is not True:
            raise ValueError(f"mutation test {index} result escaped rejection")
        mutation_hashes.append(log_hash)
        mutation_types.append(mutation_type.casefold())
    if len(set(mutation_hashes)) != len(mutation_hashes) or len(
        set(mutation_types)
    ) != len(mutation_types):
        raise ValueError("verifier-mutation logs and types must be unique")

    runtime_p95 = sorted(prequalification_runtimes)[
        math.ceil(0.95 * len(prequalification_runtimes)) - 1
    ]
    oracle_runtime_p95 = sorted(oracle_runtimes)[
        math.ceil(0.95 * len(oracle_runtimes)) - 1
    ]
    peak_memory = max(peak_memories)
    if peak_memory > 0.70 * float(observed_memory_mb):
        raise ValueError("expert peak memory exceeds the 70% admission headroom")

    evidence = {
        "candidate_id": candidate_id,
        "candidate_hash": record["candidate_hash"],
        "evidence_bundle_path": str(bundle_path),
        "evidence_bundle_sha256": _file_sha256(bundle_path),
        "expert_baseline_path": artifacts["expert_baseline"]["path"],
        "expert_baseline_sha256": artifacts["expert_baseline"]["sha256"],
        "independent_oracle_path": artifacts["independent_oracle"]["path"],
        "independent_oracle_sha256": artifacts["independent_oracle"]["sha256"],
        "evaluator_path": artifacts["evaluator"]["path"],
        "evaluator_sha256": artifacts["evaluator"]["sha256"],
        "instruction_path": artifacts["instruction"]["path"],
        "candidate_metadata_path": artifacts["candidate_metadata"]["path"],
        "task_bundle_path": artifacts["task_bundle"]["path"],
        "task_bundle_sha256": artifacts["task_bundle"]["sha256"],
        "framework_prompt_path": artifacts["framework_prompt"]["path"],
        "framework_prompt_sha256": artifacts["framework_prompt"]["sha256"],
        **materialization_contract,
        "artifact_authors": sorted(
            {artifacts[name]["author"] for name in artifact_names}, key=str.casefold
        ),
        "independent_artifact_authors": independent_authors,
        "container_image_digest": image_digest,
        "docker_image": docker_image,
        "harbor_version": harbor_version,
        "harbor_task_digest": harbor_task_digest,
        "task_checksum": task_checksum,
        "source_commit": source_commit,
        "public_api_canary_passed": True,
        "public_api_canary_log_sha256": canary_log_hash,
        "expert_runtime_p95_sec": runtime_p95,
        "independent_oracle_runtime_sec": oracle_runtime_p95,
        "gold_effective_lines": gold_effective_lines,
        "reproducibility_runs": len(verifier_runs),
        "observed_cpu_count": observed_cpu_count,
        "observed_memory_mb": float(observed_memory_mb),
        "expert_peak_memory_mb": peak_memory,
        "verifier_only_passed": True,
        "valid_alternatives_accepted": len(alternatives),
        "mutation_tests_rejected": len(mutation_tests),
        "expert_prequalified_cases": stored_prequalified_cases,
        "expert_prequalified_case_count": len(stored_prequalified_cases),
        "verifier_runs": verifier_records,
        "independent_oracle_log_sha256s": oracle_log_hashes,
        "valid_alternative_log_sha256s": alternative_hashes,
        "mutation_test_log_sha256s": mutation_hashes,
        "notes": bundle.get("notes", ""),
    }
    recorded_at = _utc_now()
    _revoke_active_authorization(
        state,
        record,
        candidate_id,
        "prototype_mutated",
        recorded_at,
    )
    record["prototype"] = {**evidence, "recorded_at": recorded_at}
    _append_event(
        state,
        {
            "event": "prototype_evidence",
            "candidate_id": candidate_id,
            "timestamp": recorded_at,
            "evidence_hash": _canonical_hash(evidence, length=64),
        },
    )
    _write_json(state_path, state)
    return candidate_gate_status(state, candidate_id)


def _validated_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValueError(
            f"protocol {label} must be 64 lowercase hexadecimal characters"
        )
    return value


def _validated_protocol_config(
    protocol_config: Any, prototype: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Validate the complete, executable protocol rather than descriptive strings."""
    if not isinstance(protocol_config, dict):
        raise ValueError("protocol config must be a JSON object")
    missing_protocol = [
        name
        for name in REQUIRED_PROTOCOL_CONFIG_FIELDS
        if name not in protocol_config
        or protocol_config.get(name) in (None, "")
        or (
            isinstance(protocol_config.get(name), str)
            and protocol_config[name].startswith("<")
        )
    ]
    if missing_protocol:
        raise ValueError(f"protocol config missing required fields: {missing_protocol}")

    for name in (
        "model",
        "provider_snapshot",
        "reasoning_effort",
        "solver_agent_version",
        "harbor_version",
        "audit_model",
        "tools_policy",
        "hardware_class",
        "docker_image",
    ):
        _normalized_nonempty_text(protocol_config[name], f"protocol {name}")
    if protocol_config["solver_agent"] not in SOLVER_AGENT_IMPORT_PATHS:
        raise ValueError(
            "protocol solver_agent must be one of "
            + ", ".join(sorted(SOLVER_AGENT_IMPORT_PATHS))
        )
    if protocol_config["reasoning_effort"] not in {
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    }:
        raise ValueError("protocol reasoning_effort is not supported")
    if protocol_config["token_budget_metric"] != "input_plus_output_tokens":
        raise ValueError(
            "protocol token_budget_metric must be input_plus_output_tokens so Harbor can verify it"
        )
    if protocol_config["token_budget_enforcement"] != "post_run_admission_only":
        raise ValueError(
            "Harbor/Codex currently supports only post_run_admission_only token accounting"
        )
    if (
        protocol_config["build_identity_policy"]
        != "require_runtime_observation_for_hardness"
    ):
        raise ValueError(
            "protocol build_identity_policy must fail closed without runtime observation"
        )
    for name in ("token_budget", "wall_time_sec", "cpus", "memory_mb"):
        value = protocol_config[name]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"protocol {name} must be a positive integer")
    if protocol_config["tools_policy"] != "harbor_default_no_mcp":
        raise ValueError("protocol tools_policy must be harbor_default_no_mcp")
    if protocol_config["network_policy"] not in {"public", "disabled"}:
        raise ValueError("protocol network_policy must be public or disabled")
    if protocol_config["proxy_policy"] not in {"none", "loopback_bridge"}:
        raise ValueError("protocol proxy_policy must be none or loopback_bridge")
    if (
        protocol_config["network_policy"] == "disabled"
        and protocol_config["proxy_policy"] != "none"
    ):
        raise ValueError("a disabled network cannot use a proxy bridge")
    if not isinstance(protocol_config["force_auth_json"], bool):
        raise ValueError("protocol force_auth_json must be boolean")
    if (
        protocol_config["force_auth_json"]
        and protocol_config["solver_agent"] != "codex-para"
    ):
        raise ValueError("force_auth_json requires solver_agent codex-para")
    if (
        "tensorcircuit" not in protocol_config["docker_image"].casefold()
        or "@" in protocol_config["docker_image"]
        or any(character.isspace() for character in protocol_config["docker_image"])
    ):
        raise ValueError(
            "protocol docker_image must be a TensorCircuit name:tag without a digest"
        )
    if not IMAGE_DIGEST_RE.fullmatch(protocol_config["container_image_digest"]):
        raise ValueError(
            "protocol container_image_digest must have the form sha256:<64 lowercase hex>"
        )
    for name in (
        "task_bundle_sha256",
        "harbor_task_digest",
        "task_checksum",
        "framework_prompt_sha256",
    ):
        _validated_sha256(protocol_config[name], name)

    if prototype is not None:
        frozen_bindings = {
            "container_image_digest": prototype.get("container_image_digest"),
            "docker_image": prototype.get("docker_image"),
            "harbor_version": prototype.get("harbor_version"),
            "task_bundle_sha256": prototype.get("task_bundle_sha256"),
            "harbor_task_digest": prototype.get("harbor_task_digest"),
            "task_checksum": prototype.get("task_checksum"),
            "framework_prompt_sha256": prototype.get("framework_prompt_sha256"),
            "cpus": prototype.get("observed_cpu_count"),
            "memory_mb": prototype.get("observed_memory_mb"),
        }
        mismatched = [
            name
            for name, expected in frozen_bindings.items()
            if protocol_config.get(name) != expected
        ]
        if mismatched:
            raise ValueError(
                "protocol config disagrees with frozen prototype evidence: "
                f"{mismatched}"
            )
    return protocol_config


def _protocol_family_hash(protocol_config: dict[str, Any]) -> str:
    return _canonical_hash(protocol_config, length=64)


def _seed_reservation_path(root: Path, manifest_hash: str, seed: int) -> Path:
    _validated_sha256(manifest_hash, "manifest hash")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("protocol seed must be a nonnegative integer")
    return (
        _repository_root(root)
        / ".artifacts"
        / "problem-discovery"
        / "seed-reservations"
        / manifest_hash
        / f"{seed}.json"
    )


@contextmanager
def _review_state_lock(root: Path) -> Iterable[None]:
    """Serialize every review-ledger read-modify-write transaction."""
    lock_path = (
        _repository_root(root)
        / ".artifacts"
        / "problem-discovery"
        / "locks"
        / "review-state.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _candidate_model_passes(record: dict[str, Any], model: str) -> list[dict[str, Any]]:
    trials = record.get("model_trials", {})
    if not isinstance(trials, dict):
        return []
    return [
        trial
        for trial in trials.values()
        if isinstance(trial, dict)
        and trial.get("model") == model
        and trial.get("passed") is True
    ]


def _candidate_model_pass_veto(
    state: dict[str, Any], record: dict[str, Any], candidate_id: str, model: str
) -> bool:
    if _candidate_model_passes(record, model):
        return True
    return any(
        isinstance(event, dict)
        and event.get("event") == "candidate_model_pass_veto"
        and event.get("candidate_id") == candidate_id
        and event.get("model") == model
        for event in state.get("event_log", [])
    )


def authorize_model_test(
    root: Path,
    candidate_id: str,
    protocol_config_path: Path,
    stage: str,
    seed_schedule: list[int],
    output: Path,
    qualifying_pilot_manifest_hash: str | None = None,
) -> dict[str, Any]:
    """Atomically issue a manifest and supersede the prior authorization."""
    with _review_state_lock(root):
        return _authorize_model_test_locked(
            root,
            candidate_id,
            protocol_config_path,
            stage,
            seed_schedule,
            output,
            qualifying_pilot_manifest_hash,
        )


def _authorize_model_test_locked(
    root: Path,
    candidate_id: str,
    protocol_config_path: Path,
    stage: str,
    seed_schedule: list[int],
    output: Path,
    qualifying_pilot_manifest_hash: str | None = None,
) -> dict[str, Any]:
    if stage not in {"pilot", "confirmation"}:
        raise ValueError("stage must be pilot or confirmation")
    minimum_trials = 5 if stage == "pilot" else 20
    if len(seed_schedule) < minimum_trials:
        raise ValueError(f"{stage} evidence requires at least {minimum_trials} trials")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seed_schedule
    ) or len(seed_schedule) != len(set(seed_schedule)):
        raise ValueError(
            "the precommitted seed schedule must contain unique nonnegative integers"
        )
    protocol_config_path = _safe_evidence_path(
        root, protocol_config_path.parent, str(protocol_config_path)
    )
    if not protocol_config_path.is_file():
        raise FileNotFoundError(
            f"protocol config does not exist: {protocol_config_path}"
        )
    protocol_config = _validated_protocol_config(_read_json(protocol_config_path))
    output = _safe_manifest_output(root, output)
    state = _read_json(root / "review_state.json")
    if _event_log_violations(state):
        raise PermissionError("review event ledger is invalid")
    status = candidate_gate_status(state, candidate_id)
    if not status["ready_for_model_test"]:
        raise PermissionError(
            "real model test remains human-gated: " + json.dumps(status, sort_keys=True)
        )
    candidate = _verified_shortlist_candidate(root, candidate_id)
    record = state["candidates"][candidate_id]
    if record["candidate_hash"] != candidate["candidate_hash"]:
        raise PermissionError(
            "candidate content changed after review; re-review is required"
        )
    _validated_protocol_config(protocol_config, record["prototype"])
    try:
        expert_prequalified_cases = _selected_prequalified_cases(
            record["prototype"], seed_schedule, root
        )
    except (KeyError, FileNotFoundError, ValueError, OSError) as error:
        raise PermissionError(str(error)) from error
    expert_prequalified_cases_sha256 = _canonical_hash(
        expert_prequalified_cases, length=64
    )
    protocol_family_hash = _protocol_family_hash(protocol_config)
    if _candidate_model_pass_veto(
        state, record, candidate_id, protocol_config["model"]
    ):
        raise PermissionError(
            "a prior valid pass for this candidate and model vetoes a hardness run"
        )

    qualifying_pilot: dict[str, Any] | None = None
    if stage == "pilot":
        if qualifying_pilot_manifest_hash is not None:
            raise ValueError("pilot authorization must not link another pilot")
    else:
        if not isinstance(
            qualifying_pilot_manifest_hash, str
        ) or not SHA256_RE.fullmatch(qualifying_pilot_manifest_hash):
            raise ValueError(
                "confirmation authorization requires --pilot-manifest-hash"
            )
        matches = [
            row
            for row in record.get("authorizations", [])
            if isinstance(row, dict)
            and row.get("manifest_hash") == qualifying_pilot_manifest_hash
        ]
        if len(matches) != 1:
            raise PermissionError("confirmation pilot link is unknown or ambiguous")
        qualifying_pilot = matches[0]
        if (
            qualifying_pilot.get("stage") != "pilot"
            or qualifying_pilot.get("model") != protocol_config["model"]
            or qualifying_pilot.get("protocol_family_hash") != protocol_family_hash
        ):
            raise PermissionError(
                "confirmation requires the same candidate, model, and protocol family"
            )
        pilot_status = model_hardness_status(
            root, candidate_id, qualifying_pilot_manifest_hash
        )
        if pilot_status.get("conclusion") != "pilot_hardness_signal_ready":
            raise PermissionError(
                "confirmation requires a completed zero-pass, two-auditor pilot"
            )
        if set(seed_schedule) & set(qualifying_pilot.get("seed_schedule", [])):
            raise ValueError("confirmation and pilot seed schedules must be disjoint")
    manifest = {
        "schema_version": 2,
        "candidate_id": candidate_id,
        "candidate_hash": candidate["candidate_hash"],
        "source_snapshot_hash": candidate["source_snapshot_hash"],
        "model": protocol_config["model"],
        "stage": stage,
        "independent_trials": len(seed_schedule),
        "seed_schedule": seed_schedule,
        "expert_prequalified_cases": expert_prequalified_cases,
        "expert_prequalified_cases_sha256": expert_prequalified_cases_sha256,
        "protocol_config_path": str(protocol_config_path),
        "protocol_config_sha256": _file_sha256(protocol_config_path),
        "protocol_config": protocol_config,
        "protocol_family_hash": protocol_family_hash,
        "qualifying_pilot_manifest_hash": (
            qualifying_pilot_manifest_hash if stage == "confirmation" else None
        ),
        "authorized_at": _utc_now(),
        "shortlist_snapshot": state["shortlist_snapshot"],
        "gate_fingerprint": _gate_fingerprint(record),
        "prototype": record["prototype"],
        "concept_reviews": record["concept_reviews"],
        "pilot_reviews": record["pilot_reviews"],
        "execution_policy": {
            "minimum_pilot_trials": 5,
            "minimum_confirmation_trials": 20,
            "frozen_prompt_required": True,
            "frozen_image_digest_required": True,
            "separate_verifier_required": True,
            "human_failure_classification_required": True,
            "confirmation_requires_zero_pass_two_auditor_pilot": True,
            "candidate_model_pass_veto": True,
            "excluded_failure_classes": [
                "infrastructure",
                "authentication",
                "policy_refusal",
                "ambiguous_specification",
                "framework_impossibility",
            ],
        },
    }
    manifest["manifest_hash"] = _canonical_hash(manifest, length=64)
    _write_json(output, manifest)
    _revoke_active_authorization(
        state,
        record,
        candidate_id,
        "superseded_by_new_authorization",
        manifest["authorized_at"],
    )
    record.setdefault("authorizations", []).append(
        {
            "manifest_hash": manifest["manifest_hash"],
            "manifest_path": str(output),
            "model": protocol_config["model"],
            "stage": stage,
            "independent_trials": len(seed_schedule),
            "seed_schedule": seed_schedule,
            "expert_prequalified_cases_sha256": expert_prequalified_cases_sha256,
            "protocol_config_sha256": manifest["protocol_config_sha256"],
            "protocol_family_hash": protocol_family_hash,
            "qualifying_pilot_manifest_hash": manifest[
                "qualifying_pilot_manifest_hash"
            ],
            "authorized_at": manifest["authorized_at"],
            "gate_fingerprint": manifest["gate_fingerprint"],
        }
    )
    record["active_authorization_hash"] = manifest["manifest_hash"]
    _append_event(
        state,
        {
            "event": "model_test_authorized",
            "candidate_id": candidate_id,
            "manifest_hash": manifest["manifest_hash"],
            "timestamp": manifest["authorized_at"],
        },
    )
    _write_json(root / "review_state.json", state)
    return manifest


def _verified_manifest(path: Path) -> dict[str, Any]:
    manifest = _read_json(path)
    recorded_hash = manifest.get("manifest_hash")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_hash"}
    if recorded_hash != _canonical_hash(unsigned, length=64):
        raise ValueError("run manifest hash does not match its contents")
    return manifest


def _active_authorization_context(
    root: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest_path = _safe_evidence_path(root, manifest_path.parent, str(manifest_path))
    manifest = _verified_manifest(manifest_path)
    if manifest.get("schema_version") != 2:
        raise PermissionError(
            "model execution requires a version 2 authorization manifest"
        )
    candidate_id = manifest.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise PermissionError("authorization manifest has no candidate binding")
    state = _read_json(root / "review_state.json")
    if _event_log_violations(state):
        raise PermissionError("review event ledger is invalid")
    record = state.get("candidates", {}).get(candidate_id)
    if not isinstance(record, dict):
        raise PermissionError(
            "authorization candidate is absent from the review ledger"
        )
    matches = [
        row
        for row in record.get("authorizations", [])
        if isinstance(row, dict)
        and row.get("manifest_hash") == manifest.get("manifest_hash")
    ]
    if len(matches) != 1:
        raise PermissionError(
            "manifest is not uniquely authorized in the review ledger"
        )
    authorization = matches[0]
    if (
        record.get("active_authorization_hash") != manifest.get("manifest_hash")
        or authorization.get("revoked_at")
        or authorization.get("revocation_reason")
    ):
        raise PermissionError("model authorization is inactive or superseded")
    expected_authorization = {
        "model": manifest.get("model"),
        "stage": manifest.get("stage"),
        "independent_trials": manifest.get("independent_trials"),
        "seed_schedule": manifest.get("seed_schedule"),
        "expert_prequalified_cases_sha256": manifest.get(
            "expert_prequalified_cases_sha256"
        ),
        "protocol_config_sha256": manifest.get("protocol_config_sha256"),
        "protocol_family_hash": manifest.get("protocol_family_hash"),
        "qualifying_pilot_manifest_hash": manifest.get(
            "qualifying_pilot_manifest_hash"
        ),
        "authorized_at": manifest.get("authorized_at"),
        "gate_fingerprint": manifest.get("gate_fingerprint"),
    }
    mismatched = [
        name
        for name, expected in expected_authorization.items()
        if authorization.get(name) != expected
    ]
    if mismatched:
        raise PermissionError(
            "authorization ledger disagrees with manifest: " + ", ".join(mismatched)
        )
    if _gate_fingerprint(record) != manifest.get("gate_fingerprint"):
        raise PermissionError("upstream human or prototype evidence changed")
    if not candidate_gate_status(state, candidate_id)["ready_for_model_test"]:
        raise PermissionError("current human gates do not authorize model execution")
    candidate = _verified_shortlist_candidate(root, candidate_id)
    if (
        candidate.get("candidate_hash") != manifest.get("candidate_hash")
        or record.get("candidate_hash") != manifest.get("candidate_hash")
        or record.get("prototype") != manifest.get("prototype")
    ):
        raise PermissionError("candidate or prototype changed after authorization")
    protocol_path = _safe_evidence_path(
        root,
        Path(manifest["protocol_config_path"]).parent,
        manifest["protocol_config_path"],
    )
    if _file_sha256(protocol_path) != manifest.get("protocol_config_sha256"):
        raise PermissionError("frozen protocol config file changed")
    protocol_config = _read_json_object(protocol_path, "protocol config")
    if protocol_config != manifest.get("protocol_config"):
        raise PermissionError("frozen protocol config payload changed")
    _validated_protocol_config(protocol_config, record["prototype"])
    if _protocol_family_hash(protocol_config) != manifest.get("protocol_family_hash"):
        raise PermissionError("protocol family binding changed")
    try:
        _revalidate_manifest_prequalified_cases(root, manifest, record["prototype"])
    except (KeyError, FileNotFoundError, ValueError, OSError) as error:
        raise PermissionError(
            "expert-prequalified manifest cases are missing or changed"
        ) from error
    if manifest.get("expert_prequalified_cases_sha256") != _canonical_hash(
        manifest.get("expert_prequalified_cases"), length=64
    ):
        raise PermissionError("expert-prequalified manifest case hash changed")
    return state, record, authorization, manifest


def authorized_model_execution(
    root: Path,
    manifest_path: Path,
    task_dir: Path,
    framework_prompt_path: Path,
    protocol_seed: int,
) -> dict[str, Any]:
    """Resolve one still-unused run from an active human authorization."""
    if not isinstance(protocol_seed, int) or isinstance(protocol_seed, bool):
        raise PermissionError("an explicit integer candidate seed is required")
    state, record, authorization, manifest = _active_authorization_context(
        root, manifest_path
    )
    if _candidate_model_pass_veto(
        state, record, manifest["candidate_id"], manifest["model"]
    ):
        raise PermissionError("candidate already has a pass for this model")
    if protocol_seed not in authorization.get("seed_schedule", []):
        raise PermissionError("candidate seed was not precommitted in the manifest")
    used_seeds = {
        row.get("protocol_seed")
        for row in record.get("model_trials", {}).values()
        if isinstance(row, dict)
        and row.get("manifest_hash") == manifest["manifest_hash"]
    }
    reserved_seeds = {
        row.get("protocol_seed")
        for row in authorization.get("seed_reservations", [])
        if isinstance(row, dict)
    }
    if protocol_seed in used_seeds or protocol_seed in reserved_seeds:
        raise PermissionError("candidate seed was already consumed or reserved")
    if _seed_reservation_path(root, manifest["manifest_hash"], protocol_seed).exists():
        raise PermissionError("candidate seed has an exclusive launch reservation")

    task_dir = task_dir.resolve()
    framework_prompt_path = framework_prompt_path.resolve()
    protocol = manifest["protocol_config"]
    selected_cases = _revalidate_manifest_prequalified_cases(
        root, manifest, record["prototype"]
    )
    selected_case = selected_cases[protocol_seed]
    if _path_sha256(task_dir) != protocol["task_bundle_sha256"]:
        raise PermissionError("materialized task hash does not match the authorization")
    if _file_sha256(framework_prompt_path) != protocol["framework_prompt_sha256"]:
        raise PermissionError("framework prompt hash does not match the authorization")
    return {
        "candidate_id": manifest["candidate_id"],
        "manifest_hash": manifest["manifest_hash"],
        "protocol_seed": protocol_seed,
        "case_digest": selected_case["case_digest"],
        "expert_admission": selected_case["admission"],
        "solver_agent": protocol["solver_agent"],
        "solver_agent_import_path": SOLVER_AGENT_IMPORT_PATHS[protocol["solver_agent"]],
        "model": protocol["model"],
        "provider_snapshot": protocol["provider_snapshot"],
        "reasoning_effort": protocol["reasoning_effort"],
        "token_budget": protocol["token_budget"],
        "token_budget_metric": protocol["token_budget_metric"],
        "token_budget_enforcement": protocol["token_budget_enforcement"],
        "wall_time_sec": protocol["wall_time_sec"],
        "solver_agent_version": protocol["solver_agent_version"],
        "harbor_version": protocol["harbor_version"],
        "audit_model": protocol["audit_model"],
        "build_identity_policy": protocol["build_identity_policy"],
        "tools_policy": protocol["tools_policy"],
        "network_policy": protocol["network_policy"],
        "proxy_policy": protocol["proxy_policy"],
        "cpus": protocol["cpus"],
        "memory_mb": protocol["memory_mb"],
        "docker_image": protocol["docker_image"],
        "container_image_digest": protocol["container_image_digest"],
        "task_bundle_sha256": protocol["task_bundle_sha256"],
        "harbor_task_digest": protocol["harbor_task_digest"],
        "task_checksum": protocol["task_checksum"],
        "framework_prompt_sha256": protocol["framework_prompt_sha256"],
        "force_auth_json": protocol["force_auth_json"],
    }


def reserve_model_trial_seed(
    root: Path,
    manifest_path: Path,
    protocol_seed: int,
    execution_id: str,
) -> dict[str, Any]:
    """Atomically consume one precommitted seed immediately before execution."""
    execution_id = _normalized_identity(execution_id, "execution id")
    with _review_state_lock(root):
        state, record, authorization, manifest = _active_authorization_context(
            root, manifest_path
        )
        if _candidate_model_pass_veto(
            state, record, manifest["candidate_id"], manifest["model"]
        ):
            raise PermissionError("candidate already has a pass for this model")
        if protocol_seed not in authorization.get("seed_schedule", []):
            raise PermissionError("candidate seed was not precommitted in the manifest")
        if any(
            isinstance(row, dict)
            and row.get("manifest_hash") == manifest["manifest_hash"]
            and row.get("protocol_seed") == protocol_seed
            for row in record.get("model_trials", {}).values()
        ) or any(
            isinstance(row, dict) and row.get("protocol_seed") == protocol_seed
            for row in authorization.get("seed_reservations", [])
        ):
            raise PermissionError("candidate seed was already consumed or reserved")
        reservation = {
            "protocol_seed": protocol_seed,
            "execution_id": execution_id,
            "reserved_at": _utc_now(),
        }
        reservation["reservation_hash"] = _canonical_hash(reservation, length=64)
        reservation_path = _seed_reservation_path(
            root, manifest["manifest_hash"], protocol_seed
        )
        reservation_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with reservation_path.open("x", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        reservation,
                        indent=2,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
        except FileExistsError as error:
            raise PermissionError(
                "candidate seed already has an exclusive launch reservation"
            ) from error
        authorization.setdefault("seed_reservations", []).append(reservation)
        _append_event(
            state,
            {
                "event": "model_seed_reserved",
                "candidate_id": manifest["candidate_id"],
                "manifest_hash": manifest["manifest_hash"],
                "protocol_seed": protocol_seed,
                "execution_id": execution_id,
                "reservation_hash": reservation["reservation_hash"],
                "timestamp": reservation["reserved_at"],
            },
        )
        _write_json(root / "review_state.json", state)
        return reservation


def _harbor_execution_status(result: dict[str, Any]) -> str:
    """Derive a coarse execution status from an immutable Harbor trial result."""
    if result.get("finished_at") in (None, ""):
        raise ValueError("raw Harbor trial is not finished")
    exception = result.get("exception_info")
    if not exception:
        return "completed"
    exception_type = str(exception.get("exception_type", "")).lower()
    message = str(exception.get("exception_message", "")).lower()
    if "timeout" in exception_type and "agent" in exception_type:
        return "agent_timeout"
    if "timeout" in exception_type and "verifier" in exception_type:
        return "verifier_timeout"
    authentication_markers = (
        "authentication",
        "unauthorized",
        "api key",
        "invalid token",
        "usage limit",
        "quota",
        "http 401",
        "http 403",
        " 401 ",
        " 403 ",
    )
    if any(marker in message for marker in authentication_markers):
        return "authentication_error"
    return "infrastructure_error"


def _functional_case_identity(path: Path) -> tuple[int, str]:
    identities: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get(CASE_IDENTITY_KEY), dict):
            identities.append(value[CASE_IDENTITY_KEY])
    if len(identities) != 1:
        raise ValueError(
            "functional output must contain exactly one structured orbit_q_case_identity"
        )
    identity = identities[0]
    seed = identity.get("protocol_seed")
    digest = identity.get("case_digest")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("functional case identity seed must be an integer")
    if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
        raise ValueError("functional case identity requires the full 64-hex digest")
    return seed, digest


def _expected_execution_bindings(
    manifest: dict[str, Any], seed: int
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    protocol = manifest["protocol_config"]
    common_env = {
        "ORBIT_Q_AUTHORIZATION_MANIFEST_HASH": manifest["manifest_hash"],
        "ORBIT_Q_CONTAINER_IMAGE_DIGEST": protocol["container_image_digest"],
        "ORBIT_Q_TASK_BUNDLE_SHA256": protocol["task_bundle_sha256"],
        "ORBIT_Q_PROVIDER_SNAPSHOT": protocol["provider_snapshot"],
        "ORBIT_Q_TOKEN_BUDGET": str(protocol["token_budget"]),
        "ORBIT_Q_TOKEN_BUDGET_ENFORCEMENT": protocol["token_budget_enforcement"],
        "ORBIT_Q_BUILD_IDENTITY_POLICY": protocol["build_identity_policy"],
        "ORBIT_Q_NETWORK_POLICY": protocol["network_policy"],
        "ORBIT_Q_PROXY_POLICY": protocol["proxy_policy"],
    }
    verifier_env = {
        "REQUIRED_QUANTUM_FRAMEWORK": "tensorcircuit",
        "ORBIT_Q_CANDIDATE_SEED": str(seed),
        **common_env,
    }
    agent_kwargs: dict[str, Any] = {
        "reasoning_effort": protocol["reasoning_effort"],
        "version": protocol["solver_agent_version"],
    }
    verifier_kwargs: dict[str, Any] = {"audit_model": protocol["audit_model"]}
    if protocol["force_auth_json"]:
        agent_kwargs["force_auth_json"] = True
        verifier_kwargs["force_auth_json"] = True
    expected = {
        "agent_import_path": SOLVER_AGENT_IMPORT_PATHS[protocol["solver_agent"]],
        "model": protocol["model"],
        "agent_kwargs": agent_kwargs,
        "verifier_kwargs": verifier_kwargs,
        "image_reference": (
            f"{protocol['docker_image']}@{protocol['container_image_digest']}"
        ),
    }
    return expected, common_env, verifier_env


def _validate_proxy_binding(
    protocol: dict[str, Any], agent_env: Any, verifier_env: Any
) -> None:
    if not isinstance(agent_env, dict) or not isinstance(verifier_env, dict):
        raise ValueError("Harbor agent and verifier env blocks must be objects")
    proxy_keys = {
        key
        for key in set(agent_env) | set(verifier_env)
        if key.casefold()
        in {
            "http_proxy",
            "https_proxy",
            "all_proxy",
            "no_proxy",
        }
    }
    if protocol["proxy_policy"] == "none":
        if proxy_keys:
            raise ValueError("Harbor config violates the frozen no-proxy policy")
        return
    expected_keys = {"HTTP_PROXY", "HTTPS_PROXY"}
    if proxy_keys != expected_keys:
        raise ValueError(
            "loopback proxy policy requires exactly HTTP_PROXY and HTTPS_PROXY"
        )
    if any(agent_env.get(key) != verifier_env.get(key) for key in expected_keys):
        raise ValueError("agent and verifier proxy bindings differ")
    proxy_pattern = re.compile(r"^https?://host\.docker\.internal:[1-9][0-9]*$")
    if any(
        not proxy_pattern.fullmatch(str(agent_env.get(key, "")))
        for key in expected_keys
    ):
        raise ValueError(
            "proxy binding is not a credential-free Docker loopback bridge"
        )


def _validate_harbor_config(config: Any, manifest: dict[str, Any], seed: int) -> None:
    if not isinstance(config, dict):
        raise ValueError("Harbor config must be a JSON object")
    protocol = manifest["protocol_config"]
    expected, _, expected_verifier_env = _expected_execution_bindings(manifest, seed)
    agent = config.get("agent")
    environment = config.get("environment")
    verifier = config.get("verifier")
    if not all(isinstance(value, dict) for value in (agent, environment, verifier)):
        raise ValueError("Harbor config is missing agent, environment, or verifier")
    allowed_top = {
        "agent",
        "agent_setup_timeout_multiplier",
        "agent_timeout_multiplier",
        "artifacts",
        "environment",
        "environment_build_timeout_multiplier",
        "extra_instruction_paths",
        "extra_instructions",
        "install_only",
        "job_id",
        "schema_version",
        "skills",
        "task",
        "timeout_multiplier",
        "trial_name",
        "trials_dir",
        "verifier",
        "verifier_timeout_multiplier",
    }
    allowed_agent = {
        "concurrency_group",
        "env",
        "extra_allowed_hosts",
        "import_path",
        "kwargs",
        "load_trajectory",
        "max_timeout_sec",
        "mcp_servers",
        "model_name",
        "n_concurrent",
        "name",
        "override_setup_timeout_sec",
        "override_timeout_sec",
        "resume_trajectory",
        "skills",
    }
    allowed_environment = {
        "cpu_enforcement_policy",
        "delete",
        "extra_allowed_hosts",
        "extra_docker_compose",
        "force_build",
        "import_path",
        "kwargs",
        "memory_enforcement_policy",
        "mounts",
        "override_cpus",
        "override_gpus",
        "override_memory_mb",
        "override_storage_mb",
        "override_tpu",
        "type",
    }
    allowed_verifier = {
        "disable",
        "env",
        "import_path",
        "kwargs",
        "max_timeout_sec",
        "override_timeout_sec",
    }
    if set(config) - allowed_top:
        raise ValueError("Harbor config contains unknown top-level controls")
    if set(agent) - allowed_agent:
        raise ValueError("Harbor agent config contains unknown controls")
    if set(environment) - allowed_environment:
        raise ValueError("Harbor environment config contains unknown controls")
    if set(verifier) - allowed_verifier:
        raise ValueError("Harbor verifier config contains unknown controls")
    required = {
        "agent import path": (agent.get("import_path"), expected["agent_import_path"]),
        "model": (agent.get("model_name"), expected["model"]),
        "agent kwargs": (agent.get("kwargs", {}), expected["agent_kwargs"]),
        "environment import path": (
            environment.get("import_path"),
            ENVIRONMENT_IMPORT_PATH,
        ),
        "CPU enforcement": (
            environment.get("cpu_enforcement_policy", "limit"),
            "limit",
        ),
        "memory enforcement": (
            environment.get("memory_enforcement_policy", "limit"),
            "limit",
        ),
        "CPU count": (environment.get("override_cpus"), protocol["cpus"]),
        "memory": (environment.get("override_memory_mb"), protocol["memory_mb"]),
        "environment kwargs": (
            environment.get("kwargs", {}),
            {
                "framework": "tensorcircuit",
                "docker_image": expected["image_reference"],
            },
        ),
        "verifier import path": (verifier.get("import_path"), VERIFIER_IMPORT_PATH),
        "verifier kwargs": (verifier.get("kwargs", {}), expected["verifier_kwargs"]),
        "verifier env bindings": (
            {key: verifier.get("env", {}).get(key) for key in expected_verifier_env},
            expected_verifier_env,
        ),
    }
    mismatched = [
        name for name, (actual, frozen) in required.items() if actual != frozen
    ]
    if mismatched:
        raise ValueError(
            "Harbor config disagrees with manifest: " + ", ".join(mismatched)
        )
    if agent.get("skills", []) or agent.get("mcp_servers", []):
        raise ValueError("Harbor config violates the frozen no-MCP tools policy")
    forbidden_agent_controls = {
        "override_timeout_sec": agent.get("override_timeout_sec"),
        "max_timeout_sec": agent.get("max_timeout_sec"),
        "load_trajectory": agent.get("load_trajectory"),
    }
    if any(value is not None for value in forbidden_agent_controls.values()):
        raise ValueError("Harbor agent timeout/trajectory override is not authorized")
    if agent.get("resume_trajectory", False) or agent.get("extra_allowed_hosts", []):
        raise ValueError("Harbor agent resume/extra-host controls are not authorized")
    if (
        agent.get("override_setup_timeout_sec") is not None
        or agent.get("n_concurrent") not in (None, 1)
        or agent.get("concurrency_group") is not None
        or agent.get("name") is not None
    ):
        raise ValueError("Harbor agent setup/concurrency controls are not authorized")
    if (
        environment.get("mounts") is not None
        or environment.get("extra_docker_compose", [])
        or environment.get("extra_allowed_hosts", [])
        or environment.get("override_gpus") not in (None, 0)
        or environment.get("override_tpu") is not None
        or environment.get("override_storage_mb") is not None
        or environment.get("type") is not None
        or environment.get("delete", True) is not True
        or environment.get("force_build", False) is not False
    ):
        raise ValueError("Harbor environment contains unauthorized expansion controls")
    if (
        verifier.get("override_timeout_sec") is not None
        or verifier.get("max_timeout_sec") is not None
    ):
        raise ValueError("Harbor verifier timeout overrides are not authorized")
    if verifier.get("disable", False) is not False:
        raise ValueError("Harbor verifier cannot be disabled")
    if config.get("skills", []):
        raise ValueError("Harbor lock contains unapproved top-level skills")
    if config.get("install_only", False) is not False:
        raise ValueError("Harbor install-only mode is not authorized")
    if config.get("artifacts", []) not in (None, []):
        raise ValueError("Harbor extra artifact injection is not authorized")
    for multiplier in (
        "agent_setup_timeout_multiplier",
        "environment_build_timeout_multiplier",
        "verifier_timeout_multiplier",
    ):
        if config.get(multiplier, 1.0) not in (None, 1.0):
            raise ValueError("Harbor setup/verifier timeout multipliers must be 1.0")
    proxy_names = (
        {"HTTP_PROXY", "HTTPS_PROXY"}
        if protocol["proxy_policy"] == "loopback_bridge"
        else set()
    )
    if set(agent.get("env", {})) != proxy_names:
        raise ValueError(
            "Harbor agent env contains fields outside the frozen proxy policy"
        )
    if set(verifier.get("env", {})) != set(expected_verifier_env) | proxy_names:
        raise ValueError("Harbor verifier env contains unapproved fields")
    if (
        config.get("timeout_multiplier", 1.0) != 1.0
        or config.get("agent_timeout_multiplier", 1.0) != 1.0
    ):
        raise ValueError("Harbor timeout multipliers must both be exactly 1.0")
    _validate_proxy_binding(protocol, agent.get("env", {}), verifier.get("env", {}))


def _parse_attested_harbor_result(
    raw_job_path: Path,
    harbor_config_path: Path,
    harbor_lock_path: Path,
    harbor_job_lock_path: Path,
    functional_output_path: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Derive trial facts from Harbor artifacts; operator claims are supplemental."""
    result = _read_json_object(raw_job_path, "raw Harbor trial result")
    harbor_config = _read_json_object(harbor_config_path, "Harbor config")
    harbor_lock = _read_json_object(harbor_lock_path, "Harbor trial lock")
    harbor_job_lock = _read_json_object(harbor_job_lock_path, "Harbor job lock")
    seed, case_digest = _functional_case_identity(functional_output_path)

    attestation = result.get("orbit_q_attestation")
    if attestation is not None:
        if (
            not isinstance(attestation, dict)
            or attestation.get("attestation_type") != "operator_attested"
        ):
            raise ValueError("supplemental operator attestation is malformed")
        _normalized_identity(attestation.get("attestor_id"), "Harbor attestor")
        _normalized_identity(attestation.get("attested_at"), "Harbor attestation time")
        supplemental_bindings = {
            "manifest_hash": manifest["manifest_hash"],
            "candidate_hash": manifest["candidate_hash"],
            "protocol_config_sha256": manifest["protocol_config_sha256"],
        }
        mismatched_claims = [
            name
            for name, expected in supplemental_bindings.items()
            if name in attestation and attestation[name] != expected
        ]
        if mismatched_claims:
            raise ValueError(
                "supplemental operator attestation contradicts machine evidence: "
                f"{mismatched_claims}"
            )

    config = result.get("config")
    _validate_harbor_config(config, manifest, seed)
    _validate_harbor_config(harbor_config, manifest, seed)
    protocol = manifest["protocol_config"]
    expected, _, _ = _expected_execution_bindings(manifest, seed)
    for section in ("agent", "environment", "verifier"):
        result_section = config.get(section, {})
        sidecar_section = harbor_config.get(section, {})
        for name in set(result_section) & set(sidecar_section):
            if result_section[name] != sidecar_section[name]:
                raise ValueError(f"result and config.json disagree in {section}.{name}")

    if harbor_lock.get("schema_version") != 1:
        raise ValueError("Harbor trial lock must use schema_version=1")
    if harbor_job_lock.get("schema_version") != 2:
        raise ValueError("Harbor job lock must use schema_version=2")
    allowed_job_lock = {
        "created_at",
        "harbor",
        "n_concurrent_trials",
        "retry",
        "schema_version",
        "trials",
    }
    if set(harbor_job_lock) - allowed_job_lock:
        raise ValueError("Harbor job lock contains unknown execution controls")
    harbor_identity = harbor_job_lock.get("harbor")
    if not isinstance(harbor_identity, dict) or set(harbor_identity) - {
        "git_commit_hash",
        "is_editable",
        "version",
    }:
        raise ValueError("Harbor job lock identity block is invalid")
    if harbor_identity.get("version") != protocol["harbor_version"]:
        raise ValueError("Harbor job lock version differs from the protocol")
    if harbor_identity.get("is_editable") is not False:
        raise ValueError("Harbor job lock must use a non-editable installation")
    if harbor_job_lock.get("n_concurrent_trials") != 1:
        raise ValueError("Harbor job lock must freeze one concurrent trial")
    retry = harbor_job_lock.get("retry")
    if (
        not isinstance(retry, dict)
        or set(retry)
        - {
            "exclude_exceptions",
            "max_retries",
            "max_wait_sec",
            "min_wait_sec",
            "wait_multiplier",
        }
        or retry.get("max_retries") != 0
    ):
        raise ValueError("Harbor job lock must disable retries")
    locked_trials = harbor_job_lock.get("trials")
    if not isinstance(locked_trials, list) or locked_trials != [harbor_lock]:
        raise ValueError("Harbor job lock must contain exactly this trial lock")
    _validate_harbor_config(harbor_lock, manifest, seed)
    allowed_trial_lock = {
        "agent",
        "environment",
        "extra_instructions",
        "install_only",
        "schema_version",
        "skills",
        "task",
        "timeout_multiplier",
        "verifier",
    }
    if set(harbor_lock) != allowed_trial_lock:
        raise ValueError("Harbor trial lock shape is not the frozen allowlist")
    locked_task = harbor_lock.get("task")
    if (
        not isinstance(locked_task, dict)
        or set(locked_task) - {"digest", "name", "path", "type"}
        or locked_task.get("type") != "local"
    ):
        raise ValueError("Harbor task lock shape is invalid")
    task_digest = harbor_lock.get("task", {}).get("digest")
    if task_digest != f"sha256:{protocol['harbor_task_digest']}":
        raise ValueError("Harbor task lock digest differs from the protocol")
    extra_instructions = harbor_lock.get("extra_instructions")
    if not isinstance(extra_instructions, list) or len(extra_instructions) != 1:
        raise ValueError("Harbor lock must contain exactly one framework prompt")
    if extra_instructions[0].get("digest") != (
        f"sha256:{protocol['framework_prompt_sha256']}"
    ):
        raise ValueError("Harbor framework prompt digest differs from the protocol")
    if result.get("task_checksum") != protocol["task_checksum"]:
        raise ValueError("raw Harbor task checksum differs from the protocol")

    agent_info = result.get("agent_info", {})
    if (
        agent_info.get("version") != protocol["solver_agent_version"]
        or agent_info.get("model_info", {}).get("name") != protocol["model"]
    ):
        raise ValueError("actual Harbor solver identity differs from the protocol")
    provider = agent_info.get("model_info", {}).get("provider")
    if provider not in (None, protocol["provider_snapshot"]):
        raise ValueError("actual Harbor provider contradicts the frozen snapshot")
    solver_provider_identity_verified = provider == protocol["provider_snapshot"]
    verifier_info = result.get("verifier_info")
    audit_model_identity_verified = False
    if verifier_info is not None:
        if not isinstance(verifier_info, dict):
            raise ValueError("Harbor verifier identity block is malformed")
        observed_audit_model = verifier_info.get("model_info", {}).get("name")
        if observed_audit_model != protocol["audit_model"]:
            raise ValueError("actual Harbor audit model differs from the protocol")
        audit_model_identity_verified = True
    agent_result = result.get("agent_result", {})
    input_tokens = agent_result.get("n_input_tokens")
    output_tokens = agent_result.get("n_output_tokens")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in (input_tokens, output_tokens)
    ):
        raise ValueError(
            "raw Harbor result needs nonnegative input/output token counts"
        )
    total_tokens = input_tokens + output_tokens
    token_budget_compliant = total_tokens <= protocol["token_budget"]

    rewards = result.get("verifier_result", {}).get("rewards")
    if not isinstance(rewards, dict):
        raise ValueError("raw Harbor result is missing verifier rewards")
    job_id = result.get("id")
    if not isinstance(job_id, str) or not job_id.strip():
        raise ValueError("raw Harbor result needs a trial UUID in id")
    machine_bindings = {
        "manifest_hash": manifest["manifest_hash"],
        "candidate_hash": manifest["candidate_hash"],
        "protocol_config_sha256": manifest["protocol_config_sha256"],
        "container_image_digest": protocol["container_image_digest"],
        "framework_prompt_sha256": protocol["framework_prompt_sha256"],
        "task_bundle_sha256": protocol["task_bundle_sha256"],
        "harbor_task_digest": protocol["harbor_task_digest"],
        "task_checksum": protocol["task_checksum"],
    }
    return {
        **machine_bindings,
        "model": expected["model"],
        "job_id": job_id,
        "protocol_seed": seed,
        "case_digest": case_digest,
        "solver_input_tokens": input_tokens,
        "solver_output_tokens": output_tokens,
        "solver_total_tokens": total_tokens,
        "token_budget_compliant": token_budget_compliant,
        "hard_token_limit_enforced": False,
        "solver_provider_identity_verified": solver_provider_identity_verified,
        "audit_model_identity_verified": audit_model_identity_verified,
        "execution_status": _harbor_execution_status(result),
        "reward": rewards.get("reward"),
        "functional_score": rewards.get("functional_score"),
        "static_policy_score": rewards.get("static_policy_score"),
        "llm_audit_score": rewards.get("llm_audit_score"),
        "runtime_sec": rewards.get("runtime_sec"),
    }


def _derived_trial_pass(evidence: dict[str, Any]) -> bool:
    return (
        evidence.get("execution_status") == "completed"
        and isinstance(evidence.get("reward"), (int, float))
        and not isinstance(evidence.get("reward"), bool)
        and evidence["reward"] > 0
        and all(
            isinstance(evidence.get(name), (int, float))
            and not isinstance(evidence.get(name), bool)
            and evidence[name] > 0
            for name in (
                "functional_score",
                "static_policy_score",
                "llm_audit_score",
            )
        )
    )


def _trial_evidence_hash(trial: dict[str, Any]) -> str:
    immutable = {
        key: value
        for key, value in trial.items()
        if key not in {"human_audits", "trial_evidence_hash"}
    }
    return _canonical_hash(immutable, length=64)


def record_model_trial(
    root: Path,
    candidate_id: str,
    manifest_path: Path,
    trial_id: str,
    result_path: Path,
) -> dict[str, Any]:
    """Atomically import one raw trial and install a pass veto if applicable."""
    with _review_state_lock(root):
        return _record_model_trial_locked(
            root, candidate_id, manifest_path, trial_id, result_path
        )


def _record_model_trial_locked(
    root: Path,
    candidate_id: str,
    manifest_path: Path,
    trial_id: str,
    result_path: Path,
) -> dict[str, Any]:
    """Record raw Harbor evidence without interpreting why a failure occurred."""
    manifest = _verified_manifest(manifest_path)
    if manifest.get("candidate_id") != candidate_id:
        raise ValueError("manifest candidate does not match the requested candidate")
    result_path = _safe_evidence_path(root, result_path.parent, str(result_path))
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    evidence = _read_json(result_path)
    if evidence.get("schema_version") != 2:
        raise ValueError("trial result must use schema_version=2")
    raw_job_path, raw_job_hash = _verified_evidence_item(
        root, result_path.parent, evidence.get("raw_job", {}), "raw Harbor job"
    )
    transcript_path, transcript_hash = _verified_evidence_item(
        root,
        result_path.parent,
        evidence.get("solver_transcript", {}),
        "solver transcript",
    )
    harbor_config_path, harbor_config_hash = _verified_evidence_item(
        root,
        result_path.parent,
        evidence.get("harbor_config", {}),
        "Harbor config",
    )
    harbor_lock_path, harbor_lock_hash = _verified_evidence_item(
        root,
        result_path.parent,
        evidence.get("harbor_lock", {}),
        "Harbor trial lock",
    )
    harbor_job_lock_path, harbor_job_lock_hash = _verified_evidence_item(
        root,
        result_path.parent,
        evidence.get("harbor_job_lock", {}),
        "Harbor job lock",
    )
    functional_output_path, functional_output_hash = _verified_evidence_item(
        root,
        result_path.parent,
        evidence.get("functional_output", {}),
        "functional output",
    )
    parsed = _parse_attested_harbor_result(
        raw_job_path,
        harbor_config_path,
        harbor_lock_path,
        harbor_job_lock_path,
        functional_output_path,
        manifest,
    )
    mismatched = [name for name, value in parsed.items() if evidence.get(name) != value]
    if mismatched:
        raise ValueError(
            f"normalized trial result disagrees with raw Harbor evidence: {mismatched}"
        )
    if evidence.get("execution_status") not in TRIAL_EXECUTION_STATUSES:
        raise ValueError("invalid trial execution status")
    for name in (
        "reward",
        "functional_score",
        "static_policy_score",
        "llm_audit_score",
    ):
        value = evidence.get(name)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not 0 <= value <= 1
        ):
            raise ValueError(f"{name} must be numeric in [0, 1]")
    expected_reward = (
        evidence["functional_score"]
        * evidence["static_policy_score"]
        * evidence["llm_audit_score"]
    )
    if abs(evidence["reward"] - expected_reward) > 1e-9:
        raise ValueError("reward must equal functional * static policy * LLM audit")
    if evidence["execution_status"] != "completed" and evidence["reward"] != 0:
        raise ValueError("an incomplete execution cannot carry a positive reward")
    runtime = evidence.get("runtime_sec")
    if (
        not isinstance(runtime, (int, float))
        or isinstance(runtime, bool)
        or runtime < -1
    ):
        raise ValueError(
            "runtime_sec must be -1 for missing data or a nonnegative number"
        )
    result_hash = _file_sha256(result_path)
    if not isinstance(evidence.get("job_id"), str) or not evidence["job_id"].strip():
        raise ValueError("job_id must be a nonempty string")
    if not isinstance(evidence.get("protocol_seed"), int) or isinstance(
        evidence["protocol_seed"], bool
    ):
        raise ValueError("protocol_seed must be an integer")

    if evidence["protocol_seed"] not in manifest.get("seed_schedule", []):
        raise ValueError("protocol_seed was not precommitted in the run manifest")
    state_path = root / "review_state.json"
    state = _read_json(state_path)
    record = state["candidates"][candidate_id]
    current_candidate = _verified_shortlist_candidate(root, candidate_id)
    if current_candidate["candidate_hash"] != manifest["candidate_hash"]:
        raise PermissionError("current shortlist content differs from the manifest")
    if not any(
        row.get("manifest_hash") == manifest["manifest_hash"]
        for row in record.get("authorizations", [])
    ):
        raise PermissionError("manifest was not authorized in this review ledger")
    if record.get("active_authorization_hash") != manifest["manifest_hash"]:
        raise PermissionError("authorization was superseded by a newer manifest")
    if _gate_fingerprint(record) != manifest.get("gate_fingerprint"):
        raise PermissionError(
            "an upstream review or prototype changed after authorization"
        )
    if not candidate_gate_status(state, candidate_id)["ready_for_model_test"]:
        raise PermissionError("current human gates no longer authorize model testing")
    if record["candidate_hash"] != manifest["candidate_hash"]:
        raise PermissionError("candidate changed after this run was authorized")
    try:
        prequalified_cases = _revalidate_manifest_prequalified_cases(
            root, manifest, record["prototype"]
        )
    except (KeyError, FileNotFoundError, ValueError, OSError) as error:
        raise PermissionError(
            "expert-prequalified manifest cases are missing or changed"
        ) from error
    selected_case = prequalified_cases[evidence["protocol_seed"]]
    if evidence.get("case_digest") != selected_case["case_digest"]:
        raise PermissionError(
            "trial case digest differs from exact expert-prequalified case"
        )
    if manifest.get("expert_prequalified_cases_sha256") != _canonical_hash(
        manifest.get("expert_prequalified_cases"), length=64
    ):
        raise PermissionError("expert-prequalified manifest case hash changed")
    matching_reservations = [
        reservation
        for reservation in next(
            row
            for row in record.get("authorizations", [])
            if row.get("manifest_hash") == manifest["manifest_hash"]
        ).get("seed_reservations", [])
        if isinstance(reservation, dict)
        and reservation.get("protocol_seed") == evidence["protocol_seed"]
    ]
    if len(matching_reservations) != 1:
        raise PermissionError(
            "trial seed was not reserved exactly once before execution"
        )
    reservation = matching_reservations[0]
    if reservation.get("reservation_hash") != _canonical_hash(
        {key: value for key, value in reservation.items() if key != "reservation_hash"},
        length=64,
    ):
        raise PermissionError("trial seed reservation hash is invalid")
    reservation_path = _seed_reservation_path(
        root, manifest["manifest_hash"], evidence["protocol_seed"]
    )
    if not reservation_path.is_file():
        raise PermissionError("exclusive seed reservation file is missing")
    if _read_json_object(reservation_path, "seed reservation") != reservation:
        raise PermissionError(
            "exclusive seed reservation file disagrees with the ledger"
        )
    if trial_id in record.setdefault("model_trials", {}):
        raise ValueError(f"trial id already exists: {trial_id}")
    if any(
        row.get("manifest_hash") == manifest["manifest_hash"]
        and (
            row.get("job_id") == evidence["job_id"]
            or row.get("protocol_seed") == evidence["protocol_seed"]
            or row.get("result_sha256") == result_hash
        )
        for row in record["model_trials"].values()
    ):
        raise ValueError(
            "job_id, protocol_seed, and result_sha256 must be unique within a manifest"
        )

    passed = _derived_trial_pass(evidence)
    trial = {
        **evidence,
        "result_path": str(result_path),
        "result_sha256": result_hash,
        "raw_job": {"path": str(raw_job_path), "sha256": raw_job_hash},
        "solver_transcript": {
            "path": str(transcript_path),
            "sha256": transcript_hash,
        },
        "harbor_config": {
            "path": str(harbor_config_path),
            "sha256": harbor_config_hash,
        },
        "harbor_lock": {
            "path": str(harbor_lock_path),
            "sha256": harbor_lock_hash,
        },
        "harbor_job_lock": {
            "path": str(harbor_job_lock_path),
            "sha256": harbor_job_lock_hash,
        },
        "functional_output": {
            "path": str(functional_output_path),
            "sha256": functional_output_hash,
        },
        "trial_id": trial_id,
        "candidate_hash": record["candidate_hash"],
        "manifest_hash": manifest["manifest_hash"],
        "model": manifest["model"],
        "passed": passed,
        "recorded_at": _utc_now(),
        "human_audits": {},
    }
    trial["trial_evidence_hash"] = _trial_evidence_hash(trial)
    record["model_trials"][trial_id] = trial
    _append_event(
        state,
        {
            "event": "model_trial_recorded",
            "candidate_id": candidate_id,
            "trial_id": trial_id,
            "manifest_hash": manifest["manifest_hash"],
            "trial_evidence_hash": trial["trial_evidence_hash"],
            "timestamp": trial["recorded_at"],
        },
    )
    if passed:
        _append_event(
            state,
            {
                "event": "candidate_model_pass_veto",
                "candidate_id": candidate_id,
                "model": manifest["model"],
                "manifest_hash": manifest["manifest_hash"],
                "trial_id": trial_id,
                "trial_evidence_hash": trial["trial_evidence_hash"],
                "timestamp": trial["recorded_at"],
            },
        )
    _write_json(state_path, state)
    return trial


def audit_model_trial(
    root: Path,
    candidate_id: str,
    trial_id: str,
    role: str,
    failure_class: str,
    reviewer: str,
    note: str,
) -> dict[str, Any]:
    """Atomically attach one append-only independent failure audit."""
    with _review_state_lock(root):
        return _audit_model_trial_locked(
            root,
            candidate_id,
            trial_id,
            role,
            failure_class,
            reviewer,
            note,
        )


def _audit_model_trial_locked(
    root: Path,
    candidate_id: str,
    trial_id: str,
    role: str,
    failure_class: str,
    reviewer: str,
    note: str,
) -> dict[str, Any]:
    """Attach an independent human explanation to one immutable raw trial."""
    if role not in FAILURE_AUDIT_ROLES:
        raise ValueError("invalid failure-audit role")
    allowed = ("success",) + SUBSTANTIVE_FAILURE_CLASSES + EXCLUDED_FAILURE_CLASSES
    if failure_class not in allowed:
        raise ValueError("invalid human failure class")
    reviewer = _normalized_identity(reviewer, "failure auditor")
    note = _normalized_nonempty_text(note, "failure-audit note")
    state_path = root / "review_state.json"
    state = _read_json(state_path)
    trial = state["candidates"][candidate_id]["model_trials"][trial_id]
    if role in trial.get("human_audits", {}):
        raise ValueError("failure audits are append-only")
    if trial["passed"] != (failure_class == "success"):
        raise ValueError(
            "success classification must agree with the compound pass result"
        )
    status = trial["execution_status"]
    allowed_by_status = {
        "infrastructure_error": {"infrastructure"},
        "authentication_error": {"authentication"},
        "verifier_timeout": {"infrastructure", "evaluator_defect", "indeterminate"},
        "agent_timeout": {"resource_strategy", "infrastructure", "indeterminate"},
    }
    if status in allowed_by_status and failure_class not in allowed_by_status[status]:
        raise ValueError(f"{failure_class} is incompatible with {status}")
    other_reviewers = {
        row.get("reviewer")
        for audit_role, row in trial.get("human_audits", {}).items()
        if audit_role != role
    }
    if _identity_key(reviewer) in {
        _identity_key(identity) for identity in other_reviewers if identity
    }:
        raise ValueError("the two failure-audit roles require distinct reviewers")
    record = state["candidates"][candidate_id]
    if _identity_key(reviewer) in _reviewer_identities(record):
        raise ValueError(
            "failure auditors must be independent of prior reviewers and authors"
        )
    audit = {
        "candidate_id": candidate_id,
        "trial_id": trial_id,
        "audit_role": role,
        "manifest_hash": trial["manifest_hash"],
        "trial_evidence_hash": trial["trial_evidence_hash"],
        "failure_class": failure_class,
        "reviewer": reviewer,
        "note": note,
        "timestamp": _utc_now(),
    }
    audit["audit_hash"] = _canonical_hash(audit, length=64)
    trial.setdefault("human_audits", {})[role] = audit
    _append_event(
        state,
        {
            "event": "model_trial_audited",
            "candidate_id": candidate_id,
            "trial_id": trial_id,
            "role": role,
            **audit,
        },
    )
    _write_json(state_path, state)
    return audit


def _event_log_violations(state: dict[str, Any]) -> list[str]:
    event_log = state.get("event_log", [])
    if not isinstance(event_log, list):
        return ["event_log_invalid"]
    violations: list[str] = []
    previous = "0" * 64
    for index, event in enumerate(event_log):
        if not isinstance(event, dict):
            violations.append(f"event_invalid:{index}")
            continue
        recorded_hash = event.get("event_hash")
        payload = {key: value for key, value in event.items() if key != "event_hash"}
        if event.get("previous_event_hash") != previous:
            violations.append(f"event_chain_broken:{index}")
        if recorded_hash != _canonical_hash(payload, length=64):
            violations.append(f"event_hash_changed:{index}")
        previous = recorded_hash if isinstance(recorded_hash, str) else ""
    return violations


def _audit_evidence_violations(
    record: dict[str, Any], candidate_id: str, trial: dict[str, Any]
) -> list[str]:
    trial_id = trial.get("trial_id", "unknown")
    audits = trial.get("human_audits", {})
    if not isinstance(audits, dict):
        return [f"human_audits_invalid:{trial_id}"]
    violations: list[str] = []
    unexpected_roles = set(audits) - set(FAILURE_AUDIT_ROLES)
    if unexpected_roles:
        violations.append(f"unexpected_audit_role:{trial_id}")
    reviewer_keys: list[str] = []
    prior_identities: set[str] = set()
    try:
        prior_identities = _reviewer_identities(record)
    except ValueError:
        violations.append("reviewer_identity_invalid")
    for role, audit in audits.items():
        if role not in FAILURE_AUDIT_ROLES or not isinstance(audit, dict):
            violations.append(f"human_audit_invalid:{trial_id}:{role}")
            continue
        expected_bindings = {
            "candidate_id": candidate_id,
            "trial_id": trial_id,
            "audit_role": role,
            "manifest_hash": trial.get("manifest_hash"),
            "trial_evidence_hash": trial.get("trial_evidence_hash"),
        }
        for name in (
            "candidate_id",
            "trial_id",
            "audit_role",
            "manifest_hash",
            "trial_evidence_hash",
        ):
            if audit.get(name) != expected_bindings[name]:
                violations.append(f"audit_binding_changed:{trial_id}:{role}:{name}")
        recorded_hash = audit.get("audit_hash")
        payload = {key: value for key, value in audit.items() if key != "audit_hash"}
        if recorded_hash != _canonical_hash(payload, length=64):
            violations.append(f"audit_hash_changed:{trial_id}:{role}")
        try:
            reviewer = _normalized_identity(audit.get("reviewer"), "failure auditor")
            note = _normalized_nonempty_text(audit.get("note"), "failure-audit note")
            if reviewer != audit.get("reviewer") or note != audit.get("note"):
                violations.append(f"audit_text_not_normalized:{trial_id}:{role}")
            reviewer_key = _identity_key(reviewer)
            reviewer_keys.append(reviewer_key)
            if reviewer_key in prior_identities:
                violations.append(f"audit_reviewer_not_independent:{trial_id}:{role}")
        except ValueError:
            violations.append(f"audit_text_invalid:{trial_id}:{role}")
        failure_class = audit.get("failure_class")
        allowed_classes = (
            ("success",) + SUBSTANTIVE_FAILURE_CLASSES + EXCLUDED_FAILURE_CLASSES
        )
        if failure_class not in allowed_classes:
            violations.append(f"audit_class_invalid:{trial_id}:{role}")
        if trial.get("passed") != (failure_class == "success"):
            violations.append(f"audit_class_disagrees_with_pass:{trial_id}:{role}")
        allowed_by_status = {
            "infrastructure_error": {"infrastructure"},
            "authentication_error": {"authentication"},
            "verifier_timeout": {
                "infrastructure",
                "evaluator_defect",
                "indeterminate",
            },
            "agent_timeout": {
                "resource_strategy",
                "infrastructure",
                "indeterminate",
            },
        }
        status = trial.get("execution_status")
        if (
            status in allowed_by_status
            and failure_class not in allowed_by_status[status]
        ):
            violations.append(f"audit_class_incompatible:{trial_id}:{role}")
    if len(reviewer_keys) != len(set(reviewer_keys)):
        violations.append(f"duplicate_failure_auditor:{trial_id}")
    return violations


def model_hardness_status(
    root: Path,
    candidate_id: str,
    manifest_hash: str,
    *,
    _allow_linked_superseded_pilot: bool = False,
) -> dict[str, Any]:
    """Summarize protocol-scoped evidence without claiming universal inability."""
    state = _read_json(root / "review_state.json")
    record = state["candidates"][candidate_id]
    authorizations = record.get("authorizations", [])
    if not isinstance(authorizations, list) or any(
        not isinstance(row, dict) for row in authorizations
    ):
        return {
            "candidate_id": candidate_id,
            "manifest_hash": manifest_hash,
            "authorization_active": False,
            "conclusion": "evidence_drift_or_gate_revoked",
            "evidence_violations": ["authorization_ledger_invalid"],
            "claim_boundary": "No model-hardness conclusion is valid for an invalid authorization ledger.",
        }
    matching_authorizations = [
        row for row in authorizations if row.get("manifest_hash") == manifest_hash
    ]
    if not matching_authorizations:
        raise KeyError("unknown authorization manifest")
    if len(matching_authorizations) != 1:
        return {
            "candidate_id": candidate_id,
            "manifest_hash": manifest_hash,
            "authorization_active": False,
            "conclusion": "evidence_drift_or_gate_revoked",
            "evidence_violations": ["authorization_ledger_duplicate"],
            "claim_boundary": "No model-hardness conclusion is valid for an ambiguous authorization ledger.",
        }
    authorization = matching_authorizations[0]
    active_authorization = next(
        (
            row
            for row in authorizations
            if row.get("manifest_hash") == record.get("active_authorization_hash")
        ),
        None,
    )
    linked_superseded_pilot = (
        _allow_linked_superseded_pilot
        and authorization.get("stage") == "pilot"
        and authorization.get("revocation_reason") == "superseded_by_new_authorization"
        and isinstance(active_authorization, dict)
        and active_authorization.get("stage") == "confirmation"
        and active_authorization.get("qualifying_pilot_manifest_hash") == manifest_hash
    )
    if not linked_superseded_pilot and (
        record.get("active_authorization_hash") != manifest_hash
        or authorization.get("revoked_at")
        or authorization.get("revocation_reason")
    ):
        return {
            "candidate_id": candidate_id,
            "manifest_hash": manifest_hash,
            "authorization_active": False,
            "conclusion": "authorization_revoked_or_superseded",
            "claim_boundary": "No model-hardness conclusion is valid for an inactive manifest.",
        }
    try:
        current_gate_fingerprint = _gate_fingerprint(record)
    except (KeyError, TypeError, ValueError):
        return {
            "candidate_id": candidate_id,
            "manifest_hash": manifest_hash,
            "authorization_active": False,
            "conclusion": "evidence_drift_or_gate_revoked",
            "evidence_violations": ["current_gate_unverifiable"],
            "claim_boundary": "No model-hardness conclusion is valid for an unverifiable gate ledger.",
        }
    if current_gate_fingerprint != authorization.get("gate_fingerprint"):
        return {
            "candidate_id": candidate_id,
            "manifest_hash": manifest_hash,
            "authorization_active": False,
            "conclusion": "authorization_revoked_by_gate_change",
            "claim_boundary": "Upstream evidence changed after this manifest was issued.",
        }
    evidence_violations = _event_log_violations(state)
    candidate: dict[str, Any] | None = None
    try:
        candidate = _verified_shortlist_candidate(root, candidate_id)
        if candidate["candidate_hash"] != record["candidate_hash"]:
            evidence_violations.append("candidate_hash_changed")
    except (KeyError, ValueError, OSError, json.JSONDecodeError):
        evidence_violations.append("candidate_content_unverifiable")
    try:
        gate_ready = candidate_gate_status(state, candidate_id)["ready_for_model_test"]
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        gate_ready = False
        evidence_violations.append("current_gate_unverifiable")
    if not gate_ready:
        evidence_violations.append("current_human_or_prototype_gate_failed")
    manifest: dict[str, Any] | None = None
    prequalified_by_seed: dict[int, dict[str, Any]] = {}
    try:
        declared_manifest_path = Path(authorization["manifest_path"])
        manifest_path = _safe_evidence_path(
            root,
            declared_manifest_path.parent,
            str(declared_manifest_path),
        )
        manifest = _verified_manifest(manifest_path)
        if manifest.get("manifest_hash") != manifest_hash:
            evidence_violations.append("manifest_hash_changed")
        manifest_bindings = {
            "candidate_id": candidate_id,
            "candidate_hash": record.get("candidate_hash"),
            "gate_fingerprint": current_gate_fingerprint,
            "prototype": record.get("prototype", {}),
            "concept_reviews": record.get("concept_reviews", {}),
            "pilot_reviews": record.get("pilot_reviews", {}),
            "shortlist_snapshot": state.get("shortlist_snapshot"),
        }
        for name, expected in manifest_bindings.items():
            if manifest.get(name) != expected:
                evidence_violations.append(f"manifest_{name}_changed")
        if candidate is not None and manifest.get(
            "source_snapshot_hash"
        ) != candidate.get("source_snapshot_hash"):
            evidence_violations.append("manifest_source_snapshot_changed")
        authorization_bindings = {
            "model": manifest.get("model"),
            "stage": manifest.get("stage"),
            "independent_trials": manifest.get("independent_trials"),
            "seed_schedule": manifest.get("seed_schedule"),
            "expert_prequalified_cases_sha256": manifest.get(
                "expert_prequalified_cases_sha256"
            ),
            "protocol_config_sha256": manifest.get("protocol_config_sha256"),
            "protocol_family_hash": manifest.get("protocol_family_hash"),
            "qualifying_pilot_manifest_hash": manifest.get(
                "qualifying_pilot_manifest_hash"
            ),
            "authorized_at": manifest.get("authorized_at"),
            "gate_fingerprint": manifest.get("gate_fingerprint"),
        }
        for name, expected in authorization_bindings.items():
            if authorization.get(name) != expected:
                evidence_violations.append(f"authorization_{name}_changed")
        prequalified_by_seed = _revalidate_manifest_prequalified_cases(
            root, manifest, record.get("prototype", {})
        )
        if manifest.get("expert_prequalified_cases_sha256") != _canonical_hash(
            manifest.get("expert_prequalified_cases"), length=64
        ):
            evidence_violations.append("manifest_prequalified_case_hash_changed")
        declared_protocol_path = Path(manifest["protocol_config_path"])
        protocol_path = _safe_evidence_path(
            root,
            declared_protocol_path.parent,
            str(declared_protocol_path),
        )
        if _file_sha256(protocol_path) != manifest["protocol_config_sha256"]:
            evidence_violations.append("protocol_config_changed")
        elif _read_json_object(protocol_path, "protocol config") != manifest.get(
            "protocol_config"
        ):
            evidence_violations.append("protocol_config_payload_changed")
        if (
            manifest.get("stage") == "confirmation"
            and not _allow_linked_superseded_pilot
        ):
            linked_pilot_hash = manifest.get("qualifying_pilot_manifest_hash")
            linked_authorizations = [
                row
                for row in authorizations
                if row.get("manifest_hash") == linked_pilot_hash
                and row.get("stage") == "pilot"
            ]
            if (
                not isinstance(linked_pilot_hash, str)
                or linked_pilot_hash == manifest_hash
                or len(linked_authorizations) != 1
            ):
                evidence_violations.append("qualifying_pilot_link_missing")
            else:
                linked_status = model_hardness_status(
                    root,
                    candidate_id,
                    linked_pilot_hash,
                    _allow_linked_superseded_pilot=True,
                )
                if linked_status.get("conclusion") != "pilot_hardness_signal_ready":
                    evidence_violations.append(
                        "qualifying_pilot_evidence_invalid_or_changed"
                    )
    except (KeyError, FileNotFoundError, ValueError, OSError, json.JSONDecodeError):
        evidence_violations.append("manifest_or_protocol_unverifiable")
    model_trials = record.get("model_trials", {})
    if not isinstance(model_trials, dict):
        evidence_violations.append("model_trial_ledger_invalid")
        model_trials = {}
    pass_veto_events = [
        event
        for event in state.get("event_log", [])
        if isinstance(event, dict)
        and event.get("event") == "candidate_model_pass_veto"
        and event.get("candidate_id") == candidate_id
        and event.get("model") == authorization.get("model")
    ]
    model_pass_rows = [
        row
        for row in model_trials.values()
        if isinstance(row, dict)
        and row.get("model") == authorization.get("model")
        and row.get("passed") is True
    ]
    for event in pass_veto_events:
        if not any(
            row.get("trial_id") == event.get("trial_id")
            and row.get("manifest_hash") == event.get("manifest_hash")
            and row.get("trial_evidence_hash") == event.get("trial_evidence_hash")
            for row in model_pass_rows
        ):
            evidence_violations.append("candidate_model_pass_veto_orphaned")
    for row in model_pass_rows:
        if not any(
            event.get("trial_id") == row.get("trial_id")
            and event.get("manifest_hash") == row.get("manifest_hash")
            and event.get("trial_evidence_hash") == row.get("trial_evidence_hash")
            for event in pass_veto_events
        ):
            evidence_violations.append("candidate_model_pass_veto_missing")
    trials_by_id = {
        trial_id: row
        for trial_id, row in model_trials.items()
        if isinstance(row, dict) and row.get("manifest_hash") == manifest_hash
    }
    if any(not isinstance(row, dict) for row in model_trials.values()):
        evidence_violations.append("model_trial_row_invalid")
    events = state.get("event_log", [])
    if not isinstance(events, list):
        events = []
    authorization_events = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("event") == "model_test_authorized"
        and event.get("candidate_id") == candidate_id
        and event.get("manifest_hash") == manifest_hash
    ]
    if len(authorization_events) != 1:
        evidence_violations.append("authorization_event_missing_or_duplicate")
    trial_events = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("event") == "model_trial_recorded"
        and event.get("candidate_id") == candidate_id
        and event.get("manifest_hash") == manifest_hash
    ]
    if any(
        not isinstance(event.get("trial_id"), str) or not event["trial_id"].strip()
        for event in trial_events
    ):
        evidence_violations.append("trial_record_event_invalid")
        trial_events = [
            event
            for event in trial_events
            if isinstance(event.get("trial_id"), str) and event["trial_id"].strip()
        ]
    event_trial_ids = {event.get("trial_id") for event in trial_events}
    ledger_trial_ids = set(trials_by_id)
    for missing_id in sorted(event_trial_ids - ledger_trial_ids, key=str):
        evidence_violations.append(f"trial_ledger_entry_missing:{missing_id}")
    for unlogged_id in sorted(ledger_trial_ids - event_trial_ids, key=str):
        evidence_violations.append(f"trial_record_event_missing:{unlogged_id}")
    trials = list(trials_by_id.values())
    if manifest is not None:
        for trial_id, trial in trials_by_id.items():
            try:
                if trial.get("trial_id") != trial_id:
                    evidence_violations.append(f"trial_id_binding_changed:{trial_id}")
                if trial.get("candidate_hash") != manifest["candidate_hash"]:
                    evidence_violations.append(
                        f"trial_candidate_binding_changed:{trial_id}"
                    )
                selected_case = prequalified_by_seed.get(trial.get("protocol_seed"))
                if not isinstance(selected_case, dict) or trial.get(
                    "case_digest"
                ) != selected_case.get("case_digest"):
                    evidence_violations.append(
                        f"trial_prequalified_case_binding_changed:{trial_id}"
                    )
                trial_reservations = [
                    row
                    for row in authorization.get("seed_reservations", [])
                    if isinstance(row, dict)
                    and row.get("protocol_seed") == trial.get("protocol_seed")
                ]
                if len(trial_reservations) != 1:
                    evidence_violations.append(
                        f"trial_seed_reservation_missing_or_duplicate:{trial_id}"
                    )
                else:
                    reservation = trial_reservations[0]
                    expected_reservation_hash = _canonical_hash(
                        {
                            key: value
                            for key, value in reservation.items()
                            if key != "reservation_hash"
                        },
                        length=64,
                    )
                    if reservation.get("reservation_hash") != expected_reservation_hash:
                        evidence_violations.append(
                            f"trial_seed_reservation_hash_changed:{trial_id}"
                        )
                    reservation_path = _seed_reservation_path(
                        root, manifest_hash, trial["protocol_seed"]
                    )
                    if (
                        not reservation_path.is_file()
                        or _read_json_object(reservation_path, "seed reservation")
                        != reservation
                    ):
                        evidence_violations.append(
                            f"trial_seed_reservation_changed:{trial_id}"
                        )
                expected_trial_hash = _trial_evidence_hash(trial)
                if trial.get("trial_evidence_hash") != expected_trial_hash:
                    evidence_violations.append(f"trial_ledger_changed:{trial_id}")
                matching_trial_events = [
                    event for event in trial_events if event.get("trial_id") == trial_id
                ]
                if len(matching_trial_events) != 1 or matching_trial_events[0].get(
                    "trial_evidence_hash"
                ) != trial.get("trial_evidence_hash"):
                    evidence_violations.append(
                        f"trial_event_binding_changed:{trial_id}"
                    )
                declared_result_path = Path(trial["result_path"])
                result_path = _safe_evidence_path(
                    root,
                    declared_result_path.parent,
                    str(declared_result_path),
                )
                if _file_sha256(result_path) != trial["result_sha256"]:
                    evidence_violations.append(f"trial_result_changed:{trial_id}")
                    continue
                normalized_result = _read_json_object(
                    result_path, f"trial result {trial_id}"
                )
                declared_raw_job_path = Path(trial["raw_job"]["path"])
                raw_job_path = _safe_evidence_path(
                    root,
                    declared_raw_job_path.parent,
                    str(declared_raw_job_path),
                )
                declared_transcript_path = Path(trial["solver_transcript"]["path"])
                transcript_path = _safe_evidence_path(
                    root,
                    declared_transcript_path.parent,
                    str(declared_transcript_path),
                )
                declared_config_path = Path(trial["harbor_config"]["path"])
                config_path = _safe_evidence_path(
                    root, declared_config_path.parent, str(declared_config_path)
                )
                declared_lock_path = Path(trial["harbor_lock"]["path"])
                lock_path = _safe_evidence_path(
                    root, declared_lock_path.parent, str(declared_lock_path)
                )
                declared_job_lock_path = Path(trial["harbor_job_lock"]["path"])
                job_lock_path = _safe_evidence_path(
                    root,
                    declared_job_lock_path.parent,
                    str(declared_job_lock_path),
                )
                declared_functional_path = Path(trial["functional_output"]["path"])
                functional_path = _safe_evidence_path(
                    root,
                    declared_functional_path.parent,
                    str(declared_functional_path),
                )
                if _path_sha256(raw_job_path) != trial["raw_job"]["sha256"]:
                    evidence_violations.append(f"raw_job_changed:{trial_id}")
                if (
                    _path_sha256(transcript_path)
                    != trial["solver_transcript"]["sha256"]
                ):
                    evidence_violations.append(f"solver_transcript_changed:{trial_id}")
                artifact_checks = (
                    (config_path, "harbor_config"),
                    (lock_path, "harbor_lock"),
                    (job_lock_path, "harbor_job_lock"),
                    (functional_path, "functional_output"),
                )
                for artifact_path, artifact_name in artifact_checks:
                    if _path_sha256(artifact_path) != trial[artifact_name]["sha256"]:
                        evidence_violations.append(
                            f"{artifact_name}_changed:{trial_id}"
                        )
                parsed = _parse_attested_harbor_result(
                    raw_job_path,
                    config_path,
                    lock_path,
                    job_lock_path,
                    functional_path,
                    manifest,
                )
                if any(trial.get(name) != value for name, value in parsed.items()):
                    evidence_violations.append(
                        f"raw_job_disagrees_with_ledger:{trial_id}"
                    )
                if any(
                    normalized_result.get(name) != value
                    for name, value in parsed.items()
                ):
                    evidence_violations.append(
                        f"raw_job_disagrees_with_normalized_result:{trial_id}"
                    )
                if trial.get("passed") != _derived_trial_pass(parsed):
                    evidence_violations.append(
                        f"trial_pass_derivation_changed:{trial_id}"
                    )
                evidence_violations.extend(
                    _audit_evidence_violations(record, candidate_id, trial)
                )
            except (
                KeyError,
                FileNotFoundError,
                ValueError,
                OSError,
                json.JSONDecodeError,
            ):
                evidence_violations.append(f"trial_evidence_unverifiable:{trial_id}")
    candidate_audit_events = [
        event
        for event in events
        if isinstance(event, dict)
        and event.get("event") == "model_trial_audited"
        and event.get("candidate_id") == candidate_id
        and event.get("manifest_hash") == manifest_hash
    ]
    if any(
        not isinstance(event.get("trial_id"), str)
        or not event["trial_id"].strip()
        or event.get("role") not in FAILURE_AUDIT_ROLES
        for event in candidate_audit_events
    ):
        evidence_violations.append("audit_event_invalid")
    audit_events = [
        event
        for event in candidate_audit_events
        if isinstance(event.get("trial_id"), str)
        and event.get("trial_id") in trials_by_id
        and event.get("role") in FAILURE_AUDIT_ROLES
    ]
    for event in audit_events:
        trial_id = event.get("trial_id")
        role = event.get("role")
        audits = trials_by_id[trial_id].get("human_audits", {})
        audit = audits.get(role) if isinstance(audits, dict) else None
        if not isinstance(audit, dict) or audit.get("audit_hash") != event.get(
            "audit_hash"
        ):
            evidence_violations.append(
                f"audit_ledger_entry_missing_or_changed:{trial_id}:{role}"
            )
    for trial_id, trial in trials_by_id.items():
        audits = trial.get("human_audits", {})
        if not isinstance(audits, dict):
            continue
        for role, audit in audits.items():
            matching_audit_events = [
                event
                for event in audit_events
                if event.get("trial_id") == trial_id
                and event.get("role") == role
                and event.get("audit_hash") == audit.get("audit_hash")
            ]
            if len(matching_audit_events) != 1:
                evidence_violations.append(
                    f"audit_event_missing_or_duplicate:{trial_id}:{role}"
                )
    if evidence_violations:
        return {
            "candidate_id": candidate_id,
            "manifest_hash": manifest_hash,
            "authorization_active": False,
            "conclusion": "evidence_drift_or_gate_revoked",
            "evidence_violations": sorted(set(evidence_violations)),
            "claim_boundary": (
                "No model-hardness conclusion is valid while current candidate, "
                "gate, manifest, protocol, or trial evidence cannot be revalidated."
            ),
        }
    unique_trials: list[dict[str, Any]] = []
    seen_runs: set[tuple[Any, Any]] = set()
    for trial in sorted(trials, key=lambda row: row["trial_id"]):
        identity = (trial.get("job_id"), trial.get("protocol_seed"))
        if identity not in seen_runs:
            unique_trials.append(trial)
            seen_runs.add(identity)

    def consensus_class(trial: dict[str, Any]) -> str | None:
        audits = trial.get("human_audits", {})
        if any(role not in audits for role in FAILURE_AUDIT_ROLES):
            return None
        classes = {audits[role].get("failure_class") for role in FAILURE_AUDIT_ROLES}
        return next(iter(classes)) if len(classes) == 1 else None

    passes = [row for row in unique_trials if row["passed"]]
    consensus_audited = [
        row for row in unique_trials if consensus_class(row) is not None
    ]
    conflicting_audits = [
        row
        for row in unique_trials
        if all(role in row.get("human_audits", {}) for role in FAILURE_AUDIT_ROLES)
        and consensus_class(row) is None
    ]
    substantive = [
        row
        for row in consensus_audited
        if consensus_class(row) in SUBSTANTIVE_FAILURE_CLASSES
        and row.get("token_budget_compliant") is True
    ]
    excluded = [
        row
        for row in consensus_audited
        if consensus_class(row) in EXCLUDED_FAILURE_CLASSES
        or row.get("token_budget_compliant") is not True
    ]
    required = authorization["independent_trials"]
    scheduled_seeds = set(authorization["seed_schedule"])
    recorded_seeds = {row["protocol_seed"] for row in unique_trials}
    missing_seeds = sorted(scheduled_seeds - recorded_seeds)
    unexpected_seeds = sorted(recorded_seeds - scheduled_seeds)
    schedule_complete = (
        not missing_seeds and not unexpected_seeds and len(unique_trials) == required
    )
    all_model_passes = _candidate_model_passes(record, authorization["model"])
    cross_manifest_passes = [
        row for row in all_model_passes if row.get("manifest_hash") != manifest_hash
    ]
    admission_limitations: list[str] = []
    admission_limitations.append("submitted_solution_process_isolation_unavailable")
    if any(row.get("hard_token_limit_enforced") is not True for row in unique_trials):
        admission_limitations.append("hard_token_execution_cap_unavailable")
    if any(
        row.get("solver_provider_identity_verified") is not True
        for row in unique_trials
    ):
        admission_limitations.append("solver_provider_build_not_runtime_observed")
    if any(
        row.get("audit_model_identity_verified") is not True for row in unique_trials
    ):
        admission_limitations.append("audit_model_build_not_runtime_observed")
    valid_count = len(passes) + len(substantive)
    if all_model_passes:
        conclusion = "solved_in_at_least_one_valid_trial"
    elif schedule_complete and len(substantive) == required:
        if authorization["stage"] == "pilot":
            conclusion = "pilot_hardness_signal_ready"
        elif admission_limitations:
            conclusion = "protocol_scoped_hardness_blocked_by_unenforced_limits"
        else:
            conclusion = "protocol_scoped_model_hard_evidence_ready"
    else:
        conclusion = "incomplete_or_invalid_trial_schedule"
    return {
        "candidate_id": candidate_id,
        "manifest_hash": manifest_hash,
        "authorization_active": True,
        "required_valid_trials": required,
        "stage": authorization["stage"],
        "scheduled_seeds": sorted(scheduled_seeds),
        "missing_seeds": missing_seeds,
        "unexpected_seeds": unexpected_seeds,
        "schedule_complete": schedule_complete,
        "recorded_unique_trials": len(unique_trials),
        "consensus_audited_trials": len(consensus_audited),
        "conflicting_audits": len(conflicting_audits),
        "valid_trials": valid_count,
        "substantive_failures": len(substantive),
        "excluded_failures": len(excluded),
        "passes": len(all_model_passes),
        "cross_manifest_passes": len(cross_manifest_passes),
        "admission_limitations": admission_limitations,
        "pass_at_k": 1 if all_model_passes else 0,
        "conclusion": conclusion,
        "claim_boundary": (
            "This conclusion applies only to the frozen manifest and model. "
            "A final hardness claim is blocked unless submitted-code isolation, "
            "runtime token enforcement, and build identity are machine-observed."
        ),
    }
