#!/usr/bin/env python3
"""Preview or execute Harbor for exactly one isolated candidate task."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlsplit, urlunsplit

from materialize_candidate_task import (
    CandidateTaskError,
    DEFAULT_OUTPUT_ROOT,
    EXPERT_ADMISSION_METRICS_KEY,
    EXPERT_ADMISSION_PROTOCOL_FILE,
    ROOT,
    discovery_candidate_binding,
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


def _is_within(path: Path, parent: Path) -> bool:
    return path == parent or path.is_relative_to(parent)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _required_expert_verifier_env(resolved: dict[str, Any]) -> dict[str, str]:
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "REQUIRED_QUANTUM_FRAMEWORK": "tensorcircuit",
        "ORBIT_Q_NON_HARDNESS_RUN": "expert_prequalification",
        "ORBIT_Q_EXPERT_PREQUALIFICATION_MODE": "expert_only_oracle",
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
            or verifier.get("kwargs", {}).get("expert_only") is not True
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
    jobs_root = validate_jobs_root(args.jobs_dir)
    evidence_root = validate_expert_evidence_root(args.expert_evidence_dir)
    job_dir = jobs_root / args.job_name
    trial_dir, derived = _validate_expert_harbor_artifacts(
        task_dir=task_dir, job_dir=job_dir, resolved=resolved
    )
    destination = (
        evidence_root / resolved["candidate_slug"] / f"seed-{resolved['protocol_seed']}"
    )
    if destination.exists():
        raise CandidateTaskError(
            f"refusing to overwrite existing expert evidence: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    sources = {
        "raw_job": trial_dir / "result.json",
        "harbor_config": trial_dir / "config.json",
        "harbor_lock": trial_dir / "lock.json",
        "harbor_job_lock": job_dir / "lock.json",
        "functional_output": trial_dir / "verifier" / "functional-stdout.txt",
        "admission_record": trial_dir / "verifier" / EXPERT_ADMISSION_FILENAME,
    }
    filenames = {
        "raw_job": "raw-job.json",
        "harbor_config": "resolved-config.json",
        "harbor_lock": "trial-lock.json",
        "harbor_job_lock": "job-lock.json",
        "functional_output": "functional-stdout.txt",
        "admission_record": EXPERT_ADMISSION_FILENAME,
    }
    try:
        raw_evidence = {}
        for name, source in sources.items():
            if source.is_symlink() or not source.is_file():
                raise CandidateTaskError(f"expert evidence source is invalid: {source}")
            target = staging / filenames[name]
            shutil.copy2(source, target)
            raw_evidence[name] = {
                "path": str((destination / target.name).resolve()),
                "sha256": _file_sha256(target),
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
        (staging / "expert-prequalified-case.json").write_text(
            json.dumps(case, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        bindings = {
            key: resolved[key]
            for key in (
                "candidate_id",
                "candidate_hash",
                "problem_id",
                "expert_baseline_sha256",
                "evaluator_sha256",
                "task_bundle_sha256",
                "framework_prompt_sha256",
                "verifier_harness_sha256",
                "instruction_sha256",
                "candidate_metadata_sha256",
                "task_policy_sha256",
                "solution_driver_sha256",
                "container_image_digest",
                "docker_image",
                "cpus",
                "memory_mb",
                "task_dir",
                "blueprint_path",
            )
        }
        bindings.update(
            {
                "harbor_task_digest": derived["harbor_task_digest"],
                "task_checksum": derived["task_checksum"],
            }
        )
        (staging / "expert-run-bindings.json").write_text(
            json.dumps(bindings, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        staging.rename(destination)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
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
        audit_model = args.audit_model or args.model or "gpt-5"
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
            "audit_model": audit_model,
            "force_auth_json": args.force_auth_json,
            "proxy_policy": (
                "loopback_bridge" if args.bridge_loopback_proxy else "none"
            ),
        }
    else:
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
            "--verifier-kwarg",
            f"audit_model={audit_model}",
            "--verifier-env",
            "REQUIRED_QUANTUM_FRAMEWORK=tensorcircuit",
        ]
    )
    if args.expert_only:
        cmd.extend(
            [
                "--verifier-kwarg",
                "expert_only=true",
                "--verifier-env",
                "ORBIT_Q_NON_HARDNESS_RUN=expert_prequalification",
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


def verify_harbor_version(harbor_bin: Path, expected: str) -> None:
    completed = subprocess.run(
        [str(harbor_bin), "--version"],
        capture_output=True,
        text=True,
        check=False,
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
    except CandidateTaskError as exc:
        print(f"error: {exc}")
        return 2

    mode = "EXECUTE" if args.execute else "DRY RUN"
    print(f"Mode: {mode}")
    print(f"Candidate task: {task_dir}")
    print("Framework: tensorcircuit")
    resolved = args._resolved_run
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
    print(f"Verifier audit model: {resolved['audit_model']}")
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
        job_path = validate_jobs_root(args.jobs_dir) / args.job_name
        evidence_path = (
            validate_expert_evidence_root(args.expert_evidence_dir)
            / resolved["candidate_slug"]
            / f"seed-{resolved['protocol_seed']}"
        )
        if job_path.exists():
            print(f"error: refusing to reuse existing Harbor job output: {job_path}")
            return 2
        if evidence_path.exists():
            print(f"error: refusing to overwrite expert evidence: {evidence_path}")
            return 2

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

    env = dict(os.environ)
    # The trial seed crosses the Harbor boundary only through --verifier-env.
    # Scrubbing a coincident host export avoids accidental adapter inheritance.
    env.pop(CANDIDATE_SEED_ENV, None)
    for name in PROXY_ENV_NAMES:
        env.pop(name, None)
    env["PYTHONPATH"] = str(ROOT) + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
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
