from __future__ import annotations

import importlib.util
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

from static_policy import check_source


DEFAULT_MAX_EFFECTIVE_CODE_LINES = 200
MIN_MAX_EFFECTIVE_CODE_LINES = 1
MAX_MAX_EFFECTIVE_CODE_LINES = 200
CASE_IDENTITY_KEY = "orbit_q_case_identity"
EXPERT_ADMISSION_METRICS_KEY = "orbit_q_expert_admission_metrics"
EXPERT_ADMISSION_PATH = Path("/logs/verifier/expert-admission.json")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
METRIC_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
EXPERT_BINDING_ENV = {
    "candidate_id": "ORBIT_Q_CANDIDATE_ID",
    "candidate_hash": "ORBIT_Q_CANDIDATE_HASH",
    "expert_baseline_sha256": "ORBIT_Q_EXPERT_BASELINE_SHA256",
    "evaluator_sha256": "ORBIT_Q_EVALUATOR_SHA256",
    "task_bundle_sha256": "ORBIT_Q_TASK_BUNDLE_SHA256",
    "framework_prompt_sha256": "ORBIT_Q_FRAMEWORK_PROMPT_SHA256",
    "verifier_harness_sha256": "ORBIT_Q_VERIFIER_HARNESS_SHA256",
    "container_image_digest": "ORBIT_Q_CONTAINER_IMAGE_DIGEST",
}


def _max_effective_code_lines(value: str | None = None) -> int:
    """Parse a verifier-only line limit without allowing policy relaxation."""

    raw_value = os.environ.get("MAX_EFFECTIVE_CODE_LINES") if value is None else value
    if raw_value is None:
        return DEFAULT_MAX_EFFECTIVE_CODE_LINES
    if not re.fullmatch(r"[0-9]+", raw_value):
        raise ValueError("MAX_EFFECTIVE_CODE_LINES must be a base-10 integer")
    parsed = int(raw_value)
    if not MIN_MAX_EFFECTIVE_CODE_LINES <= parsed <= MAX_MAX_EFFECTIVE_CODE_LINES:
        raise ValueError(
            "MAX_EFFECTIVE_CODE_LINES must be between "
            f"{MIN_MAX_EFFECTIVE_CODE_LINES} and {MAX_MAX_EFFECTIVE_CODE_LINES}"
        )
    return parsed


def _static_policy_result(source: Path, framework: str) -> dict:
    return check_source(
        source,
        framework,
        max_lines=_max_effective_code_lines(),
    )


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_score(runtime_sec: float | None) -> float:
    if runtime_sec is None:
        return 0.0
    full_score_sec = float(os.environ.get("RUNTIME_FULL_SCORE_SEC", "180"))
    zero_score_sec = float(os.environ.get("RUNTIME_ZERO_SCORE_SEC", "300"))
    if runtime_sec <= full_score_sec:
        return 1.0
    if runtime_sec >= zero_score_sec:
        return 0.0
    return float((zero_score_sec - runtime_sec) / (zero_score_sec - full_score_sec))


def _parse_runtime_sec(output: str) -> float | None:
    match = re.search(r"End-to-end solution time:\s*([0-9]+(?:\.[0-9]+)?)s", output)
    if not match:
        return None
    return float(match.group(1))


def _structured_output_rows(output: str, key: str) -> list[dict]:
    """Return exact one-key JSON records emitted by the trusted evaluator."""

    rows = []
    for line in output.splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and key in value:
            if set(value) != {key} or not isinstance(value[key], dict):
                raise ValueError(f"{key} output line must contain exactly one object")
            rows.append(value[key])
    return rows


def _sha256_words(digest: str) -> dict[str, int]:
    """Encode a SHA-256 into JSON-safe reward integers Harbor preserves."""

    return {
        f"expert_admission_sha256_word_{index}": int(digest[offset : offset + 8], 16)
        for index, offset in enumerate(range(0, 64, 8))
    }


def _expert_admission_record(output: str, functional_passed: bool) -> tuple[dict, str]:
    """Recompute strict case-admission margins from evaluator stdout.

    The operator cannot supply an admission summary.  The only accepted values are
    the structured rows captured from the evaluator subprocess whose source hash is
    frozen by the expert-run bindings.
    """

    if os.environ.get("ORBIT_Q_EXPERT_PREQUALIFICATION_MODE") != "expert_only_oracle":
        raise ValueError("expert admission requested outside expert-only mode")
    if not functional_passed:
        raise ValueError("expert admission requires a functional pass")

    identity_rows = _structured_output_rows(output, CASE_IDENTITY_KEY)
    metric_rows = _structured_output_rows(output, EXPERT_ADMISSION_METRICS_KEY)
    if len(identity_rows) != 1 or len(metric_rows) != 1:
        raise ValueError(
            "expert evaluator must emit exactly one case identity and one admission row"
        )
    identity = identity_rows[0]
    if set(identity) != {"protocol_seed", "case_digest"}:
        raise ValueError("expert case identity has an invalid shape")
    seed = identity["protocol_seed"]
    case_digest = identity["case_digest"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("expert protocol seed must be a nonnegative integer")
    if not isinstance(case_digest, str) or not SHA256_RE.fullmatch(case_digest):
        raise ValueError("expert case digest must be a full lowercase SHA-256")

    declared = metric_rows[0]
    if set(declared) != {
        "schema_version",
        "protocol_seed",
        "case_digest",
        "metrics",
    }:
        raise ValueError("expert admission metric row has an invalid shape")
    if (
        declared["schema_version"] != 1
        or declared["protocol_seed"] != seed
        or declared["case_digest"] != case_digest
    ):
        raise ValueError("expert admission metrics disagree with case identity")
    metrics = declared["metrics"]
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("expert admission metrics must be a nonempty list")

    margins = []
    seen = set()
    for index, metric in enumerate(metrics):
        if not isinstance(metric, dict) or set(metric) != {
            "metric",
            "direction",
            "observed",
            "threshold",
        }:
            raise ValueError(f"expert admission metric {index} has an invalid shape")
        name = metric["metric"]
        direction = metric["direction"]
        if not isinstance(name, str) or not METRIC_RE.fullmatch(name):
            raise ValueError(f"expert admission metric {index} has an invalid name")
        if name in seen:
            raise ValueError("expert admission metric names must be unique")
        seen.add(name)
        if direction not in {"at_least", "at_most"}:
            raise ValueError(f"expert admission metric {index} has invalid direction")
        observed = metric["observed"]
        threshold = metric["threshold"]
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (observed, threshold)
        ):
            raise ValueError(f"expert admission metric {index} must be finite numeric")
        observed = float(observed)
        threshold = float(threshold)
        margin = (
            observed - threshold if direction == "at_least" else threshold - observed
        )
        if margin <= 0:
            raise ValueError(
                f"expert admission metric {name} has no strict pass margin"
            )
        margins.append(
            {
                "metric": name,
                "direction": direction,
                "observed": observed,
                "threshold": threshold,
                "absolute_margin": margin,
                "passed": True,
            }
        )

    bindings = {}
    for name, env_name in EXPERT_BINDING_ENV.items():
        value = os.environ.get(env_name)
        if not value:
            raise ValueError(f"missing frozen expert binding {env_name}")
        bindings[name] = value
    if not bindings["candidate_id"].strip():
        raise ValueError("candidate id must be nonempty")
    for name in (
        "candidate_hash",
        "expert_baseline_sha256",
        "evaluator_sha256",
        "task_bundle_sha256",
        "framework_prompt_sha256",
        "verifier_harness_sha256",
    ):
        if not SHA256_RE.fullmatch(bindings[name]):
            raise ValueError(f"{name} must be a full lowercase SHA-256")
    if not IMAGE_DIGEST_RE.fullmatch(bindings["container_image_digest"]):
        raise ValueError("container image digest must be sha256:<64 lowercase hex>")
    if os.environ.get("ORBIT_Q_CANDIDATE_SEED") != str(seed):
        raise ValueError("verifier seed binding disagrees with evaluator case identity")

    threshold_policy = [
        {
            "metric": margin["metric"],
            "direction": margin["direction"],
            "threshold": margin["threshold"],
        }
        for margin in margins
    ]
    threshold_payload = json.dumps(
        threshold_policy, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    record = {
        "schema_version": 1,
        "producer": "orbit_q_score_submission_from_candidate_evaluator",
        **bindings,
        "protocol_seed": seed,
        "case_digest": case_digest,
        "expert_passed": True,
        "threshold_policy_sha256": hashlib.sha256(threshold_payload).hexdigest(),
        "admission": {
            "status": "admitted",
            "minimum_margin": min(row["absolute_margin"] for row in margins),
            "margins": margins,
        },
    }
    serialized = json.dumps(record, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    EXPERT_ADMISSION_PATH.write_text(serialized)
    return record, hashlib.sha256(serialized.encode()).hexdigest()


def _compound_reward(
    functional_score: float, static_policy_score: float, llm_audit_score: float
) -> float:
    """Compute pass reward; runtime remains a reporting-only diagnostic."""

    return float(functional_score * static_policy_score * llm_audit_score)


def _functional_score(problem_id: int, solution_module: str) -> dict:
    evaluate_path = Path(f"/tests/evaluate_{problem_id}.py")
    cmd = [sys.executable, str(evaluate_path), "--solution", solution_module]
    extra = os.environ.get("EVALUATE_EXTRA_ARGS", "").strip()
    if extra:
        cmd.extend(extra.split())
    env = dict(os.environ)
    env["PYTHONPATH"] = "/root:/tests" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    try:
        completed = subprocess.run(
            cmd,
            cwd="/root",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(os.environ.get("FUNCTIONAL_TIMEOUT_SEC", "900")),
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        captured = exc.stdout or ""
        output = (
            captured.decode("utf-8", errors="replace")
            if isinstance(captured, bytes)
            else str(captured)
        )
        output += "\nFUNCTIONAL_TIMEOUT"
        Path("/logs/verifier/functional-stdout.txt").write_text(output)
        return {
            "functional_score": 0.0,
            "runtime_score": 0.0,
            "runtime_sec": -1.0,
            "functional_return_code": -1,
            "functional_passed": False,
            "functional_output_tail": output[-4000:],
        }
    output = completed.stdout
    runtime_sec = _parse_runtime_sec(output)
    overall_lines = [
        line.strip()
        for line in output.splitlines()
        if line.strip() in {"Overall: PASS", "Overall: FAIL"}
    ]
    passed = (
        completed.returncode == 0
        and bool(overall_lines)
        and overall_lines[-1] == "Overall: PASS"
    )
    Path("/logs/verifier/functional-stdout.txt").write_text(output)
    return {
        "functional_score": 1.0 if passed else 0.0,
        "runtime_score": _runtime_score(runtime_sec),
        "runtime_sec": -1.0 if runtime_sec is None else runtime_sec,
        "functional_return_code": completed.returncode,
        "functional_passed": passed,
        "functional_output_tail": output[-4000:],
    }


def _llm_audit_score(
    source: Path,
    framework: str,
    evaluation_summary: str,
    problem_statement: str,
) -> dict:
    if os.environ.get("CODEX_AUDIT_ENABLED", "1") == "0":
        return {"llm_audit_score": 1.0, "llm_audit_skipped": True}
    audit_module = _load_module(Path("/tests/audit_codex.py"), "audit_codex_runtime")
    try:
        return audit_module.audit_source(
            source,
            framework,
            evaluation_summary,
            problem_statement,
        )
    except Exception as exc:
        return {"llm_audit_score": 0.0, "audit_error": str(exc)}


def main() -> None:
    problem_id = int(Path("/tests/problem_id.txt").read_text().strip())
    framework = os.environ.get("REQUIRED_QUANTUM_FRAMEWORK", "tensorcircuit")
    solution_module = f"solution_{problem_id}"
    solution_path = Path(f"/root/{solution_module}.py")

    rewards = {
        "problem_id": problem_id,
        "reward": 0.0,
        "functional_score": 0.0,
        "runtime_score": 0.0,
        "runtime_sec": -1.0,
        "static_policy_score": 0.0,
        "llm_audit_score": 0.0,
        "line_count_score": 0.0,
        "framework_score": 0.0,
        "no_forbidden_framework_score": 0.0,
        "no_raw_simulator_bypass_score": 0.0,
        "no_static_cheating_score": 0.0,
        "llm_framework_compliance_score": 0.0,
        "llm_cheating_score": 0.0,
        "llm_problem_fidelity_score": 0.0,
        "llm_implementation_correctness_score": 0.0,
        "llm_uses_required_framework_score": 0.0,
        "llm_no_other_quantum_framework_imports_score": 0.0,
        "llm_no_other_quantum_framework_score": 0.0,
        "llm_no_raw_simulator_bypass_score": 0.0,
        "llm_no_hardcoded_or_hidden_answer_score": 0.0,
        "llm_no_test_or_reward_tampering_score": 0.0,
        "llm_no_evaluator_exploit_score": 0.0,
    }
    details = {"problem_id": problem_id, "framework": framework}

    if not solution_path.exists():
        details["missing_solution"] = str(solution_path)
        Path("/logs/verifier/reward.json").write_text(json.dumps(rewards, indent=2))
        Path("/logs/verifier/audit-details.json").write_text(
            json.dumps(details, indent=2)
        )
        return

    try:
        static = _static_policy_result(solution_path, framework)
    except ValueError as exc:
        details["verifier_configuration_error"] = str(exc)
        Path("/logs/verifier/reward.json").write_text(json.dumps(rewards, indent=2))
        Path("/logs/verifier/audit-details.json").write_text(
            json.dumps(details, indent=2)
        )
        return
    functional = _functional_score(problem_id, solution_module)
    expert_admission_hash = None
    if os.environ.get("ORBIT_Q_EXPERT_PREQUALIFICATION_MODE") == "expert_only_oracle":
        try:
            _, expert_admission_hash = _expert_admission_record(
                Path("/logs/verifier/functional-stdout.txt").read_text(
                    encoding="utf-8", errors="replace"
                ),
                bool(functional.get("functional_passed")),
            )
        except (OSError, TypeError, ValueError) as exc:
            functional["functional_score"] = 0.0
            functional["functional_passed"] = False
            functional["expert_admission_error"] = str(exc)
    problem_statement_path = Path("/tests/problem_statement.md")
    problem_statement = (
        problem_statement_path.read_text(errors="replace")
        if problem_statement_path.exists()
        else ""
    )
    audit = _llm_audit_score(
        solution_path,
        framework,
        functional.get("functional_output_tail", ""),
        problem_statement,
    )

    details.update({"static": static, "functional": functional, "audit": audit})
    rewards.update(
        {
            "functional_score": float(functional["functional_score"]),
            "runtime_score": float(functional["runtime_score"]),
            "runtime_sec": float(functional["runtime_sec"]),
            "static_policy_score": float(static["static_policy_score"]),
            "llm_audit_score": float(audit.get("llm_audit_score", 0.0)),
            "line_count_score": float(static["line_count_score"]),
            "framework_score": float(static["framework_score"]),
            "no_forbidden_framework_score": float(
                static["no_forbidden_framework_score"]
            ),
            "no_raw_simulator_bypass_score": float(
                static["no_raw_simulator_bypass_score"]
            ),
            "no_static_cheating_score": float(static["no_static_cheating_score"]),
            "llm_framework_compliance_score": float(
                audit.get("llm_framework_compliance_score", 0.0)
            ),
            "llm_cheating_score": float(audit.get("llm_cheating_score", 0.0)),
            "llm_problem_fidelity_score": float(
                audit.get("llm_problem_fidelity_score", 0.0)
            ),
            "llm_implementation_correctness_score": float(
                audit.get("llm_implementation_correctness_score", 0.0)
            ),
            "llm_uses_required_framework_score": float(
                audit.get("llm_uses_required_framework_score", 0.0)
            ),
            "llm_no_other_quantum_framework_imports_score": float(
                audit.get("llm_no_other_quantum_framework_imports_score", 0.0)
            ),
            "llm_no_other_quantum_framework_score": float(
                audit.get("llm_no_other_quantum_framework_score", 0.0)
            ),
            "llm_no_raw_simulator_bypass_score": float(
                audit.get("llm_no_raw_simulator_bypass_score", 0.0)
            ),
            "llm_no_hardcoded_or_hidden_answer_score": float(
                audit.get("llm_no_hardcoded_or_hidden_answer_score", 0.0)
            ),
            "llm_no_test_or_reward_tampering_score": float(
                audit.get("llm_no_test_or_reward_tampering_score", 0.0)
            ),
            "llm_no_evaluator_exploit_score": float(
                audit.get("llm_no_evaluator_exploit_score", 0.0)
            ),
        }
    )
    rewards["reward"] = _compound_reward(
        rewards["functional_score"],
        rewards["static_policy_score"],
        rewards["llm_audit_score"],
    )
    if expert_admission_hash is not None:
        rewards.update(_sha256_words(expert_admission_hash))

    Path("/root/evaluation_summary.json").write_text(json.dumps(details, indent=2))
    Path("/logs/verifier/audit-details.json").write_text(json.dumps(details, indent=2))
    Path("/logs/verifier/reward.json").write_text(json.dumps(rewards, indent=2))


if __name__ == "__main__":
    main()
