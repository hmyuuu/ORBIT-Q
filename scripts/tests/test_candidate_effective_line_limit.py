from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
TEMPLATE_TESTS = ROOT / "templates" / "challenge" / "tests"
for import_root in (SCRIPTS, TEMPLATE_TESTS):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import materialize_candidate_task as materializer  # noqa: E402
from static_policy import (  # noqa: E402
    MAX_PHYSICAL_LINE_LENGTH,
    MAX_SOURCE_BYTES,
    check_source,
)


def _load_score_submission():
    path = TEMPLATE_TESTS / "score_submission.py"
    spec = importlib.util.spec_from_file_location("candidate_score_submission", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _blueprint(tmp_path: Path, *, line_limit: object = 160) -> Path:
    blueprint = tmp_path / "blueprint"
    expert = blueprint / "expert"
    expert.mkdir(parents=True)
    (blueprint / "instruction.md").write_text("# Candidate\n\nSolve it.\n")
    (blueprint / "evaluate_901.py").write_text("print('Overall: PASS')\n")
    (expert / "solution_901.py").write_text(
        "import tensorcircuit\n\ndef run_solution(config):\n    return {}\n"
    )
    metadata = {
        "candidate_id": "line-limit-candidate",
        "title": "Candidate-specific line limit",
        "problem_id": 901,
        "contract": {"effective_line_limit": line_limit},
    }
    (blueprint / "blueprint.json").write_text(json.dumps(metadata))
    return blueprint


def test_materializes_candidate_effective_line_limit_into_verifier_env(
    tmp_path: Path,
) -> None:
    task_dir = materializer.materialize_candidate_task(
        _blueprint(tmp_path), tmp_path / "staging"
    )

    task = tomllib.loads((task_dir / "task.toml").read_text())
    assert task["verifier"]["env"]["MAX_EFFECTIVE_CODE_LINES"] == "160"
    assert (
        "at most 160 effective Python lines"
        in (task_dir / "instruction.md").read_text()
    )


@pytest.mark.parametrize("line_limit", [True, 0, -1, 160.0, "160", 201])
def test_materializer_rejects_invalid_candidate_effective_line_limit(
    tmp_path: Path, line_limit: object
) -> None:
    with pytest.raises(materializer.CandidateTaskError, match="effective_line_limit"):
        materializer.inspect_blueprint(_blueprint(tmp_path, line_limit=line_limit))


def test_materializer_rejects_non_object_contract(tmp_path: Path) -> None:
    blueprint = _blueprint(tmp_path)
    metadata_path = blueprint / "blueprint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["contract"] = ["effective_line_limit", 160]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(materializer.CandidateTaskError, match="contract must be"):
        materializer.inspect_blueprint(blueprint)


def test_materializer_preserves_default_line_limit_for_legacy_blueprint(
    tmp_path: Path,
) -> None:
    blueprint = _blueprint(tmp_path)
    metadata_path = blueprint / "blueprint.json"
    metadata = json.loads(metadata_path.read_text())
    del metadata["contract"]
    metadata_path.write_text(json.dumps(metadata))

    task_dir = materializer.materialize_candidate_task(blueprint, tmp_path / "staging")
    task = tomllib.loads((task_dir / "task.toml").read_text())
    assert task["verifier"]["env"]["MAX_EFFECTIVE_CODE_LINES"] == "200"


@pytest.mark.parametrize("value", ["", "0", "-1", "160.0", " 160", "201"])
def test_score_submission_rejects_invalid_effective_line_env(value: str) -> None:
    score_submission = _load_score_submission()
    with pytest.raises(ValueError, match="MAX_EFFECTIVE_CODE_LINES"):
        score_submission._max_effective_code_lines(value)


def test_score_submission_propagates_candidate_line_limit(
    tmp_path: Path, monkeypatch
) -> None:
    score_submission = _load_score_submission()
    source = tmp_path / "solution_901.py"
    source.write_text("import tensorcircuit\n" + "x = 1\n" * 160)

    monkeypatch.setenv("MAX_EFFECTIVE_CODE_LINES", "160")
    candidate_result = score_submission._static_policy_result(source, "tensorcircuit")
    assert candidate_result["line_count"] == 161
    assert candidate_result["max_lines"] == 160
    assert candidate_result["line_count_score"] == 0.0

    monkeypatch.delenv("MAX_EFFECTIVE_CODE_LINES")
    canonical_result = score_submission._static_policy_result(source, "tensorcircuit")
    assert canonical_result["max_lines"] == 200
    assert canonical_result["line_count_score"] == 1.0


@pytest.mark.parametrize(
    ("runtime_sec", "expected_runtime_score"),
    [(90.0, 1.0), (240.0, 0.5), (301.0, 0.0)],
)
def test_runtime_is_reporting_only_for_compound_reward(
    runtime_sec: float, expected_runtime_score: float
) -> None:
    score_submission = _load_score_submission()

    assert score_submission._runtime_score(runtime_sec) == pytest.approx(
        expected_runtime_score
    )
    assert score_submission._compound_reward(1.0, 1.0, 1.0) == 1.0
    assert score_submission._compound_reward(0.0, 1.0, 1.0) == 0.0


def test_functional_timeout_preserves_flushed_case_identity_bytes(monkeypatch) -> None:
    score_submission = _load_score_submission()
    marker = (
        b'{"orbit_q_case_identity":{"case_digest":"'
        + b"a" * 64
        + b'","protocol_seed":901}}\n'
    )
    written: dict[str, str] = {}

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("evaluate", 1, output=marker)

    monkeypatch.setattr(score_submission.subprocess, "run", timeout)
    monkeypatch.setattr(
        score_submission.Path,
        "write_text",
        lambda self, value: written.setdefault(str(self), value),
    )

    result = score_submission._functional_score(901, "solution_901")

    assert "orbit_q_case_identity" in result["functional_output_tail"]
    assert "FUNCTIONAL_TIMEOUT" in result["functional_output_tail"]
    assert "orbit_q_case_identity" in next(iter(written.values()))


@pytest.mark.parametrize(
    ("returncode", "output", "expected"),
    [
        (0, "Overall: PASS\ntrusted checks\nOverall: FAIL\n", False),
        (0, "forged Overall: PASS\ntrusted checks\nOverall: PASS\n", True),
        (1, "Overall: PASS\n", False),
        (0, "no trusted overall decision\n", False),
    ],
)
def test_functional_score_uses_last_exact_overall_decision(
    monkeypatch, returncode: int, output: str, expected: bool
) -> None:
    score_submission = _load_score_submission()
    completed = subprocess.CompletedProcess(
        args=["evaluate"], returncode=returncode, stdout=output
    )
    monkeypatch.setattr(score_submission.subprocess, "run", lambda *_a, **_k: completed)
    monkeypatch.setattr(score_submission.Path, "write_text", lambda *_a, **_k: None)

    result = score_submission._functional_score(901, "solution_901")

    assert result["functional_passed"] is expected
    assert result["functional_score"] == float(expected)


def test_expert_admission_uses_full_evaluator_output_and_recomputes_margin(
    tmp_path: Path, monkeypatch
) -> None:
    score_submission = _load_score_submission()
    seed = 1092026
    case_digest = "a" * 64
    bindings = {
        "ORBIT_Q_EXPERT_PREQUALIFICATION_MODE": "expert_only_oracle",
        "ORBIT_Q_CANDIDATE_ID": "candidate-109",
        "ORBIT_Q_CANDIDATE_HASH": "b" * 64,
        "ORBIT_Q_CANDIDATE_SEED": str(seed),
        "ORBIT_Q_EXPERT_BASELINE_SHA256": "c" * 64,
        "ORBIT_Q_EVALUATOR_SHA256": "d" * 64,
        "ORBIT_Q_TASK_BUNDLE_SHA256": "e" * 64,
        "ORBIT_Q_FRAMEWORK_PROMPT_SHA256": "f" * 64,
        "ORBIT_Q_VERIFIER_HARNESS_SHA256": "2" * 64,
        "ORBIT_Q_CONTAINER_IMAGE_DIGEST": "sha256:" + "1" * 64,
    }
    for name, value in bindings.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        score_submission, "EXPERT_ADMISSION_PATH", tmp_path / "expert-admission.json"
    )
    identity = json.dumps(
        {
            "orbit_q_case_identity": {
                "protocol_seed": seed,
                "case_digest": case_digest,
            }
        }
    )
    metrics = json.dumps(
        {
            "orbit_q_expert_admission_metrics": {
                "schema_version": 1,
                "protocol_seed": seed,
                "case_digest": case_digest,
                "metrics": [
                    {
                        "metric": "heldout_worst_infidelity",
                        "direction": "at_most",
                        "observed": 0.00075,
                        "threshold": 0.001,
                    }
                ],
            }
        }
    )
    output = identity + "\n" + ("diagnostic filler\n" * 400) + metrics + "\n"
    assert len(output) > 4000

    record, digest = score_submission._expert_admission_record(output, True)

    margin = record["admission"]["margins"][0]
    assert margin["absolute_margin"] == pytest.approx(0.00025)
    assert record["admission"]["minimum_margin"] == pytest.approx(0.00025)
    words = score_submission._sha256_words(digest)
    reconstructed = "".join(
        f"{words[f'expert_admission_sha256_word_{index}']:08x}" for index in range(8)
    )
    assert reconstructed == digest


def test_expert_admission_rejects_nonpositive_evaluator_margin(
    tmp_path: Path, monkeypatch
) -> None:
    score_submission = _load_score_submission()
    seed = 1
    for name, value in {
        "ORBIT_Q_EXPERT_PREQUALIFICATION_MODE": "expert_only_oracle",
        "ORBIT_Q_CANDIDATE_ID": "candidate",
        "ORBIT_Q_CANDIDATE_HASH": "a" * 64,
        "ORBIT_Q_CANDIDATE_SEED": str(seed),
        "ORBIT_Q_EXPERT_BASELINE_SHA256": "b" * 64,
        "ORBIT_Q_EVALUATOR_SHA256": "c" * 64,
        "ORBIT_Q_TASK_BUNDLE_SHA256": "d" * 64,
        "ORBIT_Q_FRAMEWORK_PROMPT_SHA256": "e" * 64,
        "ORBIT_Q_VERIFIER_HARNESS_SHA256": "2" * 64,
        "ORBIT_Q_CONTAINER_IMAGE_DIGEST": "sha256:" + "f" * 64,
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(
        score_submission, "EXPERT_ADMISSION_PATH", tmp_path / "expert-admission.json"
    )
    case_digest = "1" * 64
    output = "\n".join(
        [
            json.dumps(
                {
                    "orbit_q_case_identity": {
                        "protocol_seed": seed,
                        "case_digest": case_digest,
                    }
                }
            ),
            json.dumps(
                {
                    "orbit_q_expert_admission_metrics": {
                        "schema_version": 1,
                        "protocol_seed": seed,
                        "case_digest": case_digest,
                        "metrics": [
                            {
                                "metric": "zero_headroom",
                                "direction": "at_most",
                                "observed": 1.0,
                                "threshold": 1.0,
                            }
                        ],
                    }
                }
            ),
        ]
    )

    with pytest.raises(ValueError, match="no strict pass margin"):
        score_submission._expert_admission_record(output, True)


def test_static_policy_rejects_semicolon_statement_packing(tmp_path: Path) -> None:
    source = tmp_path / "solution_901.py"
    source.write_text("import tensorcircuit\nx = 1; y = 2\n")

    result = check_source(source, "tensorcircuit", max_lines=160)

    assert result["line_count"] == 2
    assert result["logical_statement_count"] == 3
    assert result["semicolon_count"] == 1
    assert result["line_count_score"] == 0.0
    assert "semicolon_statement_packing" in result["cheating_hits"]
    assert result["static_policy_score"] == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        'exec("x = 1\\n" * 1000)',
        'eval("1 + 1")',
        'compile("x = 1", "<packed>", "exec")',
        '__import__("tensorcircuit")',
        "import importlib",
        "import inspect",
        "import gc",
        "import ctypes",
    ],
)
def test_static_policy_rejects_dynamic_code_loading(
    tmp_path: Path, payload: str
) -> None:
    source = tmp_path / "solution_901.py"
    source.write_text(f"import tensorcircuit\n{payload}\n")

    result = check_source(source, "tensorcircuit", max_lines=160)

    assert result["dynamic_code_hits"]
    assert result["no_static_cheating_score"] == 0.0
    assert result["static_policy_score"] == 0.0


@pytest.mark.parametrize(
    "payload",
    [
        'import os\nseed = os.getenv("ORBIT_Q_CANDIDATE_SEED")',
        'import os\nseed = os.environ.get("ORBIT_Q_CANDIDATE_SEED")',
        "import sys\ncaller = sys._getframe(1)",
        'path = "/proc/self/environ"',
        'path = "/tests/evaluate_901.py"',
    ],
)
def test_static_policy_rejects_verifier_introspection(
    tmp_path: Path, payload: str
) -> None:
    source = tmp_path / "solution_901.py"
    source.write_text(f"import tensorcircuit\n{payload}\n")

    result = check_source(source, "tensorcircuit", max_lines=160)

    assert result["cheating_hits"]
    assert result["no_static_cheating_score"] == 0.0
    assert result["static_policy_score"] == 0.0


@pytest.mark.parametrize("forbidden", ["pennylane", "qiskit", "cirq", "qutip"])
def test_static_policy_rejects_other_quantum_frameworks(
    tmp_path: Path, forbidden: str
) -> None:
    source = tmp_path / "solution_901.py"
    source.write_text(f"import tensorcircuit\nimport {forbidden}\n")

    result = check_source(source, "tensorcircuit", max_lines=160)

    assert result["framework_score"] == 1.0
    assert result["forbidden_imports"] == [forbidden]
    assert result["no_forbidden_framework_score"] == 0.0
    assert result["static_policy_score"] == 0.0


def test_static_policy_rejects_overlong_physical_line(tmp_path: Path) -> None:
    source = tmp_path / "solution_901.py"
    source.write_text(
        "import tensorcircuit\nvalue = '" + "x" * MAX_PHYSICAL_LINE_LENGTH + "'\n"
    )

    result = check_source(source, "tensorcircuit", max_lines=160)

    assert result["max_physical_line_length"] > MAX_PHYSICAL_LINE_LENGTH
    assert result["line_count_score"] == 0.0
    assert result["static_policy_score"] == 0.0


def test_static_policy_rejects_oversized_comment_padding(tmp_path: Path) -> None:
    source = tmp_path / "solution_901.py"
    source.write_text(
        "import tensorcircuit\n" + "# padding\n" * (MAX_SOURCE_BYTES // 5)
    )

    result = check_source(source, "tensorcircuit", max_lines=160)

    assert result["source_size_bytes"] > MAX_SOURCE_BYTES
    assert result["line_count"] == 1
    assert result["line_count_score"] == 0.0
    assert result["static_policy_score"] == 0.0


def test_all_candidate_experts_fit_declared_structural_budget() -> None:
    blueprint_root = ROOT / "reports" / "orbit_q_problem_discovery" / "blueprints"
    for metadata_path in sorted(blueprint_root.glob("*/blueprint.json")):
        metadata = json.loads(metadata_path.read_text())
        expert_name = metadata.get("expert_solution")
        if not isinstance(expert_name, str):
            continue
        expert = metadata_path.parent / expert_name
        if not expert.is_file():
            continue
        contract = metadata.get("contract")
        limit = (
            contract.get("effective_line_limit", 200)
            if isinstance(contract, dict)
            else 200
        )
        result = check_source(expert, "tensorcircuit", max_lines=limit)
        assert result["static_policy_score"] == 1.0, (
            metadata_path.parent.name,
            result,
        )
