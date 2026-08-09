from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = (
    ROOT
    / "reports"
    / "orbit_q_problem_discovery"
    / "blueprints"
    / "coherent-toric-recovery-portfolio"
    / "evaluate_116.py"
)


def _load_evaluator():
    spec = importlib.util.spec_from_file_location("orbit_q_evaluate_116", EVALUATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_submitted_recovery_requires_exact_binary_integer_lists() -> None:
    evaluator = _load_evaluator()
    valid = {
        "horizontal": [[0, 1], [1, 0]],
        "vertical": [[1, 0], [0, 1]],
    }
    horizontal, vertical = evaluator._strict_submitted_arrays(valid, 2)
    assert horizontal.tolist() == valid["horizontal"]
    assert vertical.tolist() == valid["vertical"]

    for bad_value in (0.5, 256, -1, True, "1"):
        invalid = {
            "horizontal": [[bad_value, 1], [1, 0]],
            "vertical": [[1, 0], [0, 1]],
        }
        with pytest.raises(ValueError, match="exact binary integers"):
            evaluator._strict_submitted_arrays(invalid, 2)

    with pytest.raises(ValueError, match="exactly 2 rows"):
        evaluator._strict_submitted_arrays(
            {"horizontal": [[0, 1]], "vertical": [[1, 0], [0, 1]]}, 2
        )


def test_submitted_metrics_reject_nonfinite_and_coerced_values() -> None:
    evaluator = _load_evaluator()
    assert evaluator._strict_submitted_metrics(
        {"robust_fidelity": 0.75, "total_recovery_cost": 9}
    ) == (0.75, 9)

    for bad_score in (float("nan"), float("inf"), True, "0.75"):
        with pytest.raises(ValueError, match="finite real number"):
            evaluator._strict_submitted_metrics(
                {"robust_fidelity": bad_score, "total_recovery_cost": 9}
            )

    for bad_cost in (9.5, True, "9"):
        with pytest.raises(ValueError, match="exact integer"):
            evaluator._strict_submitted_metrics(
                {"robust_fidelity": 0.75, "total_recovery_cost": bad_cost}
            )
