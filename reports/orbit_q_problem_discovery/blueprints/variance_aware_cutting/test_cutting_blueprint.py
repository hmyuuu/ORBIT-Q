import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("evaluate_104.py")
SPEC = importlib.util.spec_from_file_location("cutting_evaluator", MODULE_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def test_signed_fragment_identity_matches_uncut_cases():
    cases = EVALUATOR.make_cases(1042026)
    for case in cases:
        r, e = EVALUATOR._fragment_means(case)
        assert abs(EVALUATOR._signed(r, e) - EVALUATOR._uncut(case)) < 2e-12


def test_allocation_is_deterministic_and_exhausts_budget():
    case = EVALUATOR.make_cases(9, count=1)[0]
    r, e = EVALUATOR._fragment_means(case)
    means, coefficients = EVALUATOR._means_and_coefficients(r, e)
    first = EVALUATOR._allocate(means, coefficients, case["total_shots"], case["min_shots"])
    second = EVALUATOR._allocate(means, coefficients, case["total_shots"], case["min_shots"])
    assert np.array_equal(first, second)
    assert first.sum() == case["total_shots"]
    assert np.all(first >= case["min_shots"])
