import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("evaluate_103.py")
SPEC = importlib.util.spec_from_file_location("dynamic_evaluator", MODULE_PATH)
EVALUATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVALUATOR)


def test_generation_is_seeded_balanced_and_detectable():
    cases_a = EVALUATOR.make_cases(1032026)
    cases_b = EVALUATOR.make_cases(1032026)
    assert cases_a == cases_b
    distances = np.array([EVALUATOR._oracle_case(case)[0] for case in cases_a])
    assert np.count_nonzero(distances <= 5e-6) == 6
    assert np.all(distances[1::2] > 2e-4)


def test_oracle_handles_zero_probability_history():
    program = {
        "prefix": [],
        "rounds": [{"measure": 0, "reset": True, "branches": {"0": [], "1": []}}],
        "final": [],
    }
    branches = EVALUATOR._branch_states(program, [], 2)
    assert np.isclose(np.trace(branches[0]), 1)
    assert np.isclose(np.trace(branches[1]), 0)

    generated = EVALUATOR.make_cases(1032026)[0]
    hidden_branches = EVALUATOR._branch_states(
        generated["program_a"], generated["probes"][0], generated["n_qubits"]
    )
    assert any(np.isclose(np.trace(branch), 0) for branch in hidden_branches)
