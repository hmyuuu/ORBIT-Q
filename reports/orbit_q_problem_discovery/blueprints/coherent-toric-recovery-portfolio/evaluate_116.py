"""Independent NumPy evaluator for coherent toric recovery selection."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import os
import time

import numpy as np


DEFAULT_SEED = 1162026
LATTICE_SIZE = 7
_QUALITY_CACHE = {}


def default_seed():
    return int(
        os.environ.get(
            "ORBIT_Q_CANDIDATE_SEED",
            os.environ.get("ORBIT_COHERENT_TORIC_SEED", str(DEFAULT_SEED)),
        )
    )


def _arrays(recovery):
    return (
        np.asarray(recovery["horizontal"], dtype=np.uint8),
        np.asarray(recovery["vertical"], dtype=np.uint8),
    )


def _strict_submitted_arrays(recovery, size):
    if not isinstance(recovery, dict):
        raise ValueError("recovery must be a dictionary")

    def parse_matrix(name):
        matrix = recovery.get(name)
        if not isinstance(matrix, list) or len(matrix) != size:
            raise ValueError(f"{name} must have exactly {size} rows")
        parsed = np.empty((size, size), dtype=np.uint8)
        for row_index, row in enumerate(matrix):
            if not isinstance(row, list) or len(row) != size:
                raise ValueError(f"{name} rows must have exactly {size} entries")
            for column_index, value in enumerate(row):
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or int(value) not in (0, 1)
                ):
                    raise ValueError(f"{name} entries must be exact binary integers")
                parsed[row_index, column_index] = int(value)
        return parsed

    return parse_matrix("horizontal"), parse_matrix("vertical")


def _strict_submitted_metrics(submitted):
    score = submitted.get("robust_fidelity")
    cost = submitted.get("total_recovery_cost")
    if (
        isinstance(score, (bool, np.bool_))
        or not isinstance(score, (int, float, np.integer, np.floating))
        or not np.isfinite(score)
    ):
        raise ValueError("robust_fidelity must be a finite real number")
    if isinstance(cost, (bool, np.bool_)) or not isinstance(cost, (int, np.integer)):
        raise ValueError("total_recovery_cost must be an exact integer")
    return float(score), int(cost)


def _record(horizontal, vertical):
    return {
        "horizontal": np.asarray(horizontal, dtype=np.uint8).astype(int).tolist(),
        "vertical": np.asarray(vertical, dtype=np.uint8).astype(int).tolist(),
    }


def _syndrome(recovery):
    horizontal, vertical = _arrays(recovery)
    return (
        horizontal
        ^ np.roll(horizontal, 1, axis=1)
        ^ vertical
        ^ np.roll(vertical, 1, axis=0)
    )


def _plaquette_boundary(plaquettes):
    plaquettes = np.asarray(plaquettes, dtype=np.uint8)
    horizontal = plaquettes ^ np.roll(plaquettes, 1, axis=0)
    vertical = plaquettes ^ np.roll(plaquettes, 1, axis=1)
    return horizontal, vertical


def _relative_homology(recovery, anchor):
    horizontal, vertical = _arrays(recovery)
    anchor_h, anchor_v = _arrays(anchor)
    cycle_h, cycle_v = horizontal ^ anchor_h, vertical ^ anchor_v
    if np.any(_syndrome(_record(cycle_h, cycle_v))):
        return None
    return int(np.sum(cycle_h[:, -1]) & 1) + 2 * int(np.sum(cycle_v[-1, :]) & 1)


def _hardware_cost(case, recovery):
    horizontal, vertical = _arrays(recovery)
    cost_h = np.asarray(case["horizontal_recovery_cost"], dtype=int)
    cost_v = np.asarray(case["vertical_recovery_cost"], dtype=int)
    return int(np.sum(horizontal * cost_h) + np.sum(vertical * cost_v))


def _edge_weights(angles):
    angles = np.asarray(angles, dtype=float)
    return np.stack((np.cos(angles / 2), -1j * np.sin(angles / 2)), axis=-1)


def _amplitude_numpy(size, recovery, branch, sector):
    horizontal, vertical = _arrays(recovery)
    weights_h = _edge_weights(branch["horizontal_angles"])
    weights_v = _edge_weights(branch["vertical_angles"])
    states = np.arange(1 << size, dtype=np.uint16)
    bits = ((states[:, None] >> np.arange(size)) & 1).astype(np.uint8)
    product = np.eye(1 << size, dtype=np.complex128)
    for row in range(size):
        transfer = np.ones((1 << size, 1 << size), dtype=np.complex128)
        for column in range(size):
            base = int(horizontal[row, column]) ^ (int(sector & 1) if row == 0 else 0)
            parity = bits[:, column, None] ^ bits[None, :, column] ^ base
            transfer *= weights_h[row, column][parity]
        diagonal = np.ones(1 << size, dtype=np.complex128)
        for column in range(size):
            base = int(vertical[row, column]) ^ (
                int((sector >> 1) & 1) if column == 0 else 0
            )
            parity = bits[:, column - 1] ^ bits[:, column] ^ base
            diagonal *= weights_v[row, column][parity]
        transfer *= diagonal[None, :]
        product = product @ transfer
    return np.trace(product) / 2


def _scenario_statistics(case):
    syndrome_rows = case["syndrome_records"]
    scenario_count = len(case["noise_scenarios"])
    candidate_count = len(syndrome_rows[0]["recovery_candidates"])
    qualities = np.empty((len(syndrome_rows), candidate_count, scenario_count))
    minimum_probability = np.inf
    maximum_coherence = 0.0
    for row_index, row in enumerate(syndrome_rows):
        candidates = row["recovery_candidates"]
        anchor = next(
            candidate
            for candidate in candidates
            if candidate["candidate_id"] == row["anchor_candidate_id"]
        )["recovery"]
        homologies = [
            _relative_homology(candidate["recovery"], anchor)
            for candidate in candidates
        ]
        for scenario_index, scenario in enumerate(case["noise_scenarios"]):
            process = np.zeros((4, 4), dtype=np.complex128)
            for branch in scenario["branches"]:
                amplitudes = np.asarray(
                    [
                        _amplitude_numpy(case["lattice_size"], anchor, branch, sector)
                        for sector in range(4)
                    ]
                )
                process += branch["probability"] * np.outer(
                    amplitudes, amplitudes.conj()
                )
            probability = float(np.trace(process).real)
            normalized = process / probability
            minimum_probability = min(minimum_probability, probability)
            maximum_coherence = max(
                maximum_coherence,
                float(np.max(np.abs(normalized - np.diag(np.diag(normalized))))),
            )
            for candidate_index, homology in enumerate(homologies):
                qualities[row_index, candidate_index, scenario_index] = normalized[
                    homology, homology
                ].real
    return qualities, minimum_probability, maximum_coherence


def _portfolio_search(case, qualities, budget):
    rows = case["syndrome_records"]
    weights = np.asarray([row["syndrome_weight"] for row in rows])
    costs = [
        [
            _hardware_cost(case, candidate["recovery"])
            for candidate in row["recovery_candidates"]
        ]
        for row in rows
    ]
    best_score, best_cost, best_choice = -1.0, 0, None
    for choice in itertools.product(range(len(costs[0])), repeat=len(rows)):
        cost = sum(costs[row][selected] for row, selected in enumerate(choice))
        if cost > budget:
            continue
        scenario_values = np.sum(
            [
                weights[row] * qualities[row, selected]
                for row, selected in enumerate(choice)
            ],
            axis=0,
        )
        score = float(np.min(scenario_values))
        if score > best_score + 1e-14 or (
            abs(score - best_score) <= 1e-14
            and (best_choice is None or (cost, choice) < (best_cost, best_choice))
        ):
            best_score, best_cost, best_choice = score, cost, choice
    if best_choice is None:
        raise RuntimeError("recovery budget admits no complete portfolio")
    return best_score, best_cost, best_choice


def _choose_budget(case, qualities):
    costs = [
        [
            _hardware_cost(case, candidate["recovery"])
            for candidate in row["recovery_candidates"]
        ]
        for row in case["syndrome_records"]
    ]
    minimum = sum(min(row) for row in costs)
    maximum = sum(max(row) for row in costs)
    weights = np.asarray([row["syndrome_weight"] for row in case["syndrome_records"]])
    best_exact = {}
    for choice in itertools.product(range(len(costs[0])), repeat=len(costs)):
        cost = sum(costs[row][selected] for row, selected in enumerate(choice))
        scenario_values = np.sum(
            [
                weights[row] * qualities[row, selected]
                for row, selected in enumerate(choice)
            ],
            axis=0,
        )
        score = float(np.min(scenario_values))
        if score > best_exact.get(cost, (-1.0, None))[0]:
            best_exact[cost] = (score, choice)
    low_score = best_exact[minimum][0]
    top_score = max(value[0] for value in best_exact.values())
    choices = []
    running_score, running_cost = -1.0, 0
    for budget in range(minimum, maximum + 1):
        exact = best_exact.get(budget)
        if exact is not None and exact[0] > running_score:
            running_score, running_cost = exact[0], budget
        if (
            running_score > low_score + 2e-5
            and running_score < top_score - 2e-5
            and running_cost > minimum
        ):
            balance = abs(running_score - (low_score + top_score) / 2)
            choices.append((balance, budget, running_score, running_cost))
    if not choices:
        return None
    _, budget, _, _ = min(choices)
    return int(budget)


def _random_recovery(rng, size):
    horizontal = np.zeros((size, size), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    for _ in range(int(rng.integers(2, 5))):
        target = horizontal if rng.integers(2) == 0 else vertical
        target[tuple(rng.integers(size, size=2))] ^= 1
    return horizontal, vertical


def _candidate_records(rng, size, anchor_h, anchor_v, cost_h, cost_v):
    records = []
    for sector in range(4):
        base_h, base_v = anchor_h.copy(), anchor_v.copy()
        if sector & 1:
            base_h[0, :] ^= 1
        if sector & 2:
            base_v[:, 0] ^= 1
        for variant in range(2):
            horizontal, vertical = base_h.copy(), base_v.copy()
            if variant:
                plaquettes = np.zeros((size, size), dtype=np.uint8)
                for _ in range(int(rng.integers(1, 4))):
                    plaquettes[tuple(rng.integers(size, size=2))] ^= 1
                boundary_h, boundary_v = _plaquette_boundary(plaquettes)
                horizontal ^= boundary_h
                vertical ^= boundary_v
            recovery = _record(horizontal, vertical)
            records.append(
                {
                    "candidate_id": f"r{rng.integers(1 << 40):010x}",
                    "recovery": recovery,
                    "hardware_cost": int(
                        np.sum(horizontal * cost_h) + np.sum(vertical * cost_v)
                    ),
                }
            )
    anchor_id = records[0]["candidate_id"]
    rng.shuffle(records)
    return records, anchor_id


def _noise_scenarios(rng, size):
    scenarios = []
    for _ in range(3):
        base_h = rng.uniform(0.95, 2.15, size=(size, size)) * rng.choice(
            [-1, 1], size=(size, size)
        )
        base_v = rng.uniform(0.95, 2.15, size=(size, size)) * rng.choice(
            [-1, 1], size=(size, size)
        )
        raw = rng.uniform(0.7, 1.3, size=2)
        probabilities = raw / np.sum(raw)
        branches = []
        for branch in range(2):
            scale = 0.92 + 0.16 * branch
            perturb_h = rng.normal(0, 0.08, size=(size, size))
            perturb_v = rng.normal(0, 0.08, size=(size, size))
            branches.append(
                {
                    "branch_id": f"b{rng.integers(1 << 40):010x}",
                    "probability": float(probabilities[branch]),
                    "horizontal_angles": (scale * base_h + perturb_h).tolist(),
                    "vertical_angles": (scale * base_v + perturb_v).tolist(),
                }
            )
        scenarios.append(
            {
                "scenario_id": f"n{rng.integers(1 << 40):010x}",
                "branches": branches,
            }
        )
    return scenarios


def _make_case(rng, case_index):
    size = LATTICE_SIZE
    for _ in range(24):
        cost_h = rng.integers(1, 5, size=(size, size))
        cost_v = rng.integers(1, 5, size=(size, size))
        rows, seen = [], set()
        while len(rows) < 5:
            anchor_h, anchor_v = _random_recovery(rng, size)
            syndrome = _syndrome(_record(anchor_h, anchor_v))
            key = tuple(syndrome.ravel())
            if not np.any(syndrome) or key in seen:
                continue
            seen.add(key)
            candidates, anchor_id = _candidate_records(
                rng, size, anchor_h, anchor_v, cost_h, cost_v
            )
            rows.append(
                {
                    "syndrome_id": f"s{rng.integers(1 << 40):010x}",
                    "syndrome": syndrome.astype(int).tolist(),
                    "syndrome_weight": float(rng.uniform(0.7, 1.3)),
                    "anchor_candidate_id": anchor_id,
                    "recovery_candidates": candidates,
                }
            )
        total_weight = sum(row["syndrome_weight"] for row in rows)
        for row in rows:
            row["syndrome_weight"] /= total_weight
        frames = [
            {
                "frame_id": f"f{rng.integers(1 << 40):010x}",
                "relative_homology": [sector & 1, (sector >> 1) & 1],
            }
            for sector in range(4)
        ]
        rng.shuffle(frames)
        case = {
            "case_id": f"toric-{case_index}-{rng.integers(1 << 40):010x}",
            "lattice_size": size,
            "n_physical_qubits": 2 * size * size,
            "horizontal_recovery_cost": cost_h.astype(int).tolist(),
            "vertical_recovery_cost": cost_v.astype(int).tolist(),
            "frame_candidates": frames,
            "noise_scenarios": _noise_scenarios(rng, size),
            "syndrome_records": rows,
            "case_nonce": int(rng.integers(1 << 62)),
        }
        qualities, minimum_probability, maximum_coherence = _scenario_statistics(case)
        budget = _choose_budget(case, qualities)
        if (
            budget is not None
            and minimum_probability > 1e-20
            and maximum_coherence > 2e-4
        ):
            case["recovery_budget"] = budget
            _QUALITY_CACHE[case["case_id"]] = (
                qualities,
                minimum_probability,
                maximum_coherence,
            )
            return case
    raise RuntimeError("failed to generate an active recovery tradeoff")


def build_config(seed):
    _QUALITY_CACHE.clear()
    rng = np.random.default_rng(seed)
    cases = [_make_case(rng, index) for index in range(2)]
    payload = json.dumps(
        cases, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return {
        "cases": cases,
        "case_digest": hashlib.sha256(payload).hexdigest(),
        "convention": (
            "horizontal[y,x] joins (y,x) to (y,x+1); vertical[y,x] joins "
            "(y,x) to (y+1,x); all indices are periodic"
        ),
    }


def _bruteforce_amplitude(size, recovery, branch, sector):
    horizontal, vertical = _arrays(recovery)
    weights_h = _edge_weights(branch["horizontal_angles"])
    weights_v = _edge_weights(branch["vertical_angles"])
    total = 0.0j
    for state in range(1 << (size * size)):
        plaquettes = (
            ((state >> np.arange(size * size)) & 1).astype(np.uint8).reshape(size, size)
        )
        boundary_h, boundary_v = _plaquette_boundary(plaquettes)
        error_h = horizontal ^ boundary_h
        error_v = vertical ^ boundary_v
        if sector & 1:
            error_h[0, :] ^= 1
        if sector & 2:
            error_v[:, 0] ^= 1
        term = np.prod(np.take_along_axis(weights_h, error_h[..., None], axis=2))
        term *= np.prod(np.take_along_axis(weights_v, error_v[..., None], axis=2))
        total += term
    return total / 2


def _oracle_canary():
    rng = np.random.default_rng(481516)
    size = 3
    horizontal, vertical = _random_recovery(rng, size)
    branch = {
        "horizontal_angles": rng.uniform(-0.8, 0.8, size=(size, size)).tolist(),
        "vertical_angles": rng.uniform(-0.8, 0.8, size=(size, size)).tolist(),
    }
    recovery = _record(horizontal, vertical)
    errors = [
        abs(
            _amplitude_numpy(size, recovery, branch, sector)
            - _bruteforce_amplitude(size, recovery, branch, sector)
        )
        for sector in range(4)
    ]
    return max(errors)


def _validate_case(case):
    size = case["lattice_size"]
    if size != LATTICE_SIZE or case["n_physical_qubits"] != 2 * size * size:
        return False
    if not np.isclose(
        sum(row["syndrome_weight"] for row in case["syndrome_records"]), 1
    ):
        return False
    frame_map = {
        bits[0] + 2 * bits[1]: frame["frame_id"]
        for frame in case["frame_candidates"]
        for bits in [frame["relative_homology"]]
    }
    if len(frame_map) != 4:
        return False
    for row in case["syndrome_records"]:
        candidates = row["recovery_candidates"]
        if len(candidates) != 8 or len({c["candidate_id"] for c in candidates}) != 8:
            return False
        anchor = next(
            (
                candidate["recovery"]
                for candidate in candidates
                if candidate["candidate_id"] == row["anchor_candidate_id"]
            ),
            None,
        )
        if anchor is None:
            return False
        homologies = []
        for candidate in candidates:
            recovery = candidate["recovery"]
            horizontal, vertical = _arrays(recovery)
            if horizontal.shape != (size, size) or vertical.shape != (size, size):
                return False
            if not np.array_equal(_syndrome(recovery), row["syndrome"]):
                return False
            if candidate["hardware_cost"] != _hardware_cost(case, recovery):
                return False
            homologies.append(_relative_homology(recovery, anchor))
        if sorted(homologies) != [0, 0, 1, 1, 2, 2, 3, 3]:
            return False
    return True


def evaluate(module_name, seed):
    config = build_config(seed)
    print(
        json.dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": seed,
                    "case_digest": config["case_digest"],
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    os.environ.pop("ORBIT_Q_CANDIDATE_SEED", None)
    os.environ.pop("ORBIT_COHERENT_TORIC_SEED", None)
    expected = []
    for case in config["cases"]:
        qualities, probability, coherence = _QUALITY_CACHE[case["case_id"]]
        score, cost, choice = _portfolio_search(
            case, qualities, case["recovery_budget"]
        )
        expected.append((qualities, probability, coherence, score, cost, choice))
    canary_error = _oracle_canary()
    started = time.perf_counter()
    result = importlib.import_module(module_name).run_solution(config)
    elapsed = time.perf_counter() - started
    submitted_cases = result.get("cases", []) if isinstance(result, dict) else []
    if not isinstance(submitted_cases, list):
        submitted_cases = []
    structure = len(submitted_cases) == len(config["cases"])
    identities = structure
    masks_exact = structure
    syndromes_valid = structure
    budgets_valid = structure
    maximum_metric_error = 0.0
    portfolio_gap = 0.0
    tradeoffs_active = True
    for case, target, submitted in zip(config["cases"], expected, submitted_cases):
        qualities, _, _, target_score, _, _ = target
        if (
            not isinstance(submitted, dict)
            or submitted.get("case_id") != case["case_id"]
        ):
            structure = identities = False
            continue
        selections = submitted.get("selections", [])
        if not isinstance(selections, list) or len(selections) != len(
            case["syndrome_records"]
        ):
            structure = False
            continue
        chosen = []
        total_cost = 0
        for row, selection in zip(case["syndrome_records"], selections):
            if (
                not isinstance(selection, dict)
                or selection.get("syndrome_id") != row["syndrome_id"]
            ):
                structure = identities = False
                continue
            candidate_map = {
                candidate["candidate_id"]: (index, candidate)
                for index, candidate in enumerate(row["recovery_candidates"])
            }
            candidate_data = candidate_map.get(selection.get("candidate_id"))
            if candidate_data is None:
                identities = False
                continue
            index, candidate = candidate_data
            anchor = next(
                value["recovery"]
                for value in row["recovery_candidates"]
                if value["candidate_id"] == row["anchor_candidate_id"]
            )
            homology = _relative_homology(candidate["recovery"], anchor)
            frame_id = next(
                frame["frame_id"]
                for frame in case["frame_candidates"]
                if frame["relative_homology"] == [homology & 1, (homology >> 1) & 1]
            )
            identities &= selection.get("frame_id") == frame_id
            try:
                supplied_h, supplied_v = _strict_submitted_arrays(
                    selection["recovery"], case["lattice_size"]
                )
            except (KeyError, TypeError, ValueError):
                masks_exact = False
                continue
            candidate_h, candidate_v = _arrays(candidate["recovery"])
            masks_exact &= bool(
                supplied_h.shape == candidate_h.shape
                and supplied_v.shape == candidate_v.shape
                and np.array_equal(supplied_h, candidate_h)
                and np.array_equal(supplied_v, candidate_v)
            )
            syndromes_valid &= bool(
                supplied_h.shape == candidate_h.shape
                and supplied_v.shape == candidate_v.shape
                and np.array_equal(_syndrome(selection["recovery"]), row["syndrome"])
            )
            chosen.append(index)
            total_cost += _hardware_cost(case, candidate["recovery"])
        if len(chosen) != len(case["syndrome_records"]):
            structure = False
            continue
        weights = np.asarray(
            [row["syndrome_weight"] for row in case["syndrome_records"]]
        )
        values = np.sum(
            [weights[row] * qualities[row, index] for row, index in enumerate(chosen)],
            axis=0,
        )
        actual_score = float(np.min(values))
        portfolio_gap = max(portfolio_gap, target_score - actual_score)
        budgets_valid &= total_cost <= case["recovery_budget"]
        try:
            declared_score, declared_cost = _strict_submitted_metrics(submitted)
            maximum_metric_error = max(
                maximum_metric_error,
                abs(declared_score - actual_score),
                abs(declared_cost - total_cost),
            )
        except (KeyError, TypeError, ValueError):
            structure = False
        costs = [
            [
                _hardware_cost(case, candidate["recovery"])
                for candidate in row["recovery_candidates"]
            ]
            for row in case["syndrome_records"]
        ]
        minimum_budget = sum(min(row) for row in costs)
        cheapest_score = _portfolio_search(case, qualities, minimum_budget)[0]
        maximum_budget = sum(max(row) for row in costs)
        unconstrained_score = _portfolio_search(case, qualities, maximum_budget)[0]
        tradeoffs_active &= bool(
            target_score > cheapest_score + 2e-5
            and target_score < unconstrained_score - 2e-5
        )
    probabilities = [target[1] for target in expected]
    coherences = [target[2] for target in expected]
    criteria = {
        "output structure and opaque identities": bool(structure and identities),
        "exact recovery artifacts": bool(masks_exact),
        "all recoveries match their measured syndrome": bool(syndromes_valid),
        "shared recovery budgets satisfied": bool(budgets_valid),
        "robust portfolio is oracle-optimal": bool(
            structure and portfolio_gap < 3e-9 and maximum_metric_error < 3e-9
        ),
        "independent transfer/bruteforce canary": canary_error < 2e-11,
        "large degenerate toric regime active": all(
            _validate_case(case) for case in config["cases"]
        ),
        "coherent conditioned regime active": min(probabilities) > 1e-20
        and min(coherences) > 2e-4,
        "fidelity-cost conflict active": bool(tradeoffs_active),
    }
    print("Problem 116 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; case digest: {config['case_digest'][:16]}")
    print(f"End-to-end solution time: {elapsed:.3f}s")
    print(f"Maximum declared-metric error: {maximum_metric_error:.3e}")
    print(f"Maximum robust-optimum gap: {portfolio_gap:.3e}")
    print(
        f"Probability floor: {min(probabilities):.3e}; minimum maximum-coherence: {min(coherences):.3e}"
    )
    print(f"Independent tiny-lattice canary error: {canary_error:.3e}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    passed = all(criteria.values())
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_116")
    parser.add_argument("--seed", type=int, default=default_seed())
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.solution, args.seed) else 1)


if __name__ == "__main__":
    main()
