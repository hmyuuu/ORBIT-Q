"""Independent NumPy evaluator for process-tensor QEC policy synthesis."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import itertools
import json
import os
import sys
import time

import numpy as np


DEFAULT_SEED = 1182026
LATTICE_SIZE = 5
ROUNDS = 5
HISTORIES = 6
SCENARIOS = 3
PAULI4 = tuple(
    np.diag(
        [
            (-1)
            ** (
                ((sector & 1) * (basis & 1))
                ^ (((sector >> 1) & 1) * ((basis >> 1) & 1))
            )
            for basis in range(4)
        ]
    ).astype(np.complex128)
    for sector in range(4)
)


def default_seed():
    return int(
        os.environ.get(
            "ORBIT_Q_CANDIDATE_SEED",
            os.environ.get("ORBIT_PROCESS_TENSOR_QEC_SEED", str(DEFAULT_SEED)),
        )
    )


def _case_digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rotation(vector):
    x, y, z = np.asarray(vector, dtype=float)
    identity = np.eye(2, dtype=np.complex128)
    paulis = (
        np.array([[0, 1], [1, 0]], dtype=np.complex128),
        np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
        np.diag([1, -1]).astype(np.complex128),
    )
    output = identity
    for angle, pauli in zip((x, y, z), paulis):
        gate = np.cos(angle / 2) * identity - 1j * np.sin(angle / 2) * pauli
        output = gate @ output
    return output


def _memory_state(preparation):
    polar, azimuth = preparation
    return np.asarray(
        [np.cos(polar / 2), np.exp(1j * azimuth) * np.sin(polar / 2)],
        dtype=np.complex128,
    )


def _arrays(recovery):
    return (
        np.asarray(recovery["horizontal"], dtype=np.uint8),
        np.asarray(recovery["vertical"], dtype=np.uint8),
    )


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
    return (
        plaquettes ^ np.roll(plaquettes, 1, axis=0),
        plaquettes ^ np.roll(plaquettes, 1, axis=1),
    )


def _edge_weights(angles):
    angles = np.asarray(angles, dtype=float)
    return np.stack((np.cos(angles / 2), -1j * np.sin(angles / 2)), axis=-1)


def _spatial_amplitude(size, recovery, angle_record, sector):
    horizontal, vertical = _arrays(recovery)
    weights_h = _edge_weights(angle_record["horizontal_angles"])
    weights_v = _edge_weights(angle_record["vertical_angles"])
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
        product = product @ (transfer * diagonal[None, :])
    return np.trace(product) / 2


def _spatial_bruteforce(size, recovery, angle_record, sector):
    horizontal, vertical = _arrays(recovery)
    weights_h = _edge_weights(angle_record["horizontal_angles"])
    weights_v = _edge_weights(angle_record["vertical_angles"])
    total = 0j
    for packed in range(1 << (size * size)):
        plaquettes = np.asarray(
            [(packed >> bit) & 1 for bit in range(size * size)], dtype=np.uint8
        ).reshape(size, size)
        boundary_h, boundary_v = _plaquette_boundary(plaquettes)
        error_h, error_v = horizontal ^ boundary_h, vertical ^ boundary_v
        if sector & 1:
            error_h[0, :] ^= 1
        if sector & 2:
            error_v[:, 0] ^= 1
        total += np.prod(
            weights_h[
                np.indices(error_h.shape)[0], np.indices(error_h.shape)[1], error_h
            ]
        ) * np.prod(
            weights_v[
                np.indices(error_v.shape)[0], np.indices(error_v.shape)[1], error_v
            ]
        )
    return total / 2


def _instrument(amplitudes, memory_rotation):
    transition = _rotation(memory_rotation)
    return sum(
        np.kron(PAULI4[sector], transition @ np.diag(amplitudes[:, sector]))
        for sector in range(4)
    )


def _action_gate(action):
    return np.kron(
        PAULI4[int(action["logical_frame"])], _rotation(action["memory_kick"])
    )


def _fidelity(
    operators,
    sequence,
    actions,
    memory_preparation,
    action_gates=None,
    memory_state=None,
):
    if action_gates is None:
        action_gates = [_action_gate(action) for action in actions]
    if memory_state is None:
        memory_state = _memory_state(memory_preparation)
    evolution = np.eye(8, dtype=np.complex128)
    for instrument, selected in zip(operators, sequence):
        evolution = action_gates[selected] @ instrument @ evolution
    blocks = np.einsum(
        "ambn,n->mab",
        evolution.reshape(4, 2, 4, 2),
        memory_state,
    )
    probability = float(np.sum(np.abs(blocks) ** 2).real / 4)
    if not np.isfinite(probability) or probability <= 1e-300:
        raise RuntimeError("conditional history probability underflow")
    numerator = float(np.sum(np.abs(np.trace(blocks, axis1=1, axis2=2)) ** 2).real / 16)
    return numerator / probability, probability


def _build_tables(case):
    tables = []
    minimum_probability = np.inf
    maximum_memory_contrast = 0.0
    for history in case["syndrome_histories"]:
        history_tables = []
        for scenario in case["process_scenarios"]:
            scenario_tables = []
            for round_record, process_round in zip(
                history["rounds"], scenario["rounds"]
            ):
                amplitudes = np.empty((2, 4), dtype=np.complex128)
                for memory in range(2):
                    for sector in range(4):
                        amplitudes[memory, sector] = _spatial_amplitude(
                            case["lattice_size"],
                            round_record["anchor_recovery"],
                            process_round["memory_conditioned_angles"][memory],
                            sector,
                        )
                maximum_memory_contrast = max(
                    maximum_memory_contrast,
                    float(np.max(np.abs(amplitudes[0] - amplitudes[1]))),
                )
                scenario_tables.append(
                    _instrument(amplitudes, process_round["memory_rotation"])
                )
            history_tables.append(scenario_tables)
        tables.append(history_tables)
    reference_action = min(
        range(len(case["action_primitives"])),
        key=lambda index: (
            case["action_primitives"][index]["intervention_cost"],
            case["action_primitives"][index]["action_id"],
        ),
    )
    reference_sequence = (reference_action,) * case["round_count"]
    for history_index, history_tables in enumerate(tables):
        for scenario_index, operators in enumerate(history_tables):
            _, probability = _fidelity(
                operators,
                reference_sequence,
                case["action_primitives"],
                case["process_scenarios"][scenario_index]["memory_preparation"],
            )
            minimum_probability = min(minimum_probability, probability)
    return tables, minimum_probability, maximum_memory_contrast


def _history_frontier(case, history_tables):
    actions = case["action_primitives"]
    gates = [_action_gate(action) for action in actions]
    memory_states = [
        _memory_state(scenario["memory_preparation"])
        for scenario in case["process_scenarios"]
    ]
    frontier = {}
    for sequence in itertools.product(range(len(actions)), repeat=case["round_count"]):
        values = [
            _fidelity(
                operators,
                sequence,
                actions,
                case["process_scenarios"][scenario]["memory_preparation"],
                gates,
                memory_states[scenario],
            )[0]
            for scenario, operators in enumerate(history_tables)
        ]
        robust = float(min(values))
        cost = sum(int(actions[selected]["intervention_cost"]) for selected in sequence)
        key = tuple(actions[selected]["action_id"] for selected in sequence)
        previous = frontier.get(cost)
        if (
            previous is None
            or robust > previous[0] + 1e-14
            or (abs(robust - previous[0]) <= 1e-14 and key < previous[2])
        ):
            frontier[cost] = (robust, sequence, key)
    return frontier


def _portfolio_states(case, frontiers):
    states = {0: (0.0, (), ())}
    for history, frontier in zip(case["syndrome_histories"], frontiers):
        updated = {}
        weight = float(history["history_weight"])
        for cost, (score, sequences, keys) in states.items():
            for extra_cost, (value, sequence, key) in frontier.items():
                total_cost = cost + extra_cost
                candidate = (
                    score + weight * value,
                    sequences + (sequence,),
                    keys + (key,),
                )
                previous = updated.get(total_cost)
                if (
                    previous is None
                    or candidate[0] > previous[0] + 1e-14
                    or (
                        abs(candidate[0] - previous[0]) <= 1e-14
                        and candidate[2] < previous[2]
                    )
                ):
                    updated[total_cost] = candidate
        states = updated
    return states


def _best_state(states, budget):
    eligible = [
        (score, cost, sequences, keys)
        for cost, (score, sequences, keys) in states.items()
        if cost <= budget
    ]
    if not eligible:
        raise RuntimeError("intervention budget admits no policy portfolio")
    return min(eligible, key=lambda value: (-value[0], value[1], value[3]))


def _solve_case(case, tables, frontiers=None, states=None):
    if frontiers is None:
        frontiers = [
            _history_frontier(case, history_tables) for history_tables in tables
        ]
    if states is None:
        states = _portfolio_states(case, frontiers)
    score, cost, sequences, _ = _best_state(states, case["shared_intervention_budget"])
    policies = []
    for history, history_tables, sequence in zip(
        case["syndrome_histories"], tables, sequences
    ):
        robust = min(
            _fidelity(
                operators,
                sequence,
                case["action_primitives"],
                case["process_scenarios"][scenario]["memory_preparation"],
            )[0]
            for scenario, operators in enumerate(history_tables)
        )
        policies.append(
            {
                "history_id": history["history_id"],
                "action_ids": [
                    case["action_primitives"][selected]["action_id"]
                    for selected in sequence
                ],
                "history_robust_fidelity": float(robust),
            }
        )
    return (
        {
            "case_id": case["case_id"],
            "policies": policies,
            "robust_objective": float(score),
            "total_intervention_cost": int(cost),
        },
        frontiers,
        states,
    )


def _random_recovery(rng, size):
    horizontal = np.zeros((size, size), dtype=np.uint8)
    vertical = np.zeros_like(horizontal)
    for _ in range(int(rng.integers(2, 6))):
        target = horizontal if rng.integers(2) == 0 else vertical
        target[tuple(rng.integers(size, size=2))] ^= 1
    return _record(horizontal, vertical)


def _action_primitives(rng):
    specifications = [
        (0, [0.0, 0.0, 0.0], 0),
        (1, [0.62, -0.18, 0.31], 1),
        (2, [-0.27, 0.71, -0.46], 2),
        (3, [0.86, -0.54, 0.73], 3),
        (0, [1.31, 0.44, -0.79], 4),
    ]
    records = [
        {
            "action_id": f"a{rng.integers(1 << 44):011x}",
            "logical_frame": frame,
            "memory_kick": kick,
            "intervention_cost": cost,
        }
        for frame, kick, cost in specifications
    ]
    rng.shuffle(records)
    return records


def _process_scenarios(rng, size, rounds):
    scenarios = []
    for scenario_index in range(SCENARIOS):
        process_rounds = []
        scenario_shift = 0.06 * (scenario_index - 1)
        for _ in range(rounds):
            base_h = rng.uniform(0.72, 1.62, size=(size, size)) * rng.choice(
                [-1, 1], size=(size, size)
            )
            base_v = rng.uniform(0.72, 1.62, size=(size, size)) * rng.choice(
                [-1, 1], size=(size, size)
            )
            conditioned = []
            for memory in range(2):
                direction = -1 if memory == 0 else 1
                conditioned.append(
                    {
                        "horizontal_angles": (
                            (1 + scenario_shift + direction * 0.105) * base_h
                            + rng.normal(0, 0.045, size=(size, size))
                        ).tolist(),
                        "vertical_angles": (
                            (1 - scenario_shift + direction * 0.105) * base_v
                            + rng.normal(0, 0.045, size=(size, size))
                        ).tolist(),
                    }
                )
            process_rounds.append(
                {
                    "memory_rotation": rng.uniform(
                        [-0.48, -0.42, -0.50], [0.48, 0.42, 0.50]
                    ).tolist(),
                    "memory_conditioned_angles": conditioned,
                }
            )
        scenarios.append(
            {
                "scenario_id": f"p{rng.integers(1 << 44):011x}",
                "memory_preparation": rng.uniform([0.35, -2.4], [2.75, 2.4]).tolist(),
                "rounds": process_rounds,
            }
        )
    return scenarios


def _candidate_case(rng, case_index):
    weights = rng.uniform(0.7, 1.3, size=HISTORIES)
    weights /= np.sum(weights)
    histories = []
    seen = set()
    for history_index in range(HISTORIES):
        round_records = []
        signature = []
        for round_index in range(ROUNDS):
            recovery = _random_recovery(rng, LATTICE_SIZE)
            syndrome = _syndrome(recovery)
            signature.extend(syndrome.ravel().tolist())
            round_records.append(
                {
                    "round_index": round_index,
                    "syndrome_id": f"s{rng.integers(1 << 44):011x}",
                    "observed_star_syndrome": syndrome.astype(int).tolist(),
                    "anchor_recovery": recovery,
                }
            )
        key = tuple(signature)
        if key in seen:
            raise RuntimeError("duplicate syndrome history")
        seen.add(key)
        histories.append(
            {
                "history_id": f"h{rng.integers(1 << 44):011x}",
                "history_weight": float(weights[history_index]),
                "rounds": round_records,
            }
        )
    case = {
        "case_id": f"c{case_index}-{rng.integers(1 << 44):011x}",
        "lattice_size": LATTICE_SIZE,
        "data_qubits": 2 * LATTICE_SIZE * LATTICE_SIZE,
        "round_count": ROUNDS,
        "action_primitives": _action_primitives(rng),
        "syndrome_histories": histories,
        "process_scenarios": _process_scenarios(rng, LATTICE_SIZE, ROUNDS),
    }
    tables, minimum_probability, memory_contrast = _build_tables(case)
    frontiers = [_history_frontier(case, history_tables) for history_tables in tables]
    states = _portfolio_states(case, frontiers)
    min_cost, max_cost = min(states), max(states)
    cheap = _best_state(states, min_cost)
    top = _best_state(states, max_cost)
    options = []
    for budget in range(min_cost, max_cost + 1):
        selected = _best_state(states, budget)
        if selected[0] <= cheap[0] + 7e-4 or selected[0] >= top[0] - 7e-4:
            continue
        balance = abs(selected[0] - (cheap[0] + top[0]) / 2)
        options.append((balance, selected[1], selected))
    if not options:
        return None
    _, budget, selected = min(options)
    case["shared_intervention_budget"] = int(budget)
    oracle, _, _ = _solve_case(case, tables, frontiers, states)
    metrics = {
        "minimum_reference_policy_probability": float(minimum_probability),
        "maximum_memory_conditioned_amplitude_contrast": float(memory_contrast),
        "cheapest_objective": float(cheap[0]),
        "selected_objective": float(selected[0]),
        "unconstrained_objective": float(top[0]),
        "cheapest_to_selected_margin": float(selected[0] - cheap[0]),
        "selected_to_unconstrained_margin": float(top[0] - selected[0]),
        "selected_cost": int(selected[1]),
        "unconstrained_cost": int(top[1]),
    }
    return case, tables, metrics, oracle


def _make_case(rng, case_index):
    for _ in range(12):
        built = _candidate_case(rng, case_index)
        if built is not None:
            return built
    raise RuntimeError("failed to admit an active process-tensor policy case")


def _configuration(seed):
    rng = np.random.default_rng(seed)
    cases, tables, metrics, oracles = [], [], [], []
    for case_index in range(2):
        case, case_tables, case_metrics, oracle = _make_case(rng, case_index)
        cases.append(case)
        tables.append(case_tables)
        metrics.append(case_metrics)
        oracles.append(oracle)
    return {"cases": cases}, tables, metrics, oracles


def _assert_case_invariants(case):
    if case["data_qubits"] != 2 * case["lattice_size"] ** 2:
        raise AssertionError("data-qubit count is inconsistent")
    if (
        len(case["syndrome_histories"]) != HISTORIES
        or len(case["process_scenarios"]) != SCENARIOS
    ):
        raise AssertionError("production history or scenario count drifted")
    action_ids = [action["action_id"] for action in case["action_primitives"]]
    if len(action_ids) != len(set(action_ids)):
        raise AssertionError("action identifiers are not unique")
    history_ids = [history["history_id"] for history in case["syndrome_histories"]]
    if len(history_ids) != len(set(history_ids)):
        raise AssertionError("history identifiers are not unique")
    if (
        abs(
            sum(history["history_weight"] for history in case["syndrome_histories"]) - 1
        )
        > 2e-15
    ):
        raise AssertionError("history weights are not normalized")
    for history in case["syndrome_histories"]:
        if len(history["rounds"]) != case["round_count"]:
            raise AssertionError("history has the wrong round count")
        for round_index, record in enumerate(history["rounds"]):
            if record["round_index"] != round_index:
                raise AssertionError("round ordering drifted")
            if not np.array_equal(
                _syndrome(record["anchor_recovery"]),
                np.asarray(record["observed_star_syndrome"], dtype=np.uint8),
            ):
                raise AssertionError("anchor recovery has the wrong syndrome")
    for scenario in case["process_scenarios"]:
        if len(scenario["rounds"]) != case["round_count"]:
            raise AssertionError("process scenario has the wrong round count")


def _strict_real(value, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{label} must be a finite real number")
    return float(value)


def _strict_integer(value, label):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an exact integer")
    return int(value)


def _validate_result(config, result, tables, oracles):
    if not isinstance(result, dict) or set(result) != {"cases"}:
        raise ValueError("result must contain exactly cases")
    submitted_cases = result["cases"]
    if not isinstance(submitted_cases, list) or len(submitted_cases) != len(
        config["cases"]
    ):
        raise ValueError("cases must preserve exact configured case count and order")
    maximum_objective_error = 0.0
    maximum_history_error = 0.0
    minimum_optimality_margin = np.inf
    for case_index, (case, submitted, case_tables, oracle) in enumerate(
        zip(config["cases"], submitted_cases, tables, oracles)
    ):
        if not isinstance(submitted, dict) or set(submitted) != {
            "case_id",
            "policies",
            "robust_objective",
            "total_intervention_cost",
        }:
            raise ValueError("each case result has an exact four-field schema")
        if submitted["case_id"] != case["case_id"]:
            raise ValueError("case_id or case order mismatch")
        policies = submitted["policies"]
        if not isinstance(policies, list) or len(policies) != len(
            case["syndrome_histories"]
        ):
            raise ValueError("policies must preserve syndrome-history count and order")
        action_lookup = {
            action["action_id"]: (index, action)
            for index, action in enumerate(case["action_primitives"])
        }
        actual_objective = 0.0
        actual_cost = 0
        for history_index, (history, policy, history_tables) in enumerate(
            zip(case["syndrome_histories"], policies, case_tables)
        ):
            if not isinstance(policy, dict) or set(policy) != {
                "history_id",
                "action_ids",
                "history_robust_fidelity",
            }:
                raise ValueError("each history policy has an exact three-field schema")
            if policy["history_id"] != history["history_id"]:
                raise ValueError("history_id or policy order mismatch")
            identifiers = policy["action_ids"]
            if (
                not isinstance(identifiers, list)
                or len(identifiers) != case["round_count"]
            ):
                raise ValueError("action_ids must contain exactly one action per round")
            if not all(
                isinstance(identifier, str) and identifier in action_lookup
                for identifier in identifiers
            ):
                raise ValueError("action_ids contains an unknown or non-string action")
            sequence = tuple(action_lookup[identifier][0] for identifier in identifiers)
            values = [
                _fidelity(
                    operators,
                    sequence,
                    case["action_primitives"],
                    case["process_scenarios"][scenario]["memory_preparation"],
                )[0]
                for scenario, operators in enumerate(history_tables)
            ]
            robust = float(min(values))
            reported = _strict_real(
                policy["history_robust_fidelity"],
                f"case {case_index} history {history_index} robust fidelity",
            )
            maximum_history_error = max(maximum_history_error, abs(reported - robust))
            actual_objective += float(history["history_weight"]) * robust
            actual_cost += sum(
                int(action_lookup[identifier][1]["intervention_cost"])
                for identifier in identifiers
            )
        reported_objective = _strict_real(
            submitted["robust_objective"], "robust_objective"
        )
        reported_cost = _strict_integer(
            submitted["total_intervention_cost"], "total_intervention_cost"
        )
        if reported_cost != actual_cost:
            raise ValueError(
                "reported intervention cost does not match selected actions"
            )
        if actual_cost > case["shared_intervention_budget"]:
            raise ValueError(
                "selected policy portfolio exceeds shared intervention budget"
            )
        optimum = float(oracle["robust_objective"])
        maximum_objective_error = max(
            maximum_objective_error,
            abs(reported_objective - actual_objective),
            abs(actual_objective - optimum),
        )
        minimum_optimality_margin = min(
            minimum_optimality_margin,
            case["shared_intervention_budget"] - actual_cost,
        )
    passed = maximum_objective_error < 3e-9 and maximum_history_error < 3e-9
    return passed, {
        "maximum_objective_or_optimality_error": maximum_objective_error,
        "maximum_history_fidelity_error": maximum_history_error,
        "minimum_unused_budget": int(minimum_optimality_margin),
    }


def _canaries():
    rng = np.random.default_rng(118)
    size = 3
    recovery = _random_recovery(rng, size)
    angle_record = {
        "horizontal_angles": rng.uniform(-1.3, 1.3, size=(size, size)).tolist(),
        "vertical_angles": rng.uniform(-1.3, 1.3, size=(size, size)).tolist(),
    }
    spatial_error = max(
        abs(
            _spatial_amplitude(size, recovery, angle_record, sector)
            - _spatial_bruteforce(size, recovery, angle_record, sector)
        )
        for sector in range(4)
    )
    actions = [
        {"logical_frame": index, "memory_kick": rng.uniform(-0.7, 0.7, 3).tolist()}
        for index in range(3)
    ]
    operators = []
    for _ in range(3):
        amplitudes = rng.normal(size=(2, 4)) + 1j * rng.normal(size=(2, 4))
        operators.append(_instrument(amplitudes, rng.uniform(-0.4, 0.4, 3)))
    sequence = (0, 2, 1)
    preparation = [1.13, -0.72]
    compact, _ = _fidelity(operators, sequence, actions, preparation)
    bell = np.eye(4, dtype=np.complex128).reshape(-1) / 2
    state = np.kron(bell, _memory_state(preparation))
    for operator, selected in zip(operators, sequence):
        state = np.kron(np.eye(4), _action_gate(actions[selected]) @ operator) @ state
    probability = float(np.vdot(state, state).real)
    tensor = state.reshape(4, 4, 2)
    overlaps = np.einsum("iim->m", tensor) / 2
    dense = float(np.sum(np.abs(overlaps) ** 2).real / probability)
    return float(spatial_error), abs(compact - dense)


def _self_test(seeds):
    spatial_error, temporal_error = _canaries()
    if spatial_error >= 2e-11 or temporal_error >= 2e-12:
        raise AssertionError("independent spatial or temporal canary failed")
    records = []
    for seed in seeds:
        started = time.perf_counter()
        config, tables, metrics, oracles = _configuration(seed)
        for case in config["cases"]:
            _assert_case_invariants(case)
        oracle = {"cases": oracles}
        passed, summary = _validate_result(config, oracle, tables, oracles)
        if not passed:
            raise AssertionError("oracle result failed strict validation")
        mutations = []
        bad = copy.deepcopy(oracle)
        bad["cases"][0]["policies"][0]["history_robust_fidelity"] = float("nan")
        mutations.append(bad)
        bad = copy.deepcopy(oracle)
        bad["cases"][0]["total_intervention_cost"] += 1
        mutations.append(bad)
        bad = copy.deepcopy(oracle)
        bad["cases"][0]["policies"][0]["history_id"] = "wrong"
        mutations.append(bad)
        bad = copy.deepcopy(oracle)
        bad["cases"][0]["policies"][0]["action_ids"][0] = "unknown"
        mutations.append(bad)
        bad = copy.deepcopy(oracle)
        case = config["cases"][0]
        cheapest = min(
            range(len(case["action_primitives"])),
            key=lambda index: (
                case["action_primitives"][index]["intervention_cost"],
                case["action_primitives"][index]["action_id"],
            ),
        )
        identifier = case["action_primitives"][cheapest]["action_id"]
        objective = 0.0
        for history, policy, history_tables in zip(
            case["syndrome_histories"], bad["cases"][0]["policies"], tables[0]
        ):
            sequence = (cheapest,) * case["round_count"]
            robust = min(
                _fidelity(
                    operators,
                    sequence,
                    case["action_primitives"],
                    case["process_scenarios"][scenario]["memory_preparation"],
                )[0]
                for scenario, operators in enumerate(history_tables)
            )
            policy["action_ids"] = [identifier] * case["round_count"]
            policy["history_robust_fidelity"] = robust
            objective += history["history_weight"] * robust
        bad["cases"][0]["robust_objective"] = objective
        bad["cases"][0]["total_intervention_cost"] = (
            len(case["syndrome_histories"])
            * case["round_count"]
            * case["action_primitives"][cheapest]["intervention_cost"]
        )
        mutations.append(bad)
        rejected = 0
        for mutation in mutations:
            try:
                mutation_passed, _ = _validate_result(config, mutation, tables, oracles)
            except (ValueError, KeyError, TypeError):
                mutation_passed = False
            rejected += int(not mutation_passed)
        if rejected != len(mutations):
            raise AssertionError("a malformed-output mutation was accepted")
        records.append(
            {
                "seed": seed,
                "case_digest": _case_digest(config),
                "case_metrics": metrics,
                "validation": summary,
                "mutations_rejected": rejected,
                "runtime_sec": time.perf_counter() - started,
            }
        )
    return {
        "spatial_bruteforce_max_error": spatial_error,
        "temporal_dense_choi_error": temporal_error,
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_118")
    parser.add_argument("--seed", type=int, default=default_seed())
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--seeds", nargs="*", type=int)
    args = parser.parse_args()
    if args.self_test:
        seeds = args.seeds or [1182026, 1182037, 1182118, 9876543]
        print(json.dumps(_self_test(seeds), sort_keys=True, indent=2))
        print("Overall: PASS")
        return
    config, tables, _, oracles = _configuration(args.seed)
    digest = _case_digest(config)
    print(
        json.dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": args.seed,
                    "case_digest": digest,
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    os.environ.pop("ORBIT_Q_CANDIDATE_SEED", None)
    os.environ.pop("ORBIT_PROCESS_TENSOR_QEC_SEED", None)
    sys.argv[:] = [sys.argv[0]]
    started = time.perf_counter()
    try:
        result = importlib.import_module(args.solution).run_solution(
            copy.deepcopy(config)
        )
        elapsed = time.perf_counter() - started
        passed, summary = _validate_result(config, result, tables, oracles)
    except Exception as exc:
        elapsed = time.perf_counter() - started
        passed = False
        summary = {"error": f"{type(exc).__name__}: {exc}"}
    runtime_passed = elapsed <= 300
    summary["runtime_limit_seconds"] = 300
    summary["runtime_limit_passed"] = runtime_passed
    passed = passed and runtime_passed
    print(f"End-to-end solution time: {elapsed:.6f}s")
    print(f"Case seed: {args.seed}; case digest: {digest[:16]}")
    print(json.dumps(summary, sort_keys=True))
    print("Overall: PASS" if passed else "Overall: FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
