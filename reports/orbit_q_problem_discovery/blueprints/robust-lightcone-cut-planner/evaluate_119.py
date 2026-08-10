"""Independent NumPy evaluator for robust lightcone cut planning."""

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


DEFAULT_SEED = 1192026
QUBITS = 76
FRAGMENTS = 7
MIN_FRAGMENT = 8
MAX_FRAGMENT = 12
LAYERS = 3
OBSERVABLES = 18
STRATA = 9
SCENARIOS = 3
I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
PAULI = {"X": X, "Y": Y, "Z": Z}


def default_seed():
    return int(os.environ.get("ORBIT_Q_CANDIDATE_SEED", str(DEFAULT_SEED)))


def _canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _rotation(pauli, theta):
    return np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * pauli


def _interaction(axis, theta):
    product = np.kron(PAULI[axis[0]], PAULI[axis[1]])
    return np.cos(theta / 2) * np.eye(4) - 1j * np.sin(theta / 2) * product


def _apply(state, matrix, qubits, count):
    rest = [q for q in range(count) if q not in qubits]
    axes = list(qubits) + rest
    tensor = state.reshape((2,) * count).transpose(axes).reshape(2 ** len(qubits), -1)
    tensor = matrix @ tensor
    return tensor.reshape((2,) * count).transpose(np.argsort(axes)).reshape(-1)


def _scenario_angles(case, scenario):
    scale = 1.0 + scenario["interaction_scale"]
    return np.array(
        [
            scale * gate["theta"] + scenario["layer_shifts"][gate["layer"]]
            for gate in case["gates"]
        ]
    )


def _cone(case, observable):
    support = {observable["site"]}
    selected = set()
    for layer in range(LAYERS - 1, -1, -1):
        for index, gate in enumerate(case["gates"]):
            if gate["layer"] == layer and (
                gate["qubits"][0] in support or gate["qubits"][1] in support
            ):
                selected.add(index)
                support.update(gate["qubits"])
    return sorted(support), selected


def _expectation(case, scenario, observable, full=False):
    if full:
        qubits, selected = (
            list(range(case["qubit_count"])),
            set(range(len(case["gates"]))),
        )
    else:
        qubits, selected = _cone(case, observable)
    local = {qubit: index for index, qubit in enumerate(qubits)}
    state = np.zeros(2 ** len(qubits), dtype=np.complex128)
    state[0] = 1
    for qubit in qubits:
        angle = case["initial_ry"][qubit] * (1 + scenario["drive_scale"])
        angle += scenario["initial_shift"]
        state = _apply(state, _rotation(Y, angle), [local[qubit]], len(qubits))
        state = _apply(
            state,
            _rotation(Z, case["initial_rz"][qubit]),
            [local[qubit]],
            len(qubits),
        )
    angles = _scenario_angles(case, scenario)
    for index, gate in enumerate(case["gates"]):
        if index in selected:
            state = _apply(
                state,
                _interaction(gate["axis"], angles[index]),
                [local[q] for q in gate["qubits"]],
                len(qubits),
            )
    for qubit in qubits:
        angle = case["final_rx"][qubit] * (1 + scenario["drive_scale"])
        angle += scenario["final_shift"]
        state = _apply(state, _rotation(X, angle), [local[qubit]], len(qubits))
        state = _apply(
            state,
            _rotation(Z, case["final_rz"][qubit]),
            [local[qubit]],
            len(qubits),
        )
    transformed = _apply(
        state,
        PAULI[observable["pauli"]],
        [local[observable["site"]]],
        len(qubits),
    )
    return float(np.vdot(state, transformed).real)


def _all_expectations(case):
    return np.array(
        [
            [
                _expectation(case, scenario, observable)
                for observable in case["observables"]
            ]
            for scenario in case["scenarios"]
        ]
    )


def _make_matching(rng, count, layer):
    if layer == 0:
        positions = list(range(count))
    elif layer == 1:
        positions = list(range(1, count - 1))
    else:
        positions = []
        for start in range(0, count, 8):
            block = list(range(start, min(start + 8, count)))
            rng.shuffle(block)
            positions.extend(block)
    if layer < 2:
        return [
            positions[index : index + 2] for index in range(0, len(positions) - 1, 2)
        ]
    return [positions[index : index + 2] for index in range(0, len(positions) - 1, 2)]


def _make_case(rng, generation_attempt):
    layout = rng.permutation(QUBITS).tolist()
    axes = ("XX", "YY", "ZZ")
    gates = []
    for layer in range(LAYERS):
        for left, right in _make_matching(rng, QUBITS, layer):
            gates.append(
                {
                    "gate_id": f"g-{rng.integers(1 << 52):013x}",
                    "layer": layer,
                    "qubits": [layout[left], layout[right]],
                    "axis": axes[int(rng.integers(0, len(axes)))],
                    "theta": float(rng.uniform(0.14, 0.72)),
                }
            )
    sites = rng.choice(QUBITS, OBSERVABLES, replace=False)
    observables = [
        {
            "observable_id": f"o-{rng.integers(1 << 48):012x}",
            "site": int(site),
            "pauli": str(rng.choice(list("XYZ"))),
        }
        for site in sites
    ]
    scenarios = []
    for index in range(SCENARIOS):
        direction = index - 1
        scenarios.append(
            {
                "scenario_id": f"u-{rng.integers(1 << 44):011x}",
                "interaction_scale": float(
                    0.055 * direction + rng.uniform(-0.012, 0.012)
                ),
                "layer_shifts": rng.uniform(-0.035, 0.035, size=LAYERS).tolist(),
                "drive_scale": float(0.035 * direction + rng.uniform(-0.008, 0.008)),
                "initial_shift": float(rng.uniform(-0.025, 0.025)),
                "final_shift": float(rng.uniform(-0.025, 0.025)),
                "stratum_excess_variance": rng.uniform(
                    0.018, 0.11, size=STRATA
                ).tolist(),
            }
        )
    mixture = np.zeros((OBSERVABLES, STRATA))
    for row in range(OBSERVABLES):
        chosen = rng.choice(STRATA, int(rng.integers(3, 6)), replace=False)
        values = rng.uniform(0.2, 1.2, size=len(chosen))
        mixture[row, chosen] = values / values.sum()
    return {
        "case_id": f"robust-cut-{rng.integers(1 << 48):012x}",
        "generation_attempt": generation_attempt,
        "qubit_count": QUBITS,
        "layout_order": layout,
        "fragment_ids": [f"f-{rng.integers(1 << 40):010x}" for _ in range(FRAGMENTS)],
        "fragment_count": FRAGMENTS,
        "minimum_fragment_qubits": MIN_FRAGMENT,
        "maximum_fragment_qubits": MAX_FRAGMENT,
        "gates": gates,
        "initial_ry": rng.uniform(-1.1, 1.1, size=QUBITS).tolist(),
        "initial_rz": rng.uniform(-0.8, 0.8, size=QUBITS).tolist(),
        "final_rx": rng.uniform(-1.0, 1.0, size=QUBITS).tolist(),
        "final_rz": rng.uniform(-0.7, 0.7, size=QUBITS).tolist(),
        "scenarios": scenarios,
        "observables": observables,
        "stratum_ids": [f"s-{rng.integers(1 << 40):010x}" for _ in range(STRATA)],
        "stratum_mixture": mixture.tolist(),
        "total_shots": 90000,
        "minimum_shots_per_stratum": 64,
        "case_nonce": int(rng.integers(1 << 31)),
    }


def _compositions(total, count, lower, upper):
    for values in itertools.product(range(lower, upper + 1), repeat=count):
        if sum(values) == total:
            yield values


def _cut_mask(case, sizes):
    positions = np.empty(case["qubit_count"], dtype=int)
    for index, qubit in enumerate(case["layout_order"]):
        positions[qubit] = index
    boundaries = np.cumsum(sizes)
    labels = np.searchsorted(boundaries, positions, side="right")
    return np.array(
        [
            labels[gate["qubits"][0]] != labels[gate["qubits"][1]]
            for gate in case["gates"]
        ],
        dtype=bool,
    )


def _target_coefficients(case, expectations, cut_mask):
    mixture = np.asarray(case["stratum_mixture"], dtype=float)
    rows = []
    log_overheads = []
    cones = [_cone(case, observable)[1] for observable in case["observables"]]
    for scenario_index, scenario in enumerate(case["scenarios"]):
        angles = _scenario_angles(case, scenario)
        gamma = 1 + 2 * np.abs(np.sin(angles))
        excess = np.asarray(scenario["stratum_excess_variance"])
        for observable_index, cone in enumerate(cones):
            active = [index for index in cone if cut_mask[index]]
            log_overhead = float(2 * np.sum(np.log(gamma[active])))
            log_overheads.append(log_overhead)
            base = max(0.0, 1 - expectations[scenario_index, observable_index] ** 2)
            rows.append(
                np.exp(log_overhead) * mixture[observable_index] ** 2 * (base + excess)
            )
    return np.asarray(rows), np.asarray(log_overheads)


def _allocate(coefficients, total, minimum, iterations=96):
    target_count, stratum_count = coefficients.shape
    remaining = total - minimum * stratum_count
    if remaining < 0:
        raise ValueError("shot budget below mandatory minimum")
    mixture = np.full(target_count, 1 / target_count)
    raw = np.full(stratum_count, total / stratum_count)
    for iteration in range(iterations):
        aggregate = mixture @ coefficients + 1e-30
        root = np.sqrt(aggregate)
        raw = minimum + remaining * root / root.sum()
        variances = np.sum(coefficients / raw, axis=1)
        worst = int(np.argmax(variances))
        step = 2.0 / (iteration + 3.0)
        mixture *= 1 - step
        mixture[worst] += step
    whole = np.floor(raw).astype(int)
    residual = total - int(whole.sum())
    order = sorted(range(stratum_count), key=lambda j: (-float(raw[j] - whole[j]), j))
    whole[order[:residual]] += 1
    return whole


def _score(case, expectations, sizes, allocation=None):
    cut_mask = _cut_mask(case, sizes)
    coefficients, log_overheads = _target_coefficients(case, expectations, cut_mask)
    if allocation is None:
        allocation = _allocate(
            coefficients,
            case["total_shots"],
            case["minimum_shots_per_stratum"],
        )
    variances = np.sum(coefficients / allocation, axis=1)
    return float(np.max(variances)), allocation, variances, log_overheads, cut_mask


def _reference(case, expectations):
    best = None
    for sizes in _compositions(
        case["qubit_count"],
        case["fragment_count"],
        case["minimum_fragment_qubits"],
        case["maximum_fragment_qubits"],
    ):
        score = _score(case, expectations, sizes)
        key = (score[0], sizes)
        if best is None or key < best[0]:
            best = (key, sizes, score)
    if best is None:
        raise AssertionError("no feasible partition")
    return best[1], best[2]


def _best_fixed_allocation(case, expectations, allocation):
    return min(
        _score(case, expectations, sizes, allocation=allocation)[0]
        for sizes in _compositions(
            case["qubit_count"],
            case["fragment_count"],
            case["minimum_fragment_qubits"],
            case["maximum_fragment_qubits"],
        )
    )


def _balanced_sizes(case):
    sizes = [case["qubit_count"] // case["fragment_count"]] * case["fragment_count"]
    for index in range(case["qubit_count"] - sum(sizes)):
        sizes[index] += 1
    return tuple(sizes)


def build_config(seed):
    for attempt in range(12):
        rng = np.random.default_rng(seed + 104729 * attempt)
        case = _make_case(rng, attempt)
        expectations = _all_expectations(case)
        _, best = _reference(case, expectations)
        balanced = _score(case, expectations, _balanced_sizes(case))
        uniform = np.full(
            STRATA,
            case["total_shots"] // STRATA,
            dtype=int,
        )
        uniform[: case["total_shots"] - int(uniform.sum())] += 1
        best_uniform = _best_fixed_allocation(case, expectations, uniform)
        balanced_uniform = _score(
            case,
            expectations,
            _balanced_sizes(case),
            allocation=uniform,
        )
        margin = min(balanced[0], balanced_uniform[0], best_uniform) / best[0] - 1
        if margin >= 0.025:
            case["maximum_robust_variance"] = float(best[0] * 1.008 + 1e-14)
            case["minimum_naive_gap"] = 0.025
            break
    else:
        raise AssertionError("failed to generate an active planning tradeoff")
    config = {
        "case": case,
        "cut_model": "independent-rpp-qpd-v1",
        "variance_model": "robust-stratified-lightcone-proxy-v1",
    }
    config["case_digest"] = _digest(config)
    return config


def _strict_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(
        value, (bool, np.bool_)
    )


def _fragments(result, case):
    rows = result.get("fragments") if isinstance(result, dict) else None
    if not isinstance(rows, list) or len(rows) != case["fragment_count"]:
        return None
    if [row.get("fragment_id") for row in rows if isinstance(row, dict)] != case[
        "fragment_ids"
    ]:
        return None
    flattened, sizes = [], []
    for row in rows:
        if set(row) != {"fragment_id", "qubits"} or not isinstance(row["qubits"], list):
            return None
        if not all(_strict_int(value) for value in row["qubits"]):
            return None
        sizes.append(len(row["qubits"]))
        flattened.extend(row["qubits"])
    if flattened != case["layout_order"]:
        return None
    if not all(
        case["minimum_fragment_qubits"] <= size <= case["maximum_fragment_qubits"]
        for size in sizes
    ):
        return None
    return tuple(sizes)


def _finite_real(value):
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


def _list_of_finite(value, shape):
    def valid(node, dimensions):
        if not dimensions:
            return _finite_real(node)
        return (
            isinstance(node, list)
            and len(node) == dimensions[0]
            and all(valid(child, dimensions[1:]) for child in node)
        )

    if not valid(value, shape):
        return None
    return np.asarray(value, dtype=float)


def _validate(result, config, expectations, reference_score):
    case = config["case"]
    root_keys = {
        "case_id",
        "fragments",
        "cut_gate_ids",
        "lightcone_cut_gate_ids",
        "pilot_expectations",
        "shot_allocation",
        "predicted_variances",
        "robust_variance",
        "maximum_log_overhead",
    }
    exact_schema = isinstance(result, dict) and set(result) == root_keys
    sizes = _fragments(result, case) if exact_schema else None
    feasible_partition = sizes is not None
    expected_cut_ids = []
    expected_cone_ids = []
    expected_variances = np.array([])
    expected_robust = np.inf
    expected_log = np.inf
    allocation_valid = False
    cut_ids_valid = False
    cones_valid = False
    if feasible_partition:
        cut_mask = _cut_mask(case, sizes)
        expected_cut_ids = [
            gate["gate_id"]
            for index, gate in enumerate(case["gates"])
            if cut_mask[index]
        ]
        expected_cone_ids = [
            [
                gate["gate_id"]
                for index, gate in enumerate(case["gates"])
                if cut_mask[index] and index in _cone(case, observable)[1]
            ]
            for observable in case["observables"]
        ]
        cut_ids_valid = result.get("cut_gate_ids") == expected_cut_ids
        cones_valid = result.get("lightcone_cut_gate_ids") == expected_cone_ids
        allocations = result.get("shot_allocation")
        if isinstance(allocations, list) and len(allocations) == STRATA:
            ids, shots = [], []
            allocation_valid = True
            for row in allocations:
                if not isinstance(row, dict) or set(row) != {"stratum_id", "shots"}:
                    allocation_valid = False
                    break
                ids.append(row["stratum_id"])
                shots.append(row["shots"])
            allocation_valid &= ids == case["stratum_ids"]
            allocation_valid &= all(_strict_int(value) for value in shots)
            if allocation_valid:
                allocation = np.asarray(shots, dtype=int)
                allocation_valid &= bool(
                    np.all(allocation >= case["minimum_shots_per_stratum"])
                    and int(allocation.sum()) == case["total_shots"]
                )
                if allocation_valid:
                    score = _score(case, expectations, sizes, allocation=allocation)
                    expected_robust = score[0]
                    expected_variances = score[2].reshape(SCENARIOS, OBSERVABLES)
                    expected_log = float(np.max(score[3]))
    submitted_expectations = _list_of_finite(
        result.get("pilot_expectations") if isinstance(result, dict) else None,
        (SCENARIOS, OBSERVABLES),
    )
    submitted_variances = _list_of_finite(
        result.get("predicted_variances") if isinstance(result, dict) else None,
        (SCENARIOS, OBSERVABLES),
    )
    scalars = []
    for key in ("robust_variance", "maximum_log_overhead"):
        value = result.get(key) if isinstance(result, dict) else None
        scalars.append(float(value) if _finite_real(value) else np.nan)
    criteria = {
        "exact output schema": exact_schema,
        "case identity preserved": bool(
            exact_schema and result.get("case_id") == case["case_id"]
        ),
        "contiguous capacity-feasible partition": feasible_partition,
        "cut artifact exact": cut_ids_valid,
        "observable lightcone artifact exact": cones_valid,
        "TensorCircuit pilot expectations match independent oracle": bool(
            submitted_expectations is not None
            and np.allclose(submitted_expectations, expectations, atol=3e-7, rtol=3e-7)
        ),
        "integer shot budget exact": allocation_valid,
        "predicted target variances exact": bool(
            allocation_valid
            and submitted_variances is not None
            and np.allclose(
                submitted_variances, expected_variances, atol=2e-11, rtol=2e-7
            )
        ),
        "reported robust variance exact": bool(
            np.isfinite(scalars[0])
            and abs(scalars[0] - expected_robust) <= 2e-11 + 2e-7 * abs(expected_robust)
        ),
        "reported log overhead exact": bool(
            np.isfinite(scalars[1])
            and abs(scalars[1] - expected_log) <= 2e-10 + 2e-8 * abs(expected_log)
        ),
        "joint design meets hidden reference threshold": bool(
            expected_robust <= case["maximum_robust_variance"]
            and expected_robust <= reference_score * 1.008 + 1.1e-14
        ),
    }
    return criteria


def _format_output(config, expectations, sizes, score):
    case = config["case"]
    _, allocation, variances, logs, cut_mask = score
    fragments, start = [], 0
    for fragment_id, size in zip(case["fragment_ids"], sizes):
        fragments.append(
            {
                "fragment_id": fragment_id,
                "qubits": case["layout_order"][start : start + size],
            }
        )
        start += size
    cut_ids = [
        gate["gate_id"] for index, gate in enumerate(case["gates"]) if cut_mask[index]
    ]
    cone_ids = [
        [
            gate["gate_id"]
            for index, gate in enumerate(case["gates"])
            if cut_mask[index] and index in _cone(case, observable)[1]
        ]
        for observable in case["observables"]
    ]
    return {
        "case_id": case["case_id"],
        "fragments": fragments,
        "cut_gate_ids": cut_ids,
        "lightcone_cut_gate_ids": cone_ids,
        "pilot_expectations": expectations.tolist(),
        "shot_allocation": [
            {"stratum_id": identity, "shots": int(shots)}
            for identity, shots in zip(case["stratum_ids"], allocation)
        ],
        "predicted_variances": variances.reshape(SCENARIOS, OBSERVABLES).tolist(),
        "robust_variance": float(np.max(variances)),
        "maximum_log_overhead": float(np.max(logs)),
    }


def _reference_output(config, expectations):
    sizes, score = _reference(config["case"], expectations)
    return _format_output(config, expectations, sizes, score)


def _tiny_canary(seed):
    rng = np.random.default_rng(seed)
    count = 8
    case = _make_case(rng, 0)
    case["qubit_count"] = count
    case["layout_order"] = list(range(count))
    case["initial_ry"] = rng.uniform(-1, 1, count).tolist()
    case["initial_rz"] = rng.uniform(-1, 1, count).tolist()
    case["final_rx"] = rng.uniform(-1, 1, count).tolist()
    case["final_rz"] = rng.uniform(-1, 1, count).tolist()
    case["gates"] = []
    for layer in range(LAYERS):
        order = rng.permutation(count)
        for left, right in order.reshape(-1, 2):
            case["gates"].append(
                {
                    "gate_id": f"tiny-{layer}-{left}-{right}",
                    "layer": layer,
                    "qubits": [int(left), int(right)],
                    "axis": str(rng.choice(["XX", "YY", "ZZ"])),
                    "theta": float(rng.uniform(-0.7, 0.7)),
                }
            )
    case["observables"] = [
        {
            "observable_id": f"tiny-o-{site}",
            "site": site,
            "pauli": str(rng.choice(list("XYZ"))),
        }
        for site in range(count)
    ]
    maximum = 0.0
    for scenario in case["scenarios"]:
        for observable in case["observables"]:
            maximum = max(
                maximum,
                abs(
                    _expectation(case, scenario, observable)
                    - _expectation(case, scenario, observable, full=True)
                ),
            )
    return maximum


def self_test(seed):
    started = time.perf_counter()
    canary_error = _tiny_canary(seed + 17)
    config = build_config(seed)
    case = config["case"]
    expectations = _all_expectations(case)
    _, reference = _reference(case, expectations)
    result = _reference_output(config, expectations)
    criteria = _validate(result, config, expectations, reference[0])
    mutations = []
    mutation = copy.deepcopy(result)
    mutation["shot_allocation"][0]["shots"] += 0.5
    mutations.append(mutation)
    mutation = copy.deepcopy(result)
    mutation["fragments"][0]["qubits"][0], mutation["fragments"][1]["qubits"][0] = (
        mutation["fragments"][1]["qubits"][0],
        mutation["fragments"][0]["qubits"][0],
    )
    mutations.append(mutation)
    mutation = copy.deepcopy(result)
    mutation["pilot_expectations"][0][0] = float("nan")
    mutations.append(mutation)
    mutation = copy.deepcopy(result)
    mutation["cut_gate_ids"] = mutation["cut_gate_ids"][1:]
    mutations.append(mutation)
    mutation = copy.deepcopy(result)
    cone_index = next(
        index
        for index, values in enumerate(mutation["lightcone_cut_gate_ids"])
        if values
    )
    mutation["lightcone_cut_gate_ids"][cone_index] = mutation["lightcone_cut_gate_ids"][
        cone_index
    ][1:]
    mutations.append(mutation)
    mutation = copy.deepcopy(result)
    mutation["predicted_variances"][0][0] *= 1.2
    mutations.append(mutation)
    mutation = copy.deepcopy(result)
    mutation["case_id"] += "-wrong"
    mutations.append(mutation)
    mutation = copy.deepcopy(result)
    mutation["robust_variance"] = str(mutation["robust_variance"])
    mutations.append(mutation)
    mutation = copy.deepcopy(result)
    mutation["pilot_expectations"] = tuple(
        tuple(str(value) for value in row) for row in mutation["pilot_expectations"]
    )
    mutations.append(mutation)
    balanced_sizes = _balanced_sizes(case)
    mutations.append(
        _format_output(
            config,
            expectations,
            balanced_sizes,
            _score(case, expectations, balanced_sizes),
        )
    )
    rejected = sum(
        not all(_validate(value, config, expectations, reference[0]).values())
        for value in mutations
    )
    print("Problem 119 independent self-test")
    print(f"Seed: {seed}; case digest: {config['case_digest']}")
    print(f"Dense/lightcone canary maximum error: {canary_error:.3e}")
    print(f"Reference robust variance: {reference[0]:.12g}")
    print(f"Naive-gap requirement: {case['minimum_naive_gap']:.3f}")
    print(f"Mutations rejected: {rejected}/{len(mutations)}")
    print(f"Self-test runtime: {time.perf_counter() - started:.2f}s")
    passed = (
        canary_error < 2e-12 and all(criteria.values()) and rejected == len(mutations)
    )
    print(f"Overall self-test: {'PASS' if passed else 'FAIL'}")
    return passed


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
    sys.argv = [sys.argv[0]]
    case = config["case"]
    expectations = _all_expectations(case)
    _, reference = _reference(case, expectations)
    module = importlib.import_module(module_name)
    started = time.perf_counter()
    result = module.run_solution(copy.deepcopy(config))
    elapsed = time.perf_counter() - started
    criteria = _validate(result, config, expectations, reference[0])
    print("Problem 119 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; case digest: {config['case_digest'][:16]}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print("Runtime is reported separately and does not change functional correctness")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    passed = all(criteria.values())
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_119")
    parser.add_argument("--seed", type=int, default=default_seed())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    verifier_seed = os.environ.get("ORBIT_Q_CANDIDATE_SEED")
    if verifier_seed is not None and int(verifier_seed) != args.seed:
        raise ValueError("--seed cannot override the verifier-authorized seed")
    passed = (
        self_test(args.seed) if args.self_test else evaluate(args.solution, args.seed)
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
