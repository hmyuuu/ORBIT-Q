import itertools

import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")


def _cone(case, observable):
    support, chosen = {observable["site"]}, set()
    for layer in range(2, -1, -1):
        for index, gate in enumerate(case["gates"]):
            if gate["layer"] == layer and any(q in support for q in gate["qubits"]):
                chosen.add(index)
                support.update(gate["qubits"])
    return sorted(support), chosen


def _rpp(circuit, left, right, axis, theta):
    for sign in (1, -1):
        for qubit in (left, right):
            if axis == "XX":
                circuit.h(qubit)
            elif axis == "YY":
                circuit.rx(qubit, theta=sign * np.pi / 2)
        if sign == 1:
            circuit.cnot(left, right)
            circuit.rz(right, theta=theta)
            circuit.cnot(left, right)


def _expectation(case, scenario, observable):
    qubits, chosen = _cone(case, observable)
    local = {qubit: index for index, qubit in enumerate(qubits)}
    circuit = tc.Circuit(len(qubits))
    for qubit in qubits:
        circuit.ry(
            local[qubit],
            theta=case["initial_ry"][qubit] * (1 + scenario["drive_scale"])
            + scenario["initial_shift"],
        )
        circuit.rz(local[qubit], theta=case["initial_rz"][qubit])
    for index, gate in enumerate(case["gates"]):
        if index in chosen:
            theta = (1 + scenario["interaction_scale"]) * gate["theta"]
            theta += scenario["layer_shifts"][gate["layer"]]
            _rpp(
                circuit,
                local[gate["qubits"][0]],
                local[gate["qubits"][1]],
                gate["axis"],
                theta,
            )
    for qubit in qubits:
        circuit.rx(
            local[qubit],
            theta=case["final_rx"][qubit] * (1 + scenario["drive_scale"])
            + scenario["final_shift"],
        )
        circuit.rz(local[qubit], theta=case["final_rz"][qubit])
    ps = [0] * len(qubits)
    ps[local[observable["site"]]] = {"X": 1, "Y": 2, "Z": 3}[observable["pauli"]]
    return float(np.real(np.asarray(circuit.expectation_ps(ps=ps))))


def _cut(case, sizes):
    position = np.empty(case["qubit_count"], dtype=int)
    for index, qubit in enumerate(case["layout_order"]):
        position[qubit] = index
    labels = np.searchsorted(np.cumsum(sizes), position, side="right")
    return np.array(
        [labels[g["qubits"][0]] != labels[g["qubits"][1]] for g in case["gates"]]
    )


def _coefficients(case, expectations, cut, cones):
    mixture, rows, logs = np.asarray(case["stratum_mixture"]), [], []
    for u, scenario in enumerate(case["scenarios"]):
        angles = np.array(
            [
                (1 + scenario["interaction_scale"]) * gate["theta"]
                + scenario["layer_shifts"][gate["layer"]]
                for gate in case["gates"]
            ]
        )
        gamma = 1 + 2 * np.abs(np.sin(angles))
        excess = np.asarray(scenario["stratum_excess_variance"])
        for o, cone in enumerate(cones):
            active = [index for index in cone if cut[index]]
            log = float(2 * np.sum(np.log(gamma[active])))
            base = max(0.0, 1 - expectations[u, o] ** 2)
            logs.append(log)
            rows.append(np.exp(log) * mixture[o] ** 2 * (base + excess))
    return np.asarray(rows), np.asarray(logs)


def _allocate(coefficients, case, iterations=96):
    targets, strata = coefficients.shape
    mixture = np.full(targets, 1 / targets)
    minimum = case["minimum_shots_per_stratum"]
    remaining = case["total_shots"] - minimum * strata
    for iteration in range(iterations):
        root = np.sqrt(mixture @ coefficients + 1e-30)
        raw = minimum + remaining * root / root.sum()
        worst = int(np.argmax(np.sum(coefficients / raw, axis=1)))
        step = 2 / (iteration + 3)
        mixture *= 1 - step
        mixture[worst] += step
    whole = np.floor(raw).astype(int)
    order = sorted(range(strata), key=lambda j: (-float(raw[j] - whole[j]), j))
    whole[order[: case["total_shots"] - int(whole.sum())]] += 1
    return whole


def _solve_plan(case, expectations, cones):
    best = None
    values = range(case["minimum_fragment_qubits"], case["maximum_fragment_qubits"] + 1)
    for sizes in itertools.product(values, repeat=case["fragment_count"]):
        if sum(sizes) != case["qubit_count"]:
            continue
        cut = _cut(case, sizes)
        coefficients, logs = _coefficients(case, expectations, cut, cones)
        allocation = _allocate(coefficients, case)
        variances = np.sum(coefficients / allocation, axis=1)
        key = (float(np.max(variances)), sizes)
        if best is None or key < best[0]:
            best = (key, sizes, cut, allocation, variances, logs)
    return best


def run_solution(config):
    case = config["case"]
    cones = [_cone(case, observable)[1] for observable in case["observables"]]
    expectations = np.array(
        [
            [
                _expectation(case, scenario, observable)
                for observable in case["observables"]
            ]
            for scenario in case["scenarios"]
        ]
    )
    _, sizes, cut, allocation, variances, logs = _solve_plan(case, expectations, cones)
    fragments, start = [], 0
    for identity, size in zip(case["fragment_ids"], sizes):
        fragments.append(
            {
                "fragment_id": identity,
                "qubits": case["layout_order"][start : start + size],
            }
        )
        start += size
    return {
        "case_id": case["case_id"],
        "fragments": fragments,
        "cut_gate_ids": [
            gate["gate_id"] for index, gate in enumerate(case["gates"]) if cut[index]
        ],
        "lightcone_cut_gate_ids": [
            [
                gate["gate_id"]
                for index, gate in enumerate(case["gates"])
                if cut[index] and index in cone
            ]
            for cone in cones
        ],
        "pilot_expectations": expectations.tolist(),
        "shot_allocation": [
            {"stratum_id": identity, "shots": int(shots)}
            for identity, shots in zip(case["stratum_ids"], allocation)
        ],
        "predicted_variances": variances.reshape(len(case["scenarios"]), -1).tolist(),
        "robust_variance": float(np.max(variances)),
        "maximum_log_overhead": float(np.max(logs)),
    }
