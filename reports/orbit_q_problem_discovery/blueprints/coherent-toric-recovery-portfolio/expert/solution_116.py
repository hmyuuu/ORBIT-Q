"""TensorCircuit expert for coherent toric recovery-portfolio selection."""

import itertools

import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")


def _arrays(recovery):
    return (
        np.asarray(recovery["horizontal"], dtype=np.uint8),
        np.asarray(recovery["vertical"], dtype=np.uint8),
    )


def _syndrome(recovery):
    horizontal, vertical = _arrays(recovery)
    return (
        horizontal
        ^ np.roll(horizontal, 1, axis=1)
        ^ vertical
        ^ np.roll(vertical, 1, axis=0)
    )


def _homology(recovery, anchor):
    horizontal, vertical = _arrays(recovery)
    anchor_h, anchor_v = _arrays(anchor)
    cycle_h, cycle_v = horizontal ^ anchor_h, vertical ^ anchor_v
    assert not np.any(_syndrome({"horizontal": cycle_h, "vertical": cycle_v}))
    return int(np.sum(cycle_h[:, -1]) & 1) + 2 * int(np.sum(cycle_v[-1, :]) & 1)


def _cost(case, recovery):
    horizontal, vertical = _arrays(recovery)
    return int(
        np.sum(horizontal * np.asarray(case["horizontal_recovery_cost"]))
        + np.sum(vertical * np.asarray(case["vertical_recovery_cost"]))
    )


def _weights(angles):
    angles = np.asarray(angles)
    return np.stack((np.cos(angles / 2), -1j * np.sin(angles / 2)), axis=-1)


def _amplitudes(case, recovery, branch):
    size = case["lattice_size"]
    horizontal, vertical = _arrays(recovery)
    weights_h = _weights(branch["horizontal_angles"])
    weights_v = _weights(branch["vertical_angles"])
    output = []
    for sector in range(4):
        circuit = tc.Circuit(size)
        for row in range(size):
            for column in range(size):
                base = int(horizontal[row, column]) ^ (
                    int(sector & 1) if row == 0 else 0
                )
                w0, w1 = weights_h[row, column][base], weights_h[row, column][base ^ 1]
                gate = np.asarray([[w0, w1], [w1, w0]])
                circuit.unitary(
                    column,
                    unitary=tc.gates.Gate(gate),
                    name="horizontal_transfer",
                )
            for column in range(size):
                base = int(vertical[row, column]) ^ (
                    int((sector >> 1) & 1) if column == 0 else 0
                )
                w0, w1 = weights_v[row, column][base], weights_v[row, column][base ^ 1]
                gate = np.diag([w0, w1, w1, w0]).reshape(2, 2, 2, 2)
                circuit.unitary(
                    (column - 1) % size,
                    column,
                    unitary=tc.gates.Gate(gate),
                    name="vertical_transfer",
                )
        output.append(np.trace(np.asarray(circuit.matrix())) / 2)
    return np.asarray(output)


def _quality_table(case):
    rows = case["syndrome_records"]
    qualities = np.empty((len(rows), 8, len(case["noise_scenarios"])))
    for row_index, row in enumerate(rows):
        anchor = next(
            candidate["recovery"]
            for candidate in row["recovery_candidates"]
            if candidate["candidate_id"] == row["anchor_candidate_id"]
        )
        homologies = [
            _homology(candidate["recovery"], anchor)
            for candidate in row["recovery_candidates"]
        ]
        for scenario_index, scenario in enumerate(case["noise_scenarios"]):
            process = np.zeros((4, 4), dtype=np.complex128)
            for branch in scenario["branches"]:
                amplitudes = _amplitudes(case, anchor, branch)
                process += branch["probability"] * np.outer(
                    amplitudes, amplitudes.conj()
                )
            process /= np.trace(process).real
            for candidate_index, homology in enumerate(homologies):
                qualities[row_index, candidate_index, scenario_index] = process[
                    homology, homology
                ].real
    return qualities


def _solve(case):
    rows = case["syndrome_records"]
    qualities = _quality_table(case)
    weights = np.asarray([row["syndrome_weight"] for row in rows])
    costs = [
        [_cost(case, candidate["recovery"]) for candidate in row["recovery_candidates"]]
        for row in rows
    ]
    best_score, best_cost, best_choice = -1.0, 0, None
    for choice in itertools.product(range(8), repeat=len(rows)):
        cost = sum(costs[row][selected] for row, selected in enumerate(choice))
        if cost > case["recovery_budget"]:
            continue
        values = np.sum(
            [
                weights[row] * qualities[row, selected]
                for row, selected in enumerate(choice)
            ],
            axis=0,
        )
        score = float(np.min(values))
        if score > best_score + 1e-14 or (
            abs(score - best_score) <= 1e-14
            and (best_choice is None or (cost, choice) < (best_cost, best_choice))
        ):
            best_score, best_cost, best_choice = score, cost, choice
    frame_map = {
        bits[0] + 2 * bits[1]: frame["frame_id"]
        for frame in case["frame_candidates"]
        for bits in [frame["relative_homology"]]
    }
    selections = []
    for row, selected in zip(rows, best_choice):
        candidate = row["recovery_candidates"][selected]
        anchor = next(
            value["recovery"]
            for value in row["recovery_candidates"]
            if value["candidate_id"] == row["anchor_candidate_id"]
        )
        selections.append(
            {
                "syndrome_id": row["syndrome_id"],
                "candidate_id": candidate["candidate_id"],
                "recovery": candidate["recovery"],
                "frame_id": frame_map[_homology(candidate["recovery"], anchor)],
            }
        )
    return {
        "case_id": case["case_id"],
        "selections": selections,
        "robust_fidelity": best_score,
        "total_recovery_cost": best_cost,
    }


def run_solution(config):
    return {"cases": [_solve(case) for case in config["cases"]]}
