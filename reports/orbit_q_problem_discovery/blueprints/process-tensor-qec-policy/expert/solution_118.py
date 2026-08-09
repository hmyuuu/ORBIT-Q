"""TensorCircuit expert for process-tensor QEC policy synthesis."""

import itertools

import numpy as np
import tensorcircuit as tc


# Keep the reviewed expert within the candidate's physical-line budget. The
# compact layout is formatting-only; its AST is regression-checked separately.
# fmt: off
tc.set_backend("numpy")
tc.set_dtype("complex128")
P4 = tuple(
    np.diag([(-1) ** (((q & 1) * (b & 1)) ^ (((q >> 1) & 1) * ((b >> 1) & 1))) for b in range(4)]).astype(complex)
    for q in range(4)
)


def _rot(vector):
    identity = np.eye(2, dtype=complex)
    paulis = (
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.diag([1, -1]).astype(complex),
    )
    output = identity
    for angle, pauli in zip(vector, paulis):
        output = (np.cos(angle / 2) * identity - 1j * np.sin(angle / 2) * pauli) @ output
    return output


def _weights(angles):
    angles = np.asarray(angles)
    return np.stack((np.cos(angles / 2), -1j * np.sin(angles / 2)), axis=-1)


def _amplitude(case, recovery, angle_record, sector):
    size = case["lattice_size"]
    horizontal = np.asarray(recovery["horizontal"], dtype=np.uint8)
    vertical = np.asarray(recovery["vertical"], dtype=np.uint8)
    weights_h = _weights(angle_record["horizontal_angles"])
    weights_v = _weights(angle_record["vertical_angles"])
    circuit = tc.Circuit(size)
    for row in range(size):
        for column in range(size):
            base = int(horizontal[row, column]) ^ (int(sector & 1) if row == 0 else 0)
            w0, w1 = weights_h[row, column][base], weights_h[row, column][base ^ 1]
            circuit.unitary(column, unitary=tc.gates.Gate(np.asarray([[w0, w1], [w1, w0]])), name="spatial_h")
        for column in range(size):
            base = int(vertical[row, column]) ^ (int((sector >> 1) & 1) if column == 0 else 0)
            w0, w1 = weights_v[row, column][base], weights_v[row, column][base ^ 1]
            gate = np.diag([w0, w1, w1, w0]).reshape(2, 2, 2, 2)
            circuit.unitary((column - 1) % size, column, unitary=tc.gates.Gate(gate), name="spatial_v")
    return np.trace(np.asarray(circuit.matrix())) / 2


def _tables(case, history):
    output = []
    for scenario in case["process_scenarios"]:
        rounds = []
        for observed, process in zip(history["rounds"], scenario["rounds"]):
            amplitudes = np.asarray([
                [_amplitude(case, observed["anchor_recovery"], process["memory_conditioned_angles"][m], q) for q in range(4)]
                for m in range(2)
            ])
            transition = _rot(process["memory_rotation"])
            rounds.append(sum(
                np.kron(P4[q], transition @ np.diag(amplitudes[:, q]))
                for q in range(4)
            ))
        output.append(rounds)
    return output


def _fidelity(operators, sequence, gates, memory):
    evolution = np.eye(8, dtype=complex)
    for operator, selected in zip(operators, sequence):
        evolution = gates[selected] @ operator @ evolution
    blocks = np.einsum("ambn,n->mab", evolution.reshape(4, 2, 4, 2), memory)
    probability = np.sum(np.abs(blocks) ** 2).real / 4
    return float(np.sum(np.abs(np.trace(blocks, axis1=1, axis2=2)) ** 2).real / (16 * probability))


def _frontier(case, tables):
    actions = case["action_primitives"]
    gates = [
        np.kron(P4[action["logical_frame"]], _rot(action["memory_kick"]))
        for action in actions
    ]
    memories = [
        np.asarray([
            np.cos(scenario["memory_preparation"][0] / 2),
            np.exp(1j * scenario["memory_preparation"][1]) * np.sin(scenario["memory_preparation"][0] / 2),
        ])
        for scenario in case["process_scenarios"]
    ]
    frontier = {}

    def assess(sequence):
        return min(_fidelity(operators, sequence, gates, memories[u]) for u, operators in enumerate(tables))

    seed = [
        min(range(len(actions)), key=lambda i: (actions[i]["intervention_cost"], actions[i]["action_id"]))
    ] * case["round_count"]
    for _ in range(2):
        for position in range(case["round_count"]):
            trials = []
            for selected in range(len(actions)):
                candidate = tuple(seed[:position] + [selected] + seed[position + 1 :])
                trials.append((assess(candidate), actions[selected]["action_id"], selected))
            seed[position] = min(trials, key=lambda value: (-value[0], value[1]))[2]
    for sequence in itertools.product(range(len(actions)), repeat=case["round_count"]):
        value = assess(sequence)
        cost = sum(actions[selected]["intervention_cost"] for selected in sequence)
        key = tuple(actions[selected]["action_id"] for selected in sequence)
        old = frontier.get(cost)
        if old is None or value > old[0] + 1e-14 or (abs(value - old[0]) <= 1e-14 and key < old[2]):
            frontier[cost] = (value, sequence, key)
    return frontier


def _solve(case):
    tables = [_tables(case, history) for history in case["syndrome_histories"]]
    frontiers = [_frontier(case, value) for value in tables]
    gates = [
        np.kron(P4[action["logical_frame"]], _rot(action["memory_kick"]))
        for action in case["action_primitives"]
    ]
    memories = [
        np.asarray([
            np.cos(scenario["memory_preparation"][0] / 2),
            np.exp(1j * scenario["memory_preparation"][1]) * np.sin(scenario["memory_preparation"][0] / 2),
        ])
        for scenario in case["process_scenarios"]
    ]
    states = {0: (0.0, (), ())}
    for history, frontier in zip(case["syndrome_histories"], frontiers):
        updated = {}
        for cost, (score, sequences, keys) in states.items():
            for extra, (value, sequence, key) in frontier.items():
                total = cost + extra
                candidate = (score + history["history_weight"] * value, sequences + (sequence,), keys + (key,))
                old = updated.get(total)
                if old is None or candidate[0] > old[0] + 1e-14 or (abs(candidate[0] - old[0]) <= 1e-14 and candidate[2] < old[2]):
                    updated[total] = candidate
        states = updated
    choices = [
        (score, cost, sequences, keys)
        for cost, (score, sequences, keys) in states.items()
        if cost <= case["shared_intervention_budget"]
    ]
    score, cost, sequences, _ = min(choices, key=lambda value: (-value[0], value[1], value[3]))
    policies = []
    for history, operators, sequence in zip(case["syndrome_histories"], tables, sequences):
        robust = min(_fidelity(values, sequence, gates, memories[u]) for u, values in enumerate(operators))
        policies.append({
            "history_id": history["history_id"],
            "action_ids": [case["action_primitives"][selected]["action_id"] for selected in sequence],
            "history_robust_fidelity": robust,
        })
    return {
        "case_id": case["case_id"],
        "policies": policies,
        "robust_objective": score,
        "total_intervention_cost": cost,
    }


def run_solution(config):
    return {"cases": [_solve(case) for case in config["cases"]]}
# fmt: on
