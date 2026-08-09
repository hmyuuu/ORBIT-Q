"""TensorCircuit expert for CSS encoder-fault verification synthesis."""

import itertools

import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")
PAULIS = ((0, 0), (1, 0), (1, 1), (0, 1))
LABEL = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}


def rref(rows):
    a = np.asarray(rows, dtype=np.uint8).copy() & 1
    row, pivots = 0, []
    for column in range(a.shape[1]):
        choices = np.flatnonzero(a[row:, column])
        if not len(choices):
            continue
        pivot = row + int(choices[0])
        a[[row, pivot]] = a[[pivot, row]]
        for other in np.flatnonzero(a[:, column]):
            if other != row:
                a[other] ^= a[row]
        pivots.append(column)
        row += 1
        if row == len(a):
            break
    return a[:row], pivots


def synthesize(instance):
    rows, pivots = rref(instance["x_generators"])
    gates = []
    for row, pivot in zip(rows, pivots):
        gates.append({"name": "h", "qubits": [pivot]})
        gates += [
            {"name": "cx", "qubits": [pivot, int(q)]}
            for q in np.flatnonzero(row)
            if q != pivot
        ]
    return gates


def conjugate(x, z, gate):
    if gate["name"] == "h":
        q = gate["qubits"][0]
        x[q], z[q] = z[q], x[q]
    else:
        control, target = gate["qubits"]
        x[target] ^= x[control]
        z[control] ^= z[target]


def propagate(x, z, gates, start):
    x, z = x.copy(), z.copy()
    for gate in gates[start:]:
        conjugate(x, z, gate)
    return x, z


def final_faults(n, gates):
    faults = {}
    for q in range(n):
        x, z = np.zeros(n, np.uint8), np.zeros(n, np.uint8)
        x[q] = 1
        error = propagate(x, z, gates, 0)
        faults[tuple(np.concatenate(error))] = error
    for index, gate in enumerate(gates):
        qubits = gate["qubits"]
        local = (
            PAULIS[1:] if gate["name"] == "h" else itertools.product(PAULIS, repeat=2)
        )
        for values in local:
            values = (values,) if gate["name"] == "h" else values
            if all(value == (0, 0) for value in values):
                continue
            x, z = np.zeros(n, np.uint8), np.zeros(n, np.uint8)
            for q, value in zip(qubits, values):
                x[q], z[q] = value
            error = propagate(x, z, gates, index + 1)
            faults[tuple(np.concatenate(error))] = error
    return list(faults.values())


def stabilizer_group(instance):
    xrows = np.asarray(instance["x_generators"], np.uint8)
    zrows = np.asarray(instance["z_generators"], np.uint8)
    zero_x, zero_z = np.zeros_like(xrows), np.zeros_like(zrows)
    rows = np.concatenate(
        [np.concatenate([xrows, zero_x], 1), np.concatenate([zero_z, zrows], 1)]
    )
    group = np.zeros((1, rows.shape[1]), np.uint8)
    for row in rows:
        group = np.concatenate([group, group ^ row])
    return group


def label(error):
    return "".join(LABEL[tuple(map(int, pair))] for pair in zip(*error))


def dangerous_faults(instance, gates):
    n = instance["n_qubits"]
    group = stabilizer_group(instance)
    gx, gz = group[:, :n], group[:, n:]
    dangerous = []
    for x, z in final_faults(n, gates):
        weights = np.count_nonzero((gx ^ x) | (gz ^ z), axis=1)
        if np.min(weights) > 1:
            dangerous.append((x, z))
    return sorted(dangerous, key=label)


def circuit(n, gates, error=None):
    c = tc.Circuit(n)
    for gate in gates:
        getattr(c, "cnot" if gate["name"] == "cx" else "h")(*gate["qubits"])
    if error is not None:
        for q, (x, z) in enumerate(zip(*error)):
            if x or z:
                getattr(c, "y" if x and z else "x" if x else "z")(q)
    return c


def pauli_codes(candidate):
    code = 1 if candidate["kind"] == "x" else 3
    return [code if bit else 0 for bit in candidate["mask"]]


def minimum_cover(candidates, syndrome):
    for count in range(len(candidates) + 1):
        best = None
        for subset in itertools.combinations(range(len(candidates)), count):
            if len(syndrome) and not np.all(np.any(syndrome[:, subset], axis=1)):
                continue
            ids = tuple(sorted(candidates[i]["id"] for i in subset))
            key = (sum(candidates[i]["coupling_cost"] for i in subset), ids)
            best = key if best is None or key < best else best
        if best is not None:
            return list(best[1])
    raise RuntimeError("candidate checks do not cover the faults")


def solve_instance(instance):
    gates = synthesize(instance)
    candidates = instance["candidate_checks"]
    codes = [pauli_codes(candidate) for candidate in candidates]
    target = circuit(instance["n_qubits"], gates)
    expectations = np.asarray([np.real(target.expectation_ps(ps=ps)) for ps in codes])
    errors = dangerous_faults(instance, gates)
    rows = []
    for error in errors:
        faulty = circuit(instance["n_qubits"], gates, error)
        rows.append([np.real(faulty.expectation_ps(ps=ps)) < -0.5 for ps in codes])
    syndrome = np.asarray(rows, dtype=bool).reshape(len(errors), len(candidates))
    return {
        "instance_id": instance["instance_id"],
        "encoder": gates,
        "dangerous_errors": [label(error) for error in errors],
        "fault_syndromes": syndrome,
        "target_expectations": expectations,
        "selected_check_ids": minimum_cover(candidates, syndrome),
    }


def run_solution(config):
    return {"instances": [solve_instance(instance) for instance in config["instances"]]}
