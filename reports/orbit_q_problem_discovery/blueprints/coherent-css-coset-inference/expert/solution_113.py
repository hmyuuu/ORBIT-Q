"""TensorCircuit expert for coherent degenerate CSS logical-coset inference."""

import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")
I2 = np.eye(2, dtype=np.complex128)
PAULI = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.diag([1, -1]).astype(np.complex128),
}


def _rowspace(rows):
    words = np.zeros((1 << len(rows), rows.shape[1]), dtype=np.uint8)
    for value in range(1, len(words)):
        low = value & -value
        words[value] = words[value ^ low] ^ rows[low.bit_length() - 1]
    return words


def _codewords(case):
    hx = np.asarray(case["x_stabilizers"], dtype=np.uint8)
    logical_x = np.asarray(case["logical_x"], dtype=np.uint8)
    support = _rowspace(hx)
    n = case["n_qubits"]
    powers = 1 << np.arange(n - 1, -1, -1)
    words = np.zeros((4, 1 << n), dtype=np.complex128)
    for logical in range(4):
        shift = ((logical >> 1) & 1) * logical_x[0] ^ (logical & 1) * logical_x[1]
        words[logical, (support ^ shift) @ powers] = 1 / np.sqrt(len(support))
    return words


def _pauli(labels):
    matrix = np.array([[1]], dtype=np.complex128)
    for label in labels:
        matrix = np.kron(matrix, PAULI[label])
    return matrix


def _rotate(circuit, record):
    pauli = _pauli(record["paulis"])
    angle = record["angle"]
    unitary = np.cos(angle / 2) * np.eye(len(pauli)) - 1j * np.sin(angle / 2) * pauli
    circuit.unitary(
        *record["sites"], unitary=tc.gates.Gate(unitary), name="coherent_pauli_rotation"
    )


def _recover(states, recovery):
    recovered = []
    for state in states:
        circuit = tc.Circuit(
            len(recovery["x"]), inputs=tc.backend.convert_to_tensor(state)
        )
        for site, (x, z) in enumerate(zip(recovery["x"], recovery["z"])):
            if x and z:
                circuit.y(site)
            elif x:
                circuit.x(site)
            elif z:
                circuit.z(site)
        recovered.append(circuit.state())
    return tc.backend.stack(recovered)


def _logical_pauli(frame):
    labels = []
    for x, z in zip(frame["logical_x"], frame["logical_z"]):
        labels.append(
            "I" if not (x or z) else "X" if x and not z else "Z" if z and not x else "Y"
        )
    return _pauli(labels)


def _solve(case):
    words = _codewords(case)
    frames = [_logical_pauli(frame) for frame in case["frame_candidates"]]
    branch_states = []
    for branch in case["noise_branches"]:
        evolved = []
        for word in words:
            circuit = tc.Circuit(
                case["n_qubits"], inputs=tc.backend.convert_to_tensor(word)
            )
            for record in branch["rotations"]:
                _rotate(circuit, record)
            evolved.append(np.asarray(circuit.state()))
        branch_states.append(np.asarray(evolved))
    rows = []
    for syndrome_case in case["syndrome_cases"]:
        chi = np.zeros((16, 16), dtype=np.complex128)
        for branch, states in zip(case["noise_branches"], branch_states):
            recovered = _recover(states, syndrome_case["recovery"])
            overlap = tc.backend.einsum(
                "bi,ai->ba",
                tc.backend.conj(tc.backend.convert_to_tensor(words)),
                recovered,
            )
            kraus = np.asarray(overlap)
            coefficients = np.asarray(
                [np.trace(frame.conj().T @ kraus) / 4 for frame in frames]
            )
            chi += branch["probability"] * np.outer(coefficients, coefficients.conj())
        probability = float(np.trace(chi).real)
        chi /= probability
        weights = np.diag(chi).real
        rows.append(
            {
                "syndrome_id": syndrome_case["syndrome_id"],
                "syndrome_probability": probability,
                "logical_chi": chi,
                "coset_weights": weights,
                "best_frame_id": case["frame_candidates"][int(np.argmax(weights))][
                    "id"
                ],
            }
        )
    return {"case_id": case["case_id"], "syndromes": rows}


def run_solution(config):
    return {"cases": [_solve(case) for case in config["cases"]]}
