"""TensorCircuit expert for the differentiable truncated-MPS sweep."""

import numpy as np
import tensorcircuit as tc


tc.set_backend("jax")
tc.set_dtype("complex128")


def _build(parameters, case):
    n = case["n_qubits"]
    split = {"max_singular_values": case["max_bond_dimension"]}
    circuit = tc.MPSCircuit(n, center_position=0, split=split)
    layers = len(case["y_offsets"])
    for layer in range(layers):
        values = parameters[3 * layer : 3 * layer + 3]
        for qubit in range(n):
            circuit.ry(
                qubit,
                theta=case["y_offsets"][layer][qubit]
                + case["y_scales"][layer][qubit] * values[0],
            )
            circuit.rz(
                qubit,
                theta=case["z_offsets"][layer][qubit]
                + case["z_scales"][layer][qubit] * values[1],
            )
        if layer % 2 == 0:
            circuit.position(0)
            bonds = range(n - 1)
        else:
            circuit.position(n - 1)
            bonds = range(n - 2, -1, -1)
        for left in bonds:
            theta = (
                case["entangler_offsets"][layer][left]
                + case["entangler_scales"][layer][left] * values[2]
            )
            gate = (
                tc.gates.rxx(theta=theta)
                if layer % 2 == 0
                else tc.gates.rzz(theta=theta)
            )
            center = left + 1 if layer % 2 == 0 else left
            circuit.apply_adjacent_double_gate(
                gate, left, left + 1, center_position=center, split=split
            )
    circuit.normalize()
    return circuit


def _energy(parameters, case):
    circuit = _build(parameters, case)
    value = tc.backend.zeros([], dtype="float64")
    paulis = {"X": tc.gates.x(), "Z": tc.gates.z()}
    for coefficient, operators in case["energy_terms"]:
        ops = tuple((paulis[label], [site]) for site, label in operators)
        expectation = circuit.expectation(*ops, normalize=False)
        value = value + coefficient * tc.backend.real(expectation)
    return value


def _solve_case(case):
    parameters = tc.backend.convert_to_tensor(np.asarray(case["parameters"]))

    def objective(values):
        return _energy(values, case)

    value, gradient = tc.backend.value_and_grad(objective)(parameters)
    circuit = _build(parameters, case)
    return (
        float(np.asarray(value)),
        np.asarray(gradient, dtype=float),
        np.asarray(circuit.get_bond_dimensions()[1:-1], dtype=int),
    )


def run_solution(config):
    rows = [_solve_case(case) for case in config["cases"]]
    return {
        "energies": np.asarray([row[0] for row in rows]),
        "gradients": np.asarray([row[1] for row in rows]),
        "final_bond_dimensions": np.asarray([row[2] for row in rows]),
    }
