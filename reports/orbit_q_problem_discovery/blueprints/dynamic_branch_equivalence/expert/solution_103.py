"""TensorCircuit expert solution for branch-sensitive instrument equivalence."""

import itertools

import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")


def apply_gates(circuit, records):
    for gate in records:
        name = gate["name"]
        q = gate["qubits"]
        if name in {"rx", "ry", "rz"}:
            getattr(circuit, name)(*q, theta=gate["theta"])
        elif name == "cx":
            circuit.cnot(*q)
        else:
            getattr(circuit, name)(*q)


def execute_branch(n, probe, program, history):
    circuit = tc.Circuit(n)
    apply_gates(circuit, probe)
    apply_gates(circuit, program["prefix"])
    for bit, round_spec in zip(history, program["rounds"]):
        measured = round_spec["measure"]
        circuit.post_select(measured, keep=bit)
        apply_gates(circuit, round_spec["branches"][str(bit)])
        if round_spec["reset"] and bit:
            circuit.x(measured)
    apply_gates(circuit, program["final"])
    state = np.asarray(circuit.state(), dtype=np.complex128).reshape(-1)
    return np.outer(state, state.conj())


def branch_matrices(n, probe, program):
    histories = itertools.product((0, 1), repeat=len(program["rounds"]))
    return [execute_branch(n, probe, program, history) for history in histories]


def solve_case(case, tolerance):
    distance = 0.0
    completeness = 0.0
    for probe in case["probes"]:
        left = branch_matrices(case["n_qubits"], probe, case["program_a"])
        right = branch_matrices(case["n_qubits"], probe, case["program_b"])
        distance = max(distance, *(np.linalg.norm(a - b) for a, b in zip(left, right)))
        completeness = max(
            completeness,
            abs(sum(np.trace(r).real for r in left) - 1.0),
            abs(sum(np.trace(r).real for r in right) - 1.0),
        )
    return distance, distance <= tolerance, completeness


def run_solution(config):
    solved = [solve_case(case, config["equivalence_tolerance"]) for case in config["cases"]]
    return {
        "instrument_distances": np.array([x[0] for x in solved], dtype=float),
        "equivalent": np.array([x[1] for x in solved], dtype=bool),
        "completeness_errors": np.array([x[2] for x in solved], dtype=float),
    }
