"""TensorCircuit expert for memory-bounded selected IQP amplitudes."""

import numpy as np
import omeco
import tensorcircuit as tc
from tensorcircuit.cons import OMEOptimizer


tc.set_backend("numpy")
tc.set_dtype("complex128")


def _order(case):
    n = case["n_qubits"]
    adjacency = [set() for _ in range(n)]
    for u, v, _ in case["rzz_edges"]:
        adjacency[u].add(v)
        adjacency[v].add(u)
    remaining, answer, width = set(range(n)), [], 0
    while remaining:

        def score(vertex):
            live = sorted(adjacency[vertex] & remaining)
            missing = sum(
                right not in adjacency[left]
                for offset, left in enumerate(live)
                for right in live[offset + 1 :]
            )
            return missing, len(live), vertex

        vertex = min(remaining, key=score)
        live = sorted(adjacency[vertex] & remaining)
        width = max(width, len(live))
        for offset, left in enumerate(live):
            for right in live[offset + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        remaining.remove(vertex)
        answer.append(vertex)
    return answer, width


def _circuit(case):
    circuit = tc.Circuit(case["n_qubits"])
    for qubit in range(case["n_qubits"]):
        circuit.h(qubit)
    for qubit, angle in enumerate(case["rz_angles"]):
        circuit.rz(qubit, theta=angle)
    for left, right, angle in case["rzz_edges"]:
        circuit.rzz(left, right, theta=angle)
    for qubit in range(case["n_qubits"]):
        circuit.h(qubit)
    return circuit


def _amplitudes(case):
    circuit = _circuit(case)
    saved_path = []
    base = OMEOptimizer(omeco.TreeSA(ntrials=8, niters=36))

    def optimizer(inputs, output, sizes, memory_limit=None, **kwargs):
        if not saved_path:
            saved_path.append(base(inputs, output, sizes, memory_limit, **kwargs))
        return saved_path[0]

    contract = tc.set_contractor(
        "custom", optimizer=optimizer, set_global=False, preprocessing=True
    )
    scale = 2 ** (case["n_qubits"] / 2)
    return np.asarray(
        [
            scale * np.asarray(contract(circuit.amplitude_before(bits)).tensor)
            for bits in case["bitstrings"]
        ]
    )


def run_solution(config):
    orders, widths, amplitudes = [], [], []
    for case in config["cases"]:
        order, width = _order(case)
        orders.append(order)
        widths.append(width)
        amplitudes.append(_amplitudes(case))
    widths = np.asarray(widths, dtype=int)
    return {
        "scaled_amplitudes": np.asarray(amplitudes),
        "elimination_orders": np.asarray(orders, dtype=int),
        "induced_widths": widths,
        "peak_complex_entries": np.asarray(
            [2 ** (int(width) + 1) for width in widths], dtype=np.int64
        ),
    }
