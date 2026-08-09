"""TensorCircuit expert for reusable replica-transfer programs."""

import hashlib
import json

import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")
K = tc.backend
SWAP = tc.gates.Gate(
    np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    ).reshape(2, 2, 2, 2)
)
SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _hop(theta, phi, eta=0.0):
    gate = np.zeros((4, 4), dtype=np.complex128)
    cosine, sine = np.cos(theta), np.sin(theta)
    gate[0, 0] = 1
    gate[1:3, 1:3] = [
        [cosine, -1j * np.exp(-1j * phi) * sine],
        [-1j * np.exp(1j * phi) * sine, cosine],
    ]
    gate[3, 3] = np.exp(-1j * eta)
    return tc.gates.Gate(gate.reshape(2, 2, 2, 2))


def _two(circuit, left, gate):
    circuit.apply_adjacent_double_gate(gate, left, left + 1, center_position=left + 1)


def _state(case):
    circuit = tc.MPSCircuit(
        2 * case["system_qubits"],
        center_position=0,
        split={"max_singular_values": case["replica_bond_capacity"]},
    )
    for site, bit in enumerate(case["initial_bits"]):
        if bit:
            circuit.x(2 * site)
    for layer in case["layers"]:
        for left, theta, phi, eta in layer:
            physical = 2 * left
            _two(circuit, physical + 1, SWAP)
            _two(circuit, physical, _hop(theta, phi, eta))
            _two(circuit, physical + 1, SWAP)
    for site, gamma in enumerate(case["amplitude_damping"]):
        _two(circuit, 2 * site, _hop(np.arcsin(np.sqrt(gamma)), 0.0))
    circuit.normalize()
    return circuit.get_tensors()


def _expression(order, selector):
    cursor = 0

    def take(count):
        nonlocal cursor
        value = SYMBOLS[cursor : cursor + count]
        cursor += count
        return value

    left_bra, left_ket = take(order), take(order)
    physical, right_bra, right_ket = take(order), take(order), take(order)
    permutation = [(r + selector) % order for r in range(order)]
    inverse = [permutation.index(r) for r in range(order)]
    terms = [left_bra + left_ket]
    terms += [left_bra[r] + physical[r] + right_bra[r] for r in range(order)]
    terms += [left_ket[r] + physical[inverse[r]] + right_ket[r] for r in range(order)]
    return ",".join(terms) + "->" + right_bra + right_ket


EXPRESSIONS = {
    (order, selector): _expression(order, selector)
    for order in (2, 3, 4)
    for selector in (-1, 0, 1)
}


def _step(environment, tensor, order, selector):
    operands = [environment]
    operands += [K.conj(tensor)] * order
    operands += [tensor] * order
    return K.einsum(EXPRESSIONS[order, selector], *operands)


def _path(case, query):
    partition_a, partition_b = set(query["partition_a"]), set(query["partition_b"])
    output = []
    for physical in range(2 * case["system_qubits"]):
        if physical % 2:
            output.append(0)
        else:
            site = physical // 2
            output.append(
                1 if site in partition_a else -1 if site in partition_b else 0
            )
    return output


def _solve(case):
    tensors, queries = _state(case), case["queries"]
    roots = {order: {} for order in (2, 3, 4)}
    for query_index, query in enumerate(queries):
        node = roots[query["order"]]
        for selector in _path(case, query):
            node = node.setdefault(selector, {})
        node.setdefault(None, []).append(query_index)
    nodes, terminals = [], [-1] * len(queries)
    moments = np.empty(len(queries))
    peak = 1

    def visit(tree, parent, site, order, environment, live):
        nonlocal peak
        for selector in sorted(value for value in tree if value is not None):
            child, node_id = tree[selector], len(nodes)
            nodes.append([parent, site, order, selector])
            next_environment = _step(environment, tensors[site], order, selector)
            charge = (
                1
                if site + 1 == len(tensors)
                else case["replica_bond_capacity"] ** (2 * order)
            )
            peak = max(peak, live + charge)
            for query_index in child.get(None, []):
                terminals[query_index] = node_id
                moments[query_index] = np.asarray(next_environment).item().real
            visit(
                child,
                node_id,
                site + 1,
                order,
                next_environment,
                live + charge,
            )
            del next_environment

    for order in (2, 3, 4):
        visit(roots[order], -1, 0, order, K.ones((1,) * (2 * order)), 1)
    payload = {"nodes": nodes, "terminal_nodes": terminals}
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    program = {
        **payload,
        "transfer_operation_count": len(nodes),
        "peak_live_complex_elements": peak,
        "program_digest": digest,
    }
    return moments, program


def run_solution(config):
    rows = [_solve(case) for case in config["cases"]]
    return {
        "moments": [row[0].tolist() for row in rows],
        "programs": [row[1] for row in rows],
    }
