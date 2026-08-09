"""Independent NumPy evaluator for replica-transfer program synthesis."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import re
import sys
import time

import numpy as np


DEFAULT_SEED = 1172026
SYSTEM_QUBITS = 32
PROGRAM_KEYS = {
    "nodes",
    "terminal_nodes",
    "transfer_operation_count",
    "peak_live_complex_elements",
    "program_digest",
}
SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)


def default_seed():
    return int(os.environ.get("ORBIT_Q_CANDIDATE_SEED", str(DEFAULT_SEED)))


def _hop(theta, phi, eta=0.0):
    gate = np.zeros((4, 4), dtype=np.complex128)
    cosine, sine = np.cos(theta), np.sin(theta)
    gate[0, 0] = 1
    gate[1:3, 1:3] = [
        [cosine, -1j * np.exp(-1j * phi) * sine],
        [-1j * np.exp(1j * phi) * sine, cosine],
    ]
    gate[3, 3] = np.exp(-1j * eta)
    return gate.reshape(2, 2, 2, 2)


def _layout_blocks(rng, system_qubits):
    lengths = np.full(5, 4, dtype=int)
    for _ in range(8):
        source, target = rng.choice(5, 2, replace=False)
        if lengths[source] > 3 and lengths[target] < 5:
            lengths[source] -= 1
            lengths[target] += 1
    start = system_qubits - int(np.sum(lengths))
    blocks = []
    for length in lengths:
        blocks.append(list(range(start, start + int(length))))
        start += int(length)
    return blocks


def _query_bank(rng, system_qubits):
    blocks = _layout_blocks(rng, system_qubits)
    queries = []
    for order in (2, 3, 4):
        for mask in range(16):
            selectors = [1]
            for block in range(1, 5):
                if not (mask >> (block - 1)) & 1:
                    selectors.append(0)
                elif order == 2 or block % 2 == 0:
                    selectors.append(1)
                else:
                    selectors.append(-1)
            partition_a, partition_b = [], []
            for selector, sites in zip(selectors, blocks):
                if selector == 1:
                    partition_a.extend(sites)
                elif selector == -1:
                    partition_b.extend(sites)
            queries.append(
                {
                    "query_id": f"q{rng.integers(1 << 48):012x}",
                    "order": order,
                    "partition_a": partition_a,
                    "partition_b": partition_b,
                    "query_nonce": int(rng.integers(1 << 31)),
                }
            )
    rng.shuffle(queries)
    return blocks, queries


def _path(case, query):
    partition_a = set(query["partition_a"])
    partition_b = set(query["partition_b"])
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


def _trie(case):
    roots = {order: {} for order in (2, 3, 4)}
    for query_index, query in enumerate(case["queries"]):
        node = roots[query["order"]]
        for selector in _path(case, query):
            node = node.setdefault(selector, {})
        node.setdefault(None, []).append(query_index)
    return roots


def _canonical_program(case):
    roots = _trie(case)
    nodes, terminals = [], [-1] * len(case["queries"])

    def visit(tree, parent, site, order):
        for selector in sorted(value for value in tree if value is not None):
            child, node_id = tree[selector], len(nodes)
            nodes.append([parent, site, order, selector])
            for query_index in child.get(None, []):
                terminals[query_index] = node_id
            visit(child, node_id, site + 1, order)

    for order in (2, 3, 4):
        visit(roots[order], -1, 0, order)
    if any(value < 0 for value in terminals):
        raise AssertionError("query trie has an unterminated path")
    payload = {"nodes": nodes, "terminal_nodes": terminals}
    digest = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    return roots, nodes, terminals, digest


def _make_case(rng, case_index):
    n = SYSTEM_QUBITS
    bits = np.zeros(n, dtype=int)
    bits[rng.choice(n, n // 2, replace=False)] = 1
    layers = []
    for layer in range(2):
        records = []
        for left in range(layer % 2, n - 1, 2):
            records.append(
                [
                    left,
                    float(rng.uniform(0.28, 0.78)),
                    float(rng.uniform(-np.pi, np.pi)),
                    float(rng.uniform(-0.42, 0.42)),
                ]
            )
        layers.append(records)
    blocks, queries = _query_bank(rng, n)
    case = {
        "case_id": f"replica-program-{case_index}-{rng.integers(1 << 40):010x}",
        "system_qubits": n,
        "initial_bits": bits.tolist(),
        "layers": layers,
        "amplitude_damping": rng.uniform(0.012, 0.072, size=n).tolist(),
        "layout_blocks": blocks,
        "queries": queries,
        "replica_bond_capacity": 4,
        "max_peak_live_complex_elements": 6_000_000,
        "case_nonce": int(rng.integers(1 << 31)),
    }
    _, nodes, _, _ = _canonical_program(case)
    case["max_transfer_operations"] = len(nodes)
    return case


def build_config(seed):
    rng = np.random.default_rng(seed)
    config = {
        "cases": [_make_case(rng, index) for index in range(2)],
        "physical_order": "system_0,environment_0,system_1,environment_1,...",
        "replica_convention": "bra r equals ket r+selector modulo order",
        "program_schema": "canonical-prefix-trie-v1",
    }
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    config["case_digest"] = hashlib.sha256(payload).hexdigest()
    return config


class _MPS:
    def __init__(self, bits):
        self.tensors = []
        for bit in bits:
            vector = np.zeros(2, dtype=np.complex128)
            vector[bit] = 1
            self.tensors.append(vector.reshape(1, 2, 1))
        self.center = 0

    def position(self, site):
        while self.center < site:
            left = self.tensors[self.center]
            dl, physical, dr = left.shape
            unitary, residue = np.linalg.qr(
                left.reshape(dl * physical, dr), mode="reduced"
            )
            self.tensors[self.center] = unitary.reshape(dl, physical, -1)
            self.tensors[self.center + 1] = np.einsum(
                "ab,bsr->asr", residue, self.tensors[self.center + 1]
            )
            self.center += 1
        while self.center > site:
            right = self.tensors[self.center]
            dl, physical, dr = right.shape
            unitary, residue = np.linalg.qr(
                right.reshape(dl, physical * dr).T, mode="reduced"
            )
            self.tensors[self.center] = unitary.T.reshape(-1, physical, dr)
            self.tensors[self.center - 1] = np.einsum(
                "lsr,ra->lsa", self.tensors[self.center - 1], residue.T
            )
            self.center -= 1

    def two(self, left, gate):
        self.position(left)
        pair = np.einsum("lim,mjr->lijr", self.tensors[left], self.tensors[left + 1])
        pair = np.einsum("abij,lijr->labr", gate, pair)
        dl, _, _, dr = pair.shape
        unitary, singular, residue = np.linalg.svd(
            pair.reshape(2 * dl, 2 * dr), full_matrices=False
        )
        rank = max(1, int(np.count_nonzero(singular > singular[0] * 2e-14)))
        self.tensors[left] = unitary[:, :rank].reshape(dl, 2, rank)
        self.tensors[left + 1] = (singular[:rank, None] * residue[:rank]).reshape(
            rank, 2, dr
        )
        self.center = left + 1

    def normalize(self):
        self.position(len(self.tensors) - 1)
        self.tensors[-1] /= np.linalg.norm(self.tensors[-1])


def _logical_pair(state, left, gate):
    physical = 2 * left
    state.two(physical + 1, SWAP)
    state.two(physical, gate)
    state.two(physical + 1, SWAP)


def _state(case):
    bits = []
    for bit in case["initial_bits"]:
        bits.extend([bit, 0])
    state = _MPS(bits)
    for layer in case["layers"]:
        for left, theta, phi, eta in layer:
            _logical_pair(state, left, _hop(theta, phi, eta))
    for site, gamma in enumerate(case["amplitude_damping"]):
        state.two(2 * site, _hop(np.arcsin(np.sqrt(gamma)), 0.0))
    state.normalize()
    return state.tensors


def _step_integer(environment, tensor, order, selector):
    left_bra = list(range(order))
    left_ket = list(range(order, 2 * order))
    physical = list(range(2 * order, 3 * order))
    right_bra = list(range(3 * order, 4 * order))
    right_ket = list(range(4 * order, 5 * order))
    permutation = [(r + selector) % order for r in range(order)]
    inverse = [permutation.index(r) for r in range(order)]
    operands = [environment, left_bra + left_ket]
    for replica in range(order):
        operands.extend(
            [tensor.conj(), [left_bra[replica], physical[replica], right_bra[replica]]]
        )
    for replica in range(order):
        operands.extend(
            [
                tensor,
                [left_ket[replica], physical[inverse[replica]], right_ket[replica]],
            ]
        )
    return np.einsum(*operands, right_bra + right_ket, optimize="greedy")


def _step_string(environment, tensor, order, selector):
    symbols = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    groups = [symbols[index * order : (index + 1) * order] for index in range(5)]
    left_bra, left_ket, physical, right_bra, right_ket = groups
    permutation = [(r + selector) % order for r in range(order)]
    inverse = [permutation.index(r) for r in range(order)]
    terms = [left_bra + left_ket]
    terms += [left_bra[r] + physical[r] + right_bra[r] for r in range(order)]
    terms += [left_ket[r] + physical[inverse[r]] + right_ket[r] for r in range(order)]
    expression = ",".join(terms) + "->" + right_bra + right_ket
    return np.einsum(
        expression,
        environment,
        *([tensor.conj()] * order),
        *([tensor] * order),
        optimize="greedy",
    )


def _execute(tensors, case, step):
    roots, expected_nodes, expected_terminals, digest = _canonical_program(case)
    nodes, terminals = [], [-1] * len(case["queries"])
    moments = np.empty(len(case["queries"]))
    peak, maximum_imaginary = 1, 0.0

    def visit(tree, parent, site, order, environment, live):
        nonlocal peak, maximum_imaginary
        for selector in sorted(value for value in tree if value is not None):
            child, node_id = tree[selector], len(nodes)
            nodes.append([parent, site, order, selector])
            next_environment = step(environment, tensors[site], order, selector)
            charge = (
                1
                if site + 1 == len(tensors)
                else case["replica_bond_capacity"] ** (2 * order)
            )
            peak = max(peak, live + charge)
            for query_index in child.get(None, []):
                value = np.asarray(next_environment).item()
                terminals[query_index] = node_id
                moments[query_index] = value.real
                maximum_imaginary = max(maximum_imaginary, abs(value.imag))
            visit(child, node_id, site + 1, order, next_environment, live + charge)
            del next_environment

    for order in (2, 3, 4):
        visit(
            roots[order],
            -1,
            0,
            order,
            np.ones((1,) * (2 * order), dtype=np.complex128),
            1,
        )
    if nodes != expected_nodes or terminals != expected_terminals:
        raise AssertionError("oracle traversal disagrees with canonical program")
    program = {
        "nodes": nodes,
        "terminal_nodes": terminals,
        "transfer_operation_count": len(nodes),
        "peak_live_complex_elements": peak,
        "program_digest": digest,
    }
    return moments, program, maximum_imaginary


def _dense_apply(state, gate, left, qubits):
    shaped = state.reshape((2,) * qubits)
    moved = np.moveaxis(shaped, (left, left + 1), (0, 1)).reshape(4, -1)
    moved = gate.reshape(4, 4) @ moved
    shaped = moved.reshape((2, 2) + (2,) * (qubits - 2))
    return np.moveaxis(shaped, (0, 1), (left, left + 1)).reshape(-1)


def _dense_reduced_moment(state, system_qubits, order, partition_a, partition_b):
    physical_qubits = 2 * system_qubits
    region = sorted(set(partition_a) | set(partition_b))
    selected = [2 * site for site in region]
    complement = [site for site in range(physical_qubits) if site not in selected]
    matrix = state.reshape((2,) * physical_qubits).transpose(selected + complement)
    matrix = matrix.reshape(2 ** len(region), -1)
    density = matrix @ matrix.conj().T
    shaped = density.reshape((2,) * (2 * len(region)))
    axes = list(range(2 * len(region)))
    for site in partition_b:
        position = region.index(site)
        axes[position], axes[len(region) + position] = (
            axes[len(region) + position],
            axes[position],
        )
    partial_transpose = shaped.transpose(axes).reshape(density.shape)
    return np.trace(np.linalg.matrix_power(partial_transpose, order))


def _explicit_replica_moment(state, system_qubits, order, partition_a, partition_b):
    physical_qubits = 2 * system_qubits
    selectors = []
    for physical in range(physical_qubits):
        site = physical // 2
        selectors.append(
            0
            if physical % 2
            else 1
            if site in partition_a
            else -1
            if site in partition_b
            else 0
        )
    replicas = state
    for _ in range(order - 1):
        replicas = np.kron(replicas, state)
    axes = []
    for replica in range(order):
        for physical, selector in enumerate(selectors):
            source = (replica + selector) % order
            axes.append(source * physical_qubits + physical)
    permuted = replicas.reshape((2,) * (order * physical_qubits)).transpose(axes)
    return np.vdot(replicas, permuted.reshape(-1))


def _dense_canary():
    case = {
        "system_qubits": 2,
        "initial_bits": [1, 0],
        "layers": [[[0, 0.63, -0.37, 0.22]]],
        "amplitude_damping": [0.07, 0.11],
    }
    state = np.zeros(16, dtype=np.complex128)
    state[int("1000", 2)] = 1
    gate = _hop(0.63, -0.37, 0.22)
    state = _dense_apply(state, SWAP, 1, 4)
    state = _dense_apply(state, gate, 0, 4)
    state = _dense_apply(state, SWAP, 1, 4)
    for site, gamma in enumerate(case["amplitude_damping"]):
        state = _dense_apply(state, _hop(np.arcsin(np.sqrt(gamma)), 0.0), 2 * site, 4)
    mps = _state(case)
    errors = []
    for order in (2, 3, 4):
        for partition_a, partition_b in (([0], []), ([0], [1])):
            selectors = []
            for physical in range(4):
                site = physical // 2
                selectors.append(
                    0
                    if physical % 2
                    else 1
                    if site in partition_a
                    else -1
                    if site in partition_b
                    else 0
                )
            environment = np.ones((1,) * (2 * order), dtype=np.complex128)
            for tensor, selector in zip(mps, selectors):
                environment = _step_integer(environment, tensor, order, selector)
            transfer = environment.item()
            dense = _dense_reduced_moment(state, 2, order, partition_a, partition_b)
            explicit = _explicit_replica_moment(
                state, 2, order, partition_a, partition_b
            )
            errors.extend([abs(transfer - dense), abs(transfer - explicit)])
    return float(max(errors))


def _exact_int(value):
    return not isinstance(value, (bool, np.bool_)) and isinstance(
        value, (int, np.integer)
    )


def _parse_program(program, case):
    if not isinstance(program, dict) or set(program) != PROGRAM_KEYS:
        raise ValueError("program has the wrong keys")
    nodes = program["nodes"]
    terminals = program["terminal_nodes"]
    if not isinstance(nodes, list) or len(nodes) > case["max_transfer_operations"]:
        raise ValueError("nodes must be a bounded list")
    parsed_nodes = []
    for row in nodes:
        if (
            not isinstance(row, list)
            or len(row) != 4
            or not all(_exact_int(value) for value in row)
        ):
            raise ValueError("node records must contain four exact integers")
        parsed_nodes.append([int(value) for value in row])
    if not isinstance(terminals, list) or len(terminals) != len(case["queries"]):
        raise ValueError("terminal_nodes has the wrong length")
    if not all(_exact_int(value) for value in terminals):
        raise ValueError("terminal nodes must be exact integers")
    operation_count = program["transfer_operation_count"]
    peak = program["peak_live_complex_elements"]
    if not _exact_int(operation_count) or not _exact_int(peak):
        raise ValueError("certificates must be exact integers")
    digest = program["program_digest"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("program_digest is not canonical SHA-256 hex")
    return (
        parsed_nodes,
        [int(value) for value in terminals],
        int(operation_count),
        int(peak),
        digest,
    )


def _validate_result(result, config, expected):
    criteria = {
        "strict output schema": True,
        "finite moment arrays": True,
        "independent replica moments": True,
        "canonical shared-prefix programs": True,
        "exact operation certificates": True,
        "exact live-memory certificates": True,
        "canonical program digests": True,
    }
    maximum_error = 0.0
    if not isinstance(result, dict) or set(result) != {"moments", "programs"}:
        criteria["strict output schema"] = False
        return criteria, float("inf")
    moments, programs = result["moments"], result["programs"]
    if (
        not isinstance(moments, list)
        or not isinstance(programs, list)
        or len(moments) != len(config["cases"])
        or len(programs) != len(config["cases"])
    ):
        criteria["strict output schema"] = False
        return criteria, float("inf")
    for case, submitted_values, submitted_program, target in zip(
        config["cases"], moments, programs, expected
    ):
        target_values, target_program, _ = target
        try:
            if not isinstance(submitted_values, list):
                raise ValueError("each case's moments must be a list")
            values = np.asarray(submitted_values)
            if values.dtype.kind not in "fiu" or values.shape != target_values.shape:
                raise ValueError("moment array has the wrong shape or dtype")
            values = values.astype(float)
            if not np.all(np.isfinite(values)):
                criteria["finite moment arrays"] = False
            error = float(np.max(np.abs(values - target_values)))
            maximum_error = max(maximum_error, error)
            if not np.allclose(values, target_values, atol=4e-9, rtol=4e-8):
                criteria["independent replica moments"] = False
            nodes, terminals, operations, peak, digest = _parse_program(
                submitted_program, case
            )
        except (TypeError, ValueError, OverflowError):
            criteria["strict output schema"] = False
            criteria["finite moment arrays"] = False
            criteria["independent replica moments"] = False
            criteria["canonical shared-prefix programs"] = False
            criteria["exact operation certificates"] = False
            criteria["exact live-memory certificates"] = False
            criteria["canonical program digests"] = False
            maximum_error = float("inf")
            continue
        criteria["canonical shared-prefix programs"] &= bool(
            nodes == target_program["nodes"]
            and terminals == target_program["terminal_nodes"]
        )
        criteria["exact operation certificates"] &= bool(
            operations
            == len(nodes)
            == target_program["transfer_operation_count"]
            <= case["max_transfer_operations"]
        )
        criteria["exact live-memory certificates"] &= bool(
            peak == target_program["peak_live_complex_elements"]
            and peak <= case["max_peak_live_complex_elements"]
        )
        payload = {"nodes": nodes, "terminal_nodes": terminals}
        submitted_digest = hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()
        criteria["canonical program digests"] &= bool(
            digest == submitted_digest == target_program["program_digest"]
        )
    return criteria, maximum_error


def _oracle(config, step=_step_integer):
    output = []
    for case in config["cases"]:
        output.append(_execute(_state(case), case, step))
    return output


def _proxy_result(config):
    rows = _oracle(config, _step_string)
    return {
        "moments": [row[0].tolist() for row in rows],
        "programs": [row[1] for row in rows],
    }


def self_test(seed):
    config = build_config(seed)
    canary_error = _dense_canary()
    expected = _oracle(config)
    proxy = _proxy_result(config)
    criteria, proxy_error = _validate_result(proxy, config, expected)
    mutations = []
    for mutate in (
        lambda value: value["programs"][0]["nodes"].__setitem__(0, [-1, 0, 2, 1]),
        lambda value: value["programs"][0]["terminal_nodes"].__setitem__(0, 0),
        lambda value: value["programs"][0].__setitem__("transfer_operation_count", 1),
        lambda value: value["programs"][0].__setitem__("peak_live_complex_elements", 1),
        lambda value: value["moments"][0].__setitem__(0, float("nan")),
    ):
        candidate = copy.deepcopy(proxy)
        mutate(candidate)
        mutated_criteria, _ = _validate_result(candidate, config, expected)
        mutations.append(not all(mutated_criteria.values()))
    bonds = [
        max(tensor.shape[2] for tensor in _state(case)[:-1]) for case in config["cases"]
    ]
    naive = [
        len(case["queries"]) * 2 * case["system_qubits"] for case in config["cases"]
    ]
    compression = [
        expected[index][1]["transfer_operation_count"] / naive[index]
        for index in range(len(expected))
    ]
    passed = bool(
        canary_error < 4e-11
        and all(criteria.values())
        and proxy_error < 4e-9
        and all(mutations)
        and max(compression) < 0.34
        and min(bonds) >= 4
        and max(bonds) <= 4
        and max(row[2] for row in expected) < 2e-9
    )
    print("Problem 117 local self-test")
    print(f"Case seed: {seed}; case digest: {config['case_digest']}")
    print(f"Dense/explicit-permutation canary error: {canary_error:.3e}")
    print(f"Expert-algorithm proxy maximum error: {proxy_error:.3e}")
    print(f"Canonical/naive operation ratios: {compression}")
    print(f"Maximum MPS bonds: {bonds}")
    print(f"Semantic mutations rejected: {sum(mutations)}/{len(mutations)}")
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
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
    sys.argv[:] = [sys.argv[0]]
    canary_error = _dense_canary()
    expected = _oracle(config)
    started = time.perf_counter()
    result = importlib.import_module(module_name).run_solution(copy.deepcopy(config))
    elapsed = time.perf_counter() - started
    criteria, maximum_error = _validate_result(result, config, expected)
    bonds = [
        max(tensor.shape[2] for tensor in _state(case)[:-1]) for case in config["cases"]
    ]
    naive = [
        len(case["queries"]) * 2 * case["system_qubits"] for case in config["cases"]
    ]
    ratios = [
        target[1]["transfer_operation_count"] / naive[index]
        for index, target in enumerate(expected)
    ]
    imaginary = max(target[2] for target in expected)
    criteria.update(
        {
            "independent dense/permutation canary": canary_error < 4e-11,
            "program reuse regime active": max(ratios) < 0.34,
            "representation barrier active": min(bonds) >= 4
            and all(
                bond <= case["replica_bond_capacity"]
                for bond, case in zip(bonds, config["cases"])
            )
            and all(case["system_qubits"] >= 32 for case in config["cases"]),
            "oracle imaginary residual": imaginary < 2e-9,
            "runtime limit": elapsed <= 300,
        }
    )
    print("Problem 117 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; case digest: {config['case_digest'][:16]}")
    print(f"End-to-end solution time: {elapsed:.3f}s")
    print(f"Maximum moment error: {maximum_error:.3e}")
    print(f"Dense/explicit-permutation canary error: {canary_error:.3e}")
    print(f"Canonical/naive operation ratios: {ratios}")
    print(f"Maximum MPS bonds: {bonds}; oracle imaginary residual: {imaginary:.3e}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    passed = all(criteria.values())
    metrics = {
        "orbit_q_expert_admission_metrics": {
            "schema_version": 1,
            "protocol_seed": seed,
            "case_digest": config["case_digest"],
            "metrics": [
                {
                    "metric": "maximum_program_to_naive_operation_ratio",
                    "direction": "at_most",
                    "observed": float(max(ratios)),
                    "threshold": 0.34,
                },
                {
                    "metric": "maximum_oracle_imaginary_residual",
                    "direction": "at_most",
                    "observed": float(imaginary),
                    "threshold": 2e-9,
                },
                {
                    "metric": "minimum_maximum_mps_bond",
                    "direction": "at_least",
                    "observed": float(min(bonds)),
                    "threshold": 4.0,
                },
                {
                    "metric": "maximum_mps_bond_capacity",
                    "direction": "at_most",
                    "observed": float(max(bonds)),
                    "threshold": 4.0,
                },
            ],
        }
    }
    print(json.dumps(metrics, sort_keys=True, separators=(",", ":")), flush=True)
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_117")
    parser.add_argument("--seed", type=int, default=default_seed())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    passed = (
        self_test(args.seed) if args.self_test else evaluate(args.solution, args.seed)
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
