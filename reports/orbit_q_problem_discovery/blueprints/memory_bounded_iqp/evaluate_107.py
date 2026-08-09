"""Independent evaluator for problem 107's memory-bounded IQP contractions."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time

import numpy as np


DEFAULT_SEED = 1072026


def default_seed():
    return int(
        os.environ.get(
            "ORBIT_Q_CANDIDATE_SEED",
            os.environ.get("ORBIT_CONTRACTION_SEED", str(DEFAULT_SEED)),
        )
    )


def case_digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _grid_edges(rows, cols):
    edges = []
    for column in range(cols):
        for row in range(rows):
            edges.append((row * cols + column, ((row + 1) % rows) * cols + column))
    for column in range(cols - 1):
        for row in range(rows):
            here = row * cols + column
            edges.append((here, row * cols + column + 1))
            diagonal = (
                ((row + (1 if column % 2 == 0 else -1)) % rows) * cols + column + 1
            )
            edges.append((here, diagonal))
    return sorted({tuple(sorted(edge)) for edge in edges})


def _make_cases(seed, count=2):
    rng = np.random.default_rng(seed)
    public, private_orders = [], []
    rows, cols = 12, 6
    base_edges = _grid_edges(rows, cols)
    base_order = [row * cols + column for column in range(cols) for row in range(rows)]
    for case_index in range(count):
        permutation = rng.permutation(rows * cols)
        mapped_edges = [
            (int(permutation[u]), int(permutation[v])) for u, v in base_edges
        ]
        rng.shuffle(mapped_edges)
        records = [[u, v, float(rng.uniform(-0.82, 0.82))] for u, v in mapped_edges]
        bitstrings = [
            "".join(str(int(bit)) for bit in rng.integers(0, 2, rows * cols))
            for _ in range(12)
        ]
        public.append(
            {
                "n_qubits": rows * cols,
                "rz_angles": rng.uniform(-0.93, 0.93, rows * cols).tolist(),
                "rzz_edges": records,
                "bitstrings": bitstrings,
                "logical_width_limit": 17,
                "case_nonce": int(rng.integers(0, 2**31 - 1)) + case_index,
            }
        )
        private_orders.append([int(permutation[v]) for v in base_order])
    return public, private_orders


def build_config(seed):
    cases, _ = _make_cases(seed)
    return {
        "cases": cases,
        "case_digest": case_digest(cases),
        "amplitude_scaling": "2**(n_qubits/2)",
    }


def _induced_width(n, edges, order):
    if len(order) != n or set(order) != set(range(n)):
        raise ValueError("elimination order is not a vertex permutation")
    adjacency = [set() for _ in range(n)]
    for u, v, *_ in edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    remaining, width = set(range(n)), 0
    for vertex in order:
        neighbors = sorted(adjacency[vertex] & remaining)
        width = max(width, len(neighbors))
        for offset, left in enumerate(neighbors):
            for right in neighbors[offset + 1 :]:
                adjacency[left].add(right)
                adjacency[right].add(left)
        remaining.remove(vertex)
    return width


def _multiply_factors(factors, scope):
    joint = np.ones((2,) * len(scope), dtype=np.complex128)
    positions = {vertex: axis for axis, vertex in enumerate(scope)}
    for variables, table in factors:
        axes = sorted(
            range(len(variables)), key=lambda axis: positions[variables[axis]]
        )
        ordered = [variables[axis] for axis in axes]
        shape = [2 if vertex in ordered else 1 for vertex in scope]
        joint *= np.transpose(table, axes).reshape(shape)
    return joint


def _amplitude(case, bitstring, order):
    n = case["n_qubits"]
    factors = []
    for vertex, angle in enumerate(case["rz_angles"]):
        z = np.array([1.0, -1.0])
        sign = np.array([1.0, -1.0]) if bitstring[vertex] == "1" else 1.0
        factors.append(((vertex,), sign * np.exp(-0.5j * angle * z)))
    for u, v, angle in case["rzz_edges"]:
        z = np.array([1.0, -1.0])
        factors.append(((u, v), np.exp(-0.5j * angle * np.outer(z, z))))
    for vertex in order:
        selected = [factor for factor in factors if vertex in factor[0]]
        factors = [factor for factor in factors if vertex not in factor[0]]
        scope = sorted({item for variables, _ in selected for item in variables})
        reduced_scope = tuple(item for item in scope if item != vertex)
        joint = _multiply_factors(selected, scope)
        reduced = joint.sum(axis=scope.index(vertex))
        factors.append((reduced_scope, reduced))
    value = np.prod([np.asarray(table).item() for _, table in factors])
    return value * 2**-n


def _dense_canary():
    case = {
        "n_qubits": 4,
        "rz_angles": [0.13, -0.27, 0.41, -0.19],
        "rzz_edges": [[0, 1, 0.31], [1, 2, -0.22], [2, 3, 0.17]],
    }
    bits, order = "1010", [0, 1, 2, 3]
    expected = 0.0j
    for state in range(16):
        z = np.array([1 - 2 * ((state >> (3 - q)) & 1) for q in range(4)])
        phase = -0.5j * (
            np.dot(case["rz_angles"], z)
            + sum(angle * z[u] * z[v] for u, v, angle in case["rzz_edges"])
        )
        parity = sum(int(bits[q]) * ((state >> (3 - q)) & 1) for q in range(4))
        expected += (-1) ** parity * np.exp(phase) / 16
    return abs(_amplitude(case, bits, order) - expected) < 2e-14


def _array(result, key, dtype):
    try:
        return np.asarray(result[key], dtype=dtype)
    except (KeyError, TypeError, ValueError):
        return np.array([], dtype=dtype)


def evaluate(module_name, seed):
    cases, oracle_orders = _make_cases(seed)
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
    os.environ.pop("ORBIT_CONTRACTION_SEED", None)
    if not _dense_canary():
        raise AssertionError("internal IQP partition-function canary failed")
    expected = np.array(
        [
            [
                2 ** (case["n_qubits"] / 2) * _amplitude(case, bits, order)
                for bits in case["bitstrings"]
            ]
            for case, order in zip(cases, oracle_orders)
        ],
        dtype=np.complex128,
    )
    started = time.perf_counter()
    result = importlib.import_module(module_name).run_solution(config)
    elapsed = time.perf_counter() - started
    amplitudes = _array(result, "scaled_amplitudes", complex)
    orders = _array(result, "elimination_orders", int)
    widths = _array(result, "induced_widths", int)
    peaks = _array(result, "peak_complex_entries", int)
    shape = (len(cases), cases[0]["n_qubits"])
    certificate_valid, measured_widths = orders.shape == shape, []
    if certificate_valid:
        try:
            measured_widths = [
                _induced_width(case["n_qubits"], case["rzz_edges"], order.tolist())
                for case, order in zip(cases, orders)
            ]
        except ValueError:
            certificate_valid = False
    measured_widths = np.asarray(measured_widths, dtype=int)
    limits = np.asarray([case["logical_width_limit"] for case in cases])
    expected_peaks = np.asarray(
        [2 ** (int(width) + 1) for width in measured_widths], dtype=np.int64
    )
    criteria = {
        "output shapes": bool(
            amplitudes.shape == expected.shape
            and orders.shape == shape
            and widths.shape == (len(cases),)
            and peaks.shape == (len(cases),)
        ),
        "outputs finite": bool(np.all(np.isfinite(amplitudes))),
        "independent scaled amplitudes": bool(
            amplitudes.shape == expected.shape
            and np.allclose(amplitudes, expected, atol=5e-6, rtol=2e-5)
        ),
        "orders are vertex permutations": bool(certificate_valid),
        "logical width within budget": bool(
            certificate_valid and np.all(measured_widths <= limits)
        ),
        "width certificate exact": bool(
            certificate_valid and np.array_equal(widths, measured_widths)
        ),
        "peak table certificate exact": bool(
            certificate_valid and np.array_equal(peaks, expected_peaks)
        ),
        "representation stress active": all(case["n_qubits"] >= 70 for case in cases),
    }
    digest = config["case_digest"]
    max_error = (
        float(np.max(np.abs(amplitudes - expected)))
        if amplitudes.shape == expected.shape
        else float("inf")
    )
    print("Problem 107 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; case digest: {digest[:16]}")
    print(f"End-to-end solution time: {elapsed:.3f}s")
    print(f"Maximum scaled-amplitude error: {max_error:.3e}")
    print(f"Measured induced widths: {measured_widths.tolist()}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    passed = all(criteria.values())
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_107")
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed(),
    )
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.solution, args.seed) else 1)


if __name__ == "__main__":
    main()
