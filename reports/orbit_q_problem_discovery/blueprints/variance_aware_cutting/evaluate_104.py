"""Independent deterministic evaluator for problem 104."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import time

import numpy as np


CHANNELS = ("rX", "rY", "rZ", "eX+", "eX-", "eY+", "eY-", "eZ+", "eZ-")
I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
S = np.diag([1, 1j]).astype(np.complex128)
SD = np.diag([1, -1j]).astype(np.complex128)
CX = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)
CZ = np.diag([1, 1, 1, -1]).astype(np.complex128)
PAULI = {0: I2, 1: X, 2: Y, 3: Z}


def _gate(name, *qubits, theta=None):
    record = {"name": name, "qubits": list(qubits)}
    if theta is not None:
        record["theta"] = float(theta)
    return record


def _matrix(gate):
    name = gate["name"]
    if name in {"h", "x", "s", "sd", "cx", "cz"}:
        return {"h": H, "x": X, "s": S, "sd": SD, "cx": CX, "cz": CZ}[name]
    theta = gate["theta"]
    p = {"rx": X, "ry": Y, "rz": Z}[name]
    return math.cos(theta / 2) * I2 - 1j * math.sin(theta / 2) * p


def _apply(state, matrix, qubits, n):
    k = len(qubits)
    rest = [q for q in range(n) if q not in qubits]
    axes = list(qubits) + rest
    tensor = state.reshape([2] * n).transpose(axes).reshape(2**k, -1)
    tensor = matrix @ tensor
    return tensor.reshape([2] * n).transpose(np.argsort(axes)).reshape(-1)


def _apply_gates(state, gates, n, mapping=None):
    mapping = list(range(n)) if mapping is None else mapping
    for gate in gates:
        qubits = [mapping[q] for q in gate["qubits"]]
        state = _apply(state, _matrix(gate), qubits, n)
    return state


def _zero_state(n):
    state = np.zeros(2**n, dtype=np.complex128)
    state[0] = 1
    return state


def _expectation(state, ps, n):
    transformed = state
    for q, code in enumerate(ps):
        if code:
            transformed = _apply(transformed, PAULI[code], [q], n)
    return float(np.vdot(state, transformed).real)


def _prep(pauli, sign):
    if pauli == "Z":
        return [] if sign == "+" else [_gate("x", 0)]
    if pauli == "X":
        return [_gate("h", 0)] if sign == "+" else [_gate("x", 0), _gate("h", 0)]
    common = [_gate("h", 0), _gate("s", 0)]
    return common if sign == "+" else [_gate("x", 0)] + common


def _fragment_means(case):
    nl, nr = case["n_left"], case["n_right"]
    left = _apply_gates(_zero_state(nl), case["left_gates"], nl)
    r = np.array([_expectation(left, [0] * (nl - 1) + [p], nl) for p in (1, 2, 3)])
    responses = []
    for pauli in "XYZ":
        for sign in "+-":
            state = _apply_gates(_zero_state(nr), _prep(pauli, sign), nr)
            state = _apply_gates(state, case["right_gates"], nr)
            responses.append(_expectation(state, case["observable_ps"], nr))
    return r, np.array(responses)


def _signed(r, e):
    return float(0.5 * (e[4] + e[5]) + 0.5 * sum(r[j] * (e[2 * j] - e[2 * j + 1]) for j in range(3)))


def _uncut(case):
    nl, nr = case["n_left"], case["n_right"]
    n = nl + nr - 1
    state = _apply_gates(_zero_state(n), case["left_gates"], n)
    mapping = [nl - 1] + list(range(nl, n))
    state = _apply_gates(state, case["right_gates"], n, mapping=mapping)
    ps = [0] * (nl - 1) + list(case["observable_ps"])
    return _expectation(state, ps, n)


def _means_and_coefficients(r, e):
    means = np.concatenate([r, e])
    coefficients = np.array(
        [
            0.5 * (e[0] - e[1]),
            0.5 * (e[2] - e[3]),
            0.5 * (e[4] - e[5]),
            0.5 * r[0], -0.5 * r[0],
            0.5 * r[1], -0.5 * r[1],
            0.5 * (1 + r[2]), 0.5 * (1 - r[2]),
        ]
    )
    return means, coefficients


def _allocate(means, coefficients, total, minimum):
    remaining = total - minimum * len(CHANNELS)
    if remaining < 0:
        raise ValueError("shot budget is smaller than mandatory minimum")
    weights = np.sqrt(coefficients**2 * np.maximum(0.0, 1 - means**2) + 1e-14)
    raw = remaining * weights / weights.sum()
    extra = np.floor(raw).astype(int)
    residual = remaining - int(extra.sum())
    order = sorted(range(len(CHANNELS)), key=lambda j: (-float(raw[j] - extra[j]), j))
    extra[order[:residual]] += 1
    return extra + minimum


def _predicted_variance(means, coefficients, allocation):
    return float(np.sum(coefficients**2 * np.maximum(0.0, 1 - means**2) / allocation))


def _sample_reconstruction(r, e, allocation, seed):
    means = np.concatenate([r, e])
    rng = np.random.default_rng(seed)
    sampled = []
    for mean, shots in zip(means, allocation):
        p = float(np.clip((1 + mean) / 2, 0, 1))
        count = rng.binomial(int(shots), p)
        sampled.append((2 * count - shots) / shots)
    return _signed(np.array(sampled[:3]), np.array(sampled[3:]))


def _random_circuit(rng, n, depth):
    gates = []
    for layer in range(depth):
        for q in range(n):
            gates.append(_gate(str(rng.choice(["rx", "ry", "rz"])), q, theta=rng.uniform(-2.5, 2.5)))
        for q in range(layer % 2, n - 1, 2):
            gates.append(_gate(str(rng.choice(["cx", "cz"])), q, q + 1))
    return gates


def make_cases(seed, count=8):
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        nl, nr = int(rng.integers(3, 5)), int(rng.integers(3, 5))
        left = _random_circuit(rng, nl, 3 + index % 2)
        left += [_gate("cx", nl - 2, nl - 1), _gate("ry", nl - 1, theta=rng.uniform(-1.8, 1.8))]
        right = [_gate("cz", 0, 1), _gate("rx", 0, theta=rng.uniform(-1.8, 1.8))]
        right += _random_circuit(rng, nr, 3)
        ps = [int(rng.integers(1, 4)) for _ in range(nr)]
        cases.append(
            {
                "n_left": nl,
                "n_right": nr,
                "left_gates": left,
                "right_gates": right,
                "observable_ps": ps,
                "total_shots": 7000 + 337 * index,
                "min_shots": 32,
                "sample_seed": int(rng.integers(0, 2**32 - 1)),
            }
        )
    return cases


def build_config(seed):
    cases = make_cases(seed)
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"cases": cases, "case_digest": digest, "component_order": list(CHANNELS)}


def _oracle(case):
    r, e = _fragment_means(case)
    means, coefficients = _means_and_coefficients(r, e)
    allocation = _allocate(means, coefficients, case["total_shots"], case["min_shots"])
    return {
        "r": r,
        "e": e,
        "means": means,
        "coefficients": coefficients,
        "allocation": allocation,
        "predicted": _predicted_variance(means, coefficients, allocation),
        "exact": _signed(r, e),
        "uncut": _uncut(case),
    }


def _parse(result, key, dtype):
    try:
        return np.asarray(result[key], dtype=dtype)
    except (KeyError, TypeError, ValueError):
        return np.array([], dtype=dtype)


def evaluate(module_name, seed):
    config = build_config(seed)
    oracle = [_oracle(case) for case in config["cases"]]
    if not all(abs(x["exact"] - x["uncut"]) < 2e-12 for x in oracle):
        raise AssertionError("internal cutting identity canary failed")
    start = time.perf_counter()
    result = importlib.import_module(module_name).run_solution(config)
    elapsed = time.perf_counter() - start
    recon = _parse(result, "reconstructions", float)
    alloc = _parse(result, "allocations", int)
    predicted = _parse(result, "predicted_variances", float)
    exact = _parse(result, "exact_reconstructions", float)
    bloch = _parse(result, "upstream_bloch", float)
    responses = _parse(result, "downstream_responses", float)
    count = len(config["cases"])
    vector_shape, alloc_shape = (count,), (count, len(CHANNELS))
    shapes = (
        recon.shape == vector_shape and predicted.shape == vector_shape and exact.shape == vector_shape
        and alloc.shape == alloc_shape and bloch.shape == (count, 3) and responses.shape == (count, 6)
    )
    expected_bloch = np.array([x["r"] for x in oracle])
    expected_responses = np.array([x["e"] for x in oracle])
    expected_exact = np.array([x["uncut"] for x in oracle])
    allocation_valid = alloc.shape == alloc_shape and all(
        np.all(alloc[j] >= case["min_shots"]) and int(alloc[j].sum()) == case["total_shots"]
        for j, case in enumerate(config["cases"])
    )
    if allocation_valid:
        submitted_variance = np.array(
            [_predicted_variance(x["means"], x["coefficients"], alloc[j]) for j, x in enumerate(oracle)]
        )
        optimal_variance = np.array([x["predicted"] for x in oracle])
        expected_recon = np.array(
            [
                _sample_reconstruction(x["r"], x["e"], alloc[j], case["sample_seed"])
                for j, (x, case) in enumerate(zip(oracle, config["cases"]))
            ]
        )
    else:
        submitted_variance = np.full(count, np.inf)
        optimal_variance = np.array([x["predicted"] for x in oracle])
        expected_recon = np.full(count, np.nan)
    criteria = {
        "all output shapes": shapes,
        "outputs finite": bool(shapes and all(np.all(np.isfinite(x)) for x in (recon, predicted, exact, bloch, responses))),
        "TensorCircuit fragment means match oracle": bool(
            bloch.shape == (count, 3) and responses.shape == (count, 6)
            and np.allclose(bloch, expected_bloch, atol=2e-6, rtol=2e-6)
            and np.allclose(responses, expected_responses, atol=2e-6, rtol=2e-6)
        ),
        "signed exact reconstruction matches uncut oracle": bool(
            exact.shape == vector_shape and np.allclose(exact, expected_exact, atol=3e-6, rtol=3e-6)
        ),
        "allocation obeys budget and minimum": allocation_valid,
        "allocation is variance-aware": bool(
            allocation_valid and np.all(submitted_variance <= optimal_variance * 1.015 + 1e-15)
        ),
        "predicted variance is correct": bool(
            predicted.shape == vector_shape and np.allclose(predicted, submitted_variance, atol=2e-10, rtol=2e-5)
        ),
        "seeded finite-shot reconstruction is reproducible": bool(
            recon.shape == vector_shape and np.allclose(recon, expected_recon, atol=3e-9, rtol=3e-8)
        ),
        "runtime below 180 seconds": elapsed < 180,
    }
    print("Problem 104 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; case digest: {config['case_digest'][:16]}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Cases: {count}; total fragment shot budget: {sum(c['total_shots'] for c in config['cases'])}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")
    return all(criteria.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_104")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ORBIT_CUTTING_SEED", "1042026")))
    args = parser.parse_args()
    evaluate(args.solution, args.seed)


if __name__ == "__main__":
    main()
