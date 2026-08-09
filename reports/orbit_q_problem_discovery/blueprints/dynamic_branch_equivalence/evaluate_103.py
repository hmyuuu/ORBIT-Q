"""Independent deterministic evaluator for problem 103."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import math
import os
import time

import numpy as np


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


def _gate(name, *qubits, theta=None):
    record = {"name": name, "qubits": list(qubits)}
    if theta is not None:
        record["theta"] = float(theta)
    return record


def _random_gate(rng, n):
    if rng.random() < 0.62:
        q = int(rng.integers(n))
        name = rng.choice(["h", "rx", "ry", "rz"])
        theta = None if name == "h" else rng.uniform(-2.4, 2.4)
        return _gate(str(name), q, theta=theta)
    q0, q1 = rng.choice(n, size=2, replace=False)
    return _gate(str(rng.choice(["cx", "cz"])), int(q0), int(q1))


def _random_gates(rng, n, count):
    return [_random_gate(rng, n) for _ in range(count)]


def _rewrite(gates):
    out = []
    for index, gate in enumerate(gates):
        name, q = gate["name"], gate["qubits"]
        if name == "rx":
            out += [_gate("h", q[0]), _gate("rz", q[0], theta=gate["theta"]), _gate("h", q[0])]
        elif name == "ry":
            out += [
                _gate("rz", q[0], theta=-math.pi / 2),
                _gate("rx", q[0], theta=gate["theta"]),
                _gate("rz", q[0], theta=math.pi / 2),
            ]
        elif name == "cx":
            out += [
                _gate("h", q[0]), _gate("h", q[1]), _gate("cx", q[1], q[0]),
                _gate("h", q[0]), _gate("h", q[1]),
            ]
        else:
            out.append(dict(gate))
        if index % 3 == 1:
            out += [_gate("x", q[0]), _gate("x", q[0])]
    return out


def _equivalent_rewrite(program):
    return {
        "prefix": _rewrite(program["prefix"]),
        "rounds": [
            {
                "measure": r["measure"],
                "reset": r["reset"],
                "branches": {bit: _rewrite(r["branches"][bit]) for bit in ("0", "1")},
            }
            for r in program["rounds"]
        ],
        "final": _rewrite(program["final"]),
    }


def _matrix(gate):
    name = gate["name"]
    if name in {"h", "x", "s", "sd", "cx", "cz"}:
        return {"h": H, "x": X, "s": S, "sd": SD, "cx": CX, "cz": CZ}[name]
    theta = gate["theta"]
    pauli = {"rx": X, "ry": Y, "rz": Z}[name]
    return math.cos(theta / 2) * I2 - 1j * math.sin(theta / 2) * pauli


def _apply(state, matrix, qubits, n):
    k = len(qubits)
    rest = [q for q in range(n) if q not in qubits]
    axes = list(qubits) + rest
    tensor = state.reshape([2] * n).transpose(axes).reshape(2**k, -1)
    tensor = matrix @ tensor
    inverse = np.argsort(axes)
    return tensor.reshape([2] * n).transpose(inverse).reshape(-1)


def _apply_gates(state, gates, n):
    for gate in gates:
        state = _apply(state, _matrix(gate), gate["qubits"], n)
    return state


def _project(state, qubit, bit, n):
    tensor = state.reshape([2] * n).copy()
    selector = [slice(None)] * n
    selector[qubit] = 1 - bit
    tensor[tuple(selector)] = 0
    return tensor.reshape(-1)


def _branch_states(program, probe, n):
    histories = itertools.product((0, 1), repeat=len(program["rounds"]))
    outputs = []
    for history in histories:
        state = np.zeros(2**n, dtype=np.complex128)
        state[0] = 1
        state = _apply_gates(state, probe, n)
        state = _apply_gates(state, program["prefix"], n)
        for bit, round_spec in zip(history, program["rounds"]):
            q = round_spec["measure"]
            state = _project(state, q, bit, n)
            state = _apply_gates(state, round_spec["branches"][str(bit)], n)
            if round_spec["reset"] and bit:
                state = _apply(state, X, [q], n)
        state = _apply_gates(state, program["final"], n)
        outputs.append(np.outer(state, state.conj()))
    return outputs


def _oracle_case(case):
    distance = 0.0
    completeness = 0.0
    for probe in case["probes"]:
        left = _branch_states(case["program_a"], probe, case["n_qubits"])
        right = _branch_states(case["program_b"], probe, case["n_qubits"])
        distance = max(distance, *(np.linalg.norm(a - b) for a, b in zip(left, right)))
        completeness = max(
            completeness,
            abs(sum(np.trace(r).real for r in left) - 1),
            abs(sum(np.trace(r).real for r in right) - 1),
        )
    return float(distance), float(completeness)


def _base_program(rng, n, rounds):
    measured = list(rng.choice(n, size=rounds, replace=False))
    return {
        "prefix": [_gate("h", int(measured[0]))] + _random_gates(rng, n, 5),
        "rounds": [
            {
                "measure": int(q),
                "reset": bool(rng.integers(2)),
                "branches": {
                    "0": _random_gates(rng, n, 3),
                    "1": _random_gates(rng, n, 3),
                },
            }
            for q in measured
        ],
        "final": _random_gates(rng, n, 5),
    }


def _mutate(program, rng, n, attempt):
    mutant = json.loads(json.dumps(program))
    kind = attempt % 3
    if kind == 0:
        mutant["final"].append(_gate("rz", int(rng.integers(n)), theta=0.11 + 0.03 * attempt))
    elif kind == 1:
        target = int((mutant["rounds"][0]["measure"] + 1) % n)
        mutant["rounds"][0]["branches"]["1"].append(_gate("rx", target, theta=0.19 + 0.02 * attempt))
    else:
        mutant["rounds"][0]["reset"] = not mutant["rounds"][0]["reset"]
    return mutant


def make_cases(seed, count=12):
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        n = int(rng.integers(4, 6))
        program_a = _base_program(rng, n, 1 + index % 2)
        if index == 0:
            measured = program_a["rounds"][0]["measure"]
            program_a["prefix"] = [
                gate for gate in program_a["prefix"] if measured not in gate["qubits"]
            ] + [_gate("rz", measured, theta=0.37)]
        probes = [[]] + [_random_gates(rng, n, 3 + j) for j in range(5)]
        rewritten = _equivalent_rewrite(program_a)
        if index % 2 == 0:
            program_b = rewritten
        else:
            for attempt in range(12):
                program_b = _mutate(rewritten, rng, n, attempt)
                trial = {"n_qubits": n, "probes": probes, "program_a": program_a, "program_b": program_b}
                if _oracle_case(trial)[0] > 2e-4:
                    break
            else:
                raise RuntimeError("failed to plant a detectable instrument mutation")
        cases.append({"n_qubits": n, "probes": probes, "program_a": program_a, "program_b": program_b})
    return cases


def build_config(seed):
    cases = make_cases(seed)
    digest = hashlib.sha256(json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"cases": cases, "equivalence_tolerance": 5e-6, "case_digest": digest}


def evaluate(module_name, seed):
    config = build_config(seed)
    expected = [_oracle_case(case) for case in config["cases"]]
    start = time.perf_counter()
    result = importlib.import_module(module_name).run_solution(config)
    elapsed = time.perf_counter() - start
    distances = np.asarray(result.get("instrument_distances"), dtype=float)
    equivalent = np.asarray(result.get("equivalent"), dtype=bool)
    completeness = np.asarray(result.get("completeness_errors"), dtype=float)
    expected_distances = np.array([x[0] for x in expected])
    expected_completeness = np.array([x[1] for x in expected])
    shape = (len(config["cases"]),)
    criteria = {
        "distance shape": distances.shape == shape,
        "classification shape": equivalent.shape == shape,
        "completeness shape": completeness.shape == shape,
        "outputs finite": bool(np.all(np.isfinite(distances)) and np.all(np.isfinite(completeness))),
        "branch distances match independent oracle": bool(
            distances.shape == shape and np.allclose(distances, expected_distances, atol=2e-6, rtol=3e-6)
        ),
        "equivalence classifications correct": bool(
            equivalent.shape == shape
            and np.array_equal(equivalent, expected_distances <= config["equivalence_tolerance"])
        ),
        "branch completeness preserved": bool(
            completeness.shape == shape
            and np.allclose(completeness, expected_completeness, atol=2e-6, rtol=0)
            and np.max(completeness) < 2e-6
        ),
        "runtime below 180 seconds": elapsed < 180,
    }
    print("Problem 103 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; case digest: {config['case_digest'][:16]}")
    print(f"End-to-end solution time: {elapsed:.2f}s")
    print(f"Cases: {len(config['cases'])}; expected equivalent: {int(np.sum(expected_distances <= config['equivalence_tolerance']))}")
    print(f"Largest reported instrument distance: {float(np.max(distances)) if distances.shape == shape else float('nan'):.6e}")
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"Overall: {'PASS' if all(criteria.values()) else 'FAIL'}")
    return all(criteria.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_103")
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ORBIT_DYNAMIC_SEED", "1032026")))
    args = parser.parse_args()
    evaluate(args.solution, args.seed)


if __name__ == "__main__":
    main()
