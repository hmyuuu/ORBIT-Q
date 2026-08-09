"""Independent binary-symplectic evaluator for problem 106."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import itertools
import json
import os
import time

import numpy as np


PAULIS = ((0, 0), (1, 0), (1, 1), (0, 1))
LABEL = {(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}


def _rref(rows, width=None):
    a = np.asarray(rows, dtype=np.uint8).copy() & 1
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if a.size == 0:
        return np.zeros((0, int(width or 0)), dtype=np.uint8), []
    row = 0
    pivots = []
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


def _canonical(rows, width):
    reduced, _ = _rref(rows, width)
    return tuple(tuple(int(v) for v in row) for row in reduced)


def _nullspace(rows):
    reduced, pivots = _rref(rows)
    free = [column for column in range(reduced.shape[1]) if column not in pivots]
    basis = []
    for column in free:
        vector = np.zeros(reduced.shape[1], dtype=np.uint8)
        vector[column] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = reduced[row, column]
        basis.append(vector)
    return np.asarray(basis, dtype=np.uint8)


def _scramble(rows, rng):
    out = np.asarray(rows, dtype=np.uint8).copy()
    for _ in range(12 * len(out)):
        first, second = rng.choice(len(out), size=2, replace=False)
        if rng.random() < 0.76:
            out[first] ^= out[second]
        else:
            out[[first, second]] = out[[second, first]]
    return out


def _simplex(dimension):
    columns = range(1, 2**dimension)
    return np.asarray(
        [[(value >> row) & 1 for value in columns] for row in range(dimension)],
        dtype=np.uint8,
    )


def _new_products(basis, count, occupied, rng):
    rows = []
    while len(rows) < count:
        coefficients = rng.integers(0, 2, size=len(basis), dtype=np.uint8)
        if not np.any(coefficients):
            continue
        value = (coefficients @ basis) & 1
        key = tuple(int(v) for v in value)
        if key in occupied:
            continue
        occupied.add(key)
        rows.append(value)
    return rows


def _candidate_pool(x_rows, z_rows, rng):
    x_basis = _scramble(_rref(x_rows)[0], rng)
    z_basis = _scramble(_rref(z_rows)[0], rng)
    x_values = [row.copy() for row in x_basis]
    z_values = [row.copy() for row in z_basis]
    x_seen = {tuple(int(v) for v in row) for row in x_values}
    z_seen = {tuple(int(v) for v in row) for row in z_values}
    if x_rows.shape[1] == 7:
        x_extra, z_extra = 2, 3
    else:
        x_extra, z_extra = 2, 1
    x_values += _new_products(x_basis, x_extra, x_seen, rng)
    z_values += _new_products(z_basis, z_extra, z_seen, rng)
    records = [("x", value) for value in x_values]
    records += [("z", value) for value in z_values]
    rng.shuffle(records)
    used_ids = set()
    candidates = []
    for kind, value in records:
        while True:
            check_id = f"v{int(rng.integers(10000, 100000))}"
            if check_id not in used_ids:
                used_ids.add(check_id)
                break
        candidates.append(
            {
                "id": check_id,
                "kind": kind,
                "mask": [int(v) for v in value],
                "coupling_cost": int(np.sum(value)),
            }
        )
    return candidates


def _make_instance(dimension, rng):
    x_rows = _simplex(dimension)
    permutation = rng.permutation(x_rows.shape[1])
    x_rows = x_rows[:, permutation]
    z_rows = _nullspace(x_rows)
    x_public = _scramble(x_rows, rng)
    z_public = _scramble(z_rows, rng)
    candidates = _candidate_pool(x_rows, z_rows, rng)
    canonical_x, _ = _rref(x_rows)
    gate_budget = int(len(canonical_x) + sum(np.sum(row) - 1 for row in canonical_x))
    payload = {
        "n_qubits": int(x_rows.shape[1]),
        "x_generators": x_public.astype(int).tolist(),
        "z_generators": z_public.astype(int).tolist(),
        "gate_budget": gate_budget,
        "candidate_checks": candidates,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"instance_id": f"css-{x_rows.shape[1]}-{digest[:12]}", **payload}


def build_config(seed):
    rng = np.random.default_rng(seed)
    instances = [_make_instance(dimension, rng) for dimension in (3, 4)]
    digest = hashlib.sha256(
        json.dumps(instances, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "instances": instances,
        "case_digest": digest,
        "effective_line_limit": 160,
        "fault_model": "prep-X plus all post-H and post-CX nonidentity Paulis",
    }


def _parse_gates(raw, n_qubits, budget):
    if not isinstance(raw, list) or len(raw) > budget:
        raise ValueError("encoder must be a list within gate_budget")
    gates = []
    for record in raw:
        if not isinstance(record, dict) or set(record) != {"name", "qubits"}:
            raise ValueError("each encoder gate needs exactly name and qubits")
        name, qubits = record["name"], record["qubits"]
        arity = {"h": 1, "cx": 2}.get(name)
        if arity is None or not isinstance(qubits, list) or len(qubits) != arity:
            raise ValueError("only h and cx gates with canonical arity are allowed")
        if any(isinstance(q, bool) or not isinstance(q, int) for q in qubits):
            raise ValueError("qubit indices must be integers")
        if any(q < 0 or q >= n_qubits for q in qubits) or len(set(qubits)) < arity:
            raise ValueError("qubit indices are out of range or repeated")
        gates.append((name, *qubits))
    return gates


def _conjugate(x, z, gate):
    if gate[0] == "h":
        q = gate[1]
        x[..., q], z[..., q] = z[..., q].copy(), x[..., q].copy()
    else:
        control, target = gate[1:]
        x[..., target] ^= x[..., control]
        z[..., control] ^= z[..., target]


def _encoder_stabilizers(n_qubits, gates):
    x = np.zeros((n_qubits, n_qubits), dtype=np.uint8)
    z = np.eye(n_qubits, dtype=np.uint8)
    for gate in gates:
        _conjugate(x, z, gate)
    return np.concatenate([x, z], axis=1)


def _encoder_state(n_qubits, gates):
    state = np.zeros(2**n_qubits, dtype=np.float64)
    state[0] = 1.0
    indices = np.arange(len(state))
    for gate in gates:
        if gate[0] == "h":
            bit = 1 << (n_qubits - 1 - gate[1])
            zeros = indices[(indices & bit) == 0]
            ones = zeros | bit
            low, high = state[zeros].copy(), state[ones].copy()
            state[zeros] = (low + high) / np.sqrt(2.0)
            state[ones] = (low - high) / np.sqrt(2.0)
        else:
            control = 1 << (n_qubits - 1 - gate[1])
            target = 1 << (n_qubits - 1 - gate[2])
            source = indices[(indices & control) != 0]
            state[source ^ target] = state[source].copy()
    return state


def _target_state(instance):
    n_qubits = instance["n_qubits"]
    x_basis, _ = _rref(instance["x_generators"])
    words = _span(x_basis)
    state = np.zeros(2**n_qubits, dtype=np.float64)
    for word in words:
        index = sum(int(bit) << (n_qubits - 1 - q) for q, bit in enumerate(word))
        state[index] = 1.0 / np.sqrt(len(words))
    return state


def _target_stabilizers(instance):
    n_qubits = instance["n_qubits"]
    x_rows = np.asarray(instance["x_generators"], dtype=np.uint8)
    z_rows = np.asarray(instance["z_generators"], dtype=np.uint8)
    x_part = np.concatenate([x_rows, np.zeros_like(x_rows)], axis=1)
    z_part = np.concatenate([np.zeros_like(z_rows), z_rows], axis=1)
    target = np.concatenate([x_part, z_part], axis=0)
    if _canonical(target, 2 * n_qubits).__len__() != n_qubits:
        raise ValueError("internal target stabilizer rank is invalid")
    return target


def _propagate_error(x, z, gates, start):
    x, z = x.copy(), z.copy()
    for gate in gates[start:]:
        _conjugate(x, z, gate)
    return x, z


def _all_final_errors(n_qubits, gates):
    errors = {}
    for qubit in range(n_qubits):
        x = np.zeros(n_qubits, dtype=np.uint8)
        z = np.zeros(n_qubits, dtype=np.uint8)
        x[qubit] = 1
        final = _propagate_error(x, z, gates, 0)
        errors[tuple(np.concatenate(final))] = final
    for index, gate in enumerate(gates):
        qubits = gate[1:]
        local_faults = (
            PAULIS[1:] if gate[0] == "h" else itertools.product(PAULIS, repeat=2)
        )
        for local in local_faults:
            if gate[0] == "h":
                local = (local,)
            if all(pair == (0, 0) for pair in local):
                continue
            x = np.zeros(n_qubits, dtype=np.uint8)
            z = np.zeros(n_qubits, dtype=np.uint8)
            for qubit, (x_value, z_value) in zip(qubits, local):
                x[qubit], z[qubit] = x_value, z_value
            final = _propagate_error(x, z, gates, index + 1)
            errors[tuple(np.concatenate(final))] = final
    return list(errors.values())


def _span(rows):
    group = np.zeros((1, rows.shape[1]), dtype=np.uint8)
    for row in rows:
        group = np.concatenate([group, group ^ row], axis=0)
    return group


def _pauli_label(x, z):
    return "".join(LABEL[(int(xv), int(zv))] for xv, zv in zip(x, z))


def _dangerous_errors(instance, gates):
    n_qubits = instance["n_qubits"]
    target = _target_stabilizers(instance)
    group = _span(target)
    gx, gz = group[:, :n_qubits], group[:, n_qubits:]
    dangerous = []
    for x, z in _all_final_errors(n_qubits, gates):
        weights = np.count_nonzero((gx ^ x) | (gz ^ z), axis=1)
        if int(np.min(weights)) > 1:
            dangerous.append((x, z))
    return sorted(dangerous, key=lambda error: _pauli_label(*error))


def _candidate_pauli(candidate, n_qubits):
    mask = np.asarray(candidate["mask"], dtype=np.uint8)
    if mask.shape != (n_qubits,) or np.any(mask > 1):
        raise ValueError("invalid candidate mask")
    zero = np.zeros(n_qubits, dtype=np.uint8)
    if candidate["kind"] == "x":
        return mask, zero
    if candidate["kind"] == "z":
        return zero, mask
    raise ValueError("candidate kind must be x or z")


def _syndrome_matrix(instance, dangerous):
    n_qubits = instance["n_qubits"]
    columns = [
        _candidate_pauli(candidate, n_qubits)
        for candidate in instance["candidate_checks"]
    ]
    return np.asarray(
        [
            [int((x @ check_z + z @ check_x) & 1) for check_x, check_z in columns]
            for x, z in dangerous
        ],
        dtype=bool,
    ).reshape(len(dangerous), len(columns))


def _best_cover(instance, syndrome):
    candidates = instance["candidate_checks"]
    for count in range(len(candidates) + 1):
        best = None
        for subset in itertools.combinations(range(len(candidates)), count):
            if len(syndrome) and not np.all(np.any(syndrome[:, subset], axis=1)):
                continue
            ids = tuple(sorted(candidates[index]["id"] for index in subset))
            cost = sum(candidates[index]["coupling_cost"] for index in subset)
            key = (cost, ids)
            if best is None or key < best:
                best = key
        if best is not None:
            return list(best[1])
    raise ValueError("candidate pool does not cover the dangerous faults")


def _evaluate_record(instance, record):
    n_qubits = instance["n_qubits"]
    gates = _parse_gates(record.get("encoder"), n_qubits, instance["gate_budget"])
    target = _target_stabilizers(instance)
    if _canonical(_encoder_stabilizers(n_qubits, gates), 2 * n_qubits) != _canonical(
        target, 2 * n_qubits
    ):
        raise ValueError("encoder does not prepare the target CSS state")
    overlap = abs(np.vdot(_target_state(instance), _encoder_state(n_qubits, gates)))
    if not np.isclose(overlap, 1.0, atol=2e-12, rtol=0):
        raise ValueError(
            "encoder has the target row space but incorrect stabilizer signs"
        )
    dangerous = _dangerous_errors(instance, gates)
    labels = [_pauli_label(*error) for error in dangerous]
    syndrome = _syndrome_matrix(instance, dangerous)
    reported_labels = record.get("dangerous_errors")
    reported_syndrome = np.asarray(record.get("fault_syndromes"), dtype=bool)
    expectations = np.asarray(record.get("target_expectations"), dtype=float)
    selected = record.get("selected_check_ids")
    return {
        "encoder valid and within gate budget": True,
        "dangerous fault labels exact": reported_labels == labels,
        "fault syndrome shape exact": reported_syndrome.shape == syndrome.shape,
        "fault syndromes match independent oracle": bool(
            reported_syndrome.shape == syndrome.shape
            and np.array_equal(reported_syndrome, syndrome)
        ),
        "target stabilizer expectations are +1": bool(
            expectations.shape == (len(instance["candidate_checks"]),)
            and np.all(np.isfinite(expectations))
            and np.allclose(expectations, 1.0, atol=2e-6, rtol=0)
        ),
        "selected checks are the exact lexicographic optimum": selected
        == _best_cover(instance, syndrome),
    }


def evaluate(module_name, seed):
    config = build_config(seed)
    started = time.perf_counter()
    try:
        result = importlib.import_module(module_name).run_solution(config)
        elapsed = time.perf_counter() - started
        records = result.get("instances")
        if not isinstance(records, list):
            raise ValueError("result instances must be a list")
        by_id = {}
        for record in records:
            if not isinstance(record, dict) or not isinstance(
                record.get("instance_id"), str
            ):
                raise ValueError("each result record needs an instance_id")
            if record["instance_id"] in by_id:
                raise ValueError("duplicate result instance_id")
            by_id[record["instance_id"]] = record
        expected_ids = {instance["instance_id"] for instance in config["instances"]}
        if set(by_id) != expected_ids:
            raise ValueError("result instance IDs do not match the hidden cases")
        per_instance = [
            _evaluate_record(instance, by_id[instance["instance_id"]])
            for instance in config["instances"]
        ]
        criteria = {
            key: all(record[key] for record in per_instance) for key in per_instance[0]
        }
        criteria["runtime below 180 seconds"] = elapsed < 180
        summary = {
            "instances": len(config["instances"]),
            "seed": seed,
            "case_digest": config["case_digest"],
        }
    except Exception as exc:
        elapsed = time.perf_counter() - started
        criteria = {"solution and independent evaluation complete": False}
        summary = {"error": f"{type(exc).__name__}: {exc}", "seed": seed}
    passed = all(criteria.values())
    print("Problem 106 evaluation")
    print(f"Solution module: {module_name}")
    print(f"End-to-end solution time: {elapsed:.6f}s")
    print(json.dumps(summary, sort_keys=True))
    print("Passing criteria:")
    for name, value in criteria.items():
        print(f"  {'PASS' if value else 'FAIL'}: {name}")
    print("Overall: PASS" if passed else "Overall: FAIL")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_106")
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("ORBIT_CSS_SEED", "1062026"))
    )
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.solution, args.seed) else 1)


if __name__ == "__main__":
    main()
