"""Independent design-only evaluator for a flagged CSS measurement gadget."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
import types

import numpy as np


DEFAULT_SEED = 1202026
DEFAULT_ORACLE_SEED = 91202026
PAULI_BITS = ((0, 0), (1, 0), (1, 1), (0, 1))
I2 = np.eye(2, dtype=np.complex128)
X2 = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
Y2 = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
Z2 = np.diag([1, -1]).astype(np.complex128)
PMAT = {"I": I2, "X": X2, "Y": Y2, "Z": Z2}
CNOT = np.asarray(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
    dtype=np.complex128,
)


def _canonical_json(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _rref(rows):
    a = np.asarray(rows, dtype=np.uint8).copy() & 1
    row = 0
    for column in range(a.shape[1]):
        choices = np.flatnonzero(a[row:, column])
        if not len(choices):
            continue
        pivot = row + int(choices[0])
        a[[row, pivot]] = a[[pivot, row]]
        for other in np.flatnonzero(a[:, column]):
            if other != row:
                a[other] ^= a[row]
        row += 1
        if row == len(a):
            break
    return a[:row]


def _scramble(rows, rng):
    out = np.asarray(rows, dtype=np.uint8).copy()
    for _ in range(24):
        a, b = rng.choice(len(out), 2, replace=False)
        if rng.random() < 0.7:
            out[a] ^= out[b]
        else:
            out[[a, b]] = out[[b, a]]
    return out


def _simplex():
    return np.asarray(
        [[(column >> row) & 1 for column in range(1, 8)] for row in range(3)],
        dtype=np.uint8,
    )


def _row_products(rows):
    return [
        np.bitwise_xor.reduce(
            np.asarray([rows[i] for i in range(len(rows)) if mask >> i & 1]), axis=0
        )
        for mask in range(1, 1 << len(rows))
    ]


def _opaque(rng, prefix, used):
    while True:
        value = f"{prefix}{int(rng.integers(100000, 1000000))}"
        if value not in used:
            used.add(value)
            return value


def _make_case(rng, ordinal, preferred_kind):
    base = _simplex()
    permutation = rng.permutation(7)
    code_x = base[:, permutation]
    code_z = code_x.copy()
    public_x, public_z = _scramble(code_x, rng), _scramble(code_z, rng)
    logical_z = np.ones(7, dtype=np.uint8)
    candidates, used = [], set()
    values = _row_products(code_x if preferred_kind == "x" else code_z)
    rng.shuffle(values)
    for value in values[:5]:
        candidates.append(
            {
                "id": _opaque(rng, "q", used),
                "kind": preferred_kind,
                "mask": value.astype(int).tolist(),
            }
        )
    rng.shuffle(candidates)
    variant_records = [
        {
            "id": _opaque(rng, "p", used),
            "coefficient_transform": [1, 1, 1],
            "echo_cost": 0,
        },
        {
            "id": _opaque(rng, "p", used),
            "coefficient_transform": [0.2, -0.4, 0.3],
            "echo_cost": 1,
        },
    ]
    rng.shuffle(variant_records)
    axis_triples = (
        ("XI", "ZX", "ZZ"),
        ("IX", "XZ", "YY"),
        ("ZI", "XY", "XX"),
        ("IZ", "YX", "ZZ"),
    )
    axes = [list(axis_triples[int(v)]) for v in rng.integers(0, len(axis_triples), 6)]
    coefficients = (
        2.5
        * (1.0 if ordinal == 0 else 1.3)
        * (0.031 + 0.028 * rng.random((6, 3)))
        * rng.choice([-1.0, 1.0], (6, 3))
    )
    edge_multipliers = 0.05 + 10.0 * rng.random((7, 3))
    payload = {
        "n_data": 7,
        "target_state": "logical_zero",
        "x_stabilizers": public_x.astype(int).tolist(),
        "z_stabilizers": public_z.astype(int).tolist(),
        "logical_z": logical_z.astype(int).tolist(),
        "check_candidates": candidates,
        "flag_windows": [[0, 2], [0, 3], [0, 4], [1, 3], [1, 4], [2, 4]],
        "cx_variants": variant_records,
        "echo_cost_budget": 5,
        "coherent_model": {
            "pauli_axes": axes,
            "nominal_coefficients": coefficients.tolist(),
            "data_edge_multipliers": edge_multipliers.tolist(),
        },
        "coherent_fidelity_floor": 0.948 if ordinal == 0 else 0.9498,
        "baseline_admission_margin": 0.001,
    }
    return {"case_id": f"fg-{ordinal}-{_digest(payload)[:12]}", **payload}


def build_public_config(seed):
    rng = np.random.default_rng(seed)
    cases = [_make_case(rng, 0, "x"), _make_case(rng, 1, "z")]
    config = {
        "schema_version": 1,
        "cases": cases,
        "effective_line_limit": 160,
        "observation_order": [
            "check",
            "flag",
            "x_stabilizer_0",
            "x_stabilizer_1",
            "x_stabilizer_2",
            "z_stabilizer_0",
            "z_stabilizer_1",
            "z_stabilizer_2",
        ],
    }
    config["public_case_digest"] = _digest(config)
    return config


def build_hidden_scenarios(config, oracle_seed):
    rng = np.random.default_rng(oracle_seed)
    records = []
    for case in config["cases"]:
        nominal = np.asarray(
            case["coherent_model"]["nominal_coefficients"], dtype=float
        )
        scenarios = [nominal.tolist()]
        for _ in range(6):
            jitter = rng.uniform(-0.055, 0.055, nominal.shape)
            common = float(rng.uniform(-0.025, 0.025))
            scenarios.append((nominal * (1.0 + jitter + common)).tolist())
        records.append({"case_id": case["case_id"], "angles": scenarios})
    return records


def _bits(row):
    return sum(int(value) << i for i, value in enumerate(row))


def _bitrow(value, width):
    return [int(value >> i & 1) for i in range(width)]


def _target_generators(case):
    return (
        [(0, _bits(row), 0) for row in case["x_stabilizers"]]
        + [(0, 0, _bits(row)) for row in case["z_stabilizers"]]
        + [(0, 0, _bits(case["logical_z"]))]
    )


def _pmul(first, second):
    phase, x, z = first
    other_phase, other_x, other_z = second
    return (
        (phase + other_phase + 2 * ((z & other_x).bit_count() & 1)) & 3,
        x ^ other_x,
        z ^ other_z,
    )


def _target_group(case):
    group = [(0, 0, 0)]
    for generator in _target_generators(case):
        group += [_pmul(value, generator) for value in group]
    return group


def _validate_case_contract(case):
    generators = _target_generators(case)
    if len(generators) != 7:
        raise ValueError("target state requires exactly seven signed generators")
    for _phase, x, z in generators:
        for _other_phase, other_x, other_z in generators:
            if ((x & other_z).bit_count() ^ (z & other_x).bit_count()) & 1:
                raise ValueError("target generators do not commute")
    group = _target_group(case)
    if len({(phase, x, z) for phase, x, z in group}) != 128:
        raise ValueError("target signed stabilizer is not independent and maximal")
    if any((phase - (x & z).bit_count()) & 1 for phase, x, z in group):
        raise ValueError("target stabilizer contains a non-Hermitian signed row")


def _coset_weight(case, x, z):
    return min(
        ((x ^ sx) | (z ^ sz)).bit_count() for _phase, sx, sz in _target_group(case)
    )


def _check_positive(case, kind, mask):
    value = (0, _bits(mask), 0) if kind == "x" else (0, 0, _bits(mask))
    return value in set(_target_group(case)) and value != (0, 0, 0)


def _syndrome(x, z, generators):
    return tuple(
        ((x & gz).bit_count() ^ (z & gx).bit_count()) & 1 for _, gx, gz in generators
    )


def _correctable_syndromes(case):
    generators = _target_generators(case)
    values = {_syndrome(0, 0, generators)}
    for qubit in range(case["n_data"]):
        bit = 1 << qubit
        values |= {
            _syndrome(bit, 0, generators),
            _syndrome(0, bit, generators),
            _syndrome(bit, bit, generators),
        }
    return values


def _entanglers(case, gadget):
    first, second = gadget["flag_window"]
    variants = iter(gadget["variant_ids"])
    records = []
    for slot in range(5):
        if slot in (first, second):
            records.append(("flag", None, next(variants)))
        if slot < 4:
            records.append(("data", gadget["data_order"][slot], next(variants)))
    if len(records) != 6:
        raise ValueError("flag window did not produce six entanglers")
    return records


def _pair(kind, role, data, syndrome, flag):
    if kind == "z":
        return (flag, syndrome) if role == "flag" else (data, syndrome)
    return (syndrome, flag) if role == "flag" else (syndrome, data)


def _conj_cx(x, z, sign, control, target):
    xc, zc, xt, zt = x[control], z[control], x[target], z[target]
    sign ^= int(xc & zt & (xt ^ zc ^ 1))
    x[target] ^= xc
    z[control] ^= zt
    return sign


def _propagate(x, z, sign, remaining, kind, syndrome, flag):
    for role, data, _ in remaining:
        control, target = _pair(kind, role, data, syndrome, flag)
        sign = _conj_cx(x, z, sign, control, target)
    return x, z, sign


def _fault_branches(case, gadget):
    n, syndrome, flag = case["n_data"], case["n_data"], case["n_data"] + 1
    records = _entanglers(case, gadget)
    code_generators = _target_generators(case)[:6]
    branches = []

    def add(x, z, sign=0, check_flip=0, flag_flip=0, label="fault"):
        check = int(x[syndrome]) if gadget["kind"] == "z" else int(z[syndrome])
        flagged = int(z[flag]) if gadget["kind"] == "z" else int(x[flag])
        data_x, data_z = _bits(x[:n]), _bits(z[:n])
        observation = (check ^ check_flip, flagged ^ flag_flip) + _syndrome(
            data_x, data_z, code_generators
        )
        branches.append((observation, data_x, data_z, sign, label))

    add(np.zeros(n + 2, np.uint8), np.zeros(n + 2, np.uint8), label="ideal")
    for ancilla, name in ((syndrome, "syndrome-prep"), (flag, "flag-prep")):
        for px, pz in PAULI_BITS[1:]:
            x, z = np.zeros(n + 2, np.uint8), np.zeros(n + 2, np.uint8)
            x[ancilla], z[ancilla] = px, pz
            add(
                *_propagate(x, z, 0, records, gadget["kind"], syndrome, flag),
                label=name,
            )
    for index, record in enumerate(records):
        control, target = _pair(gadget["kind"], record[0], record[1], syndrome, flag)
        for left, right in itertools.product(PAULI_BITS, repeat=2):
            if left == right == (0, 0):
                continue
            x, z = np.zeros(n + 2, np.uint8), np.zeros(n + 2, np.uint8)
            x[control], z[control] = left
            x[target], z[target] = right
            add(
                *_propagate(
                    x, z, 0, records[index + 1 :], gadget["kind"], syndrome, flag
                ),
                label=f"cx-{index}",
            )
    zeros = np.zeros(n + 2, np.uint8)
    add(zeros.copy(), zeros.copy(), check_flip=1, label="check-measurement")
    add(zeros.copy(), zeros.copy(), flag_flip=1, label="flag-measurement")
    return branches


def _apply_one(state, gate, qubit, n_qubits):
    tensor = state.reshape((2,) * n_qubits)
    moved = np.moveaxis(tensor, qubit, 0).reshape(2, -1)
    moved = gate @ moved
    return np.moveaxis(moved.reshape((2,) + (2,) * (n_qubits - 1)), 0, qubit).reshape(
        -1
    )


def _apply_two(state, gate, first, second, n_qubits):
    order = [first, second] + [q for q in range(n_qubits) if q not in (first, second)]
    inverse = np.argsort(order)
    tensor = state.reshape((2,) * n_qubits).transpose(order).reshape(4, -1)
    tensor = (gate @ tensor).reshape((2, 2) + (2,) * (n_qubits - 2))
    return tensor.transpose(inverse).reshape(-1)


def _data_state(case):
    rows = _rref(case["x_stabilizers"])
    values = [np.zeros(case["n_data"], dtype=np.uint8)]
    for row in rows:
        values += [value ^ row for value in values]
    state = np.zeros(1 << case["n_data"], dtype=np.complex128)
    for value in values:
        index = sum(int(bit) << (case["n_data"] - 1 - q) for q, bit in enumerate(value))
        state[index] = 1 / np.sqrt(len(values))
    return state


def _initial_state(case, kind):
    zero = np.asarray([1.0, 0.0], dtype=np.complex128)
    plus = np.asarray([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
    syndrome, flag = (zero, plus) if kind == "z" else (plus, zero)
    return np.kron(np.kron(_data_state(case), syndrome), flag)


def _variant_map(case):
    return {record["id"]: record for record in case["cx_variants"]}


def _simulate(case, gadget, coefficients):
    n_total = case["n_data"] + 2
    syndrome, flag = case["n_data"], case["n_data"] + 1
    state = _initial_state(case, gadget["kind"])
    initial = state.copy()
    variants = _variant_map(case)
    axes = case["coherent_model"]["pauli_axes"]
    edge_multipliers = np.asarray(
        case["coherent_model"]["data_edge_multipliers"], dtype=float
    )
    for index, (role, data, variant_id) in enumerate(_entanglers(case, gadget)):
        control, target = _pair(gadget["kind"], role, data, syndrome, flag)
        state = _apply_two(state, CNOT, control, target, n_total)
        transform = np.asarray(
            variants[variant_id]["coefficient_transform"], dtype=float
        )
        edge = np.ones(3) if role == "flag" else edge_multipliers[data]
        hamiltonian = sum(
            transform[k]
            * float(coefficients[index][k])
            * edge[k]
            * np.kron(PMAT[axis[0]], PMAT[axis[1]])
            for k, axis in enumerate(axes[index])
        )
        values, vectors = np.linalg.eigh(hamiltonian)
        error = (vectors * np.exp(-0.5j * values)) @ vectors.conj().T
        state = _apply_two(state, error, control, target, n_total)
    return float(abs(np.vdot(initial, state)) ** 2)


def _strict_binary_list(value, length, label):
    if type(value) is not list or len(value) != length:
        raise ValueError(f"{label} must be an exact list of length {length}")
    if any(type(item) is not int or item not in (0, 1) for item in value):
        raise ValueError(f"{label} must contain exact binary integers")
    return value


def _parse_output(config, output):
    if type(output) is not dict or set(output) != {"cases"}:
        raise ValueError("output must contain exactly cases")
    if type(output["cases"]) is not list or len(output["cases"]) != len(
        config["cases"]
    ):
        raise ValueError("output cases must preserve exact case count")
    parsed = []
    expected_observations = [list(bits) for bits in itertools.product((0, 1), repeat=8)]
    for case, result in zip(config["cases"], output["cases"]):
        _validate_case_contract(case)
        if type(result) is not dict or set(result) != {"case_id", "gadget", "decoder"}:
            raise ValueError("case result has wrong fields")
        if type(result["case_id"]) is not str or result["case_id"] != case["case_id"]:
            raise ValueError("case identity/order mismatch")
        gadget = result["gadget"]
        required = {
            "check_id",
            "kind",
            "mask",
            "data_order",
            "flag_window",
            "variant_ids",
        }
        if type(gadget) is not dict or set(gadget) != required:
            raise ValueError("gadget has wrong fields")
        if type(gadget["check_id"]) is not str or type(gadget["kind"]) is not str:
            raise ValueError("check_id and kind must be exact strings")
        candidate = next(
            (
                record
                for record in case["check_candidates"]
                if record["id"] == gadget["check_id"]
            ),
            None,
        )
        if (
            candidate is None
            or gadget["kind"] != candidate["kind"]
            or gadget["mask"] != candidate["mask"]
        ):
            raise ValueError("gadget check does not exactly match a supplied candidate")
        _strict_binary_list(gadget["mask"], case["n_data"], "mask")
        if not _check_positive(case, gadget["kind"], gadget["mask"]):
            raise ValueError(
                "selected check is not a nonidentity positive target stabilizer"
            )
        support = [q for q, bit in enumerate(gadget["mask"]) if bit]
        if type(gadget["data_order"]) is not list or any(
            type(q) is not int for q in gadget["data_order"]
        ):
            raise ValueError("data_order must contain exact integers")
        if sorted(gadget["data_order"]) != support or len(gadget["data_order"]) != 4:
            raise ValueError("data_order must be the exact weight-four support")
        if type(gadget["flag_window"]) is not list or any(
            type(q) is not int for q in gadget["flag_window"]
        ):
            raise ValueError("flag_window must contain exact integers")
        if gadget["flag_window"] not in case["flag_windows"]:
            raise ValueError("flag_window is not allowed")
        variants = _variant_map(case)
        if type(gadget["variant_ids"]) is not list or len(gadget["variant_ids"]) != 6:
            raise ValueError("variant_ids must be an exact six-entry list")
        if any(
            type(value) is not str or value not in variants
            for value in gadget["variant_ids"]
        ):
            raise ValueError("unknown or non-string variant id")
        if (
            sum(variants[value]["echo_cost"] for value in gadget["variant_ids"])
            > case["echo_cost_budget"]
        ):
            raise ValueError("echo cost exceeds budget")
        decoder = result["decoder"]
        if type(decoder) is not list or len(decoder) != 256:
            raise ValueError("decoder must contain all 256 observations")
        table = {}
        for expected, record in zip(expected_observations, decoder):
            if type(record) is not dict or set(record) != {"observation", "recovery"}:
                raise ValueError("decoder row has wrong fields")
            if record["observation"] != expected:
                raise ValueError(
                    "decoder observations must be unique lexicographic rows"
                )
            _strict_binary_list(record["observation"], 8, "observation")
            recovery = record["recovery"]
            if type(recovery) is not dict or set(recovery) != {"x", "z"}:
                raise ValueError("recovery has wrong fields")
            _strict_binary_list(recovery["x"], case["n_data"], "recovery.x")
            _strict_binary_list(recovery["z"], case["n_data"], "recovery.z")
            table[tuple(expected)] = (_bits(recovery["x"]), _bits(recovery["z"]))
        parsed.append((case, gadget, table))
    return parsed


def validate_output(config, hidden, output):
    parsed = _parse_output(config, output)
    hidden_map = {record["case_id"]: record["angles"] for record in hidden}
    metrics = []
    for case, gadget, table in parsed:
        ideal_fidelity = _simulate(case, gadget, np.zeros((6, 3)))
        if ideal_fidelity < 1 - 2e-12:
            raise ValueError("ideal returned gadget does not preserve the target state")
        branches = _fault_branches(case, gadget)
        reachable = {record[0] for record in branches}
        for observation, error_x, error_z, _phase, label in branches:
            if _coset_weight(case, error_x, error_z) > 1 and observation[1] != 1:
                raise ValueError(f"dangerous raw branch {label} did not raise the flag")
            recovery_x, recovery_z = table[observation]
            if _coset_weight(case, error_x ^ recovery_x, error_z ^ recovery_z) > 1:
                raise ValueError(f"one common recovery fails reachable branch {label}")
        for observation, recovery in table.items():
            if observation not in reachable and recovery != (0, 0):
                raise ValueError(
                    "unreachable observations require canonical identity recovery"
                )
        fidelities = [
            _simulate(case, gadget, angles) for angles in hidden_map[case["case_id"]]
        ]
        minimum = min(fidelities)
        if minimum < case["coherent_fidelity_floor"]:
            raise ValueError(
                "returned implementation misses hidden coherent-fidelity floor"
            )
        metrics.append(
            {
                "case_id": case["case_id"],
                "reachable_observations": len(reachable),
                "fault_branches": len(branches),
                "minimum_coherent_fidelity": minimum,
                "ideal_fidelity": ideal_fidelity,
            }
        )
    return metrics


def _install_tc_shim():
    module = types.ModuleType("tensorcircuit")

    class Circuit:
        def __init__(self, nqubits, inputs=None):
            self.nqubits = nqubits
            self._state = np.asarray(inputs, dtype=np.complex128).copy()
            if inputs is None:
                self._state = np.zeros(1 << nqubits, dtype=np.complex128)
                self._state[0] = 1

        def h(self, qubit):
            self._state = _apply_one(
                self._state, (X2 + Z2) / np.sqrt(2), qubit, self.nqubits
            )

        def cnot(self, control, target):
            self._state = _apply_two(self._state, CNOT, control, target, self.nqubits)

        def unitary(self, first, second=None, unitary=None, **_kwargs):
            sites = first if isinstance(first, (tuple, list)) else (first, second)
            self._state = _apply_two(
                self._state, np.asarray(unitary), sites[0], sites[1], self.nqubits
            )

        def state(self):
            return self._state

    module.Circuit = Circuit
    module.gates = types.SimpleNamespace(Gate=lambda value: np.asarray(value))
    module.set_backend = lambda _value: None
    module.set_dtype = lambda _value: None
    sys.modules["tensorcircuit"] = module


def _solution_worker(connection, module_name, config, use_shim):
    try:
        if use_shim:
            _install_tc_shim()
        solution = importlib.import_module(module_name)
        connection.send(("ok", solution.run_solution(config)))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


def _run_isolated(module_name, config, use_shim, timeout=300):
    context = mp.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_solution_worker, args=(child, module_name, config, use_shim)
    )
    process.start()
    child.close()
    if not parent.poll(timeout):
        process.terminate()
        process.join(5)
        raise TimeoutError("solution worker exceeded timeout")
    status, payload = parent.recv()
    process.join(5)
    if process.exitcode != 0 or status != "ok":
        raise RuntimeError(f"isolated solution failed: {payload}")
    return payload


def _dense_two_canary():
    rng = np.random.default_rng(120)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    unitary, _ = np.linalg.qr(raw)
    state = rng.normal(size=8) + 1j * rng.normal(size=8)
    state /= np.linalg.norm(state)
    full = np.zeros((8, 8), dtype=np.complex128)
    for column in range(8):
        bits = [(column >> (2 - q)) & 1 for q in range(3)]
        source = 2 * bits[2] + bits[0]
        for target_bits in range(4):
            out = bits.copy()
            out[2], out[0] = target_bits >> 1, target_bits & 1
            row = sum(bit << (2 - q) for q, bit in enumerate(out))
            full[row, column] = unitary[target_bits, source]
    error = float(np.max(np.abs(_apply_two(state, unitary, 2, 0, 3) - full @ state)))
    if error > 2e-14:
        raise AssertionError(f"dense two-site canary failed: {error}")
    return error


def _phase_canary():
    labels = ("I", "X", "Y", "Z")
    maximum = 0.0
    for left, right in itertools.product(range(4), repeat=2):
        x = np.asarray([PAULI_BITS[left][0], PAULI_BITS[right][0]], np.uint8)
        z = np.asarray([PAULI_BITS[left][1], PAULI_BITS[right][1]], np.uint8)
        sign = _conj_cx(x, z, 0, 0, 1)
        actual = ((-1) ** sign) * np.kron(
            labels_matrix(x[0], z[0]), labels_matrix(x[1], z[1])
        )
        before = np.kron(PMAT[labels[left]], PMAT[labels[right]])
        maximum = max(maximum, float(np.max(np.abs(actual - CNOT @ before @ CNOT))))
    if maximum > 1e-14:
        raise AssertionError(f"phase-aware CNOT canary failed: {maximum}")
    return maximum


def _coset_syndrome_canary(config):
    mismatches = 0
    for case in config["cases"]:
        generators = _target_generators(case)
        stabilizers = [(x, z) for _phase, x, z in _target_group(case)]
        correctable = _correctable_syndromes(case)
        for labels in itertools.product(range(4), repeat=case["n_data"]):
            x = sum(PAULI_BITS[value][0] << qubit for qubit, value in enumerate(labels))
            z = sum(PAULI_BITS[value][1] << qubit for qubit, value in enumerate(labels))
            explicit = (
                min(((x ^ sx) | (z ^ sz)).bit_count() for sx, sz in stabilizers) <= 1
            )
            if explicit != (_syndrome(x, z, generators) in correctable):
                mismatches += 1
    if mismatches:
        raise AssertionError(
            f"target-coset/syndrome canary found {mismatches} mismatches"
        )
    return mismatches


def labels_matrix(x, z):
    return PMAT[{(0, 0): "I", (1, 0): "X", (1, 1): "Y", (0, 1): "Z"}[(int(x), int(z))]]


def _decoder_context(case):
    generators = _target_generators(case)
    representatives = {}
    for labels in itertools.product(range(4), repeat=case["n_data"]):
        x = sum(PAULI_BITS[value][0] << qubit for qubit, value in enumerate(labels))
        z = sum(PAULI_BITS[value][1] << qubit for qubit, value in enumerate(labels))
        syndrome = _syndrome(x, z, generators)
        key = (sum(value != 0 for value in labels), labels)
        if syndrome not in representatives or key < representatives[syndrome][0]:
            representatives[syndrome] = (key, x, z)
    return generators, representatives, _correctable_syndromes(case)


def _canonical_decoder(case, gadget, context):
    generators, representatives, correctable = context
    groups = {}
    for observation, x, z, _phase, _label in _fault_branches(case, gadget):
        if _coset_weight(case, x, z) > 1 and observation[1] != 1:
            return None
        groups.setdefault(observation, []).append(_syndrome(x, z, generators))
    recoveries = {}
    ordered_representatives = sorted(representatives.values())
    for observation, errors in groups.items():
        for _key, x, z in ordered_representatives:
            recovery_syndrome = _syndrome(x, z, generators)
            if all(
                tuple(a ^ b for a, b in zip(error, recovery_syndrome)) in correctable
                for error in errors
            ):
                recoveries[observation] = (x, z)
                break
        if observation not in recoveries:
            return None
    rows = []
    for observation in itertools.product((0, 1), repeat=8):
        x, z = recoveries.get(observation, (0, 0))
        rows.append(
            {
                "observation": list(observation),
                "recovery": {
                    "x": _bitrow(x, case["n_data"]),
                    "z": _bitrow(z, case["n_data"]),
                },
            }
        )
    return rows


def _variant_assignments(case):
    variants = case["cx_variants"]
    return [
        [variants[index]["id"] for index in choices]
        for choices in itertools.product(range(len(variants)), repeat=6)
        if sum(variants[index]["echo_cost"] for index in choices)
        <= case["echo_cost_budget"]
    ]


def _baseline_search(case, mode):
    context = _decoder_context(case)
    plain = next(
        value["id"] for value in case["cx_variants"] if value["echo_cost"] == 0
    )
    costly = next(
        value["id"] for value in case["cx_variants"] if value["echo_cost"] == 1
    )
    checks = (
        case["check_candidates"][:1]
        if mode in ("first", "reviewer")
        else case["check_candidates"]
    )
    assignments = [[plain] * 6] if mode == "cheap" else _variant_assignments(case)
    if mode == "reviewer":
        assignments = [
            [costly if index in (1, 3, 4, 5) else plain for index in range(6)]
        ]
    nominal = case["coherent_model"]["nominal_coefficients"]
    best = None
    for check in checks:
        support = [qubit for qubit, bit in enumerate(check["mask"]) if bit]
        if mode in ("sorted", "reviewer"):
            orders = [tuple(sorted(support))]
        elif mode == "reversed":
            orders = [tuple(sorted(support, reverse=True))]
        else:
            orders = itertools.permutations(support)
        for order in orders:
            windows = [[0, 3]] if mode == "reviewer" else case["flag_windows"]
            for window in windows:
                base = {
                    "check_id": check["id"],
                    "kind": check["kind"],
                    "mask": check["mask"],
                    "data_order": list(order),
                    "flag_window": window,
                    "variant_ids": assignments[0],
                }
                decoder = _canonical_decoder(case, base, context)
                if decoder is None:
                    continue
                for variant_ids in assignments:
                    gadget = {**base, "variant_ids": variant_ids}
                    key = (_simulate(case, gadget, nominal), _canonical_json(gadget))
                    if best is None or key > best[0]:
                        best = (
                            key,
                            {
                                "case_id": case["case_id"],
                                "gadget": gadget,
                                "decoder": decoder,
                            },
                        )
    if best is None:
        raise AssertionError(
            f"baseline {mode} produced no complete fault-valid artifact"
        )
    return best[0][0], best[1]


def _admission_baselines(config):
    scores, artifacts = {}, {}
    for case_index, case in enumerate(config["cases"]):
        for mode in ("cheap", "sorted", "reversed", "first", "reviewer"):
            score, complete_case = _baseline_search(case, mode)
            margin = case["coherent_fidelity_floor"] - score
            if margin < case["baseline_admission_margin"]:
                raise AssertionError(
                    f"baseline {mode} lacks admission margin: {margin}"
                )
            key = f"{case['case_id']}:{mode}"
            scores[key], artifacts[(case_index, mode)] = score, complete_case
    return scores, artifacts


def _flag_dependence_canary(config, output):
    no_flag_hooks = 0
    unsafe_windows = 0
    for case, result in zip(config["cases"], output["cases"]):
        gadget = result["gadget"]
        records = _entanglers(case, gadget)
        for _observation, x, z, _phase, label in _fault_branches(case, gadget):
            if label.startswith("cx-"):
                gate_index = int(label[3:])
                if records[gate_index][0] == "data" and _coset_weight(case, x, z) > 1:
                    no_flag_hooks += 1
                    break
        context = _decoder_context(case)
        for window in case["flag_windows"]:
            if window != gadget["flag_window"]:
                changed = {**gadget, "flag_window": window}
                if _canonical_decoder(case, changed, context) is None:
                    unsafe_windows += 1
                    break
    if no_flag_hooks != len(config["cases"]) or unsafe_windows != len(config["cases"]):
        raise AssertionError(
            "development cases do not make flag presence and window functional"
        )
    return {
        "no_flag_dangerous_hooks": no_flag_hooks,
        "unsafe_wrong_windows": unsafe_windows,
    }


def _expect_rejected(config, hidden, output):
    try:
        validate_output(config, hidden, output)
    except (TypeError, ValueError):
        return True
    raise AssertionError("semantic/schema mutation unexpectedly passed")


def _mutation_canaries(config, hidden, output, baseline_artifacts):
    rejected = 0
    mutation = copy.deepcopy(output)
    mutation["extra"] = 1
    rejected += _expect_rejected(config, hidden, mutation)
    mutation = copy.deepcopy(output)
    mutation["cases"][0]["gadget"] = {}
    rejected += _expect_rejected(config, hidden, mutation)
    mutation = copy.deepcopy(output)
    mutation["cases"][0]["gadget"]["mask"][0] ^= 1
    rejected += _expect_rejected(config, hidden, mutation)
    for case_index, _case in enumerate(config["cases"]):
        for mode in ("cheap", "sorted", "reversed", "first", "reviewer"):
            mutation = copy.deepcopy(output)
            mutation["cases"][case_index] = baseline_artifacts[(case_index, mode)]
            rejected += _expect_rejected(config, hidden, mutation)
    parsed = _parse_output(config, output)
    reachable = {branch[0] for branch in _fault_branches(parsed[0][0], parsed[0][1])}
    changed = False
    for index, row in enumerate(output["cases"][0]["decoder"]):
        if tuple(row["observation"]) in reachable and any(
            row["recovery"]["x"] + row["recovery"]["z"]
        ):
            mutation = copy.deepcopy(output)
            mutation["cases"][0]["decoder"][index]["recovery"] = {
                "x": [0] * 7,
                "z": [0] * 7,
            }
            rejected += _expect_rejected(config, hidden, mutation)
            changed = True
            break
    if not changed:
        raise AssertionError("development case lacks a nonidentity reachable recovery")
    return rejected


def run_evaluation(module_name, seed, oracle_seed, use_shim, self_test=False):
    config = build_public_config(seed)
    hidden = build_hidden_scenarios(config, oracle_seed)
    hidden_commitment = _digest(hidden)
    full_digest = _digest({"public": config, "hidden_commitment": hidden_commitment})
    print(
        json.dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": seed,
                    "case_digest": full_digest,
                }
            }
        ),
        flush=True,
    )
    baseline_scores, baseline_artifacts = _admission_baselines(config)
    for name in (
        "ORBIT_Q_CANDIDATE_SEED",
        "ORBIT_Q_ORACLE_SEED",
        "ORBIT_FLAGGED_CSS_SEED",
    ):
        os.environ.pop(name, None)
    sys.argv[:] = [sys.argv[0]]
    started = time.monotonic()
    output = _run_isolated(module_name, copy.deepcopy(config), use_shim)
    runtime = time.monotonic() - started
    metrics = validate_output(config, hidden, output)
    result = {"runtime_sec": runtime, "metrics": metrics, "case_digest": full_digest}
    if self_test:
        result["dense_two_site_max_error"] = _dense_two_canary()
        result["phase_cnot_max_error"] = _phase_canary()
        result["coset_syndrome_mismatches"] = _coset_syndrome_canary(config)
        result["flag_dependence"] = _flag_dependence_canary(config, output)
        result["baseline_nominal_scores"] = baseline_scores
        result["mutations_rejected"] = _mutation_canaries(
            config, hidden, output, baseline_artifacts
        )
    print(json.dumps(result, sort_keys=True), flush=True)
    print("Overall: PASS", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_120")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--oracle-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--shim", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    arguments = parser.parse_args()
    public_env = os.environ.get("ORBIT_Q_CANDIDATE_SEED")
    oracle_env = os.environ.get("ORBIT_Q_ORACLE_SEED")
    if (
        public_env is not None
        and arguments.seed is not None
        and arguments.seed != int(public_env)
    ):
        parser.error("--seed conflicts with verifier-owned ORBIT_Q_CANDIDATE_SEED")
    if (
        oracle_env is not None
        and arguments.oracle_seed is not None
        and arguments.oracle_seed != int(oracle_env)
    ):
        parser.error("oracle seed override conflicts with verifier-owned entropy")
    arguments.seed = (
        int(public_env)
        if public_env is not None
        else (DEFAULT_SEED if arguments.seed is None else arguments.seed)
    )
    arguments.oracle_seed = (
        int(oracle_env)
        if oracle_env is not None
        else (
            DEFAULT_ORACLE_SEED
            if arguments.oracle_seed is None
            else arguments.oracle_seed
        )
    )
    if arguments.self_test:
        expert_dir = os.path.join(os.path.dirname(__file__), "expert")
        if expert_dir not in sys.path:
            sys.path.insert(0, expert_dir)
        arguments.shim = True
    run_evaluation(
        arguments.solution,
        arguments.seed,
        arguments.oracle_seed,
        arguments.shim,
        self_test=arguments.self_test,
    )


if __name__ == "__main__":
    main()
