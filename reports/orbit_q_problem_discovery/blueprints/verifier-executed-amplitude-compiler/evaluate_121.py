"""Independent verifier for candidate 121's executable contraction bytecode."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.util
import itertools
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np


DEFAULT_SEED = 1212026
N_QUBITS = 64
ROWS = 8
COLS = 8
GROUP_COUNT = 6
LANES = 8
PREFIX = 24
LANE = "@lane"
SCHEMA = "orbit-q/batched-amplitude-elimination/v1"
MAX_ARTIFACT_BYTES = 1_000_000
ROOT_KEYS = {
    "schema",
    "case_id",
    "case_digest",
    "source_tensor_ids",
    "lane_groups",
    "slice_indices",
    "steps",
    "terminal_inputs",
    "certificates",
}
CERTIFICATE_KEYS = {
    "slice_count",
    "eliminate_operations",
    "terminal_contractions",
    "peak_live_elements",
    "max_intermediate_elements",
    "complex_fmas",
    "program_digest",
}


def _canonical(value):
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _opaque(rng, prefix):
    return f"{prefix}-{int(rng.integers(0, 1 << 63)):016x}"


def _ordered(indices):
    return sorted(index for index in indices if index != LANE) + (
        [LANE] if LANE in indices else []
    )


def _size(indices, lanes=LANES):
    return 2 ** sum(index != LANE for index in indices) * (
        lanes if LANE in indices else 1
    )


def _grid_edges(layout):
    layers = [[] for _ in range(4)]
    for row in range(ROWS):
        for column in range(COLS - 1):
            layers[column % 2].append(
                [layout[row * COLS + column], layout[row * COLS + column + 1]]
            )
    for row in range(ROWS - 1):
        for column in range(COLS):
            layers[2 + row % 2].append(
                [layout[row * COLS + column], layout[(row + 1) * COLS + column]]
            )
    return layers


def _expected_groups(case):
    buckets = {}
    width = case["shared_prefix_qubits"]
    for query in case["queries"]:
        buckets.setdefault(tuple(query["bits"][:width]), []).append(query["query_id"])
    return [
        {
            "prefix_bits": list(prefix),
            "query_ids": sorted(identities),
            "lane_width": len(identities),
        }
        for prefix, identities in sorted(buckets.items())
    ]


def _source_specs(case):
    variables = case["variable_ids"]
    prefix = case["shared_prefix_qubits"]
    rows = []
    for qubit, identity in enumerate(case["site_tensor_ids"]):
        rows.append((identity, [variables[qubit]] + ([] if qubit < prefix else [LANE])))
    for gate in case["entanglers"]:
        rows.append((gate["gate_id"], [variables[qubit] for qubit in gate["qubits"]]))
    return rows


def _simulate_plan(case, sliced):
    live = {
        identity: [axis for axis in axes if axis not in sliced]
        for identity, axes in _source_specs(case)
    }
    peak = sum(_size(axes) for axes in live.values())
    maximum = max(_size(axes) for axes in live.values())
    steps, order, fmas = [], [], 0
    remaining = set(case["variable_ids"]) - set(sliced)
    while remaining:
        choices = []
        for index in remaining:
            inputs = sorted(
                identity for identity, axes in live.items() if index in axes
            )
            scope = _ordered({axis for identity in inputs for axis in live[identity]})
            output = [axis for axis in scope if axis != index]
            choices.append((len(output) - (LANE in output), _size(scope), index))
        index = min(choices)[2]
        inputs = sorted(identity for identity, axes in live.items() if index in axes)
        scope = _ordered({axis for identity in inputs for axis in live[identity]})
        output_axes = [axis for axis in scope if axis != index]
        output = f"tmp-{len(steps):03d}"
        fmas += (len(inputs) - 1) * _size(scope) + _size(output_axes)
        for identity in inputs:
            del live[identity]
        live[output] = output_axes
        steps.append(["ELIM", index, inputs, output])
        order.append(index)
        maximum = max(maximum, _size(output_axes))
        peak = max(peak, sum(_size(axes) for axes in live.values()))
        remaining.remove(index)
    terminal = sorted(live)
    fmas += max(0, len(terminal) - 1) * LANES
    return steps, terminal, peak, maximum, fmas, order


def _reference_resources(case):
    candidates = []
    for sliced in itertools.combinations(sorted(case["slice_candidate_ids"]), 2):
        plan = _simulate_plan(case, sliced)
        candidates.append(((plan[2], plan[4], sliced, tuple(plan[5])), plan))
    _, plan = min(candidates)
    repetitions = GROUP_COUNT * 4
    return {
        "peak": plan[2],
        "maximum": plan[3],
        "fmas": plan[4] * repetitions,
    }


def build_config(seed):
    rng = np.random.default_rng(seed)
    layout = rng.permutation(N_QUBITS).tolist()
    variables = [_opaque(rng, "wire") for _ in range(N_QUBITS)]
    sites = [_opaque(rng, "site") for _ in range(N_QUBITS)]
    local = []
    for _ in range(N_QUBITS):
        local.append(
            {
                "initial_ry": float(rng.uniform(-1.25, 1.25)),
                "initial_rz": float(rng.uniform(-0.9, 0.9)),
                "final_rx": float(rng.uniform(-1.15, 1.15)),
                "final_ry": float(rng.uniform(-1.0, 1.0)),
            }
        )
    entanglers = []
    for layer, edges in enumerate(_grid_edges(layout)):
        rng.shuffle(edges)
        for edge in edges:
            entanglers.append(
                {
                    "gate_id": _opaque(rng, "rzz"),
                    "layer": layer,
                    "qubits": edge,
                    "theta": float(rng.uniform(-0.82, 0.82)),
                }
            )
    queries = []
    prefixes = set()
    while len(prefixes) < GROUP_COUNT:
        prefixes.add(tuple(rng.integers(0, 2, size=PREFIX).tolist()))
    for prefix in sorted(prefixes):
        suffixes = set()
        while len(suffixes) < LANES:
            suffixes.add(tuple(rng.integers(0, 2, size=N_QUBITS - PREFIX).tolist()))
        for suffix in suffixes:
            queries.append(
                {
                    "query_id": _opaque(rng, "amp"),
                    "bits": list(prefix + suffix),
                }
            )
    rng.shuffle(queries)
    central = [layout[row * COLS + column] for row in (2, 3, 4, 5) for column in (3, 4)]
    case = {
        "case_id": _opaque(rng, "amplitude-compiler"),
        "n_qubits": N_QUBITS,
        "grid_shape": [ROWS, COLS],
        "logical_to_physical": layout,
        "variable_ids": variables,
        "site_tensor_ids": sites,
        "local_gates": local,
        "entanglers": entanglers,
        "queries": queries,
        "shared_prefix_qubits": PREFIX,
        "lane_group_count": GROUP_COUNT,
        "lanes_per_group": LANES,
        "required_slice_variables": 2,
        "slice_candidate_ids": sorted(variables[qubit] for qubit in central),
        "case_nonce": int(rng.integers(0, 1 << 31)),
    }
    resources = _reference_resources(case)
    case["max_peak_live_elements"] = int(np.ceil(resources["peak"] * 1.08))
    case["max_intermediate_elements"] = int(np.ceil(resources["maximum"] * 1.08))
    case["max_complex_fmas"] = int(np.ceil(resources["fmas"] * 1.08))
    config = {
        "case": case,
        "bytecode_schema": SCHEMA,
        "output_lane_index": LANE,
        "resource_counting": "binary-index exact live entries and declared complex-FMA model",
    }
    config["case_digest"] = _digest(config)
    return config


def _is_int(value):
    return type(value) is int


def _is_string_list(value):
    return type(value) is list and all(type(item) is str for item in value)


def _core(artifact):
    return {key: artifact[key] for key in artifact if key != "certificates"}


def _validate_structure(artifact, config):
    case = config["case"]
    if type(artifact) is not dict or set(artifact) != ROOT_KEYS:
        raise ValueError("artifact must be an exact built-in dict with exact root keys")
    if (
        type(artifact["schema"]) is not str
        or artifact["schema"] != SCHEMA
        or type(artifact["case_id"]) is not str
        or artifact["case_id"] != case["case_id"]
        or type(artifact["case_digest"]) is not str
        or artifact["case_digest"] != config["case_digest"]
    ):
        raise ValueError("case identity or bytecode schema mismatch")
    expected_sources = [identity for identity, _ in _source_specs(case)]
    if not _is_string_list(artifact["source_tensor_ids"]):
        raise ValueError("source_tensor_ids must contain built-in strings")
    if artifact["source_tensor_ids"] != expected_sources:
        raise ValueError("source tensor provenance is incomplete or stale")
    groups = artifact["lane_groups"]
    if type(groups) is not list or groups != _expected_groups(case):
        raise ValueError("lane groups are not the canonical eight-lane prefix groups")
    for group in groups:
        if type(group) is not dict or set(group) != {
            "prefix_bits",
            "query_ids",
            "lane_width",
        }:
            raise ValueError("lane group schema mismatch")
        if (
            type(group["prefix_bits"]) is not list
            or not all(_is_int(bit) and bit in (0, 1) for bit in group["prefix_bits"])
            or not _is_string_list(group["query_ids"])
            or not _is_int(group["lane_width"])
        ):
            raise ValueError("lane group fields require strict built-in types")
    sliced = artifact["slice_indices"]
    if (
        not _is_string_list(sliced)
        or len(sliced) != case["required_slice_variables"]
        or sliced != sorted(set(sliced))
        or not set(sliced) <= set(case["slice_candidate_ids"])
    ):
        raise ValueError("slice plan is invalid or noncanonical")
    live = {
        identity: [axis for axis in axes if axis not in sliced]
        for identity, axes in _source_specs(case)
    }
    initial_suffix = [
        axes for _, axes in _source_specs(case)[case["shared_prefix_qubits"] : N_QUBITS]
    ]
    if not initial_suffix or any(LANE not in axes for axes in initial_suffix):
        raise ValueError("the batched output-lane hyperedge is absent")
    peak = sum(_size(axes) for axes in live.values())
    maximum = max(_size(axes) for axes in live.values())
    fmas = 0
    remaining = set(case["variable_ids"]) - set(sliced)
    steps = artifact["steps"]
    if type(steps) is not list or len(steps) != N_QUBITS - len(sliced):
        raise ValueError("the elimination program has the wrong length")
    for number, row in enumerate(steps):
        if (
            type(row) is not list
            or len(row) != 4
            or type(row[0]) is not str
            or row[0] != "ELIM"
            or type(row[1]) is not str
            or not _is_string_list(row[2])
            or type(row[3]) is not str
        ):
            raise ValueError("each bytecode record must have exact symbolic fields")
        index, inputs, output = row[1], row[2], row[3]
        if index not in remaining or output != f"tmp-{number:03d}" or output in live:
            raise ValueError("elimination index or temporary identity is noncanonical")
        expected_inputs = sorted(
            identity for identity, axes in live.items() if index in axes
        )
        if inputs != expected_inputs or not inputs:
            raise ValueError("step operands do not match live-index provenance")
        scope = _ordered({axis for identity in inputs for axis in live[identity]})
        output_axes = [axis for axis in scope if axis != index]
        fmas += (len(inputs) - 1) * _size(scope) + _size(output_axes)
        for identity in inputs:
            del live[identity]
        live[output] = output_axes
        maximum = max(maximum, _size(output_axes))
        peak = max(peak, sum(_size(axes) for axes in live.values()))
        remaining.remove(index)
    if remaining or any(axis != LANE for axes in live.values() for axis in axes):
        raise ValueError("program leaves an internal binary index live")
    terminal = artifact["terminal_inputs"]
    if not _is_string_list(terminal) or terminal != sorted(live):
        raise ValueError("terminal operand list is not canonical")
    if not any(LANE in axes for axes in live.values()):
        raise ValueError("program does not produce a batched amplitude lane")
    fmas += max(0, len(terminal) - 1) * LANES
    repetitions = len(groups) * 2 ** len(sliced)
    expected = {
        "slice_count": 2 ** len(sliced),
        "eliminate_operations": len(steps) * repetitions,
        "terminal_contractions": repetitions,
        "peak_live_elements": peak,
        "max_intermediate_elements": maximum,
        "complex_fmas": fmas * repetitions,
        "program_digest": _digest(_core(artifact)),
    }
    certificates = artifact["certificates"]
    if type(certificates) is not dict or set(certificates) != CERTIFICATE_KEYS:
        raise ValueError("resource certificate schema mismatch")
    for key in CERTIFICATE_KEYS - {"program_digest"}:
        if not _is_int(certificates[key]):
            raise ValueError("resource counts must be exact built-in integers")
    if (
        type(certificates["program_digest"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", certificates["program_digest"]) is None
        or certificates != expected
    ):
        raise ValueError("program digest or resource certificate is false")
    if (
        peak > case["max_peak_live_elements"]
        or maximum > case["max_intermediate_elements"]
        or expected["complex_fmas"] > case["max_complex_fmas"]
    ):
        raise ValueError("program exceeds the declared resource envelope")
    return {
        "live_specs": live,
        "peak": peak,
        "maximum": maximum,
        "fmas": expected["complex_fmas"],
    }


def _ry(theta):
    cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.complex128)


def _rx(theta):
    cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[cosine, -1j * sine], [-1j * sine, cosine]], dtype=np.complex128)


def _rz(theta):
    return np.diag(np.exp(np.array([-0.5j * theta, 0.5j * theta])))


def _parameters(case, mode, digest):
    if mode == "declared":
        local = [
            [
                record["initial_ry"],
                record["initial_rz"],
                record["final_rx"],
                record["final_ry"],
            ]
            for record in case["local_gates"]
        ]
        edges = [gate["theta"] for gate in case["entanglers"]]
        return np.asarray(local), np.asarray(edges)
    seed = int.from_bytes(
        hashlib.sha256((digest + ":fresh-tensors-v1").encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng(seed)
    local = rng.uniform(-1.3, 1.3, size=(N_QUBITS, 4))
    edges = rng.uniform(-0.9, 0.9, size=len(case["entanglers"]))
    return local, edges


def _numeric_sources(case, group, mode, digest):
    lookup = {query["query_id"]: query for query in case["queries"]}
    queries = [lookup[identity] for identity in group["query_ids"]]
    local_parameters, edge_parameters = _parameters(case, mode, digest)
    values, specs = {}, dict(_source_specs(case))
    for qubit, identity in enumerate(case["site_tensor_ids"]):
        initial_ry, initial_rz, final_rx, final_ry = local_parameters[qubit]
        prepared = _rz(initial_rz) @ _ry(initial_ry) @ np.array([1.0, 0.0])
        final = _ry(final_ry) @ _rx(final_rx)
        if qubit < case["shared_prefix_qubits"]:
            bit = group["prefix_bits"][qubit]
            values[identity] = prepared * final[bit, :]
        else:
            bits = np.asarray([query["bits"][qubit] for query in queries])
            values[identity] = prepared[:, None] * final[bits, :].T
    signs = np.array([1.0, -1.0])
    for gate, theta in zip(case["entanglers"], edge_parameters):
        values[gate["gate_id"]] = np.exp(
            -0.5j * theta * signs[:, None] * signs[None, :]
        )
    return values, specs


def _align(value, axes, union, lane_width):
    array = np.asarray(value, dtype=np.complex128)
    if axes:
        permutation = sorted(
            range(len(axes)), key=lambda position: union.index(axes[position])
        )
        array = np.transpose(array, permutation)
        ordered_axes = [axes[position] for position in permutation]
    else:
        ordered_axes = []
    shape = []
    for axis in union:
        shape.append((lane_width if axis == LANE else 2) if axis in ordered_axes else 1)
    return array.reshape(shape)


def _execute_values(values, specs, steps, terminal, sliced, lane_width):
    total = np.zeros(lane_width, dtype=np.complex128)
    for bits in itertools.product((0, 1), repeat=len(sliced)):
        assignment = dict(zip(sliced, bits))
        live_values, live_specs = {}, {}
        for identity, axes in specs.items():
            selector = tuple(assignment.get(axis, slice(None)) for axis in axes)
            live_values[identity] = np.asarray(values[identity])[selector]
            live_specs[identity] = [axis for axis in axes if axis not in assignment]
        for _, index, inputs, output in steps:
            scope = _ordered(
                {axis for identity in inputs for axis in live_specs[identity]}
            )
            product = np.ones(
                [lane_width if axis == LANE else 2 for axis in scope],
                dtype=np.complex128,
            )
            for identity in inputs:
                product *= _align(
                    live_values[identity], live_specs[identity], scope, lane_width
                )
                del live_values[identity]
                del live_specs[identity]
            position = scope.index(index)
            live_values[output] = np.sum(product, axis=position)
            live_specs[output] = [axis for axis in scope if axis != index]
        product = np.ones(lane_width, dtype=np.complex128)
        for identity in terminal:
            product *= _align(
                live_values[identity], live_specs[identity], [LANE], lane_width
            )
        total += product
    return total


def _row_oracle(case, group, values):
    layout = case["logical_to_physical"]
    states = ((np.arange(1 << COLS)[:, None] >> np.arange(COLS)) & 1).astype(int)
    edge_values = {
        frozenset(gate["qubits"]): values[gate["gate_id"]]
        for gate in case["entanglers"]
    }
    rows = []
    for row in range(ROWS):
        weight = np.ones((1 << COLS, group["lane_width"]), dtype=np.complex128)
        for column in range(COLS):
            qubit = layout[row * COLS + column]
            tensor = np.asarray(values[case["site_tensor_ids"][qubit]])
            if tensor.ndim == 1:
                tensor = tensor[:, None]
            weight *= tensor[states[:, column], :]
        for column in range(COLS - 1):
            left = layout[row * COLS + column]
            right = layout[row * COLS + column + 1]
            tensor = edge_values[frozenset((left, right))]
            weight *= tensor[states[:, column], states[:, column + 1]][:, None]
        rows.append(weight)
    state = rows[0]
    for row in range(1, ROWS):
        transfer = np.ones((1 << COLS, 1 << COLS), dtype=np.complex128)
        for column in range(COLS):
            upper = layout[(row - 1) * COLS + column]
            lower = layout[row * COLS + column]
            tensor = edge_values[frozenset((upper, lower))]
            transfer *= tensor[states[:, column, None], states[None, :, column]]
        state = rows[row] * (transfer.T @ state)
    return np.sum(state, axis=0)


def _execute_artifact(artifact, config):
    case = config["case"]
    maximum_error = 0.0
    norms = []
    for mode in ("declared", "fresh"):
        for group in artifact["lane_groups"]:
            values, specs = _numeric_sources(case, group, mode, config["case_digest"])
            actual = _execute_values(
                values,
                specs,
                artifact["steps"],
                artifact["terminal_inputs"],
                artifact["slice_indices"],
                group["lane_width"],
            )
            expected = _row_oracle(case, group, values)
            error = float(np.max(np.abs(actual - expected)))
            scale = float(np.max(np.abs(expected)))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError("independent amplitude oracle is degenerate")
            maximum_error = max(maximum_error, error / scale)
            norms.append(float(np.linalg.norm(expected)))
    if not np.isfinite(maximum_error) or maximum_error > 2e-10:
        raise ValueError("trusted execution disagrees with the independent row oracle")
    return maximum_error, norms


def validate_artifact(artifact, config, execute=True):
    try:
        resources = _validate_structure(artifact, config)
        execution_error, norms = (0.0, [])
        if execute:
            execution_error, norms = _execute_artifact(artifact, config)
        return True, {
            **resources,
            "execution_error": execution_error,
            "oracle_norms": norms,
        }
    except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
        return False, {"error": str(error)}


def _apply_dense(state, matrix, qubits, count):
    rest = [qubit for qubit in range(count) if qubit not in qubits]
    axes = list(qubits) + rest
    moved = state.reshape((2,) * count).transpose(axes)
    moved = matrix @ moved.reshape(2 ** len(qubits), -1)
    inverse = np.argsort(axes)
    return moved.reshape((2,) * count).transpose(inverse).reshape(-1)


def _dense_canary():
    rng = np.random.default_rng(121)
    count, lanes = 6, 5
    local = rng.uniform(-1.0, 1.0, size=(count, 4))
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]
    angles = rng.uniform(-0.7, 0.7, size=len(edges))
    outputs = rng.integers(0, 2, size=(lanes, count))
    state = np.zeros(1 << count, dtype=np.complex128)
    state[0] = 1
    for qubit, (a, b, _, _) in enumerate(local):
        state = _apply_dense(state, _rz(b) @ _ry(a), [qubit], count)
    signs = np.array([1.0, -1.0])
    factors = []
    for (left, right), theta in zip(edges, angles):
        factor = np.exp(-0.5j * theta * signs[:, None] * signs[None, :])
        factors.append(factor)
        state = _apply_dense(state, np.diag(factor.reshape(-1)), [left, right], count)
    site = []
    for qubit, (a, b, c, d) in enumerate(local):
        final = _ry(d) @ _rx(c)
        state = _apply_dense(state, final, [qubit], count)
        prepared = _rz(b) @ _ry(a) @ np.array([1.0, 0.0])
        site.append(prepared[:, None] * final[outputs[:, qubit], :].T)
    factorized = np.zeros(lanes, dtype=np.complex128)
    for internal in itertools.product((0, 1), repeat=count):
        value = np.ones(lanes, dtype=np.complex128)
        for qubit in range(count):
            value *= site[qubit][internal[qubit]]
        for factor, (left, right) in zip(factors, edges):
            value *= factor[internal[left], internal[right]]
        factorized += value
    dense = np.asarray([state.reshape((2,) * count)[tuple(bits)] for bits in outputs])
    return float(np.max(np.abs(dense - factorized)))


def _bytecode_canary():
    rng = np.random.default_rng(2121)
    values = {
        "a": rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3)),
        "b": rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)),
        "c": rng.normal(size=(2, 3)) + 1j * rng.normal(size=(2, 3)),
    }
    specs = {"a": ["x", LANE], "b": ["x", "y"], "c": ["y", LANE]}
    steps = [["ELIM", "y", ["b", "c"], "tmp-000"]]
    actual = _execute_values(values, specs, steps, ["a", "tmp-000"], ["x"], 3)
    expected = np.einsum("xl,xy,yl->l", values["a"], values["b"], values["c"])
    return float(np.max(np.abs(actual - expected)))


class _ShimCircuit:
    def __init__(self, nqubits):
        self.nqubits = nqubits
        self.records = []

    def _one(self, name, qubit, **parameters):
        self.records.append({"name": name, "index": (qubit,), "parameters": parameters})

    def ry(self, qubit, **parameters):
        self._one("ry", qubit, **parameters)

    def rz(self, qubit, **parameters):
        self._one("rz", qubit, **parameters)

    def rx(self, qubit, **parameters):
        self._one("rx", qubit, **parameters)

    def rzz(self, left, right, **parameters):
        self.records.append(
            {"name": "rzz", "index": (left, right), "parameters": parameters}
        )

    def to_qir(self):
        return copy.deepcopy(self.records)


def _load_module(solution, name):
    path = Path(solution)
    if path.suffix == ".py" or path.exists():
        spec = importlib.util.spec_from_file_location(name, path.resolve())
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {solution}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(solution)


def _run_shim_expert(config):
    shim = types.ModuleType("tensorcircuit")
    shim.Circuit = _ShimCircuit
    shim.set_backend = lambda name: None
    shim.set_dtype = lambda name: None
    previous = sys.modules.get("tensorcircuit")
    sys.modules["tensorcircuit"] = shim
    try:
        path = Path(__file__).with_name("expert") / "solution_121.py"
        module = _load_module(str(path), f"candidate121_expert_{time.time_ns()}")
        first = module.run_solution(copy.deepcopy(config))
        second = module.run_solution(copy.deepcopy(config))
    finally:
        if previous is None:
            sys.modules.pop("tensorcircuit", None)
        else:
            sys.modules["tensorcircuit"] = previous
    deterministic = _canonical(first) == _canonical(second)
    return first, deterministic


def _refresh_digest(artifact):
    if type(artifact) is dict and "certificates" in artifact:
        artifact["certificates"]["program_digest"] = _digest(_core(artifact))


def _mutant_results(proxy, config):
    mutations = []

    def attempt(name, mutate, refresh=True):
        candidate = copy.deepcopy(proxy)
        mutate(candidate)
        if refresh:
            _refresh_digest(candidate)
        passed, detail = validate_artifact(candidate, config, execute=False)
        mutations.append({"name": name, "rejected": not passed, "detail": detail})

    attempt(
        "independent-per-amplitude paths",
        lambda value: value.__setitem__(
            "lane_groups",
            [
                {
                    "prefix_bits": query["bits"][:PREFIX],
                    "query_ids": [query["query_id"]],
                    "lane_width": 1,
                }
                for query in config["case"]["queries"]
            ],
        ),
    )
    attempt(
        "missing batch lane",
        lambda value: value["lane_groups"][0].__setitem__("lane_width", 1),
    )
    attempt(
        "false cost certificate",
        lambda value: value["certificates"].__setitem__(
            "complex_fmas", value["certificates"]["complex_fmas"] - 1
        ),
        refresh=False,
    )
    attempt(
        "stale tensor identity",
        lambda value: value["source_tensor_ids"].__setitem__(0, "site-stale"),
    )
    attempt(
        "illegal embedded constants",
        lambda value: value.__setitem__("constants", [0.125]),
    )
    attempt(
        "invalid duplicate slice plan",
        lambda value: value["slice_indices"].__setitem__(1, value["slice_indices"][0]),
    )
    attempt("forged step provenance", lambda value: value["steps"][0][2].pop())
    attempt(
        "repeated elimination index",
        lambda value: value["steps"][1].__setitem__(1, value["steps"][0][1]),
    )
    attempt(
        "stale query identity",
        lambda value: value["lane_groups"][0]["query_ids"].__setitem__(0, "amp-stale"),
    )
    attempt("invalid terminal plan", lambda value: value["terminal_inputs"].pop())
    return mutations


def _effective_lines(path):
    return sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in Path(path).read_text().splitlines()
    )


def _isolation_regressions(config):
    with tempfile.TemporaryDirectory(prefix="orbit-q-121-hostile-") as directory:
        root = Path(directory)
        patcher = root / "patch_determinism.py"
        patcher.write_text(
            "import os,sys\n"
            "sys.modules['__main__']._canonical=lambda value:b'same'\n"
            "def run_solution(config): return {'worker_pid':os.getpid()}\n"
        )
        _, deterministic, patch_error = _isolated_submission(str(patcher), config, 20)
        subclass = root / "subclass.py"
        subclass.write_text(
            "class AlmostInt(int): pass\n"
            "def run_solution(config): return {'value':AlmostInt(1)}\n"
        )
        _, _, subclass_error = _isolated_submission(str(subclass), config, 20)
        marker = root / "descendant-survived"
        descendant = root / "descendant.py"
        child_code = (
            "import pathlib,time;time.sleep(0.35);"
            f"pathlib.Path({str(marker)!r}).write_text('alive')"
        )
        descendant.write_text(
            "import subprocess,sys\n"
            f"CHILD={child_code!r}\n"
            "def run_solution(config):\n"
            " subprocess.Popen([sys.executable,'-c',CHILD],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
            " return {'ok':1}\n"
        )
        _, descendant_deterministic, descendant_error = _isolated_submission(
            str(descendant), config, 20
        )
        time.sleep(0.7)
        return {
            "child_canonical_monkeypatch_rejected": patch_error is None
            and not deterministic,
            "integer_subclass_rejected_before_json": subclass_error is not None,
            "ordinary_descendant_process_group_killed": descendant_error is None
            and descendant_deterministic
            and not marker.exists(),
        }


def self_test(seed):
    config = build_config(seed)
    proxy, deterministic = _run_shim_expert(config)
    passed, details = validate_artifact(proxy, config, execute=True)
    dense_error = _dense_canary()
    bytecode_error = _bytecode_canary()
    mutations = _mutant_results(proxy, config)
    isolation_regressions = _isolation_regressions(config)
    expert_path = Path(__file__).with_name("expert") / "solution_121.py"
    lines = _effective_lines(expert_path)
    overall = bool(
        passed
        and deterministic
        and dense_error < 2e-12
        and bytecode_error < 2e-12
        and all(row["rejected"] for row in mutations)
        and all(isolation_regressions.values())
        and lines <= 160
    )
    return {
        "overall": overall,
        "case_digest": config["case_digest"],
        "deterministic": deterministic,
        "dense_canary_error": dense_error,
        "bytecode_canary_error": bytecode_error,
        "execution_relative_error": details.get("execution_error"),
        "resource_claims": {
            key: details.get(key) for key in ("peak", "maximum", "fmas")
        },
        "resource_caps": {
            key: config["case"][key]
            for key in (
                "max_peak_live_elements",
                "max_intermediate_elements",
                "max_complex_fmas",
            )
        },
        "mutants": mutations,
        "mutants_rejected": sum(row["rejected"] for row in mutations),
        "mutants_attempted": len(mutations),
        "hostile_isolation_regressions": isolation_regressions,
        "expert_effective_lines": lines,
        "validation": details,
    }


def _scrub_worker_process(solution):
    for key in list(os.environ):
        upper = key.upper()
        if any(
            token in upper for token in ("SEED", "KEY", "TOKEN", "SECRET", "PASSWORD")
        ):
            os.environ.pop(key, None)
    sys.argv[:] = [Path(solution).name]


def _worker(solution, config_file, output_file):
    config = json.loads(Path(config_file).read_text())
    _scrub_worker_process(solution)
    payload = {}
    type_, dict_, list_, str_, int_ = type, dict, list, str, int
    canonical = _canonical
    try:
        module = _load_module(solution, f"candidate121_submission_{time.time_ns()}")
        artifact = module.run_solution(copy.deepcopy(config))
        stack = [artifact]
        while stack:
            item = stack.pop()
            kind = type_(item)
            if kind is dict_:
                stack.extend(item.keys())
                stack.extend(item.values())
            elif kind is list_:
                stack.extend(item)
            elif kind is not str_ and kind is not int_:
                raise TypeError(
                    "artifact contains a non-built-in or unsupported value type"
                )
        if len(canonical(artifact)) > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact exceeds the one-megabyte transfer limit")
        payload = {"artifact": artifact}
    except BaseException as error:
        payload = {"error": f"{type(error).__name__}: {error}"}
    try:
        Path(output_file).write_text(json.dumps(payload, allow_nan=False))
    except BaseException as error:
        Path(output_file).write_text(
            json.dumps({"error": f"non-JSON artifact: {type(error).__name__}: {error}"})
        )
    return 0


def _sanitized_environment():
    allowed = {
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "LANG",
        "LC_ALL",
        "TMPDIR",
        "NUMBA_DISABLE_JIT",
        "JAX_PLATFORM_NAME",
        "XLA_FLAGS",
    }
    environment = {key: value for key, value in os.environ.items() if key in allowed}
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    return environment


def _run_worker(command, cwd, timeout):
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=_sanitized_environment(),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            start_new_session=True,
        )
    except OSError as error:
        return None, f"isolated worker launch failed: {error}"
    error = None
    try:
        process.communicate(timeout=timeout)
        if process.returncode != 0:
            error = f"isolated worker failed with code {process.returncode}"
    except subprocess.TimeoutExpired:
        error = "isolated worker exceeded the shared runtime limit"
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if process.poll() is None:
            process.communicate()
    return process.returncode, error


def _isolated_submission(solution, config, timeout):
    with tempfile.TemporaryDirectory(prefix="orbit-q-121-") as directory:
        config_file = Path(directory) / "public-config.json"
        config_file.write_text(json.dumps(config, allow_nan=False))
        deadline = time.monotonic() + timeout
        artifacts = []
        for invocation in range(2):
            output_file = Path(directory) / f"artifact-{invocation}.json"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--artifact-worker",
                "--solution",
                str(Path(solution).resolve()) if Path(solution).exists() else solution,
                "--config-file",
                str(config_file),
                "--output-file",
                str(output_file),
            ]
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None, False, "isolated workers exceeded the shared runtime limit"
            _, error = _run_worker(
                command,
                str(Path(solution).resolve().parent)
                if Path(solution).exists()
                else os.getcwd(),
                remaining,
            )
            if error is not None or not output_file.exists():
                return None, False, error or "isolated worker produced no artifact"
            if output_file.stat().st_size > MAX_ARTIFACT_BYTES + 100_000:
                return (
                    None,
                    False,
                    "isolated worker artifact exceeded the transfer limit",
                )
            payload = json.loads(output_file.read_text())
            if "error" in payload:
                return None, False, payload["error"]
            artifacts.append(payload.get("artifact"))
        deterministic = _canonical(artifacts[0]) == _canonical(artifacts[1])
        return artifacts[0], deterministic, None


def _identity(seed, config):
    return {
        "orbit_q_case_identity": {
            "problem_id": 121,
            "protocol_seed": seed,
            "case_id": config["case"]["case_id"],
            "case_digest": config["case_digest"],
        }
    }


def _resolve_seed(cli_seed):
    raw = os.environ.get("ORBIT_Q_CANDIDATE_SEED")
    env_seed = int(raw) if raw is not None else None
    if cli_seed is not None and env_seed is not None and cli_seed != env_seed:
        raise SystemExit("seed mismatch: --seed disagrees with ORBIT_Q_CANDIDATE_SEED")
    return (
        cli_seed
        if cli_seed is not None
        else env_seed
        if env_seed is not None
        else DEFAULT_SEED
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--artifact-worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--config-file", help=argparse.SUPPRESS)
    parser.add_argument("--output-file", help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)
    if arguments.artifact_worker:
        if (
            not arguments.solution
            or not arguments.config_file
            or not arguments.output_file
        ):
            parser.error("artifact worker requires solution and two files")
        return _worker(arguments.solution, arguments.config_file, arguments.output_file)
    seed = _resolve_seed(arguments.seed)
    config = build_config(seed)
    print(
        json.dumps(_identity(seed, config), sort_keys=True, separators=(",", ":")),
        flush=True,
    )
    if arguments.self_test:
        started = time.perf_counter()
        report = self_test(seed)
        print(json.dumps(report, sort_keys=True, allow_nan=False))
        print(f"Self-test time: {time.perf_counter() - started:.3f}s")
        print("Overall: PASS" if report["overall"] else "Overall: FAIL")
        return 0 if report["overall"] else 1
    if not arguments.solution:
        parser.error("provide --solution or --self-test")
    artifact, deterministic, error = _isolated_submission(
        arguments.solution, config, 300
    )
    if error is not None:
        print(f"Isolated artifact error: {error}")
        print("Overall: FAIL")
        return 1
    passed, details = validate_artifact(artifact, config, execute=True)
    passed = passed and deterministic
    print(f"Deterministic artifact: {deterministic}")
    print(json.dumps(details, sort_keys=True, allow_nan=False))
    print("Overall: PASS" if passed else "Overall: FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
