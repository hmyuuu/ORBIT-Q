"""TensorCircuit expert for verifier-executed batched-amplitude bytecode."""

import hashlib
import itertools
import json

import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")
LANE = "@lane"
SCHEMA = "orbit-q/batched-amplitude-elimination/v1"


def _qir_edges(case):
    circuit = tc.Circuit(case["n_qubits"])
    for qubit, record in enumerate(case["local_gates"]):
        circuit.ry(qubit, theta=record["initial_ry"])
        circuit.rz(qubit, theta=record["initial_rz"])
    for gate in case["entanglers"]:
        circuit.rzz(*gate["qubits"], theta=gate["theta"])
    for qubit, record in enumerate(case["local_gates"]):
        circuit.rx(qubit, theta=record["final_rx"])
        circuit.ry(qubit, theta=record["final_ry"])
    qir = circuit.to_qir()
    edges = [list(row["index"]) for row in qir if str(row["name"]).lower() == "rzz"]
    expected = [gate["qubits"] for gate in case["entanglers"]]
    if edges != expected or len(qir) != 4 * case["n_qubits"] + len(edges):
        raise ValueError("TensorCircuit QIR does not match the supplied circuit")
    return edges


def _groups(case):
    width = case["shared_prefix_qubits"]
    buckets = {}
    for query in case["queries"]:
        buckets.setdefault(tuple(query["bits"][:width]), []).append(query["query_id"])
    groups = []
    for prefix, identities in sorted(buckets.items()):
        groups.append(
            {
                "prefix_bits": list(prefix),
                "query_ids": sorted(identities),
                "lane_width": len(identities),
            }
        )
    if any(group["lane_width"] != case["lanes_per_group"] for group in groups):
        raise ValueError("query bank does not form complete batched groups")
    return groups


def _sources(case, edges):
    variables = case["variable_ids"]
    prefix = case["shared_prefix_qubits"]
    rows = []
    for qubit, identity in enumerate(case["site_tensor_ids"]):
        indices = [variables[qubit]] + ([] if qubit < prefix else [LANE])
        rows.append((identity, indices))
    for gate, edge in zip(case["entanglers"], edges):
        rows.append((gate["gate_id"], [variables[q] for q in edge]))
    return rows


def _ordered(indices):
    return sorted(index for index in indices if index != LANE) + (
        [LANE] if LANE in indices else []
    )


def _size(indices, lanes):
    return 2 ** sum(index != LANE for index in indices) * (
        lanes if LANE in indices else 1
    )


def _make_plan(case, sources, sliced):
    lanes = case["lanes_per_group"]
    live = {
        identity: [x for x in indices if x not in sliced]
        for identity, indices in sources
    }
    steps, order, fmas = [], [], 0
    peak = sum(_size(indices, lanes) for indices in live.values())
    maximum = max(_size(indices, lanes) for indices in live.values())
    remaining = set(case["variable_ids"]) - set(sliced)
    while remaining:
        choices = []
        for index in remaining:
            inputs = sorted(
                identity for identity, axes in live.items() if index in axes
            )
            scope = _ordered({axis for identity in inputs for axis in live[identity]})
            output = [axis for axis in scope if axis != index]
            choices.append((len(output) - (LANE in output), _size(scope, lanes), index))
        index = min(choices)[2]
        inputs = sorted(identity for identity, axes in live.items() if index in axes)
        scope = _ordered({axis for identity in inputs for axis in live[identity]})
        output_axes = [axis for axis in scope if axis != index]
        output = f"tmp-{len(steps):03d}"
        fmas += (len(inputs) - 1) * _size(scope, lanes) + _size(output_axes, lanes)
        for identity in inputs:
            del live[identity]
        live[output] = output_axes
        steps.append(["ELIM", index, inputs, output])
        order.append(index)
        maximum = max(maximum, _size(output_axes, lanes))
        peak = max(peak, sum(_size(axes, lanes) for axes in live.values()))
        remaining.remove(index)
    terminal = sorted(live)
    fmas += max(0, len(terminal) - 1) * lanes
    return steps, terminal, peak, maximum, fmas, order


def _compile(case, edges, digest):
    sources = _sources(case, edges)
    choices = []
    for sliced in itertools.combinations(sorted(case["slice_candidate_ids"]), 2):
        plan = _make_plan(case, sources, sliced)
        choices.append(((plan[2], plan[4], sliced, tuple(plan[5])), sliced, plan))
    _, sliced, plan = min(choices)
    steps, terminal, peak, maximum, fmas, _ = plan
    groups = _groups(case)
    core = {
        "schema": SCHEMA,
        "case_id": case["case_id"],
        "case_digest": digest,
        "source_tensor_ids": [identity for identity, _ in sources],
        "lane_groups": groups,
        "slice_indices": list(sliced),
        "steps": steps,
        "terminal_inputs": terminal,
    }
    program_digest = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    repetitions = len(groups) * 2 ** len(sliced)
    core["certificates"] = {
        "slice_count": 2 ** len(sliced),
        "eliminate_operations": len(steps) * repetitions,
        "terminal_contractions": repetitions,
        "peak_live_elements": peak,
        "max_intermediate_elements": maximum,
        "complex_fmas": fmas * repetitions,
        "program_digest": program_digest,
    }
    return core


def run_solution(config):
    case = config["case"]
    return _compile(case, _qir_edges(case), config["case_digest"])
