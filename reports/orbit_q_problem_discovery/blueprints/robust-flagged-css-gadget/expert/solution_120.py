import itertools

import numpy as np
import tensorcircuit as tc


# Keep the reviewed expert within the candidate's 160-line physical budget.
# The compact layout is formatting-only; its AST and behavior are checked.
# fmt: off
tc.set_backend("numpy"), tc.set_dtype("complex128")
PB = ((0, 0), (1, 0), (1, 1), (0, 1))
ID = np.eye(2, dtype=np.complex128)
X = np.asarray([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.asarray([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
PM = {"I": ID, "X": X, "Y": Y, "Z": Z}


def _bits(row):
    return sum(int(v) << q for q, v in enumerate(row))


def _row(value, n):
    return [int(value >> q & 1) for q in range(n)]


def _gens(case):
    return [( _bits(r), 0) for r in case["x_stabilizers"]] + [
        (0, _bits(r)) for r in case["z_stabilizers"]
    ] + [(0, _bits(case["logical_z"]))]


def _syn(x, z, generators):
    return tuple(((x & b).bit_count() ^ (z & a).bit_count()) & 1 for a, b in generators)


def _span_state(case):
    rows = [_bits(r) for r in case["x_stabilizers"]]
    words = [0]
    for value in rows:
        if value not in words:
            words += [word ^ value for word in words]
    state = np.zeros(1 << case["n_data"], dtype=np.complex128)
    for word in words:
        index = sum((word >> q & 1) << (case["n_data"] - 1 - q) for q in range(case["n_data"]))
        state[index] = 1 / np.sqrt(len(words))
    return state


def _initial(case, kind):
    zero, plus = np.asarray([1, 0]), np.asarray([1, 1]) / np.sqrt(2)
    syndrome, flag = (zero, plus) if kind == "z" else (plus, zero)
    return np.kron(np.kron(_span_state(case), syndrome), flag).astype(np.complex128)


def _ent(gadget):
    output, variants = [], iter(gadget["variant_ids"])
    for slot in range(5):
        if slot in gadget["flag_window"]:
            output.append(("flag", None, next(variants)))
        if slot < 4:
            output.append(("data", gadget["data_order"][slot], next(variants)))
    return output


def _pair(kind, role, data, syndrome, flag):
    if kind == "z":
        return (flag, syndrome) if role == "flag" else (data, syndrome)
    return (syndrome, flag) if role == "flag" else (syndrome, data)


def _prop(x, z, gates, kind, syndrome, flag):
    for role, data, _ in gates:
        control, target = _pair(kind, role, data, syndrome, flag)
        if x >> control & 1:
            x ^= 1 << target
        if z >> target & 1:
            z ^= 1 << control
    return x, z


def _branches(case, gadget):
    n, syndrome, flag = case["n_data"], case["n_data"], case["n_data"] + 1
    gates, code = _ent(gadget), _gens(case)[:6]
    output = []

    def add(x, z, check=0, flagged=0):
        check ^= (x >> syndrome & 1) if gadget["kind"] == "z" else (z >> syndrome & 1)
        flagged ^= (z >> flag & 1) if gadget["kind"] == "z" else (x >> flag & 1)
        dx, dz = x & ((1 << n) - 1), z & ((1 << n) - 1)
        output.append(((check, flagged) + _syn(dx, dz, code), dx, dz))

    add(0, 0)
    for ancilla in (syndrome, flag):
        for px, pz in PB[1:]:
            add(*_prop(px << ancilla, pz << ancilla, gates, gadget["kind"], syndrome, flag))
    for index, (role, data, _variant) in enumerate(gates):
        control, target = _pair(gadget["kind"], role, data, syndrome, flag)
        for left, right in itertools.product(PB, repeat=2):
            if left == right == (0, 0):
                continue
            x, z = left[0] << control | right[0] << target, left[1] << control | right[1] << target
            add(*_prop(x, z, gates[index + 1 :], gadget["kind"], syndrome, flag))
    add(0, 0, check=1)
    add(0, 0, flagged=1)
    return output


def _context(case):
    n, generators = case["n_data"], _gens(case)
    reps = {}
    for labels in itertools.product(range(4), repeat=n):
        x = sum(PB[p][0] << q for q, p in enumerate(labels))
        z = sum(PB[p][1] << q for q, p in enumerate(labels))
        syndrome = _syn(x, z, generators)
        key = (sum(p != 0 for p in labels), labels)
        if syndrome not in reps or key < reps[syndrome][0]:
            reps[syndrome] = (key, x, z)
    correct = {_syn(0, 0, generators)}
    for q in range(n):
        correct |= {_syn(1 << q, 0, generators), _syn(0, 1 << q, generators), _syn(1 << q, 1 << q, generators)}
    return generators, reps, correct


def _decoder(case, gadget, context):
    generators, reps, correct = context
    groups = {}
    for observation, x, z in _branches(case, gadget):
        if _syn(x, z, generators) not in correct and observation[1] != 1:
            return None
        groups.setdefault(observation, []).append(_syn(x, z, generators))
    table = {}
    for observation, errors in groups.items():
        for syndrome, (_key, x, z) in reps.items():
            if all(tuple(a ^ b for a, b in zip(error, syndrome)) in correct for error in errors):
                table[observation] = (x, z)
                break
        if observation not in table:
            return None
    rows = []
    for observation in itertools.product((0, 1), repeat=8):
        x, z = table.get(observation, (0, 0))
        rows.append({"observation": list(observation), "recovery": {"x": _row(x, case["n_data"]), "z": _row(z, case["n_data"])}})
    return rows


def _score(case, gadget):
    n, syndrome, flag = case["n_data"], case["n_data"], case["n_data"] + 1
    state = _initial(case, gadget["kind"])
    target, variants = state.copy(), {v["id"]: v for v in case["cx_variants"]}
    axes, coefficients = case["coherent_model"]["pauli_axes"], case["coherent_model"]["nominal_coefficients"]
    edge_multipliers = case["coherent_model"]["data_edge_multipliers"]
    circuit = tc.Circuit(n + 2, inputs=state)
    for index, (role, data, variant_id) in enumerate(_ent(gadget)):
        control, target_q = _pair(gadget["kind"], role, data, syndrome, flag)
        circuit.cnot(control, target_q)
        transform = variants[variant_id]["coefficient_transform"]
        edge = [1, 1, 1] if role == "flag" else edge_multipliers[data]
        hamiltonian = sum(transform[k] * coefficients[index][k] * edge[k] * np.kron(PM[p[0]], PM[p[1]]) for k, p in enumerate(axes[index]))
        values, vectors = np.linalg.eigh(hamiltonian)
        unitary = (vectors * np.exp(-0.5j * values)) @ vectors.conj().T
        circuit.unitary(control, target_q, unitary=tc.gates.Gate(unitary), name="calibrated_error")
    return float(abs(np.vdot(target, np.asarray(circuit.state()))) ** 2)


def _solve(case):
    plain = next(v["id"] for v in case["cx_variants"] if v["echo_cost"] == 0)
    context, structures = _context(case), []
    for check in case["check_candidates"]:
        support = [q for q, bit in enumerate(check["mask"]) if bit]
        for order in itertools.permutations(support):
            for window in case["flag_windows"]:
                gadget = {"check_id": check["id"], "kind": check["kind"], "mask": check["mask"], "data_order": list(order), "flag_window": window, "variant_ids": [plain] * 6}
                decoder = _decoder(case, gadget, context)
                if decoder is not None:
                    structures.append((_score(case, gadget), gadget, decoder))
    structures.sort(key=lambda item: (-item[0], item[1]["check_id"], item[1]["data_order"], item[1]["flag_window"]))
    best = None
    variants = case["cx_variants"]
    for _base, gadget, decoder in structures[:12]:
        for choices in itertools.product(range(len(variants)), repeat=6):
            selected = [variants[index] for index in choices]
            if sum(value["echo_cost"] for value in selected) > case["echo_cost_budget"]:
                continue
            candidate = {**gadget, "variant_ids": [value["id"] for value in selected]}
            key = (_score(case, candidate), tuple(candidate["variant_ids"]), candidate["check_id"], tuple(candidate["data_order"]), tuple(candidate["flag_window"]))
            if best is None or key > best[0]:
                best = (key, candidate, decoder)
    if best is None:
        raise RuntimeError("no fault-valid gadget")
    return {"case_id": case["case_id"], "gadget": best[1], "decoder": best[2]}


def run_solution(config):
    return {"cases": [_solve(case) for case in config["cases"]]}
# fmt: on
