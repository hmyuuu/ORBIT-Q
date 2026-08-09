"""Independent NumPy evaluator for coherent degenerate CSS inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time

import numpy as np


I2 = np.eye(2, dtype=np.complex128)
PAULI = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.diag([1, -1]).astype(np.complex128),
}
DEFAULT_SEED = 1132026


def default_seed():
    return int(
        os.environ.get(
            "ORBIT_Q_CANDIDATE_SEED",
            os.environ.get("ORBIT_COHERENT_CSS_SEED", str(DEFAULT_SEED)),
        )
    )


def _base_code():
    h = np.array(
        [[1, 0, 1, 0, 1, 0, 1], [0, 1, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]],
        dtype=np.uint8,
    )
    hx = np.zeros((6, 14), dtype=np.uint8)
    hx[:3, :7], hx[3:, 7:] = h, h
    hz = hx.copy()
    lx = np.zeros((2, 14), dtype=np.uint8)
    lx[0, :7], lx[1, 7:] = 1, 1
    return hx, hz, lx, lx.copy()


def _rowspace(rows):
    words = np.zeros((1 << len(rows), rows.shape[1]), dtype=np.uint8)
    for value in range(1, len(words)):
        low = value & -value
        words[value] = words[value ^ low] ^ rows[low.bit_length() - 1]
    return words


def _distance(stabilizers, logicals):
    space = _rowspace(stabilizers)
    best = stabilizers.shape[1] + 1
    for selector in range(1, 1 << len(logicals)):
        logical = np.zeros(stabilizers.shape[1], dtype=np.uint8)
        for row in range(len(logicals)):
            if selector >> row & 1:
                logical ^= logicals[row]
        best = min(best, int(np.min(np.sum(space ^ logical, axis=1))))
    return best


def _gf2_rank(rows):
    matrix = np.asarray(rows, dtype=np.uint8).copy()
    rank = 0
    for column in range(matrix.shape[1]):
        pivots = np.flatnonzero(matrix[rank:, column])
        if not len(pivots):
            continue
        pivot = rank + int(pivots[0])
        matrix[[rank, pivot]] = matrix[[pivot, rank]]
        for row in range(matrix.shape[0]):
            if row != rank and matrix[row, column]:
                matrix[row] ^= matrix[rank]
        rank += 1
        if rank == len(matrix):
            break
    return rank


def _scramble_code(rng):
    for _ in range(200):
        hx, hz, lx, lz = _base_code()
        cnots = []
        for _ in range(24):
            control, target = rng.choice(14, 2, replace=False).tolist()
            hx[:, target] ^= hx[:, control]
            lx[:, target] ^= lx[:, control]
            hz[:, control] ^= hz[:, target]
            lz[:, control] ^= lz[:, target]
            cnots.append([int(control), int(target)])
        permutation = rng.permutation(14)
        hx, hz, lx, lz = (matrix[:, permutation] for matrix in (hx, hz, lx, lz))
        if _distance(hx, lx) == 3 and _distance(hz, lz) == 3:
            break
    else:
        raise RuntimeError("failed to generate a distance-three CSS transform")
    for matrix in (hx, hz):
        for _ in range(18):
            source, target = rng.choice(6, 2, replace=False)
            matrix[target] ^= matrix[source]
        matrix[:] = matrix[rng.permutation(6)]
    assert not np.any((hx @ hz.T) & 1)
    assert np.array_equal((lx @ lz.T) & 1, np.eye(2, dtype=np.uint8))
    return hx, hz, lx, lz, cnots, permutation.tolist()


def _syndrome(hx, hz, x, z):
    return np.concatenate(((hx @ z) & 1, (hz @ x) & 1)).astype(int)


def _mask_record(x, z):
    return {"x": x.astype(int).tolist(), "z": z.astype(int).tolist()}


def _make_recoveries(rng, hx, hz):
    records, seen = [], set()
    candidates = [(np.zeros(14, dtype=np.uint8), np.zeros(14, dtype=np.uint8))]
    for weight in (1, 1, 2, 2, 2, 3):
        x, z = np.zeros(14, dtype=np.uint8), np.zeros(14, dtype=np.uint8)
        for site in rng.choice(14, weight, replace=False):
            label = rng.choice(list("XYZ"))
            x[site] = label in "XY"
            z[site] = label in "YZ"
        candidates.append((x, z))
    for x, z in candidates:
        syndrome = tuple(_syndrome(hx, hz, x, z))
        if syndrome in seen:
            continue
        seen.add(syndrome)
        records.append(
            {
                "syndrome_id": f"s{rng.integers(1 << 30):08x}",
                "syndrome": [int(bit) for bit in syndrome],
                "recovery": _mask_record(x, z),
            }
        )
        if len(records) == 4:
            return records
    raise RuntimeError("failed to generate four distinct syndromes")


def _labels_from_masks(x, z):
    labels = np.full(len(x), "I", dtype="<U1")
    labels[(x == 1) & (z == 0)] = "X"
    labels[(x == 0) & (z == 1)] = "Z"
    labels[(x == 1) & (z == 1)] = "Y"
    sites = np.flatnonzero(labels != "I")
    return sites.tolist(), labels[sites].tolist()


def _noise_branches(rng, recoveries):
    raw_weights = rng.uniform(0.6, 1.4, 3)
    weights = raw_weights / raw_weights.sum()
    branch_records = []
    forced = []
    for record in recoveries[1:]:
        x = np.asarray(record["recovery"]["x"], dtype=np.uint8)
        z = np.asarray(record["recovery"]["z"], dtype=np.uint8)
        forced.append(_labels_from_masks(x, z))
    pairs = [rng.choice(14, 2, replace=False).tolist() for _ in range(13)]
    pair_labels = [
        rng.choice(["XX", "ZZ", "XZ", "ZX", "YY", "XY", "YZ"]) for _ in pairs
    ]
    local_labels = rng.choice(list("XYZ"), size=14)
    for branch, weight in enumerate(weights):
        rotations = []
        for site, label in enumerate(local_labels):
            angle = rng.uniform(0.055, 0.19) * (1 + 0.11 * (branch - 1))
            rotations.append(
                {"sites": [site], "paulis": [str(label)], "angle": float(angle)}
            )
        for sites, labels in zip(pairs, pair_labels):
            rotations.append(
                {
                    "sites": [int(q) for q in sites],
                    "paulis": list(str(labels)),
                    "angle": float(rng.uniform(-0.14, 0.14) + 0.018 * (branch - 1)),
                }
            )
        for index, (sites, labels) in enumerate(forced):
            rotations.append(
                {
                    "sites": [int(q) for q in sites],
                    "paulis": list(labels),
                    "angle": float(
                        rng.uniform(0.22, 0.37) * (1 + 0.07 * branch) + 0.006 * index
                    ),
                }
            )
        rng.shuffle(rotations)
        branch_records.append(
            {
                "branch_id": f"b{rng.integers(1 << 30):08x}",
                "probability": float(weight),
                "rotations": rotations,
            }
        )
    return branch_records


def _frames(rng):
    frames = []
    for x in range(4):
        for z in range(4):
            frames.append(
                {
                    "id": f"f{rng.integers(1 << 30):08x}",
                    "logical_x": [(x >> 1) & 1, x & 1],
                    "logical_z": [(z >> 1) & 1, z & 1],
                }
            )
    rng.shuffle(frames)
    return frames


def _make_cases(seed, count=2):
    rng = np.random.default_rng(seed)
    cases = []
    for case_index in range(count):
        hx, hz, lx, lz, cnots, permutation = _scramble_code(rng)
        recoveries = _make_recoveries(rng, hx, hz)
        cases.append(
            {
                "case_id": f"css-{case_index}-{rng.integers(1 << 31):08x}",
                "n_qubits": 14,
                "n_logicals": 2,
                "x_stabilizers": hx.astype(int).tolist(),
                "z_stabilizers": hz.astype(int).tolist(),
                "logical_x": lx.astype(int).tolist(),
                "logical_z": lz.astype(int).tolist(),
                "syndrome_cases": recoveries,
                "frame_candidates": _frames(rng),
                "noise_branches": _noise_branches(rng, recoveries),
                "case_nonce": int(rng.integers(1 << 31)),
                "scramble_witness": {
                    "cnot_count": len(cnots),
                    "permutation_checksum": int(
                        sum((i + 1) * q for i, q in enumerate(permutation))
                    ),
                },
            }
        )
    return cases


def build_config(seed):
    cases = _make_cases(seed)
    payload = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    return {
        "cases": cases,
        "case_digest": hashlib.sha256(payload).hexdigest(),
        "pauli_convention": "P(x,z)=i^(x dot z) X^x Z^z; qubit zero is most significant",
    }


def _codewords(case):
    hx = np.asarray(case["x_stabilizers"], dtype=np.uint8)
    logical_x = np.asarray(case["logical_x"], dtype=np.uint8)
    support = _rowspace(hx)
    n = case["n_qubits"]
    powers = 1 << np.arange(n - 1, -1, -1)
    words = np.zeros((4, 1 << n), dtype=np.complex128)
    for logical in range(4):
        shift = ((logical >> 1) & 1) * logical_x[0] ^ (logical & 1) * logical_x[1]
        indices = (support ^ shift) @ powers
        words[logical, indices] = 1 / np.sqrt(len(support))
    return words


def _pauli_vector(state, x, z):
    n = len(x)
    indices = np.arange(1 << n, dtype=np.int64)
    x_integer = int(np.dot(x, 1 << np.arange(n - 1, -1, -1)))
    signs = np.ones(len(indices), dtype=np.int8)
    for site in np.flatnonzero(z):
        signs *= 1 - 2 * ((indices >> (n - 1 - site)) & 1)
    destination = indices ^ x_integer
    output = np.empty_like(state)
    output[destination] = (1j ** int(np.dot(x, z))) * signs * state
    return output


def _rotation(state, record):
    x = np.zeros(int(np.log2(state.size)), dtype=np.uint8)
    z = x.copy()
    for site, label in zip(record["sites"], record["paulis"]):
        x[site] = label in "XY"
        z[site] = label in "YZ"
    angle = record["angle"]
    return np.cos(angle / 2) * state - 1j * np.sin(angle / 2) * _pauli_vector(
        state, x, z
    )


def _logical_pauli(frame):
    matrix = np.array([[1]], dtype=np.complex128)
    for x, z in zip(frame["logical_x"], frame["logical_z"]):
        matrix = np.kron(
            matrix,
            (1j ** (x * z))
            * np.linalg.matrix_power(PAULI["X"], x)
            @ np.linalg.matrix_power(PAULI["Z"], z),
        )
    return matrix


def _oracle(case):
    words = _codewords(case)
    frames = [_logical_pauli(frame) for frame in case["frame_candidates"]]
    branch_states = []
    for branch in case["noise_branches"]:
        evolved = []
        for word in words:
            state = word.copy()
            for rotation in branch["rotations"]:
                state = _rotation(state, rotation)
            evolved.append(state)
        branch_states.append(np.asarray(evolved))
    outputs = []
    for syndrome_case in case["syndrome_cases"]:
        x = np.asarray(syndrome_case["recovery"]["x"], dtype=np.uint8)
        z = np.asarray(syndrome_case["recovery"]["z"], dtype=np.uint8)
        recovered_words = np.asarray([_pauli_vector(word, x, z) for word in words])
        chi = np.zeros((16, 16), dtype=np.complex128)
        for branch, states in zip(case["noise_branches"], branch_states):
            kraus = recovered_words.conj() @ states.T
            coefficients = np.asarray(
                [np.trace(frame.conj().T @ kraus) / 4 for frame in frames]
            )
            chi += branch["probability"] * np.outer(coefficients, coefficients.conj())
        probability = float(np.trace(chi).real)
        normalized = chi / probability
        weights = np.real(np.diag(normalized))
        outputs.append(
            {
                "syndrome_id": syndrome_case["syndrome_id"],
                "syndrome_probability": probability,
                "logical_chi": normalized,
                "coset_weights": weights,
                "best_frame_id": case["frame_candidates"][int(np.argmax(weights))][
                    "id"
                ],
            }
        )
    return {"case_id": case["case_id"], "syndromes": outputs}


def _validate_code(case):
    hx = np.asarray(case["x_stabilizers"], dtype=np.uint8)
    hz = np.asarray(case["z_stabilizers"], dtype=np.uint8)
    lx = np.asarray(case["logical_x"], dtype=np.uint8)
    lz = np.asarray(case["logical_z"], dtype=np.uint8)
    recoveries_ok = all(
        np.array_equal(
            _syndrome(
                hx,
                hz,
                np.asarray(record["recovery"]["x"], dtype=np.uint8),
                np.asarray(record["recovery"]["z"], dtype=np.uint8),
            ),
            np.asarray(record["syndrome"], dtype=int),
        )
        for record in case["syndrome_cases"]
    )
    frame_keys = {
        tuple(frame["logical_x"] + frame["logical_z"])
        for frame in case["frame_candidates"]
    }
    branch_probabilities = [branch["probability"] for branch in case["noise_branches"]]
    correlated = sum(
        len(rotation["sites"]) > 1
        for branch in case["noise_branches"]
        for rotation in branch["rotations"]
    )
    return (
        _gf2_rank(hx) == 6
        and _gf2_rank(hz) == 6
        and not np.any((hx @ hz.T) & 1)
        and not np.any((hx @ lz.T) & 1)
        and not np.any((hz @ lx.T) & 1)
        and np.array_equal((lx @ lz.T) & 1, np.eye(2, dtype=np.uint8))
        and _distance(hx, lx) == 3
        and _distance(hz, lz) == 3
        and recoveries_ok
        and len(frame_keys) == 16
        and len({frame["id"] for frame in case["frame_candidates"]}) == 16
        and np.isclose(sum(branch_probabilities), 1)
        and min(branch_probabilities) > 0
        and correlated >= 30
    )


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
    os.environ.pop("ORBIT_COHERENT_CSS_SEED", None)
    expected = [_oracle(case) for case in config["cases"]]
    started = time.perf_counter()
    result = importlib.import_module(module_name).run_solution(config)
    elapsed = time.perf_counter() - started
    submitted_cases = result.get("cases", []) if isinstance(result, dict) else []
    if not isinstance(submitted_cases, list):
        submitted_cases = []
    structure = len(submitted_cases) == len(expected)
    maximum_error, all_finite, invariants, identities = 0.0, True, True, True
    if structure:
        for case, submitted, target in zip(config["cases"], submitted_cases, expected):
            if not isinstance(submitted, dict):
                structure = False
                continue
            structure &= submitted.get("case_id") == target["case_id"]
            rows = submitted.get("syndromes", [])
            if not isinstance(rows, list):
                structure = False
                continue
            structure &= len(rows) == len(target["syndromes"])
            if len(rows) != len(target["syndromes"]):
                continue
            for row, wanted in zip(rows, target["syndromes"]):
                if not isinstance(row, dict):
                    structure = False
                    continue
                structure &= row.get("syndrome_id") == wanted["syndrome_id"]
                try:
                    probability = float(row["syndrome_probability"])
                    chi = np.asarray(row["logical_chi"], dtype=np.complex128)
                    weights = np.asarray(row["coset_weights"], dtype=float)
                    frame_id = row["best_frame_id"]
                except (KeyError, TypeError, ValueError):
                    structure = False
                    continue
                shape_ok = chi.shape == (16, 16) and weights.shape == (16,)
                structure &= shape_ok
                if not shape_ok:
                    continue
                all_finite &= bool(
                    np.all(np.isfinite(chi))
                    and np.all(np.isfinite(weights))
                    and np.isfinite(probability)
                )
                maximum_error = max(
                    maximum_error,
                    abs(probability - wanted["syndrome_probability"]),
                    float(np.max(np.abs(chi - wanted["logical_chi"]))),
                    float(np.max(np.abs(weights - wanted["coset_weights"]))),
                )
                invariants &= bool(
                    np.allclose(chi, chi.conj().T, atol=2e-9)
                    and np.isclose(np.trace(chi), 1, atol=2e-9)
                    and np.min(np.linalg.eigvalsh(chi)) > -2e-8
                    and np.allclose(weights, np.diag(chi).real, atol=2e-9)
                    and np.all(weights >= -2e-9)
                )
                identities &= frame_id == wanted["best_frame_id"]
    target_rows = [row for case in expected for row in case["syndromes"]]
    probabilities = np.asarray([row["syndrome_probability"] for row in target_rows])
    coherence = max(
        float(np.max(np.abs(row["logical_chi"] - np.diag(np.diag(row["logical_chi"])))))
        for row in target_rows
    )
    criteria = {
        "output structure and opaque identities": bool(structure and identities),
        "outputs finite": bool(all_finite),
        "independent coherent logical-coset oracle": bool(
            structure and maximum_error < 3e-8
        ),
        "conditional process invariants": bool(structure and invariants),
        "CSS degeneracy and distance active": all(
            _validate_code(case) for case in config["cases"]
        ),
        "coherent correlated regime active": coherence > 2e-4
        and np.min(probabilities) > 2e-6,
    }
    print("Problem 113 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; case digest: {config['case_digest'][:16]}")
    print(f"End-to-end solution time: {elapsed:.3f}s")
    print(f"Maximum oracle error: {maximum_error:.3e}")
    print(
        f"Syndrome-probability floor: {np.min(probabilities):.3e}; maximum logical coherence: {coherence:.3e}"
    )
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    passed = all(criteria.values())
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_113")
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed(),
    )
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.solution, args.seed) else 1)


if __name__ == "__main__":
    main()
