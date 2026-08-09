"""Independent NumPy oracle for a near-exceptional-point Liouvillian."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time

import numpy as np


I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
SM = np.array([[0, 1], [0, 0]], dtype=np.complex128)
PAULI = np.asarray([X, Y, Z])
DEFAULT_SEED = 1142026


def default_seed():
    return int(
        os.environ.get(
            "ORBIT_Q_CANDIDATE_SEED",
            os.environ.get("ORBIT_LIOUVILLIAN_EP_SEED", str(DEFAULT_SEED)),
        )
    )


def _expm(matrix):
    values, vectors = np.linalg.eig(matrix)
    return (vectors * np.exp(values)) @ np.linalg.inv(vectors)


def _rotation(axis, angle):
    operator = {"X": X, "Y": Y, "Z": Z}[axis]
    return np.cos(angle / 2) * I2 - 1j * np.sin(angle / 2) * operator


def _frame(alpha, beta):
    return _rotation("Z", alpha) @ _rotation("Y", beta)


def _liouvillian(parameters):
    down, up, dephasing, omega, alpha, beta = parameters
    frame = _frame(alpha, beta)
    hamiltonian = frame @ (0.5 * omega * X) @ frame.conj().T
    jumps = [
        np.sqrt(down) * (frame @ SM @ frame.conj().T),
        np.sqrt(up) * (frame @ SM.conj().T @ frame.conj().T),
        np.sqrt(dephasing / 2) * (frame @ Z @ frame.conj().T),
    ]
    generator = -1j * (np.kron(I2, hamiltonian) - np.kron(hamiltonian.T, I2))
    for jump in jumps:
        square = jump.conj().T @ jump
        generator += np.kron(jump.conj(), jump)
        generator -= 0.5 * (np.kron(I2, square) + np.kron(square.T, I2))
    return generator


def _apply_super(operator, matrix):
    return (operator @ matrix.reshape(-1, order="F")).reshape(2, 2, order="F")


def _affine(operator):
    transfer = np.empty((3, 3), dtype=float)
    for row, observable in enumerate(PAULI):
        for column, probe in enumerate(PAULI):
            transfer[row, column] = (
                0.5 * np.trace(observable @ _apply_super(operator, probe)).real
            )
    shift = np.asarray(
        [
            0.5 * np.trace(observable @ _apply_super(operator, I2)).real
            for observable in PAULI
        ]
    )
    return transfer, shift


def _choi(operator):
    result = np.zeros((4, 4), dtype=np.complex128)
    for ket in range(2):
        for bra in range(2):
            basis = np.zeros((2, 2), dtype=np.complex128)
            basis[ket, bra] = 1
            input_part = np.zeros((2, 2), dtype=np.complex128)
            input_part[ket, bra] = 1
            result += np.kron(input_part, _apply_super(operator, basis))
    return 0.5 * (result + result.conj().T)


def _kraus_from_super(operator):
    values, vectors = np.linalg.eigh(_choi(operator))
    return [
        np.sqrt(value) * vectors[:, index].reshape(2, 2).T
        for index, value in enumerate(values)
        if value > 2e-13
    ]


def _super_from_kraus(operators):
    return sum(np.kron(operator.conj(), operator) for operator in operators)


def _bloch(matrix):
    return np.asarray([np.trace(observable @ matrix).real for observable in PAULI])


def _initial_density(probe):
    state = _rotation("Z", probe[1]) @ _rotation("Y", probe[0]) @ np.array([1, 0])
    return np.outer(state, state.conj())


def _features(parameters, case, probes):
    channel = _expm(case["time_step"] * _liouvillian(parameters))
    axes = np.asarray(case["measurement_axes"], dtype=float)
    rows = []
    for probe in probes:
        rho = _initial_density(probe["preparation"])
        readouts = set(probe["readout_steps"])
        values = {}
        for step in range(1, max(readouts) + 1):
            rho = _apply_super(channel, rho)
            if step in readouts:
                values[step] = axes @ _bloch(rho)
        rows.append([values[step] for step in probe["readout_steps"]])
    return np.asarray(rows, dtype=float)


def _orthogonal(rng):
    matrix, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def _unit_vector(rng):
    value = rng.normal(size=3)
    return value / np.linalg.norm(value)


def _make_case(rng, case_index, discriminant_sign):
    down = float(rng.uniform(0.52, 0.60))
    up = float(rng.uniform(0.10, 0.15))
    dephasing = float(rng.uniform(0.08, 0.13))
    longitudinal = down + up
    transverse = longitudinal / 2 + dephasing
    difference = longitudinal - transverse
    normalized = discriminant_sign * float(rng.uniform(0.0045, 0.0115))
    discriminant = normalized * (longitudinal + transverse) ** 2
    omega = 0.5 * np.sqrt(difference**2 - discriminant)
    alpha = float(rng.uniform(-0.95, 0.95))
    beta = float(rng.uniform(0.48, 2.48))
    hidden = np.asarray([down, up, dephasing, omega, alpha, beta])
    training = [
        {
            "preparation": rng.uniform([0.22, -np.pi], [2.92, np.pi]).tolist(),
            "readout_steps": [2, 4, 7],
        }
        for _ in range(6)
    ]
    heldout = [
        {
            "preparation": rng.uniform([0.18, -np.pi], [2.96, np.pi]).tolist(),
            "readout_steps": [1, 3, 6, 9],
        }
        for _ in range(2)
    ]
    case = {
        "case_id": f"lep-{case_index}-{rng.integers(1 << 31):08x}",
        "parameter_names": [
            "gamma_down",
            "gamma_up",
            "gamma_phi",
            "drive_omega",
            "frame_alpha",
            "frame_beta",
        ],
        "parameter_bounds": [
            [0.43, 0.68],
            [0.05, 0.22],
            [0.035, 0.20],
            [0.025, 0.24],
            [-1.28, 1.28],
            [0.28, 2.82],
        ],
        "time_step": float(rng.uniform(0.56, 0.82)),
        "measurement_axes": _orthogonal(rng).tolist(),
        "training_probes": training,
        "heldout_probes": heldout,
        "response_pairs": [
            {
                "id": f"r{rng.integers(1 << 30):08x}",
                "source": _unit_vector(rng).tolist(),
                "readout": _unit_vector(rng).tolist(),
            }
            for _ in range(3)
        ],
        "response_frequencies": np.sort(rng.uniform(0.04, 0.72, 5)).tolist(),
        "case_nonce": int(rng.integers(1 << 31)),
    }
    observed = _features(hidden, case, training)
    for record, values in zip(training, observed):
        record["observed_features"] = values.tolist()
    return case, hidden


def _configuration(seed):
    rng = np.random.default_rng(seed)
    signs = np.asarray([1, -1])
    rng.shuffle(signs)
    cases, hidden = [], []
    for index, sign in enumerate(signs):
        case, parameters = _make_case(rng, index, int(sign))
        cases.append(case)
        hidden.append(parameters)
    return {
        "cases": cases,
        "response_convention": "chi(omega)=m^T(i*omega*I-G)^(-1)b",
        "eigenvalue_order": "mu_plus, mu_minus, mu_transverse; principal square root",
    }, hidden


def _case_digest(config):
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _analysis(parameters, case):
    generator = _liouvillian(parameters)
    transfer, shift = _affine(generator)
    down, up, dephasing, omega = parameters[:4]
    longitudinal = down + up
    transverse = longitudinal / 2 + dephasing
    discriminant = (longitudinal - transverse) ** 2 - 4 * omega**2
    root = np.sqrt(complex(discriminant))
    eigenvalues = np.asarray(
        [
            -(longitudinal + transverse) / 2 + root / 2,
            -(longitudinal + transverse) / 2 - root / 2,
            -transverse,
        ]
    )
    projectors = []
    identity = np.eye(3, dtype=np.complex128)
    for index, value in enumerate(eigenvalues):
        projector = identity.copy()
        for other_index, other in enumerate(eigenvalues):
            if other_index != index:
                projector = projector @ (transfer - other * identity) / (value - other)
        projectors.append(projector)
    residues = []
    susceptibilities = []
    for pair in case["response_pairs"]:
        source = np.asarray(pair["source"])
        readout = np.asarray(pair["readout"])
        residues.append([readout @ projector @ source for projector in projectors])
        susceptibilities.append(
            [
                readout @ np.linalg.solve(1j * frequency * identity - transfer, source)
                for frequency in case["response_frequencies"]
            ]
        )
    theta_ep = abs(longitudinal - transverse) / 2
    return {
        "liouvillian_matrix": transfer,
        "steady_bloch": -np.linalg.solve(transfer, shift),
        "ep_discriminant": float(discriminant),
        "signed_ep_offset": float(omega - theta_ep),
        "liouvillian_eigenvalues": eigenvalues,
        "modal_residues": np.asarray(residues),
        "susceptibilities": np.asarray(susceptibilities),
    }


def _canary():
    parameters = np.asarray([0.57, 0.12, 0.105, 0.115, 0.43, 1.17])
    generator = _liouvillian(parameters)
    transfer, shift = _affine(generator)
    down, up, dephasing, omega, alpha, beta = parameters
    longitudinal = down + up
    transverse = longitudinal / 2 + dephasing
    canonical = np.asarray(
        [
            [-transverse, 0, 0],
            [0, -transverse, -omega],
            [0, omega, -longitudinal],
        ]
    )
    frame = _frame(alpha, beta)
    rotation = np.asarray(
        [
            [
                0.5 * np.trace(PAULI[i] @ frame @ PAULI[j] @ frame.conj().T).real
                for j in range(3)
            ]
            for i in range(3)
        ]
    )
    expected_transfer = rotation @ canonical @ rotation.T
    expected_shift = rotation @ np.asarray([0, 0, down - up])
    channel = _expm(0.71 * generator)
    kraus = _kraus_from_super(channel)
    return (
        np.allclose(transfer, expected_transfer, atol=4e-13, rtol=4e-13)
        and np.allclose(shift, expected_shift, atol=4e-13, rtol=4e-13)
        and np.allclose(_super_from_kraus(kraus), channel, atol=2e-12, rtol=2e-12)
        and np.allclose(sum(item.conj().T @ item for item in kraus), I2, atol=2e-12)
        and np.min(np.linalg.eigvalsh(_choi(channel))) > -2e-12
    )


def _array(record, key, dtype):
    try:
        return np.asarray(record[key], dtype=dtype)
    except (KeyError, TypeError, ValueError):
        return np.asarray([], dtype=dtype)


def evaluate(module_name, seed):
    config, hidden = _configuration(seed)
    digest = _case_digest(config)
    print(
        json.dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": seed,
                    "case_digest": digest,
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    os.environ.pop("ORBIT_Q_CANDIDATE_SEED", None)
    os.environ.pop("ORBIT_LIOUVILLIAN_EP_SEED", None)
    if not _canary():
        raise AssertionError("analytic Liouvillian/Kraus canary failed")
    expected = []
    for case, parameters in zip(config["cases"], hidden):
        expected.append(
            {
                "parameters": parameters,
                "heldout_features": _features(parameters, case, case["heldout_probes"]),
                **_analysis(parameters, case),
            }
        )
    started = time.perf_counter()
    try:
        result = importlib.import_module(module_name).run_solution(config)
        elapsed = time.perf_counter() - started
        submitted = result["cases"] if isinstance(result, dict) else []
    except Exception as exc:
        elapsed = time.perf_counter() - started
        submitted = []
        print(f"Solution exception: {type(exc).__name__}: {exc}")
    structure = isinstance(submitted, list) and len(submitted) == len(expected)
    finite, identities, invariants = True, True, True
    errors = {
        name: 0.0
        for name in (
            "parameter",
            "training",
            "heldout",
            "generator",
            "steady",
            "spectral",
            "response",
        )
    }
    if structure:
        for case, row, wanted in zip(config["cases"], submitted, expected):
            if not isinstance(row, dict):
                structure = False
                continue
            identities &= row.get("case_id") == case["case_id"]
            parameters = _array(row, "estimated_parameters", float)
            heldout = _array(row, "heldout_features", float)
            generator = _array(row, "liouvillian_matrix", float)
            steady = _array(row, "steady_bloch", float)
            eigenvalues = _array(row, "liouvillian_eigenvalues", complex)
            residues = _array(row, "modal_residues", complex)
            susceptibility = _array(row, "susceptibilities", complex)
            shapes = (
                parameters.shape == (6,)
                and heldout.shape == wanted["heldout_features"].shape
                and generator.shape == (3, 3)
                and steady.shape == (3,)
                and eigenvalues.shape == (3,)
                and residues.shape == wanted["modal_residues"].shape
                and susceptibility.shape == wanted["susceptibilities"].shape
            )
            structure &= shapes
            if not shapes:
                continue
            try:
                reported_rmse = float(row["training_rmse"])
                reported_discriminant = float(row["ep_discriminant"])
                reported_offset = float(row["signed_ep_offset"])
            except (KeyError, TypeError, ValueError):
                structure = False
                continue
            arrays = [
                parameters,
                heldout,
                generator,
                steady,
                eigenvalues,
                residues,
                susceptibility,
            ]
            finite &= all(np.all(np.isfinite(value)) for value in arrays)
            finite &= np.isfinite(
                reported_rmse + reported_discriminant + reported_offset
            )
            predicted_training = _features(parameters, case, case["training_probes"])
            observed_training = np.asarray(
                [probe["observed_features"] for probe in case["training_probes"]]
            )
            verified_rmse = float(
                np.sqrt(np.mean((predicted_training - observed_training) ** 2))
            )
            errors["parameter"] = max(
                errors["parameter"],
                float(np.max(np.abs(parameters - wanted["parameters"]))),
            )
            errors["training"] = max(
                errors["training"], verified_rmse, abs(reported_rmse - verified_rmse)
            )
            errors["heldout"] = max(
                errors["heldout"],
                float(np.max(np.abs(heldout - wanted["heldout_features"]))),
            )
            errors["generator"] = max(
                errors["generator"],
                float(np.max(np.abs(generator - wanted["liouvillian_matrix"]))),
            )
            errors["steady"] = max(
                errors["steady"], float(np.max(np.abs(steady - wanted["steady_bloch"])))
            )
            errors["spectral"] = max(
                errors["spectral"],
                abs(reported_discriminant - wanted["ep_discriminant"]),
                abs(reported_offset - wanted["signed_ep_offset"]),
                float(np.max(np.abs(eigenvalues - wanted["liouvillian_eigenvalues"]))),
            )
            errors["response"] = max(
                errors["response"],
                float(np.max(np.abs(residues - wanted["modal_residues"]))),
                float(np.max(np.abs(susceptibility - wanted["susceptibilities"]))),
            )
            modal = np.asarray(
                [
                    [
                        np.sum(residues[pair] / (1j * frequency - eigenvalues))
                        for frequency in case["response_frequencies"]
                    ]
                    for pair in range(len(case["response_pairs"]))
                ]
            )
            invariants &= bool(
                np.allclose(modal, susceptibility, atol=8e-7, rtol=8e-7)
                and np.linalg.norm(steady) <= 1 + 2e-7
            )
    normalized = [
        item["ep_discriminant"]
        / (
            item["parameters"][0]
            + item["parameters"][1]
            + (item["parameters"][0] + item["parameters"][1]) / 2
            + item["parameters"][2]
        )
        ** 2
        for item in expected
    ]
    gaps = [
        abs(item["liouvillian_eigenvalues"][0] - item["liouvillian_eigenvalues"][1])
        for item in expected
    ]
    all_mode_gaps = [
        min(
            abs(
                item["liouvillian_eigenvalues"][left]
                - item["liouvillian_eigenvalues"][right]
            )
            for left in range(3)
            for right in range(left)
        )
        for item in expected
    ]
    nonnormality = [
        np.linalg.norm(
            item["liouvillian_matrix"] @ item["liouvillian_matrix"].T
            - item["liouvillian_matrix"].T @ item["liouvillian_matrix"]
        )
        for item in expected
    ]
    criteria = {
        "output structure and opaque identities": bool(structure and identities),
        "outputs finite and response identity": bool(finite and invariants),
        "independent parameter inference": errors["parameter"] < 3e-4
        and errors["training"] < 4e-7,
        "independent heldout response": errors["heldout"] < 4e-6,
        "independent Liouvillian and steady state": errors["generator"] < 4e-5
        and errors["steady"] < 4e-5,
        "exceptional-point spectral data": errors["spectral"] < 6e-5,
        "biorthogonal residues and susceptibility": errors["response"] < 6e-4,
        "near but separated non-Hermitian regime": min(
            abs(value) for value in normalized
        )
        > 0.004
        and max(abs(value) for value in normalized) < 0.012
        and np.sign(normalized[0]) != np.sign(normalized[1])
        and min(gaps) > 0.06
        and min(all_mode_gaps) > 0.03
        and min(nonnormality) > 0.02,
    }
    print("Problem 114 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; case digest: {digest[:16]}")
    print(f"End-to-end solution time: {elapsed:.3f}s")
    print(
        "Maximum errors: "
        + ", ".join(f"{name}={value:.3e}" for name, value in errors.items())
    )
    print(
        "EP margins: "
        + ", ".join(
            f"normalized_D={value:+.3e}, EP_split={gap:.3e}, all_gap={all_gap:.3e}, nonnormality={non:.3e}"
            for value, gap, all_gap, non in zip(
                normalized, gaps, all_mode_gaps, nonnormality
            )
        )
    )
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    passed = all(criteria.values())
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_114")
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed(),
    )
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.solution, args.seed) else 1)


if __name__ == "__main__":
    main()
