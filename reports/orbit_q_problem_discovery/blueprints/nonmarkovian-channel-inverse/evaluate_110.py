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
PAULI = {"I": I2, "X": X, "Y": Y, "Z": Z}
DEFAULT_SEED = 1102026


def default_seed():
    return int(os.environ.get("ORBIT_Q_CANDIDATE_SEED", str(DEFAULT_SEED)))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def case_digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rotation(axis, angle):
    return np.cos(angle / 2) * I2 - 1j * np.sin(angle / 2) * PAULI[axis]


def _collision(parameters, duration):
    g, phase, zeta = parameters[:3]
    hamiltonian = np.diag([zeta / 2, -zeta / 2, -zeta / 2, zeta / 2]).astype(
        np.complex128
    )
    hamiltonian[1, 2] = g * np.exp(-1j * phase)
    hamiltonian[2, 1] = g * np.exp(1j * phase)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    return (eigenvectors * np.exp(-1j * duration * eigenvalues)) @ eigenvectors.conj().T


def _apply_channel(rho, operators, target):
    result = np.zeros_like(rho)
    for operator in operators:
        embedded = np.kron(operator, I2) if target == 0 else np.kron(I2, operator)
        result += embedded @ rho @ embedded.conj().T
    return result


def _simulate(parameters, config, probe):
    system = _rotation("Z", probe["system_prep"][1]) @ _rotation(
        "Y", probe["system_prep"][0]
    )
    memory = _rotation("Z", probe["memory_prep"][1]) @ _rotation(
        "Y", probe["memory_prep"][0]
    )
    state = np.kron(system, memory) @ np.array([1, 0, 0, 0], dtype=np.complex128)
    rho = np.outer(state, state.conj())
    omega, rate, ground_weight = parameters[3:]
    reset = (
        np.array([[1, 0], [0, 0]], dtype=np.complex128),
        np.array([[0, 1], [0, 0]], dtype=np.complex128),
    )
    readouts = set(config["readout_after"])
    features = []
    for step_number, step in enumerate(probe["steps"], start=1):
        if step["reset_system"]:
            rho = _apply_channel(rho, reset, 0)
            polar, azimuth = step["reset_prepare"]
            preparation = _rotation("Z", azimuth) @ _rotation("Y", polar)
            embedded = np.kron(preparation, I2)
            rho = embedded @ rho @ embedded.conj().T
        control = (
            _rotation("Z", step["control"][2])
            @ _rotation("Y", step["control"][1])
            @ _rotation("X", step["control"][0])
        )
        embedded = np.kron(control, I2)
        rho = embedded @ rho @ embedded.conj().T
        duration = step["duration"]
        collision = _collision(parameters, duration)
        rho = collision @ rho @ collision.conj().T
        memory_phase = np.kron(I2, _rotation("Z", omega * duration))
        rho = memory_phase @ rho @ memory_phase.conj().T
        gamma = 1 - np.exp(-rate * duration)
        damping = (
            np.sqrt(ground_weight)
            * np.diag([1, np.sqrt(1 - gamma)]).astype(np.complex128),
            np.sqrt(ground_weight)
            * np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=np.complex128),
            np.sqrt(1 - ground_weight)
            * np.diag([np.sqrt(1 - gamma), 1]).astype(np.complex128),
            np.sqrt(1 - ground_weight)
            * np.array([[0, 0], [np.sqrt(gamma), 0]], dtype=np.complex128),
        )
        rho = _apply_channel(rho, damping, 1)
        if step_number in readouts:
            for label in config["observables"]:
                value = np.trace(np.kron(PAULI[label[0]], PAULI[label[1]]) @ rho)
                features.append(float(np.real(value)))
    return np.asarray(features)


def _probe(rng, index):
    steps = []
    reset_at = 3 + index % 3
    for step_index in range(8):
        reset_system = step_index + 1 == reset_at
        steps.append(
            {
                "duration": float(rng.uniform(0.14, 0.31)),
                "control": rng.uniform(
                    [-0.75, -0.62, -0.58], [0.75, 0.62, 0.58]
                ).tolist(),
                "reset_system": reset_system,
                "reset_prepare": (
                    rng.uniform([0.28, -2.4], [2.72, 2.4]).tolist()
                    if reset_system
                    else None
                ),
            }
        )
    return {
        "system_prep": rng.uniform([0.25, -2.5], [2.80, 2.5]).tolist(),
        "memory_prep": rng.uniform([0.30, -2.4], [2.65, 2.4]).tolist(),
        "steps": steps,
    }


def _configuration(seed):
    rng = np.random.default_rng(seed)
    bounds = np.array(
        [
            [0.38, 1.18],
            [-1.25, 0.95],
            [-0.42, 0.68],
            [-0.88, 0.72],
            [0.16, 0.88],
            [0.62, 0.94],
        ]
    )
    hidden = np.array([0.79, -0.38, 0.21, -0.27, 0.49, 0.79]) + rng.normal(
        0, [0.035, 0.065, 0.035, 0.05, 0.025, 0.012]
    )
    probes = [_probe(rng, index) for index in range(8)]
    base = {
        "n_qubits": 2,
        "system_qubit": 0,
        "memory_qubit": 1,
        "parameter_names": [
            "g",
            "phi",
            "zeta",
            "omega",
            "damping_rate",
            "bath_ground_weight",
        ],
        "parameter_bounds": bounds.tolist(),
        "readout_after": [2, 5, 8],
        "observables": ["XI", "YI", "ZI"],
    }
    training = [
        {**probe, "observed_features": _simulate(hidden, base, probe).tolist()}
        for probe in probes[:5]
    ]
    return {**base, "training_probes": training, "heldout_probes": probes[5:]}, hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_110")
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed(),
    )
    args = parser.parse_args()
    config, hidden = _configuration(args.seed)
    digest = case_digest(config)
    print(
        json.dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": args.seed,
                    "case_digest": digest,
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    os.environ.pop("ORBIT_Q_CANDIDATE_SEED", None)
    started = time.perf_counter()
    try:
        result = importlib.import_module(args.solution).run_solution(config)
        elapsed = time.perf_counter() - started
        estimated = np.asarray(result["estimated_parameters"], dtype=float)
        predicted = np.asarray(result["heldout_features"], dtype=float)
        expected = np.stack(
            [_simulate(hidden, config, probe) for probe in config["heldout_probes"]]
        )
        reported_training_rmse = float(result["training_rmse"])
        training_observed = np.concatenate(
            [
                np.asarray(probe["observed_features"], dtype=float)
                for probe in config["training_probes"]
            ]
        )
        training_predicted = np.concatenate(
            [_simulate(estimated, config, probe) for probe in config["training_probes"]]
        )
        verified_training_rmse = float(
            np.sqrt(np.mean((training_predicted - training_observed) ** 2))
        )
        training_rmse_report_error = abs(
            reported_training_rmse - verified_training_rmse
        )
        parameter_error = float(np.max(np.abs(estimated - hidden)))
        prediction_error = float(np.max(np.abs(predicted - expected)))
        summary = {
            "parameter_max_error": parameter_error,
            "heldout_max_error": prediction_error,
            "reported_training_rmse": reported_training_rmse,
            "verified_training_rmse": verified_training_rmse,
            "training_rmse_report_error": training_rmse_report_error,
            "parameter_shape": list(estimated.shape),
            "heldout_shape": list(predicted.shape),
        }
        passed = (
            set(result) == {"estimated_parameters", "training_rmse", "heldout_features"}
            and estimated.shape == (6,)
            and predicted.shape == expected.shape
            and np.all(np.isfinite(estimated))
            and np.all(np.isfinite(predicted))
            and np.isfinite(reported_training_rmse)
            and parameter_error < 0.012
            and prediction_error < 2e-4
            and verified_training_rmse < 5e-5
            and training_rmse_report_error < 1e-8
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        passed = False
        summary = {"error": f"{type(exc).__name__}: {exc}"}
    print(f"End-to-end solution time: {elapsed:.6f}s")
    print(f"Case seed: {args.seed}; case digest: {digest[:16]}")
    print(json.dumps(summary, sort_keys=True))
    print("Overall: PASS" if passed else "Overall: FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
