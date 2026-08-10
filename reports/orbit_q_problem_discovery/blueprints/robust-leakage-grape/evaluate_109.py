from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import sys
import time
from itertools import product

import numpy as np
from scipy.linalg import expm


DEFAULT_SEED = 1092026


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


def _operators():
    lowering = np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2.0)], [0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
    number = np.diag([0.0, 1.0, 2.0]).astype(np.complex128)
    projector = np.diag([0.0, 0.0, 1.0]).astype(np.complex128)
    return (
        number,
        projector,
        lowering + lowering.conj().T,
        -1j * (lowering - lowering.conj().T),
    )


def _metrics(controls, config, members):
    number, projector, x_drive, y_drive = _operators()
    angle = config["target_angle"]
    target = np.array(
        [
            [np.cos(angle / 2), -1j * np.sin(angle / 2)],
            [-1j * np.sin(angle / 2), np.cos(angle / 2)],
        ],
        dtype=np.complex128,
    )
    infidelities, leakages = [], []
    for member in members:
        unitary = np.eye(3, dtype=np.complex128)
        drift = (
            member["detuning"] * number
            + (config["anharmonicity"] + member["anharmonic_shift"]) * projector
        )
        scale = 1.0 + member["amplitude_fraction"]
        for ux, uy in controls:
            hamiltonian = drift + 0.5 * scale * (ux * x_drive + uy * y_drive)
            unitary = expm(-1j * config["slice_duration"] * hamiltonian) @ unitary
        block = unitary[:2, :2]
        overlap = np.trace(target.conj().T @ block)
        infidelities.append(1.0 - abs(overlap) ** 2 / 4.0)
        leakages.append(max(0.0, 1.0 - np.sum(abs(block) ** 2) / 2.0))
    return np.asarray(infidelities), np.asarray(leakages)


def _configuration(seed):
    rng = np.random.default_rng(seed)
    widths = np.array([0.105, 0.075, 0.24]) * rng.uniform(0.96, 1.04, 3)
    keys = ("detuning", "amplitude_fraction", "anharmonic_shift")
    bounds = {key: [-float(width), float(width)] for key, width in zip(keys, widths)}
    anchors = [{key: 0.0 for key in keys}]
    for key, width in zip(keys, widths):
        for sign in (-1.0, 1.0):
            anchors.append(
                {name: (sign * float(width) if name == key else 0.0) for name in keys}
            )
    return {
        "n_levels": 3,
        "n_slices": 44,
        "slice_duration": 0.22,
        "anharmonicity": -4.85 + float(rng.uniform(-0.08, 0.08)),
        "target_angle": float(np.pi / 2 + rng.uniform(-0.025, 0.025)),
        "uncertainty_bounds": bounds,
        "training_ensemble": anchors,
        "max_drive_amplitude": 3.15,
        "max_slew_per_slice": 1.15,
        "max_edge_amplitude": 0.035,
        "maximum_worst_infidelity": 0.00125,
        "maximum_p95_infidelity": 0.0007,
        "maximum_worst_leakage": 0.00015,
        # Public solver-local randomness only: base seed modulo 2**31 - 1.
        "optimization_seed": int(seed % (2**31 - 1)),
    }


def _heldout(config, seed):
    keys = ("detuning", "amplitude_fraction", "anharmonic_shift")
    limits = [config["uncertainty_bounds"][key] for key in keys]
    corners = [
        {key: float(value) for key, value in zip(keys, values)}
        for values in product(*[(low, high) for low, high in limits])
    ]
    rng = np.random.default_rng(seed ^ 0x5A17C3)
    interior = []
    for point in rng.uniform(-1.0, 1.0, size=(48, 3)):
        interior.append(
            {
                key: float((high - low) * (value + 1) / 2 + low)
                for key, value, (low, high) in zip(keys, point, limits)
            }
        )
    return corners + interior


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_109")
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed(),
    )
    args = parser.parse_args()
    verifier_seed = os.environ.get("ORBIT_Q_CANDIDATE_SEED")
    if verifier_seed is not None and int(verifier_seed) != args.seed:
        raise ValueError("--seed cannot override the verifier-authorized seed")
    config = _configuration(args.seed)
    hidden_members = _heldout(config, args.seed)
    solution_config = copy.deepcopy(config)
    digest = case_digest(config)
    trusted_print = print
    trusted_json_dumps = json.dumps
    trusted_metrics = _metrics
    trusted_clock = time.perf_counter
    trusted_print(
        trusted_json_dumps(
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
    sys.argv[:] = [sys.argv[0]]
    started = trusted_clock()
    try:
        result = importlib.import_module(args.solution).run_solution(solution_config)
        elapsed = trusted_clock() - started
        required_keys = {
            "controls",
            "training_worst_infidelity",
            "training_worst_leakage",
        }
        if not isinstance(result, dict) or set(result) != required_keys:
            raise ValueError(
                "run_solution must return exactly controls, "
                "training_worst_infidelity, and training_worst_leakage"
            )
        controls = np.asarray(result["controls"], dtype=float)
        training_i, training_l = trusted_metrics(
            controls, config, config["training_ensemble"]
        )
        hidden_i, hidden_l = trusted_metrics(controls, config, hidden_members)
        amplitudes = np.linalg.norm(controls, axis=1)
        slew = np.linalg.norm(np.diff(controls, axis=0), axis=1)
        reported_i = float(result["training_worst_infidelity"])
        reported_l = float(result["training_worst_leakage"])
        summary = {
            "training_worst_infidelity": float(np.max(training_i)),
            "training_worst_leakage": float(np.max(training_l)),
            "heldout_worst_infidelity": float(np.max(hidden_i)),
            "heldout_p95_infidelity": float(np.quantile(hidden_i, 0.95)),
            "heldout_worst_leakage": float(np.max(hidden_l)),
            "max_drive_amplitude": float(np.max(amplitudes)),
            "max_slew_per_slice": float(np.max(slew)),
            "max_edge_amplitude": float(max(amplitudes[0], amplitudes[-1])),
            "controls_shape": list(controls.shape),
        }
        passed = (
            controls.shape == (config["n_slices"], 2)
            and np.all(np.isfinite(controls))
            and abs(reported_i - summary["training_worst_infidelity"]) < 2e-6
            and abs(reported_l - summary["training_worst_leakage"]) < 2e-6
            and summary["heldout_worst_infidelity"]
            <= config["maximum_worst_infidelity"]
            and summary["heldout_p95_infidelity"] <= config["maximum_p95_infidelity"]
            and summary["heldout_worst_leakage"] <= config["maximum_worst_leakage"]
            and summary["max_drive_amplitude"] <= config["max_drive_amplitude"] + 1e-9
            and summary["max_slew_per_slice"] <= config["max_slew_per_slice"] + 1e-9
            and summary["max_edge_amplitude"] <= config["max_edge_amplitude"] + 1e-9
        )
    except Exception as exc:
        elapsed = trusted_clock() - started
        passed = False
        summary = {"error": f"{type(exc).__name__}: {exc}"}
    trusted_print(f"End-to-end solution time: {elapsed:.6f}s")
    trusted_print(f"Case digest: {digest[:16]}")
    trusted_print(trusted_json_dumps(summary, sort_keys=True))
    if passed:
        admission_metrics = {
            "orbit_q_expert_admission_metrics": {
                "schema_version": 1,
                "protocol_seed": args.seed,
                "case_digest": digest,
                "metrics": [
                    {
                        "metric": "heldout_worst_infidelity",
                        "direction": "at_most",
                        "observed": summary["heldout_worst_infidelity"],
                        "threshold": config["maximum_worst_infidelity"],
                    },
                    {
                        "metric": "heldout_p95_infidelity",
                        "direction": "at_most",
                        "observed": summary["heldout_p95_infidelity"],
                        "threshold": config["maximum_p95_infidelity"],
                    },
                    {
                        "metric": "heldout_worst_leakage",
                        "direction": "at_most",
                        "observed": summary["heldout_worst_leakage"],
                        "threshold": config["maximum_worst_leakage"],
                    },
                    {
                        "metric": "maximum_drive_amplitude",
                        "direction": "at_most",
                        "observed": summary["max_drive_amplitude"],
                        "threshold": config["max_drive_amplitude"] + 1e-9,
                    },
                    {
                        "metric": "maximum_slew_per_slice",
                        "direction": "at_most",
                        "observed": summary["max_slew_per_slice"],
                        "threshold": config["max_slew_per_slice"] + 1e-9,
                    },
                    {
                        "metric": "maximum_edge_amplitude",
                        "direction": "at_most",
                        "observed": summary["max_edge_amplitude"],
                        "threshold": config["max_edge_amplitude"] + 1e-9,
                    },
                ],
            }
        }
        trusted_print(
            trusted_json_dumps(
                admission_metrics, sort_keys=True, separators=(",", ":")
            ),
            flush=True,
        )
    trusted_print("Overall: PASS" if passed else "Overall: FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
