from __future__ import annotations

import argparse
import importlib
import json
import time
from itertools import product

import numpy as np
from scipy.linalg import expm


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
    parser.add_argument("--seed", type=int, default=1092026)
    args = parser.parse_args()
    config = _configuration(args.seed)
    started = time.perf_counter()
    try:
        result = importlib.import_module(args.solution).run_solution(config)
        elapsed = time.perf_counter() - started
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
        training_i, training_l = _metrics(controls, config, config["training_ensemble"])
        hidden_i, hidden_l = _metrics(controls, config, _heldout(config, args.seed))
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
        elapsed = time.perf_counter() - started
        passed = False
        summary = {"error": f"{type(exc).__name__}: {exc}"}
    print(f"End-to-end solution time: {elapsed:.6f}s")
    print(json.dumps(summary, sort_keys=True))
    print("Overall: PASS" if passed else "Overall: FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
