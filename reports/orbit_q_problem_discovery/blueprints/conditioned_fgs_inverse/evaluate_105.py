from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time

import numpy as np


DEFAULT_SEED = 1052026


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


def _post_select(alpha, site, keep, n):
    row = site + n * (1 - keep)
    partner = (row + n) % (2 * n)
    pivot = int(np.argmax(np.abs(alpha[partner])))
    denominator = alpha[partner, pivot]
    if abs(denominator) < 1e-11:
        raise ValueError("postselection branch is numerically impossible")
    updated = alpha - np.outer(alpha[:, pivot], alpha[partner] / denominator)
    saved = alpha[:, pivot].copy()
    updated[:, pivot] = saved
    updated[row, :] = 0
    updated[row, pivot] = saved[row]
    updated[:, pivot] = 0
    updated[partner, pivot] = 1
    return np.linalg.qr(updated)[0]


def _simulate(theta, config, probe):
    n = config["n_sites"]
    alpha = np.zeros((2 * n, n), dtype=np.complex128)
    filled = set(config["filled"])
    for i in range(n):
        alpha[i + n if i in filled else i, i] = 1
    hopping, pairing, potential = theta
    phase, dt = probe["phase"], probe["dt"]
    for _ in range(config["steps"]):
        for i in range(n):
            chi = dt * potential * (-1) ** i * (1 + 0.13 * np.cos(phase + 0.31 * i))
            alpha[i] *= np.exp(-0.5j * chi)
            alpha[i + n] *= np.exp(0.5j * chi)
        for parity in (0, 1):
            for i in range(parity, n - 1, 2):
                chi = dt * hopping * (1 + 0.11 * np.sin(phase + 0.47 * i))
                c, s = np.cos(abs(chi) / 2), np.sin(abs(chi) / 2)
                p = chi / (abs(chi) + 1e-30)
                ri, rj, ril, rjl = (
                    alpha[i].copy(),
                    alpha[i + 1].copy(),
                    alpha[i + n].copy(),
                    alpha[i + 1 + n].copy(),
                )
                alpha[i] = c * ri - 1j * p * s * rj
                alpha[i + 1] = -1j * np.conj(p) * s * ri + c * rj
                alpha[i + n] = 1j * np.conj(p) * s * rjl + c * ril
                alpha[i + 1 + n] = 1j * p * s * ril + c * rjl
        for parity in (0, 1):
            for i in range(parity, n - 1, 2):
                chi = dt * pairing * np.exp(1j * (phase + 0.19 * i))
                c, s = np.cos(abs(chi) / 2), np.sin(abs(chi) / 2)
                p = chi / (abs(chi) + 1e-30)
                ri, rj, ril, rjl = (
                    alpha[i].copy(),
                    alpha[i + 1].copy(),
                    alpha[i + n].copy(),
                    alpha[i + 1 + n].copy(),
                )
                alpha[i] = c * ri - 1j * p * s * rjl
                alpha[i + 1] = c * rj + 1j * p * s * ril
                alpha[i + n] = 1j * np.conj(p) * s * rj + c * ril
                alpha[i + 1 + n] = -1j * np.conj(p) * s * ri + c * rjl
    log_probability = 0.0
    for site, outcome in zip(probe["measure_sites"], probe["outcomes"]):
        correlation = alpha @ alpha.conj().T
        p0 = float(np.real(correlation[site, site]))
        probability = p0 if outcome == 0 else 1 - p0
        log_probability += np.log(max(probability, 1e-15))
        alpha = _post_select(alpha, site, outcome, n)
    correlation = alpha @ alpha.conj().T
    features = [log_probability]
    features.extend(float(np.real(correlation[i, i])) for i in config["feature_sites"])
    for i, j in config["feature_pairs"]:
        features.extend(
            (float(np.real(correlation[i, j])), float(np.imag(correlation[i, j])))
        )
    for i, j in config["feature_pairs"]:
        value = correlation[i, j + n]
        features.extend((float(np.real(value)), float(np.imag(value))))
    return np.asarray(features)


def _configuration(seed):
    rng = np.random.default_rng(seed)
    n = 36
    hidden = np.array([0.73, 0.31, -0.22]) + rng.normal(0, [0.015, 0.01, 0.012])
    base = {
        "n_sites": n,
        "filled": list(range(0, n, 2)),
        "steps": 4,
        "feature_sites": [2, 5, 9, 14, 20, 27, 32],
        "feature_pairs": [[1, 4], [6, 11], [13, 18], [21, 25], [29, 34]],
        "parameter_bounds": [[0.35, 1.05], [0.08, 0.62], [-0.55, 0.28]],
    }
    probes = []
    for index, (phase, dt) in enumerate(
        ((0.17, 0.23), (0.83, 0.19), (1.41, 0.27), (2.02, 0.21), (2.63, 0.25))
    ):
        probes.append(
            {
                "phase": phase,
                "dt": dt,
                "measure_sites": [3 + index, 25 - index],
                "outcomes": [index % 2, (index // 2) % 2],
            }
        )
    training = []
    for probe in probes[:3]:
        training.append(
            {**probe, "observed_features": _simulate(hidden, base, probe).tolist()}
        )
    config = {**base, "training_probes": training, "heldout_probes": probes[3:]}
    return config, hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_105")
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
        module = importlib.import_module(args.solution)
        result = module.run_solution(config)
        elapsed = time.perf_counter() - started
        estimated = np.asarray(result["estimated_parameters"], dtype=float)
        predicted = np.asarray(result["heldout_features"], dtype=float)
        expected = np.stack(
            [_simulate(hidden, config, probe) for probe in config["heldout_probes"]]
        )
        parameter_error = float(np.max(np.abs(estimated - hidden)))
        prediction_error = float(np.max(np.abs(predicted - expected)))
        training_rmse = float(result["training_rmse"])
        passed = (
            estimated.shape == (3,)
            and predicted.shape == expected.shape
            and parameter_error < 0.012
            and prediction_error < 2e-4
            and training_rmse < 5e-5
            and np.all(np.isfinite(predicted))
        )
        summary = {
            "parameter_max_error": parameter_error,
            "heldout_max_error": prediction_error,
            "training_rmse": training_rmse,
            "parameter_shape": list(estimated.shape),
            "heldout_shape": list(predicted.shape),
        }
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
