"""Independent evaluator for Problem 102.

The oracle deliberately uses only NumPy 2x2 matrix products and Chebyshev
identities.  It does not import or share circuit code with the expert solution.
"""

import argparse
import hashlib
import importlib
import json
import os
import time

import numpy as np
from numpy.polynomial.chebyshev import chebfit


DEFAULT_SEEDS = (102031, 102043, 102059)
BASE_SEED_OFFSETS = tuple(seed - DEFAULT_SEEDS[0] for seed in DEFAULT_SEEDS)


def seeds_from_base(base_seed):
    """Expand one verifier-only base seed to the three-instance schedule."""

    if base_seed < 0:
        raise ValueError("base seed must be non-negative")
    return tuple(base_seed + offset for offset in BASE_SEED_OFFSETS)


def default_seeds():
    """Preserve local defaults unless Harbor supplies its verifier-only seed."""

    base_seed = os.environ.get("ORBIT_Q_CANDIDATE_SEED")
    return DEFAULT_SEEDS if base_seed is None else seeds_from_base(int(base_seed))


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


def z_phase(phi):
    return np.diag([np.exp(1j * phi), np.exp(-1j * phi)])


def signal_matrix(x):
    off = 1j * np.sqrt(max(0.0, 1.0 - float(x) ** 2))
    return np.array([[x, off], [off, x]], dtype=np.complex128)


def oracle_unitary(phases, x):
    unitary = z_phase(float(phases[0]))
    signal = signal_matrix(x)
    for phase in phases[1:]:
        unitary = unitary @ signal @ z_phase(float(phase))
    return unitary


def oracle_response(phases, points):
    return np.array([oracle_unitary(phases, x)[0, 0] for x in points])


def symmetric_phases(rng, degree):
    half_size = degree // 2 + 1
    half = rng.uniform(-0.62, 0.62, size=half_size)
    half += rng.choice((-1.0, 1.0), size=half_size) * np.linspace(0.04, 0.19, half_size)
    return np.concatenate([half, half[-2::-1]])


def generate_case(seed, degree=8):
    rng = np.random.default_rng(seed)
    planted = symmetric_phases(rng, degree)
    fit_x = np.cos(np.pi * (np.arange(4 * degree + 9) + 0.5) / (4 * degree + 9))
    fit_y = oracle_response(planted, fit_x)
    coeff = chebfit(fit_x, fit_y, degree)
    coeff[np.arange(degree + 1) % 2 != degree % 2] = 0.0
    training_x = np.cos(np.pi * (np.arange(3 * degree + 7) + 0.37) / (3 * degree + 7))
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "degree": degree,
                "real": np.real(coeff).round(14).tolist(),
                "imag": np.imag(coeff).round(14).tolist(),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]
    config = {
        "degree": degree,
        "target_chebyshev_real": np.real(coeff),
        "target_chebyshev_imag": np.imag(coeff),
        "training_x": training_x,
        "response_tolerance": 1.5e-3,
        "instance_fingerprint": fingerprint,
        "max_iterations": 900,
    }
    return config, planted


def evaluate_case(module, seed):
    config, planted = generate_case(seed)
    started = time.perf_counter()
    result = module.run_solution(config)
    elapsed = time.perf_counter() - started

    phases = np.asarray(result.get("phases"), dtype=float)
    degree = config["degree"]
    hidden_rng = np.random.default_rng(seed ^ 0x5A17)
    hidden_x = np.concatenate(
        [np.linspace(-0.997, 0.997, 257), hidden_rng.uniform(-0.999, 0.999, 191)]
    )
    target = oracle_response(planted, hidden_x)
    candidate = (
        oracle_response(phases, hidden_x)
        if phases.shape == (degree + 1,)
        else np.full_like(target, np.nan)
    )
    errors = np.abs(candidate - target)
    rmse = float(np.sqrt(np.mean(errors**2)))
    max_error = float(np.max(errors))
    symmetry_error = (
        float(np.max(np.abs(phases - phases[::-1])))
        if phases.shape == (degree + 1,)
        else np.inf
    )
    principal = bool(
        phases.shape == (degree + 1,) and np.all(np.abs(phases) <= np.pi + 1e-12)
    )
    finite_real = bool(phases.shape == (degree + 1,) and np.all(np.isfinite(phases)))

    probe_x = hidden_x[::41]
    unitarity = (
        max(
            float(
                np.linalg.norm(
                    oracle_unitary(phases, x).conj().T @ oracle_unitary(phases, x)
                    - np.eye(2)
                )
            )
            for x in probe_x
        )
        if finite_real
        else np.inf
    )
    parity = (
        float(
            np.max(
                np.abs(
                    oracle_response(phases, probe_x)
                    - ((-1) ** degree) * oracle_response(phases, -probe_x)
                )
            )
        )
        if finite_real
        else np.inf
    )

    criteria = {
        "result keys exactly phases": isinstance(result, dict)
        and set(result) == {"phases"},
        "phase shape": phases.shape == (degree + 1,),
        "finite real phases": finite_real,
        "principal phase representatives": principal,
        "reflection symmetry": symmetry_error <= 2e-5,
        "hidden response rmse": rmse <= 7.5e-4,
        "hidden response maximum error": max_error <= config["response_tolerance"],
        "unitarity identity": unitarity <= 2e-12,
        "parity identity": parity <= 2e-10,
    }
    metrics = {
        "seed": seed,
        "fingerprint": config["instance_fingerprint"],
        "elapsed": elapsed,
        "rmse": rmse,
        "max_error": max_error,
        "symmetry_error": symmetry_error,
        "unitarity_defect": unitarity,
        "parity_error": parity,
    }
    return criteria, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_102")
    parser.add_argument("--seeds", default=",".join(map(str, default_seeds())))
    args = parser.parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds or seeds != seeds_from_base(seeds[0]):
        raise ValueError("--seeds must be an ordered (b, b + 12, b + 28) schedule")
    verifier_seed = os.environ.get("ORBIT_Q_CANDIDATE_SEED")
    if verifier_seed is not None and int(verifier_seed) != seeds[0]:
        raise ValueError("--seeds cannot override the verifier-authorized base seed")
    digest = case_digest([generate_case(seed)[0] for seed in seeds])
    print(
        json.dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": seeds[0],
                    "case_digest": digest,
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    os.environ.pop("ORBIT_Q_CANDIDATE_SEED", None)
    module = importlib.import_module(args.solution)

    all_pass = True
    total_time = 0.0
    print("Problem 102 evaluation")
    print(f"Solution module: {args.solution}")
    print(f"Case seeds: {','.join(map(str, seeds))}")
    print(f"Case digest: {digest[:16]}")
    for seed in seeds:
        criteria, metrics = evaluate_case(module, seed)
        total_time += metrics["elapsed"]
        all_pass &= all(criteria.values())
        print(
            f"Instance {metrics['fingerprint']}: RMSE={metrics['rmse']:.3e}, "
            f"max={metrics['max_error']:.3e}, symmetry={metrics['symmetry_error']:.3e}"
        )
        for name, passed in criteria.items():
            print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"End-to-end solution time: {total_time:.2f}s")
    print("Runtime is reported separately and does not change functional correctness")
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
