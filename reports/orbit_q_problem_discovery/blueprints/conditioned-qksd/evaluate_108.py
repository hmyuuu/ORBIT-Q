"""Independent oracle for shot-conditioned fermionic QKSD (problem 108)."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time

import numpy as np


def _one_body(case):
    matrix = np.diag(np.asarray(case["onsite"], dtype=float)).astype(complex)
    for left, right, real, imag in case["hoppings"]:
        value = complex(real, imag)
        matrix[left, right] += value
        matrix[right, left] += value.conjugate()
    return matrix


def _moments(case, lags):
    matrix = _one_body(case)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    occupied = np.eye(case["n_sites"], dtype=complex)[:, case["filled"]]
    coefficients = eigenvectors.conj().T @ occupied
    answer_g, answer_h = [], []
    for lag in lags:
        evolved = eigenvectors @ (
            np.exp(-1j * lag * case["time_step"] * eigenvalues)[:, None] * coefficients
        )
        overlap = occupied.conj().T @ evolved
        insertion = occupied.conj().T @ matrix @ evolved
        determinant = np.linalg.det(overlap)
        answer_g.append(determinant)
        answer_h.append(determinant * np.trace(np.linalg.solve(overlap, insertion)))
    return np.asarray(answer_g), np.asarray(answer_h)


def _signed_mean(counts):
    counts = np.asarray(counts, dtype=float)
    return float((counts[0] - counts[1]) / np.sum(counts))


def _calibrate(counts, calibration):
    observed = _signed_mean(counts)
    response_plus = _signed_mean(calibration[0])
    response_minus = _signed_mean(calibration[1])
    slope = 0.5 * (response_plus - response_minus)
    offset = 0.5 * (response_plus + response_minus)
    return float(np.clip((observed - offset) / slope, -1.0, 1.0))


def _toeplitz(moments):
    size = len(moments)
    result = np.empty((size, size), dtype=complex)
    for row in range(size):
        for column in range(size):
            lag = column - row
            result[row, column] = (
                moments[lag] if lag >= 0 else moments[-lag].conjugate()
            )
    return 0.5 * (result + result.conj().T)


def _conditioned_solution(case):
    calibration = np.asarray(case["calibration_counts"], dtype=float)
    overlap_counts = np.asarray(case["overlap_counts"], dtype=float)
    hamiltonian_counts = np.asarray(case["hamiltonian_counts"], dtype=float)
    size = case["krylov_dimension"]
    overlap_moments, hamiltonian_moments = [], []
    for lag in range(size):
        overlap_moments.append(
            _calibrate(overlap_counts[lag, 0], calibration[0])
            + 1j * _calibrate(overlap_counts[lag, 1], calibration[1])
        )
        hamiltonian_moments.append(
            case["hamiltonian_normalization"]
            * (
                _calibrate(hamiltonian_counts[lag, 0], calibration[2])
                + 1j * _calibrate(hamiltonian_counts[lag, 1], calibration[3])
            )
        )
    overlap_moments[0] = 1.0 + 0.0j
    hamiltonian_moments[0] = hamiltonian_moments[0].real + 0.0j
    overlap = _toeplitz(np.asarray(overlap_moments))
    hamiltonian = _toeplitz(np.asarray(hamiltonian_moments))
    spectrum, vectors = np.linalg.eigh(overlap)
    minimum_shots = min(int(np.sum(record)) for lag in overlap_counts for record in lag)
    cutoff = case["noise_filter_z"] * np.sqrt(size / minimum_shots)
    retained = spectrum > cutoff
    if np.count_nonzero(retained) < 3:
        raise AssertionError("hidden case retained fewer than three QKSD modes")
    whitening = vectors[:, retained] / np.sqrt(spectrum[retained])[None, :]
    projected = whitening.conj().T @ hamiltonian @ whitening
    projected = 0.5 * (projected + projected.conj().T)
    energies = np.linalg.eigvalsh(projected)[:3]
    condition = float(spectrum[retained][-1] / spectrum[retained][0])
    return {
        "energies": energies,
        "rank": int(np.count_nonzero(retained)),
        "cutoff": float(cutoff),
        "condition": condition,
        "spectrum": spectrum,
        "overlap": overlap,
        "hamiltonian": hamiltonian,
    }


def _sample_channel(rng, signal, response_plus, response_minus, shots):
    signal = float(np.clip(signal, -1.0, 1.0))
    probability = 0.5 * (1 + signal) * response_plus
    probability += 0.5 * (1 - signal) * response_minus
    plus = int(rng.binomial(shots, probability))
    return [plus, shots - plus]


def _make_case(rng, index):
    n = 56 + 4 * index
    particle_count = 9 + index
    permutation = rng.permutation(n)
    onsite_base = 0.24 * rng.normal(size=n) + 0.12 * np.cos(
        2 * np.pi * np.arange(n) / n + rng.uniform(-np.pi, np.pi)
    )
    onsite = np.empty(n)
    onsite[permutation] = onsite_base
    edge_map = {}
    for site in range(n):
        neighbors = (
            ((site + 1) % n, -0.48 * (1 + 0.08 * rng.normal())),
            ((site + 2) % n, 0.105 * (1 + 0.10 * rng.normal())),
            ((site + n // 2) % n, -0.21 * (1 + 0.08 * rng.normal())),
        )
        for other, magnitude in neighbors:
            left, right = sorted((int(permutation[site]), int(permutation[other])))
            if left == right or (left, right) in edge_map:
                continue
            phase = rng.uniform(-0.42, 0.42)
            value = magnitude * np.exp(1j * phase)
            edge_map[left, right] = (
                value if left == permutation[site] else value.conjugate()
            )
    hoppings = [
        [left, right, float(value.real), float(value.imag)]
        for (left, right), value in edge_map.items()
    ]
    rng.shuffle(hoppings)
    filled_base = rng.choice(n, particle_count, replace=False)
    filled = sorted(int(permutation[site]) for site in filled_base)
    case = {
        "n_sites": n,
        "filled": filled,
        "onsite": onsite.tolist(),
        "hoppings": hoppings,
        "time_step": float(0.052 + 0.006 * index + rng.uniform(-0.002, 0.002)),
        "krylov_dimension": 12,
        "control_lags": [1, 11, 15, 25],
        "noise_filter_z": 4.25,
        "case_nonce": int(rng.integers(0, 2**31 - 1)),
    }
    matrix = _one_body(case)
    case["hamiltonian_normalization"] = float(
        particle_count * np.max(np.sum(np.abs(matrix), axis=1))
    )
    exact_g, exact_h = _moments(case, range(case["krylov_dimension"]))
    calibration = []
    responses = []
    calibration_shots = 120_000
    for _ in range(4):
        response_plus = float(rng.uniform(0.945, 0.978))
        response_minus = float(rng.uniform(0.022, 0.057))
        responses.append((response_plus, response_minus))
        calibration.append(
            [
                _sample_channel(
                    rng, 1.0, response_plus, response_minus, calibration_shots
                ),
                _sample_channel(
                    rng, -1.0, response_plus, response_minus, calibration_shots
                ),
            ]
        )
    signal_shots = 240_000
    overlap_counts, hamiltonian_counts = [], []
    for overlap, hamiltonian in zip(exact_g, exact_h):
        overlap_counts.append(
            [
                _sample_channel(rng, overlap.real, *responses[0], signal_shots),
                _sample_channel(rng, overlap.imag, *responses[1], signal_shots),
            ]
        )
        normalized = hamiltonian / case["hamiltonian_normalization"]
        hamiltonian_counts.append(
            [
                _sample_channel(rng, normalized.real, *responses[2], signal_shots),
                _sample_channel(rng, normalized.imag, *responses[3], signal_shots),
            ]
        )
    case["calibration_counts"] = calibration
    case["overlap_counts"] = overlap_counts
    case["hamiltonian_counts"] = hamiltonian_counts
    return case


def build_config(seed):
    rng = np.random.default_rng(seed)
    cases = [_make_case(rng, index) for index in range(2)]
    encoded = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    return {
        "cases": cases,
        "case_digest": hashlib.sha256(encoded).hexdigest(),
        "measurement_model": "asymmetric binary readout after normalized Hadamard tests",
    }


def _annihilator(n, site):
    identity = np.eye(2)
    z = np.diag([1.0, -1.0])
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]])
    result = np.array([[1.0]])
    for qubit in range(n):
        result = np.kron(
            result, z if qubit < site else lowering if qubit == site else identity
        )
    return result


def _dense_canary():
    matrix = np.array(
        [
            [0.21, -0.37 + 0.08j, 0.0, 0.0],
            [-0.37 - 0.08j, -0.16, 0.24 - 0.05j, 0.0],
            [0.0, 0.24 + 0.05j, 0.09, -0.31 + 0.03j],
            [0.0, 0.0, -0.31 - 0.03j, -0.11],
        ]
    )
    operators = [_annihilator(4, site) for site in range(4)]
    many_body = sum(
        matrix[i, j] * operators[i].conj().T @ operators[j]
        for i in range(4)
        for j in range(4)
    )
    state = np.zeros(16, dtype=complex)
    state[int("1010", 2)] = 1.0
    values, vectors = np.linalg.eigh(many_body)
    evolved = vectors @ (np.exp(-0.19j * values) * (vectors.conj().T @ state))
    exact = np.array([state.conj() @ evolved, state.conj() @ many_body @ evolved])
    orbitals = np.eye(4, dtype=complex)[:, [0, 2]]
    one_values, one_vectors = np.linalg.eigh(matrix)
    moved = one_vectors @ (
        np.exp(-0.19j * one_values)[:, None] * (one_vectors.conj().T @ orbitals)
    )
    overlap = orbitals.conj().T @ moved
    determinant = np.linalg.det(overlap)
    transition = determinant * np.trace(
        np.linalg.solve(overlap, orbitals.conj().T @ matrix @ moved)
    )
    return np.allclose(exact, [determinant, transition], atol=2e-13, rtol=2e-13)


def _array(result, key, dtype):
    try:
        return np.asarray(result[key], dtype=dtype)
    except (KeyError, TypeError, ValueError):
        return np.asarray([], dtype=dtype)


def evaluate(module_name, seed):
    if not _dense_canary():
        raise AssertionError("independent Slater-transition canary failed")
    config = build_config(seed)
    expected = [_conditioned_solution(case) for case in config["cases"]]
    control = [_moments(case, case["control_lags"]) for case in config["cases"]]
    started = time.perf_counter()
    try:
        result = importlib.import_module(module_name).run_solution(config)
        elapsed = time.perf_counter() - started
        energies = _array(result, "ritz_energies", float)
        ranks = _array(result, "retained_ranks", int)
        cutoffs = _array(result, "overlap_cutoffs", float)
        conditions = _array(result, "retained_condition_numbers", float)
        spectra = _array(result, "overlap_spectra", float)
        overlaps = _array(result, "sampled_overlap_matrices", complex)
        hamiltonians = _array(result, "sampled_hamiltonian_matrices", complex)
        control_g = _array(result, "control_overlaps", complex)
        control_h = _array(result, "control_hamiltonian_moments", complex)
        expected_energies = np.stack([item["energies"] for item in expected])
        expected_ranks = np.asarray([item["rank"] for item in expected])
        expected_cutoffs = np.asarray([item["cutoff"] for item in expected])
        expected_conditions = np.asarray([item["condition"] for item in expected])
        expected_spectra = np.stack([item["spectrum"] for item in expected])
        expected_overlaps = np.stack([item["overlap"] for item in expected])
        expected_hamiltonians = np.stack([item["hamiltonian"] for item in expected])
        expected_g = np.stack([item[0] for item in control])
        expected_h = np.stack([item[1] for item in control])
        criteria = {
            "all output shapes": bool(
                energies.shape == (2, 3)
                and ranks.shape == (2,)
                and cutoffs.shape == (2,)
                and conditions.shape == (2,)
                and spectra.shape == (2, 12)
                and overlaps.shape == (2, 12, 12)
                and hamiltonians.shape == (2, 12, 12)
                and control_g.shape == (2, 4)
                and control_h.shape == (2, 4)
            ),
            "outputs finite": bool(
                all(
                    np.all(np.isfinite(value))
                    for value in (
                        energies,
                        cutoffs,
                        conditions,
                        spectra,
                        overlaps,
                        hamiltonians,
                        control_g,
                        control_h,
                    )
                )
            ),
            "calibrated sampled matrices": bool(
                np.allclose(overlaps, expected_overlaps, atol=3e-9, rtol=3e-9)
                and np.allclose(
                    hamiltonians, expected_hamiltonians, atol=3e-8, rtol=3e-9
                )
            ),
            "canonical overlap filtering": bool(
                np.array_equal(ranks, expected_ranks)
                and np.allclose(cutoffs, expected_cutoffs, atol=2e-12, rtol=2e-12)
                and np.allclose(conditions, expected_conditions, atol=3e-8, rtol=3e-9)
                and np.allclose(spectra, expected_spectra, atol=3e-9, rtol=3e-9)
            ),
            "three filtered Ritz energies": bool(
                np.allclose(energies, expected_energies, atol=3e-8, rtol=3e-9)
            ),
            "phase-sensitive TensorCircuit controls": bool(
                np.allclose(control_g, expected_g, atol=2e-8, rtol=2e-8)
                and np.allclose(control_h, expected_h, atol=3e-8, rtol=2e-8)
            ),
            "conditioning stress active": bool(
                np.all(expected_ranks >= 3)
                and np.all(expected_ranks < 12)
                and all(item["spectrum"][0] < -1e-4 for item in expected)
            ),
            "runtime below 300 seconds": elapsed < 300,
        }
        max_energy_error = float(np.max(np.abs(energies - expected_energies)))
        max_control_error = float(
            max(
                np.max(np.abs(control_g - expected_g)),
                np.max(np.abs(control_h - expected_h)),
            )
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        criteria = {"solution execution": False}
        max_energy_error = float("inf")
        max_control_error = float("inf")
        print(f"Solution error: {type(exc).__name__}: {exc}")
    print("Problem 108 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; digest: {config['case_digest'][:16]}")
    print(f"End-to-end solution time: {elapsed:.3f}s")
    print(f"Maximum Ritz-energy error: {max_energy_error:.3e}")
    print(f"Maximum control-moment error: {max_control_error:.3e}")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    passed = all(criteria.values())
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_108")
    parser.add_argument(
        "--seed",
        type=int,
        default=int(os.environ.get("ORBIT_QKSD_SEED", "1082026")),
    )
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.solution, args.seed) else 1)


if __name__ == "__main__":
    main()
