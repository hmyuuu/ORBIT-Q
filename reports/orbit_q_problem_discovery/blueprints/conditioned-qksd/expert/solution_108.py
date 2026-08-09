"""TensorCircuit expert for phase-sensitive, shot-conditioned QKSD."""

from __future__ import annotations

import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")


def _matrix(case):
    matrix = np.diag(np.asarray(case["onsite"], dtype=float)).astype(complex)
    for left, right, real, imag in case["hoppings"]:
        value = complex(real, imag)
        matrix[left, right] += value
        matrix[right, left] += value.conjugate()
    return matrix


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


def _sampled_qksd(case):
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
    whitening = vectors[:, retained] / np.sqrt(spectrum[retained])[None, :]
    projected = whitening.conj().T @ hamiltonian @ whitening
    energies = np.linalg.eigvalsh(0.5 * (projected + projected.conj().T))[:3]
    return {
        "energies": energies,
        "rank": int(np.count_nonzero(retained)),
        "cutoff": float(cutoff),
        "condition": float(spectrum[retained][-1] / spectrum[retained][0]),
        "spectrum": spectrum,
        "overlap": overlap,
        "hamiltonian": hamiltonian,
    }


def _controls(case):
    n = case["n_sites"]
    filled = case["filled"]
    matrix = _matrix(case)
    generator = np.zeros((2 * n, 2 * n), dtype=complex)
    generator[:n, :n] = case["time_step"] * matrix
    generator[n:, n:] = -case["time_step"] * matrix.T
    simulator = tc.FGSSimulator(n, filled=filled)
    initial = np.asarray(simulator.get_alpha())[n:, filled].conj()
    overlaps, hamiltonian_moments, previous = [], [], 0
    for lag in case["control_lags"]:
        simulator.evol_hamiltonian((lag - previous) * generator)
        previous = lag
        evolved = np.asarray(simulator.get_alpha())[n:, filled].conj()
        transition_overlap = initial.conj().T @ evolved
        transition_hamiltonian = initial.conj().T @ matrix @ evolved
        determinant = np.linalg.det(transition_overlap)
        overlaps.append(determinant)
        hamiltonian_moments.append(
            determinant
            * np.trace(np.linalg.solve(transition_overlap, transition_hamiltonian))
        )
    return np.asarray(overlaps), np.asarray(hamiltonian_moments)


def run_solution(config):
    sampled = [_sampled_qksd(case) for case in config["cases"]]
    controls = [_controls(case) for case in config["cases"]]
    return {
        "ritz_energies": np.stack([item["energies"] for item in sampled]),
        "retained_ranks": np.asarray([item["rank"] for item in sampled]),
        "overlap_cutoffs": np.asarray([item["cutoff"] for item in sampled]),
        "retained_condition_numbers": np.asarray(
            [item["condition"] for item in sampled]
        ),
        "overlap_spectra": np.stack([item["spectrum"] for item in sampled]),
        "sampled_overlap_matrices": np.stack([item["overlap"] for item in sampled]),
        "sampled_hamiltonian_matrices": np.stack(
            [item["hamiltonian"] for item in sampled]
        ),
        "control_overlaps": np.stack([item[0] for item in controls]),
        "control_hamiltonian_moments": np.stack([item[1] for item in controls]),
    }
