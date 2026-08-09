from __future__ import annotations

import numpy as np
import tensorcircuit as tc
from scipy.optimize import least_squares


def _simulate(theta, config, probe):
    n = config["n_sites"]
    simulator = tc.FGSSimulator(n, filled=config["filled"])
    hopping, pairing, potential = theta
    phase, dt = probe["phase"], probe["dt"]
    for _ in range(config["steps"]):
        for i in range(n):
            simulator.evol_cp(
                i, dt * potential * (-1) ** i * (1 + 0.13 * np.cos(phase + 0.31 * i))
            )
        for parity in (0, 1):
            for i in range(parity, n - 1, 2):
                simulator.evol_hp(
                    i, i + 1, dt * hopping * (1 + 0.11 * np.sin(phase + 0.47 * i))
                )
        for parity in (0, 1):
            for i in range(parity, n - 1, 2):
                simulator.evol_sp(
                    i, i + 1, dt * pairing * np.exp(1j * (phase + 0.19 * i))
                )
    log_probability = 0.0
    for site, outcome in zip(probe["measure_sites"], probe["outcomes"]):
        correlation = np.asarray(simulator.get_cmatrix())
        p0 = float(np.real(correlation[site, site]))
        log_probability += np.log(max(p0 if outcome == 0 else 1 - p0, 1e-15))
        simulator.post_select(site, outcome)
    correlation = np.asarray(simulator.get_cmatrix())
    features = [log_probability]
    features.extend(float(np.real(correlation[i, i])) for i in config["feature_sites"])
    for i, j in config["feature_pairs"]:
        value = correlation[i, j]
        features.extend((float(np.real(value)), float(np.imag(value))))
    for i, j in config["feature_pairs"]:
        value = correlation[i, j + n]
        features.extend((float(np.real(value)), float(np.imag(value))))
    return np.asarray(features)


def run_solution(config):
    tc.set_backend("numpy")
    tc.set_dtype("complex128")
    probes = config["training_probes"]
    observed = [np.asarray(probe["observed_features"], dtype=float) for probe in probes]

    def residual(theta):
        return np.concatenate(
            [
                _simulate(theta, config, probe) - target
                for probe, target in zip(probes, observed)
            ]
        )

    lower, upper = np.asarray(config["parameter_bounds"], dtype=float).T
    starts = [
        (lower + upper) / 2,
        lower * 0.35 + upper * 0.65,
        lower * 0.65 + upper * 0.35,
    ]
    fits = [
        least_squares(
            residual,
            start,
            bounds=(lower, upper),
            diff_step=2e-5,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
            max_nfev=120,
        )
        for start in starts
    ]
    fit = min(fits, key=lambda value: np.dot(value.fun, value.fun))
    heldout = np.stack(
        [_simulate(fit.x, config, probe) for probe in config["heldout_probes"]]
    )
    return {
        "estimated_parameters": np.asarray(fit.x),
        "training_rmse": float(np.sqrt(np.mean(fit.fun**2))),
        "heldout_features": heldout,
    }
