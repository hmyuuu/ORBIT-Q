from __future__ import annotations

import numpy as np
import tensorcircuit as tc
from scipy.optimize import least_squares


tc.set_backend("numpy")
tc.set_dtype("complex128")
I2 = np.eye(2, dtype=np.complex128)
PAULI = {
    "I": I2,
    "X": np.array([[0, 1], [1, 0]], dtype=np.complex128),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    "Z": np.diag([1, -1]).astype(np.complex128),
}


def _collision(parameters, duration):
    g, phase, zeta = parameters[:3]
    common = np.exp(0.5j * zeta * duration)
    edge = np.exp(-0.5j * zeta * duration)
    cosine, sine = np.cos(g * duration), np.sin(g * duration)
    unitary = np.zeros((4, 4), dtype=np.complex128)
    unitary[0, 0] = unitary[3, 3] = edge
    unitary[1, 1] = unitary[2, 2] = common * cosine
    unitary[1, 2] = -1j * common * np.exp(-1j * phase) * sine
    unitary[2, 1] = -1j * common * np.exp(1j * phase) * sine
    return tc.gates.any(unitary=unitary, name="collision")


def _simulate(parameters, config, probe):
    circuit = tc.DMCircuit(config["n_qubits"])
    circuit.ry(0, theta=probe["system_prep"][0])
    circuit.rz(0, theta=probe["system_prep"][1])
    circuit.ry(1, theta=probe["memory_prep"][0])
    circuit.rz(1, theta=probe["memory_prep"][1])
    omega, rate, bath_p = parameters[3:]
    readouts = set(config["readout_after"])
    features = []
    for step_number, step in enumerate(probe["steps"], start=1):
        if step["reset_system"]:
            circuit.reset(0)
            circuit.ry(0, theta=step["reset_prepare"][0])
            circuit.rz(0, theta=step["reset_prepare"][1])
        circuit.rx(0, theta=step["control"][0])
        circuit.ry(0, theta=step["control"][1])
        circuit.rz(0, theta=step["control"][2])
        duration = step["duration"]
        circuit.unitary(
            0, 1, unitary=_collision(parameters, duration), name="collision"
        )
        circuit.rz(1, theta=omega * duration)
        circuit.amplitudedamping(1, gamma=1 - np.exp(-rate * duration), p=bath_p)
        if step_number in readouts:
            rho = np.asarray(circuit.densitymatrix(reuse=False))
            for label in config["observables"]:
                observable = np.einsum(
                    "ab,cd->acbd", PAULI[label[0]], PAULI[label[1]]
                ).reshape(4, 4)
                features.append(float(np.real(np.trace(observable @ rho))))
    return np.asarray(features)


def run_solution(config):
    probes = config["training_probes"]
    targets = [np.asarray(probe["observed_features"], dtype=float) for probe in probes]

    def residual(parameters):
        return np.concatenate(
            [
                _simulate(parameters, config, probe) - target
                for probe, target in zip(probes, targets)
            ]
        )

    lower, upper = np.asarray(config["parameter_bounds"], dtype=float).T
    fractions = np.asarray(
        [
            [0.50, 0.50, 0.50, 0.50, 0.50, 0.50],
            [0.28, 0.72, 0.36, 0.64, 0.31, 0.69],
            [0.72, 0.31, 0.67, 0.27, 0.69, 0.38],
            [0.43, 0.23, 0.78, 0.74, 0.46, 0.27],
        ]
    )
    fits = []
    for fraction in fractions:
        fits.append(
            least_squares(
                residual,
                lower + fraction * (upper - lower),
                bounds=(lower, upper),
                diff_step=3e-5,
                xtol=2e-10,
                ftol=2e-10,
                gtol=2e-10,
                max_nfev=180,
            )
        )
    fit = min(fits, key=lambda candidate: np.dot(candidate.fun, candidate.fun))
    heldout = np.stack(
        [_simulate(fit.x, config, probe) for probe in config["heldout_probes"]]
    )
    return {
        "estimated_parameters": np.asarray(fit.x),
        "training_rmse": float(np.sqrt(np.mean(fit.fun**2))),
        "heldout_features": heldout,
    }
