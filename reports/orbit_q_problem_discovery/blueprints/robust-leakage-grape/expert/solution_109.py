from __future__ import annotations

from itertools import product

import numpy as np
import tensorcircuit as tc
from scipy.optimize import minimize


K = tc.set_backend("jax")
tc.set_dtype("complex128")
LOWERING = K.convert_to_tensor(
    np.array(
        [[0.0, 1.0, 0.0], [0.0, 0.0, np.sqrt(2.0)], [0.0, 0.0, 0.0]],
        dtype=np.complex128,
    )
)
NUMBER = K.convert_to_tensor(np.diag([0.0, 1.0, 2.0]).astype(np.complex128))
PROJECTOR = K.convert_to_tensor(np.diag([0.0, 0.0, 1.0]).astype(np.complex128))
X_DRIVE = LOWERING + K.conj(K.transpose(LOWERING))
Y_DRIVE = -1j * (LOWERING - K.conj(K.transpose(LOWERING)))


def _members(config):
    keys = ("detuning", "amplitude_fraction", "anharmonic_shift")
    limits = [config["uncertainty_bounds"][key] for key in keys]
    values = [[0.0, low, high] for low, high in limits]
    points = [dict(zip(keys, point)) for point in product(*values)]
    return np.asarray([[point[key] for key in keys] for point in points], dtype=float)


def _unitary(controls, member, config):
    circuit = tc.QuditCircuit(1, dim=3)
    drift = member[0] * NUMBER + (config["anharmonicity"] + member[2]) * PROJECTOR
    scale = 1.0 + member[1]
    for index in range(config["n_slices"]):
        ux, uy = controls[index]
        hamiltonian = drift + 0.5 * scale * (ux * X_DRIVE + uy * Y_DRIVE)
        propagator = K.expm(-1j * config["slice_duration"] * hamiltonian)
        circuit.unitary(0, unitary=tc.gates.Gate(propagator), name="grape_slice")
    return circuit.matrix()


def _metrics(controls, members, config):
    angle = config["target_angle"]
    target = K.cast(
        K.stack(
            [
                K.stack([K.cos(angle / 2), -1j * K.sin(angle / 2)]),
                K.stack([-1j * K.sin(angle / 2), K.cos(angle / 2)]),
            ]
        ),
        "complex128",
    )

    def one(member):
        block = _unitary(controls, member, config)[:2, :2]
        overlap = K.trace(K.conj(K.transpose(target)) @ block)
        infidelity = 1.0 - K.abs(overlap) ** 2 / 4.0
        leakage = 1.0 - K.sum(K.abs(block) ** 2) / 2.0
        return K.stack([K.real(infidelity), K.real(leakage)])

    return K.vmap(one)(members)


def _warm_start(config):
    n, dt, theta = config["n_slices"], config["slice_duration"], config["target_angle"]
    phase = np.arccos(-theta / (4 * np.pi))
    areas = np.array([theta / 2, np.pi, 2 * np.pi, np.pi, theta / 2])
    phases = np.array([0.0, phase, 3 * phase, phase, 0.0])
    counts = np.maximum(2, np.rint((n - 2) * areas / np.sum(areas)).astype(int))
    while np.sum(counts) != n - 2:
        index = (
            int(np.argmax(areas / counts))
            if np.sum(counts) < n - 2
            else int(np.argmax(counts))
        )
        counts[index] += 1 if np.sum(counts) < n - 2 else -1
    controls = np.zeros((n, 2))
    start = 1
    for area, phi, count in zip(areas, phases, counts):
        amplitude = area / (count * dt)
        controls[start : start + count] = amplitude * np.array(
            [np.cos(phi), np.sin(phi)]
        )
        start += count
    controls[1:-1] = np.apply_along_axis(
        lambda value: np.convolve(value, [0.2, 0.6, 0.2], mode="same"),
        0,
        controls[1:-1],
    )
    return controls


def run_solution(config):
    members = K.convert_to_tensor(_members(config))
    n = config["n_slices"]
    max_drive = config["max_drive_amplitude"]
    initial = _warm_start(config)

    def objective(flat, risk):
        controls = K.reshape(flat, (n, 2))
        metrics = _metrics(controls, members, config)
        errors, leakage = metrics[:, 0], metrics[:, 1]
        risk_mass = K.softmax(risk * errors)
        differences = controls[1:] - controls[:-1]
        slew_square = K.sum(differences**2, axis=1)
        smooth = K.mean(slew_square)
        slew_excess = K.relu(slew_square - config["max_slew_per_slice"] ** 2)
        return 100.0 * (
            0.3 * K.mean(errors)
            + 0.7 * K.sum(risk_mass * errors)
            + 1.8 * K.mean(leakage)
            + 3e-4 * smooth
            + 0.12 * K.mean(slew_excess**2)
        )

    bounds = []
    for index in range(n):
        limit = 0.0 if index in (0, n - 1) else max_drive / np.sqrt(2)
        bounds.extend([(-limit, limit), (-limit, limit)])
    flat = initial.reshape(-1)
    for risk, iterations in ((25.0, 90), (180.0, 150)):
        value_gradient = K.jit(K.value_and_grad(lambda value: objective(value, risk)))

        def evaluate(value):
            loss, gradient = value_gradient(K.convert_to_tensor(value))
            return float(K.numpy(loss)), np.asarray(K.numpy(gradient), dtype=float)

        flat = minimize(
            evaluate,
            flat,
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": iterations, "ftol": 2e-12, "gtol": 2e-8, "maxls": 30},
        ).x
    controls = flat.reshape(n, 2)
    amplitudes = np.linalg.norm(controls, axis=1)
    controls *= np.minimum(1.0, max_drive / np.maximum(amplitudes, 1e-15))[:, None]
    public = np.asarray(
        K.numpy(
            _metrics(
                K.convert_to_tensor(controls),
                K.convert_to_tensor(
                    [
                        [
                            member[key]
                            for key in (
                                "detuning",
                                "amplitude_fraction",
                                "anharmonic_shift",
                            )
                        ]
                        for member in config["training_ensemble"]
                    ]
                ),
                config,
            )
        )
    )
    return {
        "controls": controls,
        "training_worst_infidelity": float(np.max(public[:, 0])),
        "training_worst_leakage": float(np.max(public[:, 1])),
    }
