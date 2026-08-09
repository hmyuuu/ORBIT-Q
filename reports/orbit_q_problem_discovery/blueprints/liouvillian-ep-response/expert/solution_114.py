"""TensorCircuit solution for near-EP Liouvillian inference and response."""

import numpy as np
import tensorcircuit as tc
from scipy.optimize import least_squares


tc.set_backend("numpy")
tc.set_dtype("complex128")
K = tc.backend
I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
SM = np.array([[0, 1], [0, 0]], dtype=np.complex128)
PAULI = np.asarray([X, Y, Z])


def _rotation(axis, angle):
    operator = {"X": X, "Y": Y, "Z": Z}[axis]
    return np.cos(angle / 2) * I2 - 1j * np.sin(angle / 2) * operator


# fmt: off
def _liouvillian(parameters):
    down, up, dephasing, omega, alpha, beta = parameters
    frame = _rotation("Z", alpha) @ _rotation("Y", beta)
    hamiltonian = frame @ (omega * X / 2) @ frame.conj().T
    jumps = [np.sqrt(down) * frame @ SM @ frame.conj().T, np.sqrt(up) * frame @ SM.conj().T @ frame.conj().T, np.sqrt(dephasing / 2) * frame @ Z @ frame.conj().T]
    result = -1j * (np.kron(I2, hamiltonian) - np.kron(hamiltonian.T, I2))
    for jump in jumps:
        square = jump.conj().T @ jump
        result += np.kron(jump.conj(), jump)
        result -= 0.5 * (np.kron(I2, square) + np.kron(square.T, I2))
    return result


def _channel(parameters, duration):
    values, vectors = np.linalg.eig(duration * _liouvillian(parameters))
    return (vectors * np.exp(values)) @ np.linalg.inv(vectors)


def _kraus(parameters, duration):
    channel = _channel(parameters, duration)
    choi = np.zeros((4, 4), dtype=np.complex128)
    for ket in range(2):
        for bra in range(2):
            basis = np.zeros((2, 2), dtype=np.complex128)
            basis[ket, bra] = 1
            output = (channel @ basis.reshape(-1, order="F")).reshape(2, 2, order="F")
            choi += np.kron(basis, output)
    values, vectors = np.linalg.eigh(0.5 * (choi + choi.conj().T))
    return [K.convert_to_tensor(np.sqrt(value) * vectors[:, index].reshape(2, 2).T) for index, value in enumerate(values) if value > 2e-13]


def _evolve(kraus, preparation, steps):
    circuit = tc.DMCircuit(1)
    circuit.ry(0, theta=preparation[0])
    circuit.rz(0, theta=preparation[1])
    result = {}
    readouts = set(steps)
    observables = K.convert_to_tensor(PAULI)
    for step in range(1, max(readouts) + 1):
        circuit.apply_general_kraus(kraus, 0)
        if step in readouts:
            rho = circuit.densitymatrix(reuse=False)
            result[step] = np.asarray(K.real(K.einsum("aij,ji->a", observables, rho)))
    return np.asarray([result[step] for step in steps])


def _features(parameters, case, probes):
    kraus = _kraus(parameters, case["time_step"])
    axes = np.asarray(case["measurement_axes"])
    return np.asarray([
        np.asarray([axes @ value for value in _evolve(kraus, probe["preparation"], probe["readout_steps"])])
        for probe in probes
    ])


def _tomography(parameters, case):
    kraus = _kraus(parameters, case["time_step"])
    preparations = [(np.pi / 2, 0), (np.pi / 2, np.pi), (np.pi / 2, np.pi / 2), (np.pi / 2, -np.pi / 2), (0, 0), (np.pi, 0)]
    outputs = np.asarray([_evolve(kraus, prep, [1])[0] for prep in preparations])
    transfer = np.column_stack(
        [(outputs[2 * q] - outputs[2 * q + 1]) / 2 for q in range(3)]
    )
    shift = np.mean(
        [(outputs[2 * q] + outputs[2 * q + 1]) / 2 for q in range(3)], axis=0
    )
    values, vectors = np.linalg.eig(transfer)
    generator = (vectors * (np.log(values) / case["time_step"])) @ np.linalg.inv(vectors)
    return np.real_if_close(generator).real, np.linalg.solve(
        np.eye(3) - transfer, shift
    )
# fmt: on


def _analysis(parameters, generator, steady, case):
    down, up, dephasing, omega = parameters[:4]
    longitudinal = down + up
    transverse = longitudinal / 2 + dephasing
    discriminant = (longitudinal - transverse) ** 2 - 4 * omega**2
    root = np.sqrt(complex(discriminant))
    eigenvalues = np.asarray(
        [
            -(longitudinal + transverse) / 2 + root / 2,
            -(longitudinal + transverse) / 2 - root / 2,
            -transverse,
        ]
    )
    identity = np.eye(3, dtype=np.complex128)
    projectors = []
    for index, value in enumerate(eigenvalues):
        projector = identity.copy()
        for other_index, other in enumerate(eigenvalues):
            if index != other_index:
                projector = projector @ (generator - other * identity) / (value - other)
        projectors.append(projector)
    residues, susceptibilities = [], []
    for pair in case["response_pairs"]:
        source, readout = np.asarray(pair["source"]), np.asarray(pair["readout"])
        residues.append([readout @ projector @ source for projector in projectors])
        susceptibilities.append(
            [
                readout @ np.linalg.solve(1j * frequency * identity - generator, source)
                for frequency in case["response_frequencies"]
            ]
        )
    return {
        "liouvillian_matrix": generator,
        "steady_bloch": steady,
        "ep_discriminant": float(discriminant),
        "signed_ep_offset": float(omega - abs(longitudinal - transverse) / 2),
        "liouvillian_eigenvalues": eigenvalues,
        "modal_residues": np.asarray(residues),
        "susceptibilities": np.asarray(susceptibilities),
    }


def _solve(case):
    lower, upper = np.asarray(case["parameter_bounds"], dtype=float).T
    targets = np.asarray(
        [probe["observed_features"] for probe in case["training_probes"]]
    )

    def residual(parameters):
        return (_features(parameters, case, case["training_probes"]) - targets).reshape(
            -1
        )

    fractions = np.asarray(
        [
            [0.5] * 6,
            [0.62, 0.38, 0.40, 0.34, 0.32, 0.45],
            [0.42, 0.58, 0.62, 0.56, 0.68, 0.62],
        ]
    )
    fits = [
        least_squares(
            residual,
            lower + fraction * (upper - lower),
            bounds=(lower, upper),
            diff_step=2e-5,
            max_nfev=110,
            xtol=2e-11,
            ftol=2e-11,
            gtol=2e-11,
        )
        for fraction in fractions
    ]
    fit = min(fits, key=lambda item: np.dot(item.fun, item.fun))
    generator, steady = _tomography(fit.x, case)
    return {
        "case_id": case["case_id"],
        "estimated_parameters": fit.x,
        "training_rmse": float(np.sqrt(np.mean(fit.fun**2))),
        "heldout_features": _features(fit.x, case, case["heldout_probes"]),
        **_analysis(fit.x, generator, steady, case),
    }


def run_solution(config):
    return {"cases": [_solve(case) for case in config["cases"]]}
