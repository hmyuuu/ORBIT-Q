"""TensorCircuit expert for charged-replica symmetry-resolved Renyi-2."""

import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")


def _gate(theta, phi, eta):
    gate = np.zeros((4, 4), dtype=np.complex128)
    cosine, sine = np.cos(theta), np.sin(theta)
    gate[0, 0] = 1
    gate[1:3, 1:3] = [
        [cosine, -1j * np.exp(-1j * phi) * sine],
        [-1j * np.exp(1j * phi) * sine, cosine],
    ]
    gate[3, 3] = np.exp(-1j * eta)
    return tc.gates.Gate(gate.reshape(2, 2, 2, 2))


def _state(case):
    circuit = tc.MPSCircuit(case["n_qubits"], center_position=0, split={})
    for site, bit in enumerate(case["initial_bits"]):
        if bit:
            circuit.x(site)
    for layer in case["layers"]:
        for left, theta, phi, eta in layer:
            circuit.apply_adjacent_double_gate(
                _gate(theta, phi, eta), left, left + 1, center_position=left + 1
            )
    circuit.normalize()
    return circuit.get_tensors()


def _characteristic(tensors, selected, alphas, replica):
    values = []
    for alpha in alphas:
        shape = (1, 1, 1, 1) if replica else (1, 1)
        environment = tc.backend.ones(shape, dtype="complex128")
        phase = tc.backend.convert_to_tensor(np.array([1, np.exp(1j * alpha)]))
        for site, tensor in enumerate(tensors):
            if replica and site in selected:
                environment = tc.backend.einsum(
                    "abcd,ate,bsf,csg,dth,s->efgh",
                    environment,
                    tc.backend.conj(tensor),
                    tensor,
                    tc.backend.conj(tensor),
                    tensor,
                    phase,
                )
            elif replica:
                environment = tc.backend.einsum(
                    "abcd,ase,bsf,ctg,dth->efgh",
                    environment,
                    tc.backend.conj(tensor),
                    tensor,
                    tc.backend.conj(tensor),
                    tensor,
                )
            elif site in selected:
                environment = tc.backend.einsum(
                    "ab,ase,bsf,s->ef",
                    environment,
                    tc.backend.conj(tensor),
                    tensor,
                    phase,
                )
            else:
                environment = tc.backend.einsum(
                    "ab,ase,bsf->ef",
                    environment,
                    tc.backend.conj(tensor),
                    tensor,
                )
        values.append(np.asarray(environment).item())
    return np.asarray(values)


def _solve(case):
    tensors = _state(case)
    count = case["fourier_size"]
    offset = case["fourier_offset"]
    alphas = offset + 2 * np.pi * np.arange(count) / count
    selected = set(case["subsystem"])
    first = _characteristic(tensors, selected, alphas, False)
    second = _characteristic(tensors, selected, alphas, True)
    correction = np.exp(-1j * offset * np.arange(count))
    probabilities = (np.fft.fft(first) / count * correction).real
    moments = (np.fft.fft(second) / count * correction).real
    queries = np.asarray(case["sector_queries"], dtype=int)
    entropies = -np.log(moments[queries] / probabilities[queries] ** 2)
    return second, probabilities, moments, entropies


def run_solution(config):
    rows = [_solve(case) for case in config["cases"]]
    return {
        "charged_moments": np.asarray([row[0] for row in rows]),
        "sector_probabilities": np.asarray([row[1] for row in rows]),
        "sector_second_moments": np.asarray([row[2] for row in rows]),
        "resolved_renyi2": np.asarray([row[3] for row in rows]),
    }
