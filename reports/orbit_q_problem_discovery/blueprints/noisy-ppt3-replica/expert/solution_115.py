"""TensorCircuit expert for noisy third-moment PPT certification."""

import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")
K = tc.backend
SWAP = tc.gates.Gate(
    np.array(
        [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.complex128,
    ).reshape(2, 2, 2, 2)
)


def _hop(theta, phi, eta=0.0):
    gate = np.zeros((4, 4), dtype=np.complex128)
    cosine, sine = np.cos(theta), np.sin(theta)
    gate[0, 0] = 1
    gate[1:3, 1:3] = [
        [cosine, -1j * np.exp(-1j * phi) * sine],
        [-1j * np.exp(1j * phi) * sine, cosine],
    ]
    gate[3, 3] = np.exp(-1j * eta)
    return tc.gates.Gate(gate.reshape(2, 2, 2, 2))


def _two(circuit, left, gate):
    circuit.apply_adjacent_double_gate(gate, left, left + 1, center_position=left + 1)


def _logical_pair(circuit, left, gate):
    physical = 2 * left
    _two(circuit, physical + 1, SWAP)
    _two(circuit, physical, gate)
    _two(circuit, physical + 1, SWAP)


def _state(case):
    circuit = tc.MPSCircuit(2 * case["system_qubits"], center_position=0, split={})
    for site, bit in enumerate(case["initial_bits"]):
        if bit:
            circuit.x(2 * site)
    for layer in case["layers"]:
        for left, theta, phi, eta in layer:
            _logical_pair(circuit, left, _hop(theta, phi, eta))
    for site, gamma in enumerate(case["amplitude_damping"]):
        _two(circuit, 2 * site, _hop(np.arcsin(np.sqrt(gamma)), 0.0))
    circuit.normalize()
    return circuit.get_tensors()


def _moment2(tensors, system_sites):
    selected = {2 * site for site in system_sites}
    environment = K.ones((1, 1, 1, 1), dtype="complex128")
    for site, tensor in enumerate(tensors):
        if site in selected:
            expression = "abcd,ate,bsf,csg,dth->efgh"
        else:
            expression = "abcd,ase,bsf,ctg,dth->efgh"
        environment = K.einsum(
            expression,
            environment,
            K.conj(tensor),
            tensor,
            K.conj(tensor),
            tensor,
        )
    return np.asarray(environment).item()


def _moment3(tensors, partition_a, partition_b):
    forward = {2 * site for site in partition_a}
    backward = {2 * site for site in partition_b}
    environment = K.ones((1, 1, 1, 1, 1, 1), dtype="complex128")
    for site, tensor in enumerate(tensors):
        if site in forward:
            expression = "abcdef,ayg,bxh,czi,dyj,exk,fzl->ghijkl"
        elif site in backward:
            expression = "abcdef,azg,bxh,cxi,dyj,eyk,fzl->ghijkl"
        else:
            expression = "abcdef,axg,bxh,cyi,dyj,ezk,fzl->ghijkl"
        environment = K.einsum(
            expression,
            environment,
            K.conj(tensor),
            tensor,
            K.conj(tensor),
            tensor,
            K.conj(tensor),
            tensor,
        )
    return np.asarray(environment).item()


def _solve(case):
    tensors = _state(case)
    system_sites = case["partition_a"] + case["partition_b"]
    purity = _moment2(tensors, system_sites).real
    moment = _moment3(tensors, case["partition_a"], case["partition_b"]).real
    gap = purity**2 - moment
    return purity, moment, gap, gap > case["certificate_tolerance"]


def run_solution(config):
    rows = [_solve(case) for case in config["cases"]]
    return {
        "purity2": np.asarray([row[0] for row in rows]),
        "ppt_moment3": np.asarray([row[1] for row in rows]),
        "ppt_gap": np.asarray([row[2] for row in rows]),
        "certified_npt": np.asarray([row[3] for row in rows]),
    }
