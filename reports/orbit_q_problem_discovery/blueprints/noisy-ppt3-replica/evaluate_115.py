"""Independent NumPy oracle for noisy third-moment PPT certification."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time

import numpy as np


DEFAULT_SEED = 1152026
SWAP = np.array(
    [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.complex128,
).reshape(2, 2, 2, 2)


def default_seed():
    return int(os.environ.get("ORBIT_Q_CANDIDATE_SEED", str(DEFAULT_SEED)))


def _hop(theta, phi, eta=0.0):
    gate = np.zeros((4, 4), dtype=np.complex128)
    cosine, sine = np.cos(theta), np.sin(theta)
    gate[0, 0] = 1
    gate[1:3, 1:3] = [
        [cosine, -1j * np.exp(-1j * phi) * sine],
        [-1j * np.exp(1j * phi) * sine, cosine],
    ]
    gate[3, 3] = np.exp(-1j * eta)
    return gate.reshape(2, 2, 2, 2)


def _make_case(rng, index, coupled):
    n = 32
    bits = np.zeros(n, dtype=int)
    bits[rng.choice(n, n // 2, replace=False)] = 1
    labels = np.zeros(n, dtype=int)
    labels[rng.choice(n, n // 2, replace=False)] = 1
    partition_a = np.flatnonzero(labels == 0).tolist()
    partition_b = np.flatnonzero(labels == 1).tolist()
    layers = []
    for layer in range(4):
        records = []
        for left in range(layer % 2, n - 1, 2):
            crosses = labels[left] != labels[left + 1]
            if coupled or not crosses:
                lower, upper = (0.52, 0.92) if crosses else (0.26, 0.74)
                records.append(
                    [
                        left,
                        float(rng.uniform(lower, upper)),
                        float(rng.uniform(-np.pi, np.pi)),
                        float(rng.uniform(-0.48, 0.48)),
                    ]
                )
        layers.append(records)
    noise_range = (0.006, 0.024) if coupled else (0.035, 0.085)
    return {
        "case_id": f"ppt3-{index}-{rng.integers(1 << 31):08x}",
        "system_qubits": n,
        "initial_bits": bits.tolist(),
        "layers": layers,
        "amplitude_damping": rng.uniform(*noise_range, size=n).tolist(),
        "partition_a": partition_a,
        "partition_b": partition_b,
        "certificate_tolerance": 2e-8,
        "case_nonce": int(rng.integers(1 << 31)),
    }


def build_config(seed):
    rng = np.random.default_rng(seed)
    pattern = [True, False, True]
    rng.shuffle(pattern)
    cases = [_make_case(rng, index, coupled) for index, coupled in enumerate(pattern)]
    config = {
        "cases": cases,
        "physical_order": "system_0,environment_0,system_1,environment_1,...",
        "moment_convention": "p3=Tr[(rho_AB^{T_B})^3]",
        "ppt_witness": "certified_npt iff purity2^2-ppt_moment3 > tolerance",
    }
    payload = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    config["case_digest"] = hashlib.sha256(payload).hexdigest()
    return config


class _MPS:
    def __init__(self, bits):
        self.tensors = []
        for bit in bits:
            vector = np.zeros(2, dtype=np.complex128)
            vector[bit] = 1
            self.tensors.append(vector.reshape(1, 2, 1))
        self.center = 0

    def position(self, site):
        while self.center < site:
            q = self.center
            left, physical, right = self.tensors[q].shape
            unitary, residue = np.linalg.qr(
                self.tensors[q].reshape(left * physical, right), mode="reduced"
            )
            self.tensors[q] = unitary.reshape(left, physical, -1)
            self.tensors[q + 1] = np.einsum("ab,bsr->asr", residue, self.tensors[q + 1])
            self.center += 1
        while self.center > site:
            q = self.center
            left, physical, right = self.tensors[q].shape
            unitary, residue = np.linalg.qr(
                self.tensors[q].reshape(left, physical * right).T, mode="reduced"
            )
            self.tensors[q] = unitary.T.reshape(-1, physical, right)
            self.tensors[q - 1] = np.einsum(
                "lsr,ra->lsa", self.tensors[q - 1], residue.T
            )
            self.center -= 1

    def two(self, left, gate):
        self.position(left)
        pair = np.einsum("lim,mjr->lijr", self.tensors[left], self.tensors[left + 1])
        pair = np.einsum("abij,lijr->labr", gate, pair)
        dl, _, _, dr = pair.shape
        unitary, singular, residue = np.linalg.svd(
            pair.reshape(2 * dl, 2 * dr), full_matrices=False
        )
        rank = max(1, int(np.count_nonzero(singular > singular[0] * 2e-14)))
        self.tensors[left] = unitary[:, :rank].reshape(dl, 2, rank)
        self.tensors[left + 1] = (singular[:rank, None] * residue[:rank]).reshape(
            rank, 2, dr
        )
        self.center = left + 1

    def normalize(self):
        self.position(len(self.tensors) - 1)
        self.tensors[-1] /= np.linalg.norm(self.tensors[-1])


def _apply_logical_pair(state, left, gate):
    physical = 2 * left
    state.two(physical + 1, SWAP)
    state.two(physical, gate)
    state.two(physical + 1, SWAP)


def _state(case):
    physical_bits = []
    for bit in case["initial_bits"]:
        physical_bits.extend([bit, 0])
    state = _MPS(physical_bits)
    for layer in case["layers"]:
        for left, theta, phi, eta in layer:
            _apply_logical_pair(state, left, _hop(theta, phi, eta))
    for site, gamma in enumerate(case["amplitude_damping"]):
        state.two(2 * site, _hop(np.arcsin(np.sqrt(gamma)), 0.0))
    state.normalize()
    return state.tensors


def _moment2(tensors, system_sites):
    selected = {2 * site for site in system_sites}
    environment = np.ones((1, 1, 1, 1), dtype=np.complex128)
    for site, tensor in enumerate(tensors):
        if site in selected:
            environment = np.einsum(
                "abcd,ate,bsf,csg,dth->efgh",
                environment,
                tensor.conj(),
                tensor,
                tensor.conj(),
                tensor,
                optimize="greedy",
            )
        else:
            environment = np.einsum(
                "abcd,ase,bsf,ctg,dth->efgh",
                environment,
                tensor.conj(),
                tensor,
                tensor.conj(),
                tensor,
                optimize="greedy",
            )
    return environment.item()


def _moment3(tensors, partition_a, partition_b):
    forward = {2 * site for site in partition_a}
    backward = {2 * site for site in partition_b}
    environment = np.ones((1, 1, 1, 1, 1, 1), dtype=np.complex128)
    for site, tensor in enumerate(tensors):
        if site in forward:
            expression = "abcdef,ayg,bxh,czi,dyj,exk,fzl->ghijkl"
        elif site in backward:
            expression = "abcdef,azg,bxh,cxi,dyj,eyk,fzl->ghijkl"
        else:
            expression = "abcdef,axg,bxh,cyi,dyj,ezk,fzl->ghijkl"
        environment = np.einsum(
            expression,
            environment,
            tensor.conj(),
            tensor,
            tensor.conj(),
            tensor,
            tensor.conj(),
            tensor,
            optimize="greedy",
        )
    return environment.item()


def _oracle(case):
    tensors = _state(case)
    sites = case["partition_a"] + case["partition_b"]
    purity = _moment2(tensors, sites)
    moment = _moment3(tensors, case["partition_a"], case["partition_b"])
    gap = purity.real**2 - moment.real
    bonds = [tensor.shape[-1] for tensor in tensors[:-1]]
    return purity, moment, gap, gap > case["certificate_tolerance"], bonds


def _dense_apply(state, gate, left, n):
    shaped = state.reshape((2,) * n)
    moved = np.moveaxis(shaped, (left, left + 1), (0, 1)).reshape(4, -1)
    moved = gate.reshape(4, 4) @ moved
    shaped = moved.reshape((2, 2) + (2,) * (n - 2))
    return np.moveaxis(shaped, (0, 1), (left, left + 1)).reshape(-1)


def _dense_canary():
    n = 4
    bits = [1, 0, 1, 0]
    records = [(0, 0.63, -0.41, 0.17), (2, 0.52, 0.37, -0.23), (1, 0.71, 0.22, 0.31)]
    gammas = [0.03, 0.07, 0.11, 0.05]
    physical_bits = sum(([bit, 0] for bit in bits), [])
    state = np.zeros(2 ** (2 * n), dtype=np.complex128)
    index = sum(bit << (2 * n - 1 - q) for q, bit in enumerate(physical_bits))
    state[index] = 1
    mps = _MPS(physical_bits)
    for left, theta, phi, eta in records:
        gate = _hop(theta, phi, eta)
        _apply_logical_pair(mps, left, gate)
        physical = 2 * left
        state = _dense_apply(state, SWAP, physical + 1, 2 * n)
        state = _dense_apply(state, gate, physical, 2 * n)
        state = _dense_apply(state, SWAP, physical + 1, 2 * n)
    for site, gamma in enumerate(gammas):
        gate = _hop(np.arcsin(np.sqrt(gamma)), 0.0)
        mps.two(2 * site, gate)
        state = _dense_apply(state, gate, 2 * site, 2 * n)
    mps.normalize()
    system = list(range(0, 2 * n, 2))
    environment = list(range(1, 2 * n, 2))
    wavefunction = state.reshape((2,) * (2 * n)).transpose(system + environment)
    wavefunction = wavefunction.reshape(2**n, 2**n)
    rho = wavefunction @ wavefunction.conj().T
    shaped = rho.reshape((2,) * (2 * n))
    axes = list(range(2 * n))
    for site in [1, 3]:
        axes[site], axes[n + site] = axes[n + site], axes[site]
    partial_transpose = shaped.transpose(axes).reshape(2**n, 2**n)
    dense2 = np.trace(rho @ rho)
    dense3 = np.trace(partial_transpose @ partial_transpose @ partial_transpose)
    replica2 = _moment2(mps.tensors, range(n))
    replica3 = _moment3(mps.tensors, [0, 2], [1, 3])
    return np.allclose(replica2, dense2, atol=4e-13, rtol=4e-13) and np.allclose(
        replica3, dense3, atol=4e-13, rtol=4e-13
    )


def _array(result, key, dtype):
    try:
        return np.asarray(result[key], dtype=dtype)
    except (KeyError, TypeError, ValueError):
        return np.array([], dtype=dtype)


def evaluate(module_name, seed):
    config = build_config(seed)
    print(
        json.dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": seed,
                    "case_digest": config["case_digest"],
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    os.environ.pop("ORBIT_Q_CANDIDATE_SEED", None)
    if not _dense_canary():
        raise AssertionError("independent partial-transpose replica canary failed")
    expected = [_oracle(case) for case in config["cases"]]
    expected_purity = np.asarray([row[0].real for row in expected])
    expected_moment = np.asarray([row[1].real for row in expected])
    expected_gap = np.asarray([row[2] for row in expected])
    expected_certificate = np.asarray([row[3] for row in expected])
    bonds = [bond for row in expected for bond in row[4]]
    imaginary_residual = max(abs(row[0].imag) + abs(row[1].imag) for row in expected)
    started = time.perf_counter()
    result = importlib.import_module(module_name).run_solution(config)
    elapsed = time.perf_counter() - started
    purity = _array(result, "purity2", float)
    moment = _array(result, "ppt_moment3", float)
    gap = _array(result, "ppt_gap", float)
    certificate = _array(result, "certified_npt", bool)
    shape = expected_purity.shape
    finite = all(np.all(np.isfinite(value)) for value in (purity, moment, gap))
    criteria = {
        "output shapes": purity.shape
        == moment.shape
        == gap.shape
        == certificate.shape
        == shape,
        "outputs finite": finite,
        "independent two-replica purity": purity.shape == shape
        and np.allclose(purity, expected_purity, atol=3e-9, rtol=3e-8),
        "independent opposite-cycle PPT moment": moment.shape == shape
        and np.allclose(moment, expected_moment, atol=3e-9, rtol=3e-8),
        "submitted PPT-gap identity": gap.shape == shape
        and purity.shape == shape
        and moment.shape == shape
        and np.allclose(gap, purity**2 - moment, atol=2e-10, rtol=2e-9),
        "independent PPT gaps": gap.shape == shape
        and np.allclose(gap, expected_gap, atol=4e-9, rtol=4e-8),
        "PPT certificates": certificate.shape == shape
        and np.array_equal(certificate, expected_certificate),
        "oracle numerical admission": imaginary_residual < 2e-10
        and np.min(np.abs(expected_gap)) > 2e-6
        and np.any(expected_certificate)
        and np.any(~expected_certificate),
        "representation stress active": len(config["cases"]) == 3
        and min(2 * case["system_qubits"] for case in config["cases"]) >= 64
        and max(bonds) >= 4,
    }
    errors = []
    for actual, target in (
        (purity, expected_purity),
        (moment, expected_moment),
        (gap, expected_gap),
    ):
        errors.append(
            float(np.max(np.abs(actual - target)))
            if actual.shape == shape
            else float("inf")
        )
    print("Problem 115 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; case digest: {config['case_digest'][:16]}")
    print(f"End-to-end solution time: {elapsed:.3f}s")
    print(
        f"Maximum purity/p3/gap errors: {errors[0]:.3e} / {errors[1]:.3e} / {errors[2]:.3e}"
    )
    print(
        f"Oracle gap range: {expected_gap.min():.3e} to {expected_gap.max():.3e}; maximum bond={max(bonds)}"
    )
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    passed = all(criteria.values())
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_115")
    parser.add_argument("--seed", type=int, default=default_seed())
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.solution, args.seed) else 1)


if __name__ == "__main__":
    main()
