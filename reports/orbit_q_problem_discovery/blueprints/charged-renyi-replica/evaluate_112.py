"""Independent evaluator for problem 112's charged replica moments."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time

import numpy as np


DEFAULT_SEED = 1122026


def default_seed():
    return int(
        os.environ.get(
            "ORBIT_Q_CANDIDATE_SEED",
            os.environ.get("ORBIT_RENYI_SEED", str(DEFAULT_SEED)),
        )
    )


def _gate(theta, phi, eta):
    gate = np.zeros((4, 4), dtype=np.complex128)
    cosine, sine = np.cos(theta), np.sin(theta)
    gate[0, 0] = 1
    gate[1:3, 1:3] = [
        [cosine, -1j * np.exp(-1j * phi) * sine],
        [-1j * np.exp(1j * phi) * sine, cosine],
    ]
    gate[3, 3] = np.exp(-1j * eta)
    return gate.reshape(2, 2, 2, 2)


def _make_cases(seed, count=2):
    rng = np.random.default_rng(seed)
    cases = []
    n, depth, subsystem_size = 72, 4, 24
    for case_index in range(count):
        bits = np.zeros(n, dtype=int)
        bits[rng.choice(n, n // 2, replace=False)] = 1
        start_parity = int(rng.integers(0, 2))
        layers = []
        for layer in range(depth):
            records = []
            for left in range((start_parity + layer) % 2, n - 1, 2):
                records.append(
                    [
                        left,
                        float(rng.uniform(0.34, 1.04)),
                        float(rng.uniform(-np.pi, np.pi)),
                        float(rng.uniform(-0.72, 0.72)),
                    ]
                )
            layers.append(records)
        subsystem = rng.choice(n, subsystem_size, replace=False).tolist()
        rng.shuffle(subsystem)
        center = int(bits[subsystem].sum())
        queries = list(range(max(0, center - 2), min(subsystem_size, center + 2) + 1))
        cases.append(
            {
                "n_qubits": n,
                "initial_bits": bits.tolist(),
                "layers": layers,
                "subsystem": [int(site) for site in subsystem],
                "fourier_offset": float(rng.uniform(-0.37, 0.37)),
                "fourier_size": subsystem_size + 1,
                "sector_queries": queries,
                "case_nonce": int(rng.integers(0, 2**31 - 1)) + case_index,
            }
        )
    return cases


def build_config(seed):
    cases = _make_cases(seed)
    payload = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    return {
        "cases": cases,
        "case_digest": hashlib.sha256(payload).hexdigest(),
        "charge_convention": "Q_A=sum |1><1|; positive Fourier phase",
    }


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
        rank = int(np.count_nonzero(singular > singular[0] * 2e-14))
        unitary, singular, residue = (
            unitary[:, :rank],
            singular[:rank],
            residue[:rank],
        )
        self.tensors[left] = unitary.reshape(dl, 2, rank)
        self.tensors[left + 1] = (singular[:, None] * residue).reshape(rank, 2, dr)
        self.center = left + 1

    def normalize(self):
        self.position(len(self.tensors) - 1)
        self.tensors[-1] /= np.linalg.norm(self.tensors[-1])


def _state(case):
    state = _MPS(case["initial_bits"])
    for layer in case["layers"]:
        for left, theta, phi, eta in layer:
            state.two(left, _gate(theta, phi, eta))
    state.normalize()
    return state.tensors


def _single_characteristic(tensors, subsystem, alphas):
    selected = set(subsystem)
    values = []
    for alpha in alphas:
        environment = np.ones((1, 1), dtype=np.complex128)
        phase = np.array([1, np.exp(1j * alpha)])
        for site, tensor in enumerate(tensors):
            if site in selected:
                environment = np.einsum(
                    "ab,ase,bsf,s->ef",
                    environment,
                    tensor.conj(),
                    tensor,
                    phase,
                    optimize="greedy",
                )
            else:
                environment = np.einsum(
                    "ab,ase,bsf->ef",
                    environment,
                    tensor.conj(),
                    tensor,
                    optimize="greedy",
                )
        values.append(environment.item())
    return np.asarray(values)


def _replica_characteristic(tensors, subsystem, alphas):
    selected = set(subsystem)
    values = []
    for alpha in alphas:
        environment = np.ones((1, 1, 1, 1), dtype=np.complex128)
        phase = np.array([1, np.exp(1j * alpha)])
        for site, tensor in enumerate(tensors):
            if site in selected:
                environment = np.einsum(
                    "abcd,ate,bsf,csg,dth,s->efgh",
                    environment,
                    tensor.conj(),
                    tensor,
                    tensor.conj(),
                    tensor,
                    phase,
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
        values.append(environment.item())
    return np.asarray(values)


def _sectors(characteristic, offset):
    count = characteristic.size
    charges = np.arange(count)
    values = np.fft.fft(characteristic) / count
    return values * np.exp(-1j * offset * charges)


def _oracle(case):
    tensors = _state(case)
    count = case["fourier_size"]
    alphas = case["fourier_offset"] + 2 * np.pi * np.arange(count) / count
    first = _single_characteristic(tensors, case["subsystem"], alphas)
    second = _replica_characteristic(tensors, case["subsystem"], alphas)
    probabilities = _sectors(first, case["fourier_offset"]).real
    moments = _sectors(second, case["fourier_offset"]).real
    queries = np.asarray(case["sector_queries"], dtype=int)
    entropies = -np.log(moments[queries] / probabilities[queries] ** 2)
    bonds = [tensor.shape[-1] for tensor in tensors[:-1]]
    return second, probabilities, moments, entropies, bonds


def _dense_apply(state, gate, left, n):
    shaped = state.reshape((2,) * n)
    moved = np.moveaxis(shaped, (left, left + 1), (0, 1)).reshape(4, -1)
    moved = gate.reshape(4, 4) @ moved
    shaped = moved.reshape((2, 2) + (2,) * (n - 2))
    return np.moveaxis(shaped, (0, 1), (left, left + 1)).reshape(-1)


def _dense_canary():
    n, bits, subsystem = 6, [1, 0, 1, 0, 0, 1], [4, 0, 2]
    records = [
        (0, 0.37, -0.41, 0.19),
        (2, 0.62, 0.23, -0.31),
        (4, 0.48, -0.72, 0.27),
        (1, 0.55, 0.64, -0.17),
        (3, 0.29, -0.33, 0.38),
    ]
    state = np.zeros(2**n, dtype=np.complex128)
    index = sum(bit << (n - 1 - q) for q, bit in enumerate(bits))
    state[index] = 1
    mps = _MPS(bits)
    for left, theta, phi, eta in records:
        gate = _gate(theta, phi, eta)
        state = _dense_apply(state, gate, left, n)
        mps.two(left, gate)
    mps.normalize()
    complement = [site for site in range(n) if site not in subsystem]
    wavefunction = state.reshape((2,) * n).transpose(subsystem + complement)
    wavefunction = wavefunction.reshape(2 ** len(subsystem), -1)
    rho = wavefunction @ wavefunction.conj().T
    count, offset = len(subsystem) + 1, 0.173
    alphas = offset + 2 * np.pi * np.arange(count) / count
    occupations = np.array([int(index).bit_count() for index in range(2**3)])
    first = np.asarray(
        [np.sum(np.diag(rho) * np.exp(1j * alpha * occupations)) for alpha in alphas]
    )
    rho2 = rho @ rho
    second = np.asarray(
        [np.sum(np.diag(rho2) * np.exp(1j * alpha * occupations)) for alpha in alphas]
    )
    first_mps = _single_characteristic(mps.tensors, subsystem, alphas)
    second_mps = _replica_characteristic(mps.tensors, subsystem, alphas)
    return np.allclose(first_mps, first, atol=3e-13, rtol=3e-13) and np.allclose(
        second_mps, second, atol=3e-13, rtol=3e-13
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
    os.environ.pop("ORBIT_RENYI_SEED", None)
    if not _dense_canary():
        raise AssertionError("independent charged-replica dense canary failed")
    expected = [_oracle(case) for case in config["cases"]]
    expected_charged = np.asarray([row[0] for row in expected])
    expected_probabilities = np.asarray([row[1] for row in expected])
    expected_moments = np.asarray([row[2] for row in expected])
    expected_entropies = np.asarray([row[3] for row in expected])
    all_bonds = [bond for row in expected for bond in row[4]]
    started = time.perf_counter()
    result = importlib.import_module(module_name).run_solution(config)
    elapsed = time.perf_counter() - started
    charged = _array(result, "charged_moments", complex)
    probabilities = _array(result, "sector_probabilities", float)
    moments = _array(result, "sector_second_moments", float)
    entropies = _array(result, "resolved_renyi2", float)
    reconstructed = []
    for case, values in zip(config["cases"], charged):
        reconstructed.append(_sectors(values, case["fourier_offset"]))
    reconstructed = np.asarray(reconstructed)
    queried_probabilities = np.asarray(
        [row[1][case["sector_queries"]] for row, case in zip(expected, config["cases"])]
    )
    queried_moments = np.asarray(
        [row[2][case["sector_queries"]] for row, case in zip(expected, config["cases"])]
    )
    criteria = {
        "output shapes": charged.shape == expected_charged.shape
        and probabilities.shape == expected_probabilities.shape
        and moments.shape == expected_moments.shape
        and entropies.shape == expected_entropies.shape,
        "outputs finite": np.all(np.isfinite(charged))
        and np.all(np.isfinite(probabilities))
        and np.all(np.isfinite(moments))
        and np.all(np.isfinite(entropies)),
        "independent charged moments": charged.shape == expected_charged.shape
        and np.allclose(charged, expected_charged, atol=2e-9, rtol=3e-8),
        "independent sector probabilities": probabilities.shape
        == expected_probabilities.shape
        and np.allclose(probabilities, expected_probabilities, atol=2e-9, rtol=3e-8),
        "independent sector second moments": moments.shape == expected_moments.shape
        and np.allclose(moments, expected_moments, atol=2e-9, rtol=3e-8),
        "independent resolved entropies": entropies.shape == expected_entropies.shape
        and np.allclose(entropies, expected_entropies, atol=3e-8, rtol=3e-8),
        "submitted shifted Fourier identity": reconstructed.shape
        == expected_moments.shape
        and np.allclose(reconstructed.imag, 0, atol=3e-9)
        and np.allclose(reconstructed.real, moments, atol=3e-9, rtol=3e-8),
        "normalization and sector positivity": probabilities.shape
        == expected_probabilities.shape
        and moments.shape == expected_moments.shape
        and np.allclose(probabilities.sum(axis=1), 1, atol=3e-9)
        and np.all(probabilities >= -3e-9)
        and np.all(moments >= -3e-9)
        and np.all(moments <= probabilities**2 + 3e-9),
        "queried sectors numerically admitted": np.min(queried_probabilities) > 2e-5
        and np.min(queried_moments) > 2e-8,
        "representation stress active": min(
            case["n_qubits"] for case in config["cases"]
        )
        >= 70
        and max(all_bonds) >= 4,
    }
    charged_error = (
        float(np.max(np.abs(charged - expected_charged)))
        if charged.shape == expected_charged.shape
        else float("inf")
    )
    probability_error = (
        float(np.max(np.abs(probabilities - expected_probabilities)))
        if probabilities.shape == expected_probabilities.shape
        else float("inf")
    )
    moment_error = (
        float(np.max(np.abs(moments - expected_moments)))
        if moments.shape == expected_moments.shape
        else float("inf")
    )
    entropy_error = (
        float(np.max(np.abs(entropies - expected_entropies)))
        if entropies.shape == expected_entropies.shape
        else float("inf")
    )
    print("Problem 112 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; case digest: {config['case_digest'][:16]}")
    print(f"End-to-end solution time: {elapsed:.3f}s")
    print(f"Maximum charged-moment error: {charged_error:.3e}")
    print(f"Maximum sector-probability error: {probability_error:.3e}")
    print(f"Maximum sector-second-moment error: {moment_error:.3e}")
    print(f"Maximum resolved-entropy error: {entropy_error:.3e}")
    print(
        "Queried-sector floors: "
        f"p={np.min(queried_probabilities):.3e}, Z2={np.min(queried_moments):.3e}; "
        f"maximum oracle bond={max(all_bonds)}"
    )
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    passed = all(criteria.values())
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_112")
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed(),
    )
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.solution, args.seed) else 1)


if __name__ == "__main__":
    main()
