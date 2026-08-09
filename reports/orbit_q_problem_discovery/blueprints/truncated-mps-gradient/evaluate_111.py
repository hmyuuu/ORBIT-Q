"""Independent evaluator for problem 111's truncated-MPS gradient."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time

import numpy as np


X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Z = np.diag([1.0, -1.0]).astype(np.complex128)
I4 = np.eye(4, dtype=np.complex128)
DEFAULT_SEED = 1112026


def default_seed():
    return int(
        os.environ.get(
            "ORBIT_Q_CANDIDATE_SEED",
            os.environ.get("ORBIT_MPS_SEED", str(DEFAULT_SEED)),
        )
    )


def _make_cases(seed, count=2):
    rng = np.random.default_rng(seed)
    cases = []
    for case_index in range(count):
        n = 36
        layers = 4
        terms = []
        for _ in range(5):
            terms.append(
                [
                    float(rng.uniform(-0.7, 0.7)),
                    [[int(rng.integers(0, n)), rng.choice(["X", "Z"]).item()]],
                ]
            )
        for _ in range(7):
            sites = sorted(rng.choice(n, size=2, replace=False).tolist())
            labels = rng.choice(["X", "Z"], size=2).tolist()
            terms.append(
                [float(rng.uniform(-0.55, 0.55)), list(map(list, zip(sites, labels)))]
            )
        case = {
            "n_qubits": n,
            "max_bond_dimension": 8 + 2 * case_index,
            "parameters": rng.uniform(-0.58, 0.58, 3 * layers).tolist(),
            "y_offsets": rng.uniform(-0.45, 0.45, (layers, n)).tolist(),
            "y_scales": rng.uniform(0.55, 1.15, (layers, n)).tolist(),
            "z_offsets": rng.uniform(-0.38, 0.38, (layers, n)).tolist(),
            "z_scales": rng.uniform(-1.1, 1.1, (layers, n)).tolist(),
            "entangler_offsets": rng.uniform(0.38, 0.92, (layers, n - 1)).tolist(),
            "entangler_scales": rng.uniform(-1.05, 1.05, (layers, n - 1)).tolist(),
            "energy_terms": terms,
            "case_nonce": int(rng.integers(0, 2**31 - 1)),
        }
        cases.append(case)
    return cases


def build_config(seed):
    cases = _make_cases(seed)
    payload = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode()
    return {
        "cases": cases,
        "case_digest": hashlib.sha256(payload).hexdigest(),
        "gradient_convention": "fixed-rank retained-SVD derivative",
    }


def _single(axis, theta):
    if axis == "Y":
        generator = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    else:
        generator = Z
    return np.cos(theta / 2) * np.eye(2) - 1j * np.sin(theta / 2) * generator


def _double(axis, theta):
    generator = np.kron(X, X) if axis == "XX" else np.kron(Z, Z)
    return (np.cos(theta / 2) * I4 - 1j * np.sin(theta / 2) * generator).reshape(
        2, 2, 2, 2
    )


class _MPS:
    def __init__(self, n, chi):
        zero = np.array([1.0, 0.0], dtype=np.complex128).reshape(1, 2, 1)
        self.tensors = [zero.copy() for _ in range(n)]
        self.center = 0
        self.chi = chi
        self.cut_gaps = []

    def position(self, site):
        while self.center < site:
            q = self.center
            left, physical, right = self.tensors[q].shape
            matrix = self.tensors[q].reshape(left * physical, right)
            unitary, residue = np.linalg.qr(matrix, mode="reduced")
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

    def one(self, site, gate):
        self.position(site)
        self.tensors[site] = np.einsum("st,ltr->lsr", gate, self.tensors[site])

    def two(self, left, gate, center):
        self.position(left if center == left else left + 1)
        pair = np.einsum("lim,mjr->lijr", self.tensors[left], self.tensors[left + 1])
        pair = np.einsum("abij,lijr->labr", gate, pair)
        dl, _, _, dr = pair.shape
        u, singular, vh = np.linalg.svd(
            pair.reshape(2 * dl, 2 * dr), full_matrices=False
        )
        if singular.size > self.chi:
            gap = (singular[self.chi - 1] - singular[self.chi]) / singular[0]
            self.cut_gaps.append(float(gap))
        keep = min(self.chi, singular.size)
        u, singular, vh = u[:, :keep], singular[:keep], vh[:keep]
        if center == left:
            self.tensors[left] = (u * singular).reshape(dl, 2, keep)
            self.tensors[left + 1] = vh.reshape(keep, 2, dr)
        else:
            self.tensors[left] = u.reshape(dl, 2, keep)
            self.tensors[left + 1] = (singular[:, None] * vh).reshape(keep, 2, dr)
        self.center = center

    def normalize(self):
        self.tensors[self.center] /= np.linalg.norm(self.tensors[self.center])

    def expectation(self, operators):
        by_site = {int(site): X if label == "X" else Z for site, label in operators}
        environment = np.ones((1, 1), dtype=np.complex128)
        for site, tensor in enumerate(self.tensors):
            operator = by_site.get(site, np.eye(2))
            environment = np.einsum(
                "ab,asr,st,btu->ru",
                environment,
                tensor.conj(),
                operator,
                tensor,
                optimize=True,
            )
        return environment.item()


def _simulate(parameters, case, diagnostics=False):
    n = case["n_qubits"]
    state = _MPS(n, case["max_bond_dimension"])
    for layer in range(len(case["y_offsets"])):
        values = parameters[3 * layer : 3 * layer + 3]
        for site in range(n):
            ya = (
                case["y_offsets"][layer][site]
                + case["y_scales"][layer][site] * values[0]
            )
            za = (
                case["z_offsets"][layer][site]
                + case["z_scales"][layer][site] * values[1]
            )
            state.one(site, _single("Y", ya))
            state.one(site, _single("Z", za))
        bonds = range(n - 1) if layer % 2 == 0 else range(n - 2, -1, -1)
        for left in bonds:
            angle = (
                case["entangler_offsets"][layer][left]
                + case["entangler_scales"][layer][left] * values[2]
            )
            center = left + 1 if layer % 2 == 0 else left
            state.two(left, _double("XX" if layer % 2 == 0 else "ZZ", angle), center)
    state.normalize()
    energy = sum(
        weight * state.expectation(operators)
        for weight, operators in case["energy_terms"]
    ).real
    if diagnostics:
        bonds = [tensor.shape[-1] for tensor in state.tensors[:-1]]
        return float(energy), np.asarray(bonds, dtype=int), state.cut_gaps
    return float(energy)


def _oracle(case, step=2e-5):
    parameters = np.asarray(case["parameters"], dtype=float)
    energy, bonds, gaps = _simulate(parameters, case, diagnostics=True)
    gradient = np.empty_like(parameters)
    for index in range(parameters.size):
        plus, minus = parameters.copy(), parameters.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (_simulate(plus, case) - _simulate(minus, case)) / (2 * step)
    return energy, gradient, bonds, gaps


def _array(result, key, dtype):
    try:
        return np.asarray(result[key], dtype=dtype)
    except (KeyError, TypeError, ValueError):
        return np.array([], dtype=dtype)


def _dense_canary():
    rng = np.random.default_rng(111)
    mps = _MPS(4, 16)
    dense = np.zeros(16, dtype=np.complex128)
    dense[0] = 1

    def embed_one(gate, site):
        factors = [np.eye(2, dtype=np.complex128) for _ in range(4)]
        factors[site] = gate
        result = factors[0]
        for factor in factors[1:]:
            result = np.kron(result, factor)
        return result

    def embed_two(gate, left):
        factors = []
        site = 0
        while site < 4:
            if site == left:
                factors.append(gate.reshape(4, 4))
                site += 2
            else:
                factors.append(np.eye(2, dtype=np.complex128))
                site += 1
        result = factors[0]
        for factor in factors[1:]:
            result = np.kron(result, factor)
        return result

    for site in range(4):
        for axis in ("Y", "Z"):
            gate = _single(axis, float(rng.uniform(-0.5, 0.5)))
            mps.one(site, gate)
            dense = embed_one(gate, site) @ dense
    for left, axis, angle, center in (
        (0, "XX", 0.37, 1),
        (1, "XX", -0.29, 2),
        (2, "ZZ", 0.41, 2),
        (1, "ZZ", -0.23, 1),
        (0, "ZZ", 0.17, 0),
    ):
        gate = _double(axis, angle)
        mps.two(left, gate, center)
        dense = embed_two(gate, left) @ dense
    mps.normalize()
    dense /= np.linalg.norm(dense)
    reconstructed = mps.tensors[0]
    for tensor in mps.tensors[1:]:
        reconstructed = np.tensordot(reconstructed, tensor, axes=(-1, 0))
    reconstructed = np.squeeze(reconstructed).reshape(-1)
    pauli = np.kron(np.kron(np.kron(Z, X), Z), X)
    expectation_error = abs(
        mps.expectation([[0, "Z"], [1, "X"], [2, "Z"], [3, "X"]])
        - np.vdot(dense, pauli @ dense)
    )
    return np.allclose(reconstructed, dense, atol=2e-13, rtol=2e-13) and (
        expectation_error < 2e-13
    )


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
    os.environ.pop("ORBIT_MPS_SEED", None)
    if not _dense_canary():
        raise AssertionError("independent TEBD dense canary failed")
    expected = [_oracle(case) for case in config["cases"]]
    expected_energy = np.asarray([row[0] for row in expected])
    expected_gradient = np.asarray([row[1] for row in expected])
    expected_bonds = np.asarray([row[2] for row in expected])
    all_gaps = [gap for row in expected for gap in row[3]]
    started = time.perf_counter()
    result = importlib.import_module(module_name).run_solution(config)
    elapsed = time.perf_counter() - started
    energies = _array(result, "energies", float)
    gradients = _array(result, "gradients", float)
    bonds = _array(result, "final_bond_dimensions", int)
    criteria = {
        "output shapes": energies.shape == expected_energy.shape
        and gradients.shape == expected_gradient.shape
        and bonds.shape == expected_bonds.shape,
        "outputs finite": np.all(np.isfinite(energies))
        and np.all(np.isfinite(gradients)),
        "independent normalized energies": energies.shape == expected_energy.shape
        and np.allclose(energies, expected_energy, atol=4e-8, rtol=3e-7),
        "independent retained-SVD gradients": gradients.shape == expected_gradient.shape
        and np.allclose(gradients, expected_gradient, atol=3e-5, rtol=3e-4),
        "bond dimensions exact": np.array_equal(bonds, expected_bonds),
        "truncation active": len(all_gaps) >= 20,
        "singular-value cuts nondegenerate": bool(all_gaps) and min(all_gaps) > 2e-7,
        "representation stress active": min(
            case["n_qubits"] for case in config["cases"]
        )
        >= 34,
    }
    energy_error = (
        float(np.max(np.abs(energies - expected_energy)))
        if energies.shape == expected_energy.shape
        else float("inf")
    )
    gradient_error = (
        float(np.max(np.abs(gradients - expected_gradient)))
        if gradients.shape == expected_gradient.shape
        else float("inf")
    )
    print("Problem 111 evaluation")
    print(f"Solution module: {module_name}")
    print(f"Case seed: {seed}; case digest: {config['case_digest'][:16]}")
    print(f"End-to-end solution time: {elapsed:.3f}s")
    print(f"Maximum energy error: {energy_error:.3e}")
    print(f"Maximum gradient error: {gradient_error:.3e}")
    print(
        f"Active cuts: {len(all_gaps)}; minimum relative cut gap: {min(all_gaps, default=0.0):.3e}"
    )
    print("Passing criteria:")
    for name, passed in criteria.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    passed = all(criteria.values())
    print(f"Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_111")
    parser.add_argument(
        "--seed",
        type=int,
        default=default_seed(),
    )
    args = parser.parse_args()
    raise SystemExit(0 if evaluate(args.solution, args.seed) else 1)


if __name__ == "__main__":
    main()
