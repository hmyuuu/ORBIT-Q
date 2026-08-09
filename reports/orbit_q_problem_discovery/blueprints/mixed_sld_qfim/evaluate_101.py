"""Independent NumPy/SciPy evaluator for Problem 101."""

import argparse
import hashlib
import importlib
import json
import time

import numpy as np
from scipy.linalg import expm


DEFAULT_SEEDS = (101021, 101033, 101051)
I2 = np.eye(2, dtype=np.complex128)
X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
Z = np.diag([1, -1]).astype(np.complex128)
PAULIS = (X, Y, Z)


def rotation(pauli, theta):
    return expm(-0.5j * float(theta) * pauli)


def apply_unitary(rho, unitary):
    return unitary @ rho @ unitary.conj().T


def local(pauli, qubit):
    return np.kron(pauli, I2) if qubit == 0 else np.kron(I2, pauli)


def oracle_density(probe, theta, weights, noise):
    rho = np.zeros((4, 4), dtype=np.complex128)
    rho[0, 0] = 1.0
    probe_gates = (
        (local(Y, 0), probe[0]),
        (local(Z, 0), probe[1]),
        (local(Y, 1), probe[2]),
        (local(Z, 1), probe[3]),
        (np.kron(X, X), probe[4]),
        (np.kron(Y, Y), probe[5]),
    )
    for generator, angle in probe_gates:
        rho = apply_unitary(rho, rotation(generator, angle))
    for p in range(3):
        gates = (
            (local(X, 0), 2 * theta[p] * weights[p, 0]),
            (local(Y, 1), 2 * theta[p] * weights[p, 1]),
            (np.kron(Z, Z), 2 * theta[p] * weights[p, 2]),
            (local(Z, 1), 2 * theta[p] * weights[p, 3]),
            (np.kron(X, X), 2 * theta[p] * weights[p, 4]),
        )
        for generator, angle in gates:
            rho = apply_unitary(rho, rotation(generator, angle))
    for qubit in range(2):
        original = rho
        rho = (1.0 - float(np.sum(noise[qubit]))) * original
        for probability, pauli in zip(noise[qubit], PAULIS):
            embedded = local(pauli, qubit)
            rho = rho + probability * (embedded @ original @ embedded)
    return rho


def finite_derivatives(probe, theta, weights, noise, step=2e-5):
    derivatives = []
    for index in range(3):
        delta = np.zeros(3)
        delta[index] = step
        plus = oracle_density(probe, theta + delta, weights, noise)
        minus = oracle_density(probe, theta - delta, weights, noise)
        derivatives.append((plus - minus) / (2.0 * step))
    return derivatives


def qfim_spectral(rho, derivatives):
    values, vectors = np.linalg.eigh(rho)
    transformed = [
        vectors.conj().T @ derivative @ vectors for derivative in derivatives
    ]
    qfim = np.empty((3, 3), dtype=float)
    denominator = values[:, None] + values[None, :]
    for a in range(3):
        for b in range(3):
            numerator = transformed[a] * transformed[b].T
            qfim[a, b] = float(np.sum(2.0 * np.real(numerator) / denominator))
    return 0.5 * (qfim + qfim.T)


def qfim_sld(rho, derivatives):
    identity = np.eye(4, dtype=np.complex128)
    linear_map = 0.5 * (np.kron(identity, rho) + np.kron(rho.T, identity))
    slds = [
        np.linalg.solve(linear_map, derivative.reshape(-1, order="F")).reshape(
            4, 4, order="F"
        )
        for derivative in derivatives
    ]
    qfim = np.array(
        [
            [np.real(np.trace(derivatives[a] @ slds[b])) for b in range(3)]
            for a in range(3)
        ]
    )
    return 0.5 * (qfim + qfim.T)


def generate_case(seed):
    rng = np.random.default_rng(seed)
    base = np.array(
        [
            [1.0, 0.20, 0.36, -0.12, 0.18],
            [0.14, 1.0, -0.24, 0.31, 0.22],
            [0.23, -0.16, 0.92, 0.18, 0.47],
        ]
    )
    weights = base + rng.normal(0.0, 0.055, base.shape)
    training_theta = rng.uniform(-0.48, 0.48, size=(4, 3))
    base_noise = np.array([[0.030, 0.022, 0.016], [0.024, 0.034, 0.019]])
    training_noise = np.stack(
        [
            np.clip(base_noise + rng.normal(0.0, 0.0035, base_noise.shape), 0.009, 0.05)
            for _ in range(3)
        ]
    )
    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "weights": weights.round(12).tolist(),
                "theta": training_theta.round(12).tolist(),
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:20]
    return {
        "generator_weights": weights,
        "training_theta": training_theta,
        "training_noise": training_noise,
        "support_floor": 2e-5,
        "score_regularizer": 0.04,
        "required_score_gain": 0.32,
        "required_min_qfim_eigenvalue": 0.025,
        "max_steps": 650,
        "learning_rate": 0.035,
        "instance_fingerprint": fingerprint,
    }


def hidden_ensemble(seed, config):
    rng = np.random.default_rng(seed ^ 0x713B)
    theta = rng.uniform(-0.53, 0.53, size=(7, 3))
    center = np.mean(np.asarray(config["training_noise"]), axis=0)
    noise = np.stack(
        [
            np.clip(center + rng.normal(0.0, 0.0045, center.shape), 0.008, 0.052)
            for _ in range(5)
        ]
    )
    return [(point, channel) for point in theta for channel in noise]


def probe_metrics(probe, seed, config):
    weights = np.asarray(config["generator_weights"])
    scores = []
    min_eigenvalues = []
    min_support = np.inf
    identity_errors = []
    physical = True
    for item, (theta, noise) in enumerate(hidden_ensemble(seed, config)):
        rho = oracle_density(probe, theta, weights, noise)
        derivatives = finite_derivatives(probe, theta, weights, noise)
        qfim = qfim_spectral(rho, derivatives)
        rho_values = np.linalg.eigvalsh(rho)
        qfim_values = np.linalg.eigvalsh(qfim)
        min_support = min(min_support, float(rho_values[0]))
        min_eigenvalues.append(float(qfim_values[0]))
        scores.append(
            float(np.linalg.slogdet(qfim + config["score_regularizer"] * np.eye(3))[1])
        )
        physical &= np.linalg.norm(rho - rho.conj().T) <= 2e-10
        physical &= abs(np.trace(rho) - 1.0) <= 2e-10
        physical &= np.linalg.norm(qfim - qfim.T) <= 3e-8 and qfim_values[0] >= -2e-7
        if item in (0, 17, 34):
            identity_errors.append(
                float(np.max(np.abs(qfim - qfim_sld(rho, derivatives))))
            )
    return {
        "robust_score": min(scores),
        "minimum_qfim_eigenvalue": min(min_eigenvalues),
        "minimum_support_eigenvalue": min_support,
        "sld_identity_error": max(identity_errors),
        "physical": physical,
    }


def evaluate_case(module, seed):
    config = generate_case(seed)
    started = time.perf_counter()
    result = module.run_solution(config)
    elapsed = time.perf_counter() - started
    probe = np.asarray(result.get("probe_parameters"), dtype=float)
    valid_shape = probe.shape == (6,)
    finite = bool(valid_shape and np.all(np.isfinite(probe)))
    candidate = (
        probe_metrics(probe, seed, config)
        if finite
        else {
            "robust_score": -np.inf,
            "minimum_qfim_eigenvalue": -np.inf,
            "minimum_support_eigenvalue": -np.inf,
            "sld_identity_error": np.inf,
            "physical": False,
        }
    )
    baseline = probe_metrics(np.zeros(6), seed, config)
    score_gain = candidate["robust_score"] - baseline["robust_score"]
    criteria = {
        "result keys exactly probe_parameters": isinstance(result, dict)
        and set(result) == {"probe_parameters"},
        "probe shape": valid_shape,
        "finite principal parameters": finite
        and bool(np.all(np.abs(probe) <= np.pi + 1e-12)),
        "physical fixed-support states": candidate["physical"]
        and candidate["minimum_support_eigenvalue"] >= config["support_floor"],
        "SLD and spectral identities agree": candidate["sld_identity_error"] <= 3e-7,
        "held-out robust score gain": score_gain >= config["required_score_gain"],
        "held-out minimum QFIM eigenvalue": candidate["minimum_qfim_eigenvalue"]
        >= config["required_min_qfim_eigenvalue"],
    }
    return criteria, {
        "fingerprint": config["instance_fingerprint"],
        "elapsed": elapsed,
        "score_gain": score_gain,
        **candidate,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_101")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    args = parser.parse_args()
    module = importlib.import_module(args.solution)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    total_time = 0.0
    all_pass = True
    print("Problem 101 evaluation")
    print(f"Solution module: {args.solution}")
    for seed in seeds:
        criteria, metrics = evaluate_case(module, seed)
        total_time += metrics["elapsed"]
        all_pass &= all(criteria.values())
        print(
            f"Instance {metrics['fingerprint']}: gain={metrics['score_gain']:.6f}, "
            f"min-qfim={metrics['minimum_qfim_eigenvalue']:.6f}, "
            f"min-support={metrics['minimum_support_eigenvalue']:.3e}"
        )
        for name, passed in criteria.items():
            print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    all_pass &= total_time <= 300.0
    print(f"End-to-end solution time: {total_time:.2f}s")
    print(
        f"Timed execution within 300 seconds: {'PASS' if total_time <= 300.0 else 'FAIL'}"
    )
    print(f"Overall: {'PASS' if all_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
