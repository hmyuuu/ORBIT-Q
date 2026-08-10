"""Expert TensorCircuit-NG solution for robust mixed-state probe design."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc

jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")
tc.set_dtype("complex128")
OPTIMIZER = optax.adam(0.04)


def density_matrix(probe, theta, weights, noise):
    circuit = tc.DMCircuit(2)
    circuit.ry(0, theta=probe[0])
    circuit.rz(0, theta=probe[1])
    circuit.ry(1, theta=probe[2])
    circuit.rz(1, theta=probe[3])
    circuit.rxx(0, 1, theta=probe[4])
    circuit.ryy(0, 1, theta=probe[5])
    for p in range(3):
        circuit.rx(0, theta=2.0 * theta[p] * weights[p, 0])
        circuit.ry(1, theta=2.0 * theta[p] * weights[p, 1])
        circuit.rzz(0, 1, theta=2.0 * theta[p] * weights[p, 2])
        circuit.rz(1, theta=2.0 * theta[p] * weights[p, 3])
        circuit.rxx(0, 1, theta=2.0 * theta[p] * weights[p, 4])
    for qubit in range(2):
        circuit.depolarizing(
            qubit,
            px=noise[qubit, 0],
            py=noise[qubit, 1],
            pz=noise[qubit, 2],
        )
    return circuit.densitymatrix()


def qfim(probe, theta, weights, noise):
    rho = density_matrix(probe, theta, weights, noise)
    derivatives = jnp.moveaxis(
        jax.jacfwd(lambda value: density_matrix(probe, value, weights, noise))(theta),
        -1,
        0,
    )
    basis = jnp.eye(16, dtype=jnp.complex128).reshape(16, 4, 4)
    mapped = jax.vmap(lambda item: 0.5 * (rho @ item + item @ rho))(basis)
    linear_map = mapped.reshape(16, 16).T
    slds = jax.vmap(
        lambda derivative: jnp.linalg.solve(linear_map, derivative.reshape(-1))
    )(derivatives).reshape(3, 4, 4)
    fisher = jax.vmap(
        lambda derivative: jax.vmap(lambda sld: jnp.real(jnp.trace(derivative @ sld)))(
            slds
        )
    )(derivatives)
    return 0.5 * (fisher + fisher.T)


def design_ensemble(config):
    theta = 0.51 * jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            [0.72, 0.72, 0.72],
            [0.72, -0.72, -0.72],
            [-0.72, 0.72, -0.72],
            [-0.72, -0.72, 0.72],
        ]
    )
    signs = jnp.asarray(
        [
            [[0, 0, 0], [0, 0, 0]],
            [[1, -1, 1], [-1, 1, -1]],
            [[-1, 1, -1], [1, -1, 1]],
        ],
        dtype=float,
    )
    center = jnp.mean(jnp.asarray(config["training_noise"]), axis=0)
    return theta, jnp.clip(center + 0.0048 * signs, 0.008, 0.052)


def robust_objective(probe, theta_points, noise_points, weights, regularizer):
    scores, eigenvalues = [], []
    for theta in theta_points:
        for noise in noise_points:
            fisher = qfim(probe, theta, weights, noise)
            scores.append(jnp.linalg.slogdet(fisher + regularizer * jnp.eye(3))[1])
            eigenvalues.append(jnp.linalg.eigvalsh(fisher)[0])
    scores, eigenvalues = jnp.stack(scores), jnp.stack(eigenvalues)
    return -jax.nn.logsumexp(-10.0 * scores) / 10.0 + 0.22 * (
        -jax.nn.logsumexp(-12.0 * eigenvalues) / 12.0
    )


def batch_loss(probes, theta, noise, weights, regularizer):
    return -jnp.sum(
        jax.vmap(
            lambda probe: robust_objective(probe, theta, noise, weights, regularizer)
        )(probes)
    )


@jax.jit
def step(probes, state, theta, noise, weights, regularizer):
    _, gradient = jax.value_and_grad(batch_loss)(
        probes, theta, noise, weights, regularizer
    )
    updates, state = OPTIMIZER.update(gradient, state, probes)
    return optax.apply_updates(probes, updates), state


def hard_score(probe, theta, noise, weights, regularizer):
    values = [
        jnp.linalg.slogdet(
            qfim(probe, point, weights, channel) + regularizer * jnp.eye(3)
        )[1]
        for point in theta
        for channel in noise
    ]
    return jnp.min(jnp.stack(values))


@jax.jit
def hard_scores(probes, theta, noise, weights, regularizer):
    return jax.vmap(
        lambda probe: hard_score(probe, theta, noise, weights, regularizer)
    )(probes)


def optimize(config, starts):
    weights = jnp.asarray(config["generator_weights"])
    theta, noise = design_ensemble(config)
    regularizer = jnp.asarray(config["score_regularizer"])
    state = OPTIMIZER.init(starts)
    probes = starts
    for _ in range(160):
        probes, state = step(probes, state, theta, noise, weights, regularizer)
    return probes, hard_scores(probes, theta, noise, weights, regularizer)


def run_solution(config):
    seed = int(str(config["instance_fingerprint"])[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    starts = jnp.asarray(
        [
            np.zeros(6),
            rng.normal(0.0, 0.55, 6),
            *rng.uniform(-1.6, 1.6, size=(3, 6)),
            *rng.uniform(-np.pi, np.pi, size=(3, 6)),
        ]
    )
    probes, scores = optimize(config, starts)
    best_probe = probes[int(jnp.argmax(scores))]
    result = (np.asarray(best_probe) + np.pi) % (2.0 * np.pi) - np.pi
    return {"probe_parameters": result}
