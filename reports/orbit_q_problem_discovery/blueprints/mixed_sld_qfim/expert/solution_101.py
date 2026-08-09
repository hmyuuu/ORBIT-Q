"""Expert TensorCircuit-NG solution for robust mixed-state probe design."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import tensorcircuit as tc

jax.config.update("jax_enable_x64", True)
K = tc.set_backend("jax")
tc.set_dtype("complex128")


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


def robust_objective(probe, theta_points, noise_points, weights, regularizer):
    scores = []
    eigenvalue_rewards = []
    for theta in theta_points:
        for noise in noise_points:
            fisher = qfim(probe, theta, weights, noise)
            scores.append(jnp.linalg.slogdet(fisher + regularizer * jnp.eye(3))[1])
            eigenvalue_rewards.append(jnp.linalg.eigvalsh(fisher)[0])
    scores = jnp.stack(scores)
    eigenvalue_rewards = jnp.stack(eigenvalue_rewards)
    soft_min = -jax.nn.logsumexp(-10.0 * scores) / 10.0
    eigen_soft_min = -jax.nn.logsumexp(-14.0 * eigenvalue_rewards) / 14.0
    return soft_min + 0.35 * eigen_soft_min


def optimize(config, starts):
    weights = jnp.asarray(config["generator_weights"])
    theta = jnp.asarray(config["training_theta"])
    noise = jnp.asarray(config["training_noise"])
    regularizer = float(config["score_regularizer"])
    optimizer = optax.adam(float(config["learning_rate"]))
    state = optimizer.init(starts)

    def loss_fn(probe):
        return -robust_objective(probe, theta, noise, weights, regularizer)

    def batch_losses(probes):
        return jax.vmap(loss_fn)(probes)

    @jax.jit
    def step(probes, opt_state):
        losses, gradient = jax.value_and_grad(
            lambda values: jnp.sum(batch_losses(values))
        )(probes)
        updates, opt_state = optimizer.update(gradient, opt_state, probes)
        return optax.apply_updates(probes, updates), opt_state, losses

    probes = starts
    for _ in range(int(config["max_steps"])):
        probes, state, _ = step(probes, state)
    return probes, batch_losses(probes)


def training_minimum_eigenvalues(config, probes):
    weights = jnp.asarray(config["generator_weights"])
    theta_points = jnp.asarray(config["training_theta"])
    noise_points = jnp.asarray(config["training_noise"])

    def score(probe):
        values = [
            jnp.linalg.eigvalsh(qfim(probe, theta, weights, noise))[0]
            for theta in theta_points
            for noise in noise_points
        ]
        return jnp.min(jnp.stack(values))

    return jax.jit(jax.vmap(score))(probes)


def run_solution(config):
    seed = int(str(config["instance_fingerprint"])[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    starts = jnp.asarray(
        [
            np.zeros(6),
            rng.normal(0.0, 0.45, 6),
            rng.uniform(-1.2, 1.2, 6),
            rng.uniform(-np.pi, np.pi, 6),
            rng.uniform(-np.pi, np.pi, 6),
        ]
    )
    probes, losses = optimize(config, starts)
    del losses
    selection_scores = training_minimum_eigenvalues(config, probes)
    best_probe = probes[int(jnp.argmax(selection_scores))]
    result = (np.asarray(best_probe) + np.pi) % (2.0 * np.pi) - np.pi
    return {"probe_parameters": result}
