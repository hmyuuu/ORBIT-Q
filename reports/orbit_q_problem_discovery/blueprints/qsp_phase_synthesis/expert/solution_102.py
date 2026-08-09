"""Expert TensorCircuit-NG solution for QSP phase synthesis."""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import differential_evolution, least_squares
import tensorcircuit as tc

K = tc.set_backend("jax")
tc.set_dtype("complex64")


def expand_symmetric(half, degree):
    del degree
    return jnp.concatenate([half, half[-2::-1]])


def qsp_response_one(half, x, degree):
    phases = expand_symmetric(half, degree)
    circuit = tc.Circuit(1)
    circuit.rz(0, theta=-2.0 * phases[0])
    off = 1j * jnp.sqrt(jnp.maximum(0.0, 1.0 - x * x))
    signal = jnp.array([[x, off], [off, x]], dtype=jnp.complex64)
    for phase in phases[1:]:
        circuit.any(0, unitary=signal)
        circuit.rz(0, theta=-2.0 * phase)
    return circuit.matrix()[0, 0]


def residual_function(config):
    degree = int(config["degree"])
    x = jnp.asarray(config["training_x"], dtype=jnp.float32)
    coeff = np.asarray(config["target_chebyshev_real"]) + 1j * np.asarray(
        config["target_chebyshev_imag"]
    )
    target = jnp.asarray(np.polynomial.chebyshev.chebval(np.asarray(x), coeff))

    def residual(half):
        response = jax.vmap(lambda point: qsp_response_one(half, point, degree))(x)
        delta = response - target
        return jnp.concatenate([jnp.real(delta), jnp.imag(delta)])

    return jax.jit(residual), jax.jit(jax.jacfwd(residual))


def principal_symmetric(half, degree):
    phases = np.concatenate([half, half[-2::-1]])
    phases = (phases + np.pi) % (2.0 * np.pi) - np.pi
    return 0.5 * (phases + phases[::-1])


def run_solution(config):
    degree = int(config["degree"])
    size = degree // 2 + 1
    residual, jacobian = residual_function(config)

    def fun(values):
        return np.asarray(residual(jnp.asarray(values, dtype=jnp.float32)), dtype=float)

    def jac(values):
        return np.asarray(jacobian(jnp.asarray(values, dtype=jnp.float32)), dtype=float)

    fingerprint = int(str(config["instance_fingerprint"])[:16], 16)

    def objective(values):
        residuals = fun(values)
        return float(np.dot(residuals, residuals))

    base_seed = fingerprint % (2**32)
    searches = ((base_seed, 16, 540), ((base_seed + 0x85EBCA6B) % (2**32), 20, 800))
    best = None
    for seed, population, iterations in searches:
        global_fit = differential_evolution(
            objective,
            [(-np.pi, np.pi)] * size,
            seed=seed,
            popsize=population,
            maxiter=min(int(config["max_iterations"]), iterations),
            tol=1e-9,
            polish=False,
            updating="immediate",
            workers=1,
        )
        fit = least_squares(
            fun,
            global_fit.x,
            jac=jac,
            bounds=(-np.pi, np.pi),
            max_nfev=int(config["max_iterations"]),
            ftol=2e-12,
            xtol=2e-12,
            gtol=2e-12,
        )
        if best is None or np.dot(fit.fun, fit.fun) < np.dot(best.fun, best.fun):
            best = fit
        if np.max(np.abs(best.fun)) < 2e-5:
            break
    return {"phases": principal_symmetric(best.x, degree)}
