"""TensorCircuit expert solution for variance-aware signed wire cutting."""

import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")
CHANNELS = ("rX", "rY", "rZ", "eX+", "eX-", "eY+", "eY-", "eZ+", "eZ-")


def apply_gates(circuit, records):
    for gate in records:
        name, q = gate["name"], gate["qubits"]
        if name in {"rx", "ry", "rz"}:
            getattr(circuit, name)(*q, theta=gate["theta"])
        elif name == "cx":
            circuit.cnot(*q)
        else:
            getattr(circuit, name)(*q)


def expectation(circuit, ps):
    return float(np.real(np.asarray(circuit.expectation_ps(ps=ps))))


def fragment_means(case):
    left = tc.Circuit(case["n_left"])
    apply_gates(left, case["left_gates"])
    boundary = case["n_left"] - 1
    bloch = np.array(
        [expectation(left, [0] * boundary + [p]) for p in (1, 2, 3)], dtype=float
    )
    responses = []
    for pauli in "XYZ":
        for sign in "+-":
            right = tc.Circuit(case["n_right"])
            if sign == "-":
                right.x(0)
            if pauli in "XY":
                right.h(0)
            if pauli == "Y":
                right.s(0)
            apply_gates(right, case["right_gates"])
            responses.append(expectation(right, case["observable_ps"]))
    return bloch, np.array(responses)


def signed(r, e):
    return float(
        0.5 * (e[4] + e[5])
        + 0.5 * sum(r[j] * (e[2 * j] - e[2 * j + 1]) for j in range(3))
    )


def means_coefficients(r, e):
    means = np.concatenate([r, e])
    coefficients = np.array(
        [
            0.5 * (e[0] - e[1]),
            0.5 * (e[2] - e[3]),
            0.5 * (e[4] - e[5]),
            0.5 * r[0], -0.5 * r[0],
            0.5 * r[1], -0.5 * r[1],
            0.5 * (1 + r[2]), 0.5 * (1 - r[2]),
        ]
    )
    return means, coefficients


def allocate(means, coefficients, total, minimum):
    remaining = total - minimum * len(CHANNELS)
    weights = np.sqrt(coefficients**2 * np.maximum(0.0, 1 - means**2) + 1e-14)
    raw = remaining * weights / weights.sum()
    extra = np.floor(raw).astype(int)
    residual = remaining - int(extra.sum())
    order = sorted(range(len(CHANNELS)), key=lambda j: (-float(raw[j] - extra[j]), j))
    extra[order[:residual]] += 1
    return extra + minimum


def sample(means, allocation, seed):
    rng = np.random.default_rng(seed)
    values = []
    for mean, shots in zip(means, allocation):
        count = rng.binomial(int(shots), float(np.clip((1 + mean) / 2, 0, 1)))
        values.append((2 * count - shots) / shots)
    return np.array(values)


def solve_case(case):
    r, e = fragment_means(case)
    means, coefficients = means_coefficients(r, e)
    allocation = allocate(means, coefficients, case["total_shots"], case["min_shots"])
    predicted = np.sum(coefficients**2 * np.maximum(0.0, 1 - means**2) / allocation)
    sampled = sample(means, allocation, case["sample_seed"])
    return signed(sampled[:3], sampled[3:]), allocation, predicted, signed(r, e), r, e


def run_solution(config):
    solved = [solve_case(case) for case in config["cases"]]
    return {
        "reconstructions": np.array([x[0] for x in solved]),
        "allocations": np.array([x[1] for x in solved], dtype=int),
        "predicted_variances": np.array([x[2] for x in solved]),
        "exact_reconstructions": np.array([x[3] for x in solved]),
        "upstream_bloch": np.array([x[4] for x in solved]),
        "downstream_responses": np.array([x[5] for x in solved]),
    }
