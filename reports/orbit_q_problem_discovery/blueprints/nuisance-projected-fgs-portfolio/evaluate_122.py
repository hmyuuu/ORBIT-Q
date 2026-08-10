"""Independent NumPy evaluator for the design-only Problem 122 prototype."""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import importlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np


DEFAULT_SEED = 1222026
PUBLIC_DEVELOPMENT_KEY = b"orbit-q-r122-public-development-key-not-private"
PARAMETER_COUNT = 5
TARGET_COUNT = 3
DERIVATIVE_STEP = 2e-4


def _default_seed():
    return int(os.environ.get("ORBIT_Q_CANDIDATE_SEED", str(DEFAULT_SEED)))


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _case_digest(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_json_default,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _initial_isometry(n_sites, filled):
    alpha = np.zeros((2 * n_sites, n_sites), dtype=np.complex128)
    occupied = set(filled)
    for site in range(n_sites):
        alpha[site + n_sites if site in occupied else site, site] = 1.0
    return alpha


def _apply_cp(alpha, site, angle):
    n_sites = alpha.shape[0] // 2
    alpha[site] *= np.exp(-0.5j * angle)
    alpha[site + n_sites] *= np.exp(0.5j * angle)


def _apply_hp(alpha, left, right, angle):
    n_sites = alpha.shape[0] // 2
    cosine, sine = np.cos(abs(angle) / 2), np.sin(abs(angle) / 2)
    phase = angle / (abs(angle) + 1e-30)
    left_row, right_row = alpha[left].copy(), alpha[right].copy()
    left_hole, right_hole = (
        alpha[left + n_sites].copy(),
        alpha[right + n_sites].copy(),
    )
    alpha[left] = cosine * left_row - 1j * phase * sine * right_row
    alpha[right] = -1j * np.conj(phase) * sine * left_row + cosine * right_row
    alpha[left + n_sites] = 1j * np.conj(phase) * sine * right_hole + cosine * left_hole
    alpha[right + n_sites] = 1j * phase * sine * left_hole + cosine * right_hole


def _apply_sp(alpha, left, right, angle):
    n_sites = alpha.shape[0] // 2
    cosine, sine = np.cos(abs(angle) / 2), np.sin(abs(angle) / 2)
    phase = angle / (abs(angle) + 1e-30)
    left_row, right_row = alpha[left].copy(), alpha[right].copy()
    left_hole, right_hole = (
        alpha[left + n_sites].copy(),
        alpha[right + n_sites].copy(),
    )
    alpha[left] = cosine * left_row - 1j * phase * sine * right_hole
    alpha[right] = cosine * right_row + 1j * phase * sine * left_hole
    alpha[left + n_sites] = 1j * np.conj(phase) * sine * right_row + cosine * left_hole
    alpha[right + n_sites] = (
        -1j * np.conj(phase) * sine * left_row + cosine * right_hole
    )


def _physical_parameters(z, config):
    return np.asarray(config["parameter_centers"]) + np.asarray(
        config["parameter_half_widths"]
    ) * np.asarray(z)


def _evolve_isometry(z, config, action):
    hopping, pairing, potential, offset, _ = _physical_parameters(z, config)
    n_sites = config["n_sites"]
    alpha = _initial_isometry(n_sites, config["filled"])
    for layer, word in enumerate(action["control_word"]):
        phase = action["phase"] + offset + 0.37 * layer * word
        for site in range(n_sites):
            angle = (
                action["dt"]
                * potential
                * (-1) ** site
                * (1.0 + 0.13 * np.cos(phase + 0.31 * site))
            )
            _apply_cp(alpha, site, angle)
        for parity in (0, 1):
            for site in range(parity, n_sites - 1, 2):
                angle = (
                    action["dt"]
                    * hopping
                    * word
                    * (1.0 + 0.11 * np.sin(phase + 0.47 * site))
                )
                _apply_hp(alpha, site, site + 1, angle)
        for parity in (0, 1):
            for site in range(parity, n_sites - 1, 2):
                angle = action["dt"] * pairing * np.exp(1j * (phase + 0.19 * site))
                _apply_sp(alpha, site, site + 1, angle)
    return alpha


def _probability(z, config, action):
    alpha = _evolve_isometry(z, config, action)
    correlation = alpha @ alpha.conj().T
    site = action["measurement_site"]
    q = float(np.real(correlation[site, site]))
    contrast = float(_physical_parameters(z, config)[4])
    return 0.5 + contrast * (q - 0.5), q


def _action_fisher(z, config, action, step=DERIVATIVE_STEP):
    probability, q = _probability(z, config, action)
    gradient = np.empty(PARAMETER_COUNT)
    for coordinate in range(4):
        delta = np.zeros(PARAMETER_COUNT)
        delta[coordinate] = step
        plus = _probability(z + delta, config, action)[0]
        minus = _probability(z - delta, config, action)[0]
        gradient[coordinate] = (plus - minus) / (2 * step)
    gradient[4] = config["parameter_half_widths"][4] * (q - 0.5)
    fisher = np.outer(gradient, gradient) / (probability * (1.0 - probability))
    return fisher, probability, gradient


def _fisher_tables(config, points):
    points = np.asarray(points, dtype=float)
    tables = np.empty((len(points), len(config["actions"]), 5, 5))
    probabilities = np.empty((len(points), len(config["actions"])))
    for point_index, point in enumerate(points):
        for action_index, action in enumerate(config["actions"]):
            fisher, probability, _ = _action_fisher(point, config, action)
            tables[point_index, action_index] = fisher
            probabilities[point_index, action_index] = probability
    return tables, probabilities


def _portfolio_metrics(allocation, tables):
    prior = np.diag([0.0, 0.0, 0.0, 4.0, 9.0])
    matrices = prior + np.einsum("a,paij->pij", allocation, tables)
    logdet = np.empty(len(matrices))
    minimum = np.empty(len(matrices))
    condition = np.empty(len(matrices))
    for index, matrix in enumerate(matrices):
        effective = matrix[:3, :3] - matrix[:3, 3:] @ np.linalg.solve(
            matrix[3:, 3:], matrix[3:, :3]
        )
        effective = 0.5 * (effective + effective.T)
        values = np.linalg.eigvalsh(effective)
        regularized = values + 0.25
        logdet[index] = np.sum(np.log(regularized))
        minimum[index] = values[0]
        condition[index] = regularized[-1] / regularized[0]
    return logdet, minimum, condition


def _configuration(seed):
    rng = np.random.default_rng(seed)
    n_sites = 16
    phases = np.asarray([0.17, 0.83, 1.51, 2.29]) + rng.uniform(-0.04, 0.04, 4)
    durations = np.asarray([0.18, 0.29]) + rng.uniform(-0.008, 0.008, 2)
    candidate_sites = np.arange(2, n_sites - 2)
    sites = sorted(rng.choice(candidate_sites, size=3, replace=False).tolist())
    words = list(itertools.product((-1, 1), repeat=3))
    actions = []
    for index, (phase, duration, site) in enumerate(
        itertools.product(phases, durations, sites)
    ):
        word = words[(index + seed) % len(words)]
        identity = hashlib.sha256(
            f"r122:{seed}:{index}:{phase:.17g}:{duration:.17g}:{site}:{word}".encode()
        ).hexdigest()[:16]
        switches = sum(word[i] != word[i - 1] for i in (1, 2))
        actions.append(
            {
                "action_id": identity,
                "phase": float(phase),
                "dt": float(duration),
                "measurement_site": int(site),
                "control_word": list(word),
                "cost_units": int(1 + (duration > 0.23) + (switches == 2)),
            }
        )
    rng.shuffle(actions)
    centers = np.asarray([0.74, 0.32, -0.25, 0.0, 0.92]) + rng.normal(
        0.0, [0.008, 0.006, 0.007, 0.0, 0.003]
    )
    half_widths = np.asarray([0.14, 0.11, 0.12, 0.065, 0.05])
    public = [np.zeros(5)]
    for coordinate in range(5):
        for sign in (-1.0, 1.0):
            point = np.zeros(5)
            point[coordinate] = 0.65 * sign
            public.append(point)
    baseline = np.zeros(len(actions), dtype=np.int64)
    baseline_indices = sorted(
        range(len(actions)),
        key=lambda i: (actions[i]["cost_units"], actions[i]["action_id"]),
    )[:8]
    baseline[baseline_indices] = 128
    return {
        "schema_version": 1,
        "prototype_scale": "reduced_local_design_canary",
        "n_sites": n_sites,
        "filled": list(range(0, n_sites, 2)),
        "parameter_order": [
            "hopping",
            "pairing",
            "staggered_potential",
            "phase_offset",
            "readout_contrast",
        ],
        "parameter_centers": centers.tolist(),
        "parameter_half_widths": half_widths.tolist(),
        "normalized_hidden_radius": 0.82,
        "derivative_step": DERIVATIVE_STEP,
        "actions": actions,
        "training_points": np.asarray(public).tolist(),
        "baseline_allocation": baseline.tolist(),
        "shot_budget": 1024,
        "allocation_granularity": 16,
        "minimum_active_actions": 6,
        "maximum_active_actions": 8,
        "minimum_active_shots": 64,
        "maximum_active_shots": 384,
        "interrogation_budget_units": 2650,
        "minimum_logdet_gain": 1.32,
        "e_ratio_floor": 1e-8,
        "minimum_e_ratio": 1.05,
        "maximum_regularized_condition": 450.0,
        "minimum_probability_margin": 0.01,
        "certificate_tolerance": 2e-6,
        "case_nonce": int(rng.integers(0, 2**63 - 1)),
    }


def _hidden_points(config, public_digest, key):
    radius = config["normalized_hidden_radius"]
    corners = np.asarray(list(itertools.product((-radius, radius), repeat=5)))
    material = hmac.new(
        key,
        b"r122-hidden-v1:" + public_digest.encode(),
        hashlib.sha256,
    ).digest()
    rng = np.random.default_rng(int.from_bytes(material[:8], "big"))
    interior = []
    candidates = rng.uniform(-radius, radius, size=(256, 5))
    while len(interior) < 8:
        if not interior:
            selected = int(np.argmax(np.linalg.norm(candidates, axis=1)))
        else:
            distances = np.min(
                np.linalg.norm(
                    candidates[:, None] - np.asarray(interior)[None], axis=2
                ),
                axis=1,
            )
            selected = int(np.argmax(distances))
        interior.append(candidates[selected].copy())
        candidates = np.delete(candidates, selected, axis=0)
    return np.vstack((corners, np.asarray(interior)))


def _result_artifact(allocation, config, public_tables=None):
    if public_tables is None:
        public_tables, _ = _fisher_tables(config, config["training_points"])
    logdet, minimum, _ = _portfolio_metrics(allocation, public_tables)
    return {
        "shot_allocation": np.asarray(allocation, dtype=np.int64),
        "public_logdet": np.asarray(logdet, dtype=np.float64),
        "public_min_eigenvalue": np.asarray(minimum, dtype=np.float64),
    }


def _validate_result(result, config, hidden_tables, hidden_probabilities, elapsed):
    errors = []
    required = {"shot_allocation", "public_logdet", "public_min_eigenvalue"}
    if type(result) is not dict or set(result) != required:
        return False, {"errors": ["result must be an exact three-key dict"]}
    allocation = result["shot_allocation"]
    public_logdet = result["public_logdet"]
    public_minimum = result["public_min_eigenvalue"]
    if type(allocation) is not np.ndarray or allocation.dtype != np.dtype("int64"):
        errors.append("shot_allocation must be an int64 ndarray")
    for name, value in (
        ("public_logdet", public_logdet),
        ("public_min_eigenvalue", public_minimum),
    ):
        if type(value) is not np.ndarray or value.dtype != np.dtype("float64"):
            errors.append(f"{name} must be a float64 ndarray")
    expected_shape = (len(config["actions"]),)
    certificate_shape = (len(config["training_points"]),)
    if type(allocation) is not np.ndarray or allocation.shape != expected_shape:
        errors.append(f"allocation shape must be {expected_shape}")
    if (
        type(public_logdet) is not np.ndarray
        or public_logdet.shape != certificate_shape
    ):
        errors.append(f"public_logdet shape must be {certificate_shape}")
    if (
        type(public_minimum) is not np.ndarray
        or public_minimum.shape != certificate_shape
    ):
        errors.append(f"public_min_eigenvalue shape must be {certificate_shape}")
    if errors:
        return False, {"errors": errors}
    if not np.all(np.isfinite(public_logdet)) or not np.all(
        np.isfinite(public_minimum)
    ):
        errors.append("certificates must be finite")
    if np.any(allocation < 0) or np.any(
        allocation % config["allocation_granularity"] != 0
    ):
        errors.append("allocation entries violate nonnegative granularity")
    active = allocation > 0
    active_count = int(np.count_nonzero(active))
    if (
        not config["minimum_active_actions"]
        <= active_count
        <= config["maximum_active_actions"]
    ):
        errors.append("active-action count is outside the declared range")
    if np.any(allocation[active] < config["minimum_active_shots"]) or np.any(
        allocation[active] > config["maximum_active_shots"]
    ):
        errors.append("active shots violate per-action bounds")
    if int(np.sum(allocation)) != config["shot_budget"]:
        errors.append("shot budget is not exact")
    costs = np.asarray([action["cost_units"] for action in config["actions"]])
    interrogation_cost = int(allocation @ costs)
    if interrogation_cost > config["interrogation_budget_units"]:
        errors.append("interrogation budget exceeded")
    public_tables, _ = _fisher_tables(config, config["training_points"])
    expected_logdet, expected_minimum, _ = _portfolio_metrics(allocation, public_tables)
    certificate_error = float(
        max(
            np.max(np.abs(public_logdet - expected_logdet)),
            np.max(np.abs(public_minimum - expected_minimum)),
        )
    )
    if certificate_error > config["certificate_tolerance"]:
        errors.append("public certificate mismatch")
    hidden_logdet, hidden_minimum, hidden_condition = _portfolio_metrics(
        allocation, hidden_tables
    )
    baseline = np.asarray(config["baseline_allocation"], dtype=np.int64)
    base_logdet, base_minimum, _ = _portfolio_metrics(baseline, hidden_tables)
    minimum_gain = float(np.min(hidden_logdet - base_logdet))
    minimum_ratio = float(
        np.min(
            (hidden_minimum + config["e_ratio_floor"])
            / (base_minimum + config["e_ratio_floor"])
        )
    )
    maximum_condition = float(np.max(hidden_condition))
    probability_margin = float(
        np.min(
            np.minimum(
                hidden_probabilities[:, active], 1 - hidden_probabilities[:, active]
            )
        )
    )
    if minimum_gain < config["minimum_logdet_gain"]:
        errors.append("hidden logdet gain below threshold")
    if minimum_ratio < config["minimum_e_ratio"]:
        errors.append("hidden E-optimality ratio below threshold")
    if maximum_condition > config["maximum_regularized_condition"]:
        errors.append("hidden regularized condition number above threshold")
    if probability_margin < config["minimum_probability_margin"]:
        errors.append("hidden probability margin below threshold")
    summary = {
        "minimum_hidden_logdet_gain": minimum_gain,
        "minimum_hidden_e_ratio": minimum_ratio,
        "maximum_hidden_regularized_condition": maximum_condition,
        "minimum_hidden_probability_margin": probability_margin,
        "maximum_public_certificate_error": certificate_error,
        "active_action_count": active_count,
        "interrogation_cost_units": interrogation_cost,
        "solution_runtime_seconds": float(elapsed),
        "errors": errors,
    }
    return not errors, summary


def _fermion_operators(n_sites):
    identity = np.eye(2, dtype=np.complex128)
    z = np.diag([1.0, -1.0]).astype(np.complex128)
    lowering = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    operators = []
    for site in range(n_sites):
        operator = np.ones((1, 1), dtype=np.complex128)
        for position in range(n_sites):
            factor = (
                z if position < site else lowering if position == site else identity
            )
            operator = np.kron(operator, factor)
        operators.append(operator)
    return operators


def _apply_dense_hamiltonian(state, hamiltonian):
    values, vectors = np.linalg.eigh(hamiltonian)
    return vectors @ (np.exp(-1j * values) * (vectors.conj().T @ state))


def _dense_q_values(z, config, action):
    n_sites = config["n_sites"]
    annihilation = _fermion_operators(n_sites)
    number = [operator.conj().T @ operator for operator in annihilation]
    dimension = 1 << n_sites
    identity = np.eye(dimension, dtype=np.complex128)
    basis = sum(1 << (n_sites - 1 - site) for site in config["filled"])
    state = np.zeros(dimension, dtype=np.complex128)
    state[basis] = 1.0
    hopping, pairing, potential, offset, _ = _physical_parameters(z, config)
    for layer, word in enumerate(action["control_word"]):
        phase = action["phase"] + offset + 0.37 * layer * word
        for site in range(n_sites):
            angle = (
                action["dt"]
                * potential
                * (-1) ** site
                * (1.0 + 0.13 * np.cos(phase + 0.31 * site))
            )
            state = _apply_dense_hamiltonian(
                state, 0.5 * angle * (number[site] - 0.5 * identity)
            )
        for parity in (0, 1):
            for site in range(parity, n_sites - 1, 2):
                angle = (
                    action["dt"]
                    * hopping
                    * word
                    * (1.0 + 0.11 * np.sin(phase + 0.47 * site))
                )
                pair = annihilation[site].conj().T @ annihilation[site + 1]
                hamiltonian = 0.5 * (angle * pair + np.conj(angle) * pair.conj().T)
                state = _apply_dense_hamiltonian(state, hamiltonian)
        for parity in (0, 1):
            for site in range(parity, n_sites - 1, 2):
                angle = action["dt"] * pairing * np.exp(1j * (phase + 0.19 * site))
                pair = annihilation[site].conj().T @ annihilation[site + 1].conj().T
                hamiltonian = 0.5 * (angle * pair + np.conj(angle) * pair.conj().T)
                state = _apply_dense_hamiltonian(state, hamiltonian)
    return np.asarray(
        [1.0 - float(np.real(np.vdot(state, operator @ state))) for operator in number]
    )


def _dense_fock_canary(config):
    tiny = copy.deepcopy(config)
    tiny["n_sites"] = 4
    tiny["filled"] = [0, 2]
    action = copy.deepcopy(config["actions"][0])
    action["measurement_site"] = 1
    point = np.asarray([0.23, -0.31, 0.17, -0.27, 0.11])
    covariance = _evolve_isometry(point, tiny, action)
    correlation = covariance @ covariance.conj().T
    covariance_q = np.real(np.diag(correlation)[: tiny["n_sites"]])
    dense_q = _dense_q_values(point, tiny, action)
    return float(np.max(np.abs(covariance_q - dense_q)))


def _derivative_canary(config):
    maximum = 0.0
    for point, action in zip(
        np.asarray(config["training_points"])[[0, 3, 8]], config["actions"][:3]
    ):
        for coordinate in range(4):
            h = DERIVATIVE_STEP
            delta = np.zeros(5)
            delta[coordinate] = h
            three = (
                _probability(point + delta, config, action)[0]
                - _probability(point - delta, config, action)[0]
            ) / (2 * h)
            five = (
                -_probability(point + 2 * delta, config, action)[0]
                + 8 * _probability(point + delta, config, action)[0]
                - 8 * _probability(point - delta, config, action)[0]
                + _probability(point - 2 * delta, config, action)[0]
            ) / (12 * h)
            maximum = max(maximum, abs(three - five))
    return float(maximum)


class _SemanticFGSSimulator:
    """API shim for expert semantics only; it is not TensorCircuit evidence."""

    def __init__(self, n_sites, filled):
        self.n_sites = n_sites
        self.alpha = _initial_isometry(n_sites, filled)

    def evol_cp(self, site, angle):
        _apply_cp(self.alpha, site, angle)

    def evol_hp(self, left, right, angle):
        _apply_hp(self.alpha, left, right, angle)

    def evol_sp(self, left, right, angle):
        _apply_sp(self.alpha, left, right, angle)

    def get_cmatrix(self):
        return self.alpha @ self.alpha.conj().T


def _run_expert(config):
    inserted_shim = False
    try:
        importlib.import_module("tensorcircuit")
        scope = "local_tensorcircuit_import"
    except ModuleNotFoundError:
        shim = types.ModuleType("tensorcircuit")
        shim.FGSSimulator = _SemanticFGSSimulator
        shim.set_backend = lambda _name: None
        shim.set_dtype = lambda _name: None
        sys.modules["tensorcircuit"] = shim
        inserted_shim = True
        scope = "semantic_fgs_shim_not_tensorcircuit_evidence"
    path = Path(__file__).resolve().parent / "expert" / "solution_122.py"
    spec = importlib.util.spec_from_file_location("r122_expert_proxy", path)
    module = importlib.util.module_from_spec(spec)
    started = time.perf_counter()
    try:
        spec.loader.exec_module(module)
        result = module.run_solution(copy.deepcopy(config))
    finally:
        if inserted_shim:
            sys.modules.pop("tensorcircuit", None)
    return result, time.perf_counter() - started, scope


def _ranked_equal_mutant(config, tables, mode):
    scores = []
    prior_nuisance = np.diag([4.0, 9.0])
    for action in range(len(config["actions"])):
        if mode == "nominal_only":
            score = float(np.trace(tables[0, action, :3, :3]))
        elif mode == "no_nuisance":
            score = float(np.min(np.trace(tables[:, action, :3, :3], axis1=1, axis2=2)))
        else:
            values = []
            for matrix in tables[:, action]:
                effective = matrix[:3, :3] - matrix[:3, 3:] @ np.linalg.solve(
                    matrix[3:, 3:] + prior_nuisance, matrix[3:, :3]
                )
                values.append(np.trace(effective))
            score = float(np.min(values))
        scores.append((score, config["actions"][action]["action_id"], action))
    allocation = np.zeros(len(config["actions"]), dtype=np.int64)
    allocation[[record[2] for record in sorted(scores, reverse=True)[:8]]] = 128
    return allocation


def _self_test(seed):
    config = _configuration(seed)
    digest = _case_digest(config)
    hidden = _hidden_points(config, digest, PUBLIC_DEVELOPMENT_KEY)
    hidden_tables, hidden_probabilities = _fisher_tables(config, hidden)
    dense_error = _dense_fock_canary(config)
    derivative_error = _derivative_canary(config)
    result, elapsed, expert_scope = _run_expert(config)
    passed, summary = _validate_result(
        result, config, hidden_tables, hidden_probabilities, elapsed
    )
    if not passed:
        raise AssertionError(f"expert semantic proxy failed: {summary}")
    public_tables, _ = _fisher_tables(config, config["training_points"])
    mutant_artifacts = {
        "baseline": _result_artifact(
            np.asarray(config["baseline_allocation"], dtype=np.int64),
            config,
            public_tables,
        ),
        "nominal_only": _result_artifact(
            _ranked_equal_mutant(config, public_tables, "nominal_only"),
            config,
            public_tables,
        ),
        "no_nuisance_schur": _result_artifact(
            _ranked_equal_mutant(config, public_tables, "no_nuisance"),
            config,
            public_tables,
        ),
        "axis_only": _result_artifact(
            _ranked_equal_mutant(config, public_tables, "axis_only"),
            config,
            public_tables,
        ),
    }
    uniform = np.zeros(len(config["actions"]), dtype=np.int64)
    uniform[np.linspace(0, len(uniform) - 1, 8, dtype=int)] = 128
    mutant_artifacts["uniform_eight_action"] = _result_artifact(
        uniform, config, public_tables
    )
    stale = np.roll(result["shot_allocation"], 1)
    mutant_artifacts["stale_shuffled_action_order"] = _result_artifact(
        stale, config, public_tables
    )
    naive = result["shot_allocation"].copy()
    first, second = np.flatnonzero(naive)[:2]
    naive[first] += 8
    naive[second] -= 8
    mutant_artifacts["naive_independent_rounding"] = _result_artifact(
        naive, config, public_tables
    )
    float_schema = copy.deepcopy(result)
    float_schema["shot_allocation"] = result["shot_allocation"].astype(float)
    mutant_artifacts["float_allocation"] = float_schema
    subclass_schema = copy.deepcopy(result)
    array_subclass = type("ArraySubclass", (np.ndarray,), {})
    subclass_schema["shot_allocation"] = result["shot_allocation"].view(array_subclass)
    mutant_artifacts["ndarray_subclass"] = subclass_schema
    nan_certificate = copy.deepcopy(result)
    nan_certificate["public_logdet"] = result["public_logdet"].copy()
    nan_certificate["public_logdet"][0] = np.nan
    mutant_artifacts["nan_certificate"] = nan_certificate
    mutation_results = {}
    for name, artifact in mutant_artifacts.items():
        accepted, details = _validate_result(
            artifact, config, hidden_tables, hidden_probabilities, 0.0
        )
        mutation_results[name] = {
            "rejected": not accepted,
            "errors": details.get("errors", []),
        }
    if not all(record["rejected"] for record in mutation_results.values()):
        raise AssertionError(f"scientific/schema mutant survived: {mutation_results}")
    if dense_error > 2e-12 or derivative_error > 2e-7:
        raise AssertionError(
            f"oracle canary failed: dense={dense_error}, derivative={derivative_error}"
        )
    return {
        "case_digest": digest,
        "expert_scope": expert_scope,
        "expert_summary": summary,
        "dense_fock_max_probability_error": dense_error,
        "three_vs_five_point_max_derivative_error": derivative_error,
        "mutants": mutation_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", default="solution_122")
    parser.add_argument("--seed", type=int, default=_default_seed())
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(_self_test(args.seed), sort_keys=True, default=_json_default))
        print("Overall: PASS (design-only local self-test)")
        return
    verifier_seed = os.environ.get("ORBIT_Q_CANDIDATE_SEED")
    if verifier_seed is not None and int(verifier_seed) != args.seed:
        raise ValueError("--seed cannot override the verifier-authorized public seed")
    config = _configuration(args.seed)
    digest = _case_digest(config)
    hidden_key_text = os.environ.get("ORBIT_Q_HIDDEN_KEY")
    hidden_key = (
        hidden_key_text.encode()
        if hidden_key_text is not None
        else PUBLIC_DEVELOPMENT_KEY
    )
    hidden_points = _hidden_points(config, digest, hidden_key)
    hidden_tables, hidden_probabilities = _fisher_tables(config, hidden_points)
    print(
        json.dumps(
            {
                "orbit_q_case_identity": {
                    "protocol_seed": args.seed,
                    "case_digest": digest,
                }
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )
    os.environ.pop("ORBIT_Q_CANDIDATE_SEED", None)
    os.environ.pop("ORBIT_Q_HIDDEN_KEY", None)
    sys.argv[:] = [sys.argv[0]]
    started = time.perf_counter()
    try:
        result = importlib.import_module(args.solution).run_solution(
            copy.deepcopy(config)
        )
        elapsed = time.perf_counter() - started
        passed, summary = _validate_result(
            result, config, hidden_tables, hidden_probabilities, elapsed
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        passed = False
        summary = {"errors": [f"{type(exc).__name__}: {exc}"]}
    print(f"End-to-end solution time: {elapsed:.6f}s")
    print(json.dumps(summary, sort_keys=True, default=_json_default))
    print("Overall: PASS" if passed else "Overall: FAIL")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
