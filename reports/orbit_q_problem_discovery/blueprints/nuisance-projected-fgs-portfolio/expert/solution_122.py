"""TensorCircuit expert for the design-only Problem 122 prototype."""

from __future__ import annotations

import numpy as np
import tensorcircuit as tc


tc.set_backend("numpy")
tc.set_dtype("complex128")


def _probability(z, config, action):
    x = np.asarray(config["parameter_centers"]) + np.asarray(
        config["parameter_half_widths"]
    ) * np.asarray(z)
    hopping, pairing, potential, offset, contrast = x
    simulator = tc.FGSSimulator(config["n_sites"], filled=config["filled"])
    for layer, word in enumerate(action["control_word"]):
        phase = action["phase"] + offset + 0.37 * layer * word
        for site in range(config["n_sites"]):
            angle = (
                action["dt"]
                * potential
                * (-1) ** site
                * (1.0 + 0.13 * np.cos(phase + 0.31 * site))
            )
            simulator.evol_cp(site, angle)
        for parity in (0, 1):
            for site in range(parity, config["n_sites"] - 1, 2):
                angle = (
                    action["dt"]
                    * hopping
                    * word
                    * (1.0 + 0.11 * np.sin(phase + 0.47 * site))
                )
                simulator.evol_hp(site, site + 1, angle)
        for parity in (0, 1):
            for site in range(parity, config["n_sites"] - 1, 2):
                angle = action["dt"] * pairing * np.exp(1j * (phase + 0.19 * site))
                simulator.evol_sp(site, site + 1, angle)
    correlation = np.asarray(simulator.get_cmatrix())
    q = float(
        np.real(correlation[action["measurement_site"], action["measurement_site"]])
    )
    return 0.5 + contrast * (q - 0.5), q


def _fisher(config, points):
    step = config["derivative_step"]
    half_widths = np.asarray(config["parameter_half_widths"])
    tables = np.empty((len(points), len(config["actions"]), 5, 5))
    for point_index, point in enumerate(points):
        for action_index, action in enumerate(config["actions"]):
            probability, q = _probability(point, config, action)
            gradient = np.empty(5)
            for coordinate in range(4):
                delta = np.zeros(5)
                delta[coordinate] = step
                plus = _probability(point + delta, config, action)[0]
                minus = _probability(point - delta, config, action)[0]
                gradient[coordinate] = (plus - minus) / (2.0 * step)
            gradient[4] = half_widths[4] * (q - 0.5)
            tables[point_index, action_index] = np.outer(gradient, gradient) / (
                probability * (1.0 - probability)
            )
    return tables


def _metrics(allocation, tables):
    prior = np.diag([0.0, 0.0, 0.0, 4.0, 9.0])
    matrices = prior + np.einsum("a,paij->pij", allocation, tables)
    logdet, minimum, condition = [], [], []
    for matrix in matrices:
        effective = matrix[:3, :3] - matrix[:3, 3:] @ np.linalg.solve(
            matrix[3:, 3:], matrix[3:, :3]
        )
        effective = 0.5 * (effective + effective.T)
        values = np.linalg.eigvalsh(effective)
        regularized = values + 0.25
        logdet.append(float(np.sum(np.log(regularized))))
        minimum.append(float(values[0]))
        condition.append(float(regularized[-1] / regularized[0]))
    return np.asarray(logdet), np.asarray(minimum), np.asarray(condition)


def _design_points(config):
    public = np.asarray(config["training_points"], dtype=float)
    coupled = np.asarray(
        [
            [(-1.0) ** (bin(row & (column + 1)).count("1")) for column in range(5)]
            for row in range(8)
        ]
    )
    return np.vstack((public, 0.78 * coupled))


def _score(allocation, tables, baseline):
    logdet, minimum, condition = _metrics(allocation, tables)
    base_logdet, base_minimum, _ = baseline
    combined = (
        logdet - base_logdet + 0.85 * np.log((minimum + 1e-8) / (base_minimum + 1e-8))
    )
    return float(np.min(combined) - 2e-4 * np.max(condition))


def _allocate(config, tables):
    count = len(config["actions"])
    allocation = np.zeros(count, dtype=np.int64)
    baseline = _metrics(np.asarray(config["baseline_allocation"]), tables)
    costs = np.asarray([action["cost_units"] for action in config["actions"]])
    minimum_cost = int(np.min(costs))
    while int(np.sum(allocation)) < config["shot_budget"]:
        total = int(np.sum(allocation))
        active = int(np.count_nonzero(allocation))
        choices = []
        for action in range(count):
            new = allocation[action] == 0
            if active < config["minimum_active_actions"] and not new:
                continue
            if new and active >= config["maximum_active_actions"]:
                continue
            increment = (
                config["minimum_active_shots"]
                if new
                else config["allocation_granularity"]
            )
            if total + increment > config["shot_budget"]:
                continue
            if allocation[action] + increment > config["maximum_active_shots"]:
                continue
            trial = allocation.copy()
            trial[action] += increment
            remainder = config["shot_budget"] - int(np.sum(trial))
            cost = int(trial @ costs) + remainder * minimum_cost
            if cost > config["interrogation_budget_units"]:
                continue
            choices.append((_score(trial, tables, baseline), -action, trial))
        if not choices:
            raise RuntimeError("allocation construction became infeasible")
        allocation = max(choices, key=lambda item: item[:2])[2]
    return allocation


def run_solution(config):
    design_points = _design_points(config)
    tables = _fisher(config, design_points)
    allocation = _allocate(config, tables)
    public_logdet, public_minimum, _ = _metrics(
        allocation, tables[: len(config["training_points"])]
    )
    return {
        "shot_allocation": allocation,
        "public_logdet": np.asarray(public_logdet, dtype=np.float64),
        "public_min_eigenvalue": np.asarray(public_minimum, dtype=np.float64),
    }
