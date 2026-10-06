"""Two-tier fitness: valid states graded by coverage and quality, invalid
states floored strictly below them.
"""

from __future__ import annotations

import numpy as np


class FitnessServiceV3:
    """Score states with an explicit, self-documented feasibility hierarchy.

    ``quality`` blends two energy signals -- overall network efficiency
    (30%) and the worst-covered target's depletion (70%), since a WSN's
    lifetime is usually decided by whichever target fails first, not by
    the network-wide average::

        quality = 0.3 * (1 - global_depletion) + 0.7 * (1 - worst_target_depletion)

    The total fitness is selected by state status::

        feasible (fully covered, connected, energy-feasible):
            fitness = quality                                            # in [0, 1]

        incomplete (some targets uncovered, still connected and
        energy-feasible):
            fitness = INCOMPLETE_QUALITY_WEIGHT * quality - uncovered_percent
                      - INCOMPLETE_QUALITY_WEIGHT                        # in [-1.5, 0)

        infeasible (a disconnected active sensor, or a sensor with
        negative remaining energy):
            fitness = INVALID_BASE_SCORE - uncovered_percent
                      - disconnected_ratio - energy_deficit_ratio        # below -2

    The incomplete tier equals ``-uncovered_percent - 0.5 * (1 - quality)``:
    the uncovered ratio plus half of the energy loss.  Subtracting the
    weight shifts the whole tier below zero, so every feasible state beats
    every incomplete one while the ordering inside the tier is unchanged.
    The tiers never overlap: incomplete states are at least -1.5, and the
    infeasible base of -2 is below that.
    """

    OBJECTIVE_NAMES = (
        "energy_efficiency",
        "target_region_balance",
        "constraint_score",
    )
    DEFAULT_WEIGHTS = (0.3, 0.7)
    EPSILON = 1e-15
    INCOMPLETE_QUALITY_WEIGHT = 0.5
    INVALID_BASE_SCORE = -2.0

    def __init__(self, problem, w1=0.3, w2=0.7):
        self.problem = problem
        weights = np.asarray([w1, w2], dtype=float)
        if np.any(weights < 0.0) or float(np.sum(weights)) <= self.EPSILON:
            raise ValueError("fitness weights must be non-negative")
        weights /= float(np.sum(weights))
        self.w1, self.w2 = (float(value) for value in weights)
        # Existing result writers expect the final weight to be available.
        self.w3 = self.w2
        self.objective_names = list(self.OBJECTIVE_NAMES)

    def _quality_components(self, energy, cost):
        """Return the two energy-quality components for one schedule."""
        positive_cost = np.maximum(cost, 0.0)
        total_energy = float(np.sum(np.maximum(energy, 0.0)))
        if total_energy <= self.EPSILON:
            global_depletion = 1.0
        else:
            global_depletion = float(
                np.clip(np.sum(positive_cost) / total_energy, 0.0, 1.0)
            )

        target_energy = np.asarray(
            self.problem.target_sensor_mask @ energy,
            dtype=float,
        )
        target_cost = np.asarray(
            self.problem.target_sensor_mask @ positive_cost,
            dtype=float,
        )
        if len(target_energy) == 0:
            worst_target_depletion = 0.0
        else:
            target_depletion = np.ones_like(target_energy, dtype=float)
            np.divide(
                target_cost,
                target_energy,
                out=target_depletion,
                where=target_energy > self.EPSILON,
            )
            worst_target_depletion = float(
                np.clip(np.max(target_depletion), 0.0, 1.0)
            )

        energy_component = self.w1 * (1.0 - global_depletion)
        target_component = self.w2 * (1.0 - worst_target_depletion)
        return energy_component, target_component, total_energy

    def evaluate_state(self, state, debug=False):
        """Return three components whose sum is the fitness."""
        problem = self.problem
        energy = np.asarray(problem.energy, dtype=float)
        sensing_radii = problem.state_radius(state)
        cost = np.asarray(
            problem.calculate_total_cost(
                state,
                sensing_radii=sensing_radii,
            ),
            dtype=float,
        )
        remaining_energy = energy - cost
        problem.target_red = problem.calculate_target_remaining_energy(
            remaining_energy
        )

        energy_component, target_component, total_energy = (
            self._quality_components(energy, cost)
        )
        quality = energy_component + target_component

        uncovered_count = len(problem.find_uncovered_targets(state))
        target_count = int(problem.TARGET_NUMBER)
        uncovered_percent = (
            0.0 if target_count == 0 else uncovered_count / target_count
        )
        disconnected_ids = problem.find_disconnected(
            state,
            sensing_radii=sensing_radii,
        )
        disconnected_count = len(disconnected_ids)
        energy_failed = np.flatnonzero(
            (~np.isfinite(remaining_energy)) | (remaining_energy < 0.0)
        )

        if len(energy_failed) or disconnected_count:
            active_count = max(1, int(np.count_nonzero(sensing_radii > 0)))
            disconnected_ratio = disconnected_count / active_count
            finite_deficit = np.where(
                np.isfinite(remaining_energy),
                np.maximum(-remaining_energy, 0.0),
                np.maximum(energy, 0.0),
            )
            energy_deficit_ratio = float(
                np.sum(finite_deficit) / max(total_energy, self.EPSILON)
            )
            desired_fitness = (
                self.INVALID_BASE_SCORE
                - uncovered_percent
                - disconnected_ratio
                - energy_deficit_ratio
            )
            if debug:
                print("energy failed sensors", energy_failed)
                print("disconnected sensors", disconnected_ids)
        elif uncovered_count:
            desired_fitness = (
                self.INCOMPLETE_QUALITY_WEIGHT * quality
                - uncovered_percent
                - self.INCOMPLETE_QUALITY_WEIGHT
            )
        else:
            desired_fitness = quality

        constraint_component = (
            desired_fitness - energy_component - target_component
        )
        return (
            float(energy_component),
            float(target_component),
            float(constraint_component),
        )


__all__ = ["FitnessServiceV3"]
