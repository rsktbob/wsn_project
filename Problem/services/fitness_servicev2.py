"""Depletion-based fitness with a feasible maximum of one."""

from __future__ import annotations

import numpy as np


class FitnessServiceV2:
    """Score current-energy depletion while preserving feasibility guidance.

    For a fully covered, connected, energy-feasible state::

        fitness = 0.3 * (1 - global_depletion)
                + 0.7 * (1 - worst_target_depletion)

    The maximum is therefore exactly ``1``.  Incomplete states remain below
    zero but are ranked by uncovered-target ratio first and energy loss
    second.  Routing or energy failures remain below ``-2``.
    """

    OBJECTIVE_NAMES = (
        "energy_efficiency",
        "target_region_balance",
        "constraint_score",
    )
    DEFAULT_WEIGHTS = (0.3, 0.7)
    EPSILON = 1e-15
    INVALID_BASE_SCORE = -2.0

    def __init__(self, problem, w1=0.3, w2=0.7):
        self.problem = problem
        weights = np.asarray([w1, w2], dtype=float)
        if np.any(weights < 0.0) or float(np.sum(weights)) <= self.EPSILON:
            raise ValueError("V5 fitness weights must be non-negative")
        weights /= float(np.sum(weights))
        self.w1, self.w2 = (float(value) for value in weights)
        # Keep w3 as the target-weight alias used by existing reports/tests.
        self.w3 = self.w2
        self.objective_names = list(self.OBJECTIVE_NAMES)

    def _aggregate_depletion(
        self,
        total_energy,
        positive_cost,
    ):
        """Return V5's network-wide cost-to-current-energy ratio."""
        if total_energy <= self.EPSILON:
            return 1.0
        return float(
            np.clip(np.sum(positive_cost) / total_energy, 0.0, 1.0)
        )

    def evaluate_state(self, state, debug=False):
        """Return weighted energy qualities and a feasibility contribution."""
        problem = self.problem
        energy = np.asarray(problem.energy, dtype=float)
        # Decode already resolved physical radii. Share them during evaluation;
        # only a manually edited or unresolved state requires conversion.
        sensing_radii = problem.state_radius(state)
        cost = np.asarray(
            problem.calculate_total_cost(
                state,
                sensing_radii=sensing_radii,
            ),
            dtype=float,
        )
        remaining_energy = energy - cost
        target_remaining_energy = problem.calculate_target_remaining_energy(
            remaining_energy
        )
        problem.target_red = target_remaining_energy

        positive_cost = np.maximum(cost, 0.0)
        total_energy = float(np.sum(np.maximum(energy, 0.0)))
        target_energy = np.asarray(
            problem.target_sensor_mask @ energy, dtype=float
        )
        target_cost = np.asarray(
            problem.target_sensor_mask @ positive_cost,
            dtype=float,
        )
        aggregate_depletion = self._aggregate_depletion(
            total_energy,
            positive_cost,
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

        energy_component = self.w1 * (1.0 - aggregate_depletion)
        target_component = self.w2 * (1.0 - worst_target_depletion)
        quality = energy_component + target_component
        loss = 1.0 - quality

        uncovered_count = len(problem.find_uncovered_targets(state))
        target_count = int(problem.TARGET_NUMBER)
        coverage_deficit = (
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
            active_count = max(
                1,
                int(
                    np.count_nonzero(sensing_radii > 0)
                ),
            )
            disconnected_ratio = disconnected_count / active_count
            finite_deficit = np.where(
                np.isfinite(remaining_energy),
                np.maximum(-remaining_energy, 0.0),
                np.maximum(energy, 0.0),
            )
            energy_deficit = float(
                np.sum(finite_deficit) / max(total_energy, self.EPSILON)
            )
            desired_fitness = (
                self.INVALID_BASE_SCORE
                - coverage_deficit
                - disconnected_ratio
                - energy_deficit
            )
            if debug:
                print("energy failed sensors", energy_failed)
                print(
                    "disconnected sensors",
                    disconnected_ids,
                )
        elif uncovered_count:
            # Half of one target's coverage increment guarantees that covering
            # one more target dominates every possible energy-loss difference.
            energy_tiebreak = 0.5 / max(1, target_count)
            desired_fitness = -coverage_deficit - energy_tiebreak * loss
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


__all__ = ["FitnessServiceV2"]
