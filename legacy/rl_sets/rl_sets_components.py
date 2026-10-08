"""RL_SETS 各版本共用的元件。

RL_SETSv2~v5 都直接繼承 :class:`RL_SETS`，再以 mixin 引入自己需要的元件，
不再一版接一版繼承。這樣修改某一版不會悄悄改變其他版本的行為；要共用的
部分必須明確放在這裡，並在類別定義列出。

* ``EnergyConstraintMetrics``：候選解摘要加上能量耗損與限制違反（v2 起）。
* ``LifetimeMetrics``：再加上預估壽命（v4 起）。
* ``EnvironmentObservation``：保存 problem，提供全域能量觀察特徵（v3 起）。
* ``ScaledPerturbation``：依比例決定 sensor 數與相鄰值突變的工具（v4 起）。
* ``RandomPolicyOption``：``random_policy=True`` 時改用均勻隨機策略（v4 起）。
"""

from __future__ import annotations

import numpy as np


class EnergyConstraintMetrics:
    """Summaries with energy-depletion and constraint-violation fields.

    Extends the five RL_SETS summary fields with indices 5-8.
    """

    REWARD_EPSILON = 1e-15

    _GLOBAL_DEPLETION = 5
    _WORST_TARGET_DEPLETION = 6
    _CONSTRAINT_VIOLATION = 7
    _FEASIBLE = 8

    def _summarize_state(self, problem, state, fitness):
        """Cache observation fields plus energy and constraint reward fields."""
        sensor_count = max(1, int(problem.SENSOR_NUMBER))
        target_count = max(1, int(problem.TARGET_NUMBER))
        sensing_radii = np.asarray(problem.state_radius(state), dtype=float)
        active_count = int(np.count_nonzero(sensing_radii > 0.0))
        uncovered_count = len(problem.find_uncovered_targets(state))
        disconnected_count = len(
            problem.find_disconnected(state, sensing_radii=sensing_radii)
        )

        energy = np.asarray(problem.energy, dtype=float)
        cost = np.asarray(
            problem.calculate_total_cost(state, sensing_radii=sensing_radii),
            dtype=float,
        )
        positive_cost = np.maximum(cost, 0.0)
        total_energy = float(np.sum(np.maximum(energy, 0.0)))
        global_depletion = float(
            np.clip(
                np.sum(positive_cost) / max(total_energy, self.REWARD_EPSILON),
                0.0,
                1.0,
            )
        )

        target_energy = np.asarray(
            problem.target_sensor_mask @ energy, dtype=float
        )
        target_cost = np.asarray(
            problem.target_sensor_mask @ positive_cost, dtype=float
        )
        if len(target_energy):
            target_depletion = np.ones_like(target_energy, dtype=float)
            np.divide(
                target_cost,
                target_energy,
                out=target_depletion,
                where=target_energy > self.REWARD_EPSILON,
            )
            worst_target_depletion = float(
                np.clip(np.max(target_depletion), 0.0, 1.0)
            )
        else:
            worst_target_depletion = 0.0

        remaining_energy = energy - cost
        energy_failed = np.any(
            (~np.isfinite(remaining_energy)) | (remaining_energy < 0.0)
        )
        coverage_deficit = uncovered_count / target_count
        disconnected_ratio = disconnected_count / max(1, active_count)
        finite_deficit = np.where(
            np.isfinite(remaining_energy),
            np.maximum(-remaining_energy, 0.0),
            np.maximum(energy, 0.0),
        )
        energy_deficit = float(
            np.sum(finite_deficit) / max(total_energy, self.REWARD_EPSILON)
        )
        violation = float(
            coverage_deficit + disconnected_ratio + energy_deficit
        )
        feasible = (
            uncovered_count == 0
            and disconnected_count == 0
            and not bool(energy_failed)
        )

        return np.asarray(
            [
                float(fitness),
                1.0 - coverage_deficit,
                global_depletion,
                active_count / sensor_count,
                disconnected_ratio,
                global_depletion,
                worst_target_depletion,
                violation,
                float(feasible),
            ],
            dtype=float,
        )


class LifetimeMetrics(EnergyConstraintMetrics):
    """Append the predicted network lifetime as summary index 9."""

    _PREDICTED_LIFETIME = 9

    def _summarize_state(self, problem, state, fitness):
        base = super()._summarize_state(problem, state, fitness)
        sensing_radii = np.asarray(problem.state_radius(state), dtype=float)
        cost = np.asarray(
            problem.calculate_total_cost(state, sensing_radii=sensing_radii),
            dtype=float,
        )
        energy = np.maximum(np.asarray(problem.energy, dtype=float), 0.0)
        active = np.isfinite(cost) & (cost > self.REWARD_EPSILON)
        if np.any(active):
            predicted_lifetime = float(np.min(energy[active] / cost[active]))
            if not np.isfinite(predicted_lifetime):
                predicted_lifetime = 0.0
            predicted_lifetime = max(0.0, predicted_lifetime)
        else:
            predicted_lifetime = 0.0
        return np.concatenate((base, np.asarray([predicted_lifetime])))


class EnvironmentObservation:
    """Keep the problem so observations can include global energy features."""

    def __init__(self, problem, *args, **kwargs):
        self._observation_problem = problem
        super().__init__(problem, *args, **kwargs)

    def initialize_market(self, problem, initial_state=None):
        self._observation_problem = problem
        super().initialize_market(problem, initial_state)

    def _environment_features(self, problem):
        """Return four fixed-size lifetime energy summaries in [0, 1]."""
        energy = np.maximum(np.asarray(problem.energy, dtype=float), 0.0)
        initial = max(float(problem.initial_energy), self.REWARD_EPSILON)
        sensor_ratios = np.clip(energy / initial, 0.0, 1.0)

        target_initial = np.asarray(
            problem.target_sensor_mask @ np.full_like(energy, initial),
            dtype=float,
        )
        target_current = np.asarray(
            problem.target_sensor_mask @ energy,
            dtype=float,
        )
        target_ratios = np.zeros_like(target_current, dtype=float)
        np.divide(
            target_current,
            target_initial,
            out=target_ratios,
            where=target_initial > self.REWARD_EPSILON,
        )
        vulnerable_target = (
            float(np.quantile(np.clip(target_ratios, 0.0, 1.0), 0.1))
            if len(target_ratios)
            else 0.0
        )
        return np.asarray(
            [
                float(np.mean(sensor_ratios)),
                float(np.clip(np.std(sensor_ratios), 0.0, 1.0)),
                float(np.mean(sensor_ratios < 0.2)),
                vulnerable_target,
            ],
            dtype=np.float32,
        )


class ScaledPerturbation:
    """Helpers for fraction-sized crossover and adjacent-value mutation."""

    def _sensor_count_for_fraction(self, sensor_count, bounds):
        fraction = self.random.uniform(*bounds)
        return min(sensor_count, max(1, int(np.ceil(sensor_count * fraction))))

    def _adjacent_value(self, current, option_count):
        if option_count <= 1:
            return current
        choices = []
        if current > 0:
            choices.append(current - 1)
        if current + 1 < option_count:
            choices.append(current + 1)
        return self.random.choice(choices)


class RandomPolicyOption:
    """Optionally replace the D3QN policy with a uniform random baseline."""

    def __init__(self, problem, *args, random_policy=False, **kwargs):
        self.random_policy = bool(random_policy)
        super().__init__(problem, *args, **kwargs)

    def _select_actions(self, observations, masks):
        """Use an untrained uniform policy only for the explicit baseline."""
        if not self.random_policy:
            return self.agent.select_actions(observations, masks)
        return np.asarray(
            [self.random.randrange(len(self.ACTION_NAMES)) for _ in observations],
            dtype=int,
        )

    def policy_statistics(self):
        statistics = super().policy_statistics()
        statistics["random_policy"] = self.random_policy
        return statistics


__all__ = [
    "EnergyConstraintMetrics",
    "EnvironmentObservation",
    "LifetimeMetrics",
    "RandomPolicyOption",
    "ScaledPerturbation",
]
