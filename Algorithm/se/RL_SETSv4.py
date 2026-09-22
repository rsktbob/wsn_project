"""Lifetime-aware three-scale D3QN hyper-heuristic on SI-SETSv2."""

from __future__ import annotations

import numpy as np

from Algorithm.se.RL_SETSv3 import RL_SETSv3
from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding
from Algorithm.se.RL_SETS import check_environment_contract


class RL_SETSv4(RL_SETSv3):
    """Select small, medium or large crossover-plus-mutation investments.

    Region selection and elitist role updates are the same as SI-SETSv2.  The
    policy changes only the perturbation scale.  Every generated child is
    scored for learning, including children rejected by the elitist update.
    """

    OBSERVATION_NAMES = (
        "searcher_fitness",
        "searcher_predicted_lifetime",
        "searcher_coverage_ratio",
        "searcher_energy_cost_ratio",
        "searcher_active_sensor_ratio",
        "searcher_disconnected_ratio",
        "region_goods_fitness_mean",
        "region_goods_fitness_best",
        "region_goods_fitness_std",
        "region_goods_predicted_lifetime_mean",
        "region_goods_predicted_lifetime_best",
        "region_goods_energy_cost_mean",
        "evaluation_progress",
        "stagnation_ratio",
        "global_remaining_energy_ratio",
        "sensor_remaining_energy_std",
        "low_energy_sensor_ratio",
        "vulnerable_target_energy_ratio",
    )
    OBSERVATION_SIZE = len(OBSERVATION_NAMES)
    ACTION_NAMES = (
        "small_crossover_and_mutation",
        "medium_crossover_and_mutation",
        "large_crossover_and_mutation",
    )
    CHECKPOINT_SCHEMA = "rl_setsv6/1"
    REWARD_VERSION = "signed_energy_lifetime_improvement/1"

    _PREDICTED_LIFETIME = 9
    LIFETIME_NORMALIZATION = 5000.0
    CROSSOVER_FRACTIONS = ((0.05, 0.10), (0.20, 0.30), (0.45, 0.60))
    MUTATION_FRACTIONS = ((0.01, 0.02), (0.05, 0.08), (0.15, 0.20))

    def __init__(self, problem, *args, random_policy=False, **kwargs):
        self.random_policy = bool(random_policy)
        super().__init__(problem, *args, **kwargs)
        self.name = (
            f"RL_SETSv4_{self.n}_{self.h}_{self.w}_{self.mutation_rate}"
        )
        self.agent.checkpoint_metadata = self._checkpoint_metadata(problem)

    def _check_checkpoint(self, problem):
        metadata = self.agent.checkpoint_metadata
        if metadata.get("schema") != self.CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint is not an RL_SETSv4 model")
        if tuple(metadata.get("observation_names", ())) != self.OBSERVATION_NAMES:
            raise ValueError("RL_SETSv4 checkpoint observation schema changed")
        if tuple(metadata.get("action_names", ())) != self.ACTION_NAMES:
            raise ValueError("RL_SETSv4 checkpoint action schema changed")
        if metadata.get("reward_version") != self.REWARD_VERSION:
            raise ValueError("RL_SETSv4 checkpoint reward version changed")
        check_environment_contract(metadata.get("environment"), problem)

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

    def _lifetime_value(self, value):
        value = max(0.0, float(value))
        return float(
            np.clip(
                np.log1p(value) / np.log1p(self.LIFETIME_NORMALIZATION),
                0.0,
                1.0,
            )
        )

    def _build_observations(self):
        observations = np.zeros(
            (self.n, self.OBSERVATION_SIZE), dtype=np.float32
        )
        progress = float(
            np.clip(self.evatime / max(1, self.evaluation_limit), 0.0, 1.0)
        )
        stagnation = float(
            np.clip(
                self.stagnation_rounds / self.stagnation_window, 0.0, 1.0
            )
        )
        environment = self._environment_features(self._observation_problem)
        for searcher_id, region in enumerate(self.selected_regions):
            region_metrics = self.goods_metrics[int(region)]
            searcher = self.searcher_metrics[searcher_id]
            observations[searcher_id] = np.asarray(
                [
                    self._fitness_value(searcher[self._FITNESS]),
                    self._lifetime_value(searcher[self._PREDICTED_LIFETIME]),
                    searcher[self._COVERAGE],
                    searcher[self._ENERGY_COST],
                    searcher[self._ACTIVE],
                    searcher[self._DISCONNECTED],
                    self._fitness_value(
                        np.mean(region_metrics[:, self._FITNESS])
                    ),
                    self._fitness_value(
                        np.max(region_metrics[:, self._FITNESS])
                    ),
                    self._fitness_spread(
                        np.std(region_metrics[:, self._FITNESS])
                    ),
                    self._lifetime_value(
                        np.mean(region_metrics[:, self._PREDICTED_LIFETIME])
                    ),
                    self._lifetime_value(
                        np.max(region_metrics[:, self._PREDICTED_LIFETIME])
                    ),
                    float(np.mean(region_metrics[:, self._ENERGY_COST])),
                    progress,
                    stagnation,
                    *environment,
                ],
                dtype=np.float32,
            )
        return np.nan_to_num(
            observations, nan=0.0, posinf=1.0, neginf=0.0
        )

    def _sensor_count_for_fraction(self, sensor_count, bounds):
        fraction = self.random.uniform(*bounds)
        return min(sensor_count, max(1, int(np.ceil(sensor_count * fraction))))

    def _crossover_for_action(self, searcher, good, action):
        sensor_count = self.code_length // 2
        count = self._sensor_count_for_fraction(
            sensor_count, self.CROSSOVER_FRACTIONS[action]
        )
        start = self.random.randrange(sensor_count - count + 1)
        return swap_segment(
            searcher, good, start * 2, (start + count) * 2
        )

    def _adjacent_value(self, current, option_count):
        if option_count <= 1:
            return current
        choices = []
        if current > 0:
            choices.append(current - 1)
        if current + 1 < option_count:
            choices.append(current + 1)
        return self.random.choice(choices)

    def _mutate_scale(self, problem, candidate, action):
        sensor_count = int(problem.SENSOR_NUMBER)
        count = self._sensor_count_for_fraction(
            sensor_count, self.MUTATION_FRACTIONS[action]
        )
        sensor_ids = self.random.sample(range(sensor_count), count)
        for sensor_id in sensor_ids:
            level_id = sensor_id * 2
            priority_id = level_id + 1
            option_count = int(problem.sensing_option_count(sensor_id))
            current_level = int(candidate.code[level_id]) % option_count
            current_priority = (
                int(candidate.code[priority_id]) % SensorEncoding.RANK_PRECISION
            )

            if action == 0:
                mutate_level = self.random.random() < 0.5 and option_count > 1
                if mutate_level:
                    candidate.code[level_id] = self._adjacent_value(
                        current_level, option_count
                    )
                else:
                    candidate.code[priority_id] = self._adjacent_value(
                        current_priority, SensorEncoding.RANK_PRECISION
                    )
            elif action == 1:
                if option_count > 1:
                    candidate.code[level_id] = self._adjacent_value(
                        current_level, option_count
                    )
                candidate.code[priority_id] = self._adjacent_value(
                    current_priority, SensorEncoding.RANK_PRECISION
                )
            elif action == 2:
                if option_count > 1:
                    candidate.code[level_id] = self._other_value(
                        current_level, option_count
                    )
                candidate.code[priority_id] = self._other_value(
                    current_priority, SensorEncoding.RANK_PRECISION
                )
            else:
                raise ValueError(f"unknown RL-SETSv4 action: {action}")
        return candidate

    def _make_children(self, problem, searcher, good, action):
        action = int(action)
        if action < 0 or action >= len(self.ACTION_NAMES):
            raise ValueError(f"unknown RL-SETSv4 action: {action}")
        children = self._crossover_for_action(searcher, good, action)
        for child in children:
            self._mutate_scale(problem, child, action)
        return children

    def _transition_reward(self, parent_metrics, child_metrics):
        parent_feasible = bool(parent_metrics[self._FEASIBLE] > 0.5)
        child_feasible = bool(child_metrics[self._FEASIBLE] > 0.5)
        if not parent_feasible and child_feasible:
            return 1.0
        if parent_feasible and not child_feasible:
            return -1.0
        if not parent_feasible:
            parent_violation = float(parent_metrics[self._CONSTRAINT_VIOLATION])
            child_violation = float(child_metrics[self._CONSTRAINT_VIOLATION])
            return float(
                np.clip(
                    (parent_violation - child_violation)
                    / max(parent_violation, self.REWARD_EPSILON),
                    -1.0,
                    1.0,
                )
            )

        parent_depletion = (
            0.3 * float(parent_metrics[self._GLOBAL_DEPLETION])
            + 0.7 * float(parent_metrics[self._WORST_TARGET_DEPLETION])
        )
        child_depletion = (
            0.3 * float(child_metrics[self._GLOBAL_DEPLETION])
            + 0.7 * float(child_metrics[self._WORST_TARGET_DEPLETION])
        )
        energy_reward = (parent_depletion - child_depletion) / max(
            parent_depletion, self.REWARD_EPSILON
        )
        parent_lifetime = float(parent_metrics[self._PREDICTED_LIFETIME])
        child_lifetime = float(child_metrics[self._PREDICTED_LIFETIME])
        lifetime_reward = (child_lifetime - parent_lifetime) / max(
            parent_lifetime, 1.0
        )
        return float(
            np.clip(
                0.5 * np.clip(energy_reward, -1.0, 1.0)
                + 0.5 * np.clip(lifetime_reward, -1.0, 1.0),
                -1.0,
                1.0,
            )
        )

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

    def vision_search(self, problem):
        """Evaluate both roles, learn from all proposals, update elitistically."""
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        self.current_adaptive_step = self.adaptive_step * (1.0 - progress)
        best_before = float(self.fitness)
        active_regions = self.selected_regions.copy()
        goods_fitness_before = self.goods_fitness.copy()
        goods_metrics_before = self.goods_metrics.copy()
        searcher_fitness_before = self.searcher_fitness.copy()
        searcher_metrics_before = self.searcher_metrics.copy()

        observations = self._build_observations()
        masks = np.ones((self.n, len(self.ACTION_NAMES)), dtype=bool)
        actions = self._select_actions(observations, masks)

        child_rows = []
        for searcher_id, region in enumerate(active_regions):
            row = []
            for good in self.goods[int(region)]:
                child1, child2 = self._make_children(
                    problem,
                    self.searchers[searcher_id],
                    good,
                    int(actions[searcher_id]),
                )
                row.append(self.align_region(problem, child1, int(region)))
                row.append(self.align_region(problem, child2, int(region)))
            child_rows.append(row)

        child_scores, child_metrics = self._evaluate_child_rows(
            problem, child_rows
        )
        child1_scores = child_scores[:, 0::2]
        child2_scores = child_scores[:, 1::2]
        child1_metrics = child_metrics[:, 0::2]
        child2_metrics = child_metrics[:, 1::2]

        rewards = np.zeros(self.n, dtype=float)
        for searcher_id, region in enumerate(active_regions):
            region = int(region)
            reward1 = np.asarray(
                [
                    self._transition_reward(
                        searcher_metrics_before[searcher_id],
                        child1_metrics[searcher_id, good_id],
                    )
                    for good_id in range(self.w)
                ],
                dtype=float,
            )
            reward2 = np.asarray(
                [
                    self._transition_reward(
                        goods_metrics_before[region, good_id],
                        child2_metrics[searcher_id, good_id],
                    )
                    for good_id in range(self.w)
                ],
                dtype=float,
            )
            rewards[searcher_id] = float(
                np.clip(0.5 * np.max(reward1) + 0.5 * np.mean(reward2), -1.0, 1.0)
            )
            all_rewards = np.concatenate((reward1, reward2))
            improved = int(np.sum(all_rewards > self.IMPROVEMENT_TOLERANCE))
            degraded = int(np.sum(all_rewards < -self.IMPROVEMENT_TOLERANCE))
            action = int(actions[searcher_id])
            self.action_uses[action] += 2 * self.w
            self.action_successes[action] += improved
            self.reward_records.append(
                {
                    "action": action,
                    "progress": progress,
                    "reward": rewards[searcher_id],
                    "improved_goods": improved,
                    "degraded_goods": degraded,
                }
            )
            self.investment_quality[region, searcher_id] = float(
                np.mean(child1_scores[searcher_id])
            )

        probabilities = self.region_probabilities(
            goods_fitness_before, self.investment_quality
        )
        selected = self.select_regions(probabilities)

        for searcher_id in range(self.n):
            good_id = int(np.argmax(child1_scores[searcher_id]))
            score = float(child1_scores[searcher_id, good_id])
            if (
                score
                > searcher_fitness_before[searcher_id]
                + self.IMPROVEMENT_TOLERANCE
            ):
                self.searchers[searcher_id] = child_rows[searcher_id][
                    good_id * 2
                ].copy()
                self.searcher_fitness[searcher_id] = score
                self.searcher_metrics[searcher_id] = child1_metrics[
                    searcher_id, good_id
                ]

        for region in range(self.h):
            visitors = np.flatnonzero(active_regions == region)
            if not len(visitors):
                continue
            for good_id in range(self.w):
                winner = int(
                    max(
                        visitors,
                        key=lambda index: child2_scores[int(index), good_id],
                    )
                )
                winner_score = float(child2_scores[winner, good_id])
                if (
                    winner_score
                    <= goods_fitness_before[region, good_id]
                    + self.IMPROVEMENT_TOLERANCE
                ):
                    continue
                self.goods[region][good_id] = child_rows[winner][
                    good_id * 2 + 1
                ].copy()
                self.goods_fitness[region, good_id] = winner_score
                self.goods_metrics[region, good_id] = child2_metrics[
                    winner, good_id
                ]

        self.selected_regions = selected
        if self.fitness > best_before + self.IMPROVEMENT_TOLERANCE:
            self.stagnation_rounds = 0
        else:
            self.stagnation_rounds += 1

        next_observations = self._build_observations()
        done = (
            self.evatime >= self.evaluation_limit
            or self.iterations_completed + 1 >= self.max_iterations
        )
        for searcher_id in range(self.n):
            self.agent.remember(
                observations[searcher_id],
                actions[searcher_id],
                rewards[searcher_id],
                next_observations[searcher_id],
                done,
                masks[searcher_id],
            )
        self.reward_history.append(float(np.mean(rewards)))
        for _ in range(self.gradient_steps):
            loss = self.agent.learn()
            if loss is not None:
                self.loss_history.append(float(loss))


__all__ = ["RL_SETSv4"]
