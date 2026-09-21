"""RL hyper-heuristic on the elitist SI-SETSv2 market update."""

from __future__ import annotations

import numpy as np

from Algorithm.se.RL_SETSv2 import RL_SETSv2
from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding
from environment_defaults import check_environment_contract


class RL_SETSv3(RL_SETSv2):
    """Choose ordinary or heavy-tailed mutation after pairwise crossover.

    Both actions retain the same crossover and mutation probability.  The
    market update matches SI-SETSv2: child1 competes with its searcher and the
    best visiting child2 competes with its good.  Neither role may deteriorate.
    """

    OBSERVATION_NAMES = RL_SETSv2.OBSERVATION_NAMES + (
        "global_remaining_energy_ratio",
        "sensor_remaining_energy_std",
        "low_energy_sensor_ratio",
        "vulnerable_target_energy_ratio",
    )
    OBSERVATION_SIZE = len(OBSERVATION_NAMES)
    ACTION_NAMES = (
        "crossover_and_local_mutation",
        "crossover_and_levy_mutation",
    )
    CHECKPOINT_SCHEMA = "rl_setsv5/1"
    REWARD_VERSION = "accepted_role_energy_improvement/1"

    LEVY_EXPONENT = 1.5
    LEVY_MAX_SENSOR_FRACTION = 0.2
    INFEASIBLE_PROPOSAL_PENALTY = 0.05

    def __init__(self, problem, *args, **kwargs):
        self._observation_problem = problem
        super().__init__(problem, *args, **kwargs)
        self.name = (
            f"RL_SETSv3_{self.n}_{self.h}_{self.w}_{self.mutation_rate}"
        )
        self.agent.checkpoint_metadata = self._checkpoint_metadata(problem)

    def initialize_market(self, problem, initial_state=None):
        self._observation_problem = problem
        super().initialize_market(problem, initial_state)

    def _check_checkpoint(self, problem):
        metadata = self.agent.checkpoint_metadata
        if metadata.get("schema") != self.CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint is not an RL_SETSv3 model")
        if tuple(metadata.get("observation_names", ())) != self.OBSERVATION_NAMES:
            raise ValueError("RL_SETSv3 checkpoint observation schema changed")
        if tuple(metadata.get("action_names", ())) != self.ACTION_NAMES:
            raise ValueError("RL_SETSv3 checkpoint action schema changed")
        if metadata.get("reward_version") != self.REWARD_VERSION:
            raise ValueError("RL_SETSv3 checkpoint reward version changed")
        check_environment_contract(metadata.get("environment"), problem)

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

    def _build_observations(self):
        local = np.zeros((self.n, 14), dtype=np.float32)
        progress = np.clip(
            self.evatime / max(1, self.evaluation_limit), 0.0, 1.0
        )
        stagnation = np.clip(
            self.stagnation_rounds / self.stagnation_window, 0.0, 1.0
        )
        for searcher_id, region in enumerate(self.selected_regions):
            region_metrics = self.goods_metrics[int(region)]
            searcher = self.searcher_metrics[searcher_id]
            local[searcher_id] = np.asarray(
                [
                    self._fitness_value(searcher[self._FITNESS]),
                    searcher[self._COVERAGE],
                    searcher[self._ENERGY_COST],
                    searcher[self._ACTIVE],
                    searcher[self._DISCONNECTED],
                    self._fitness_value(np.mean(region_metrics[:, self._FITNESS])),
                    self._fitness_value(np.max(region_metrics[:, self._FITNESS])),
                    self._fitness_spread(np.std(region_metrics[:, self._FITNESS])),
                    np.mean(region_metrics[:, self._COVERAGE]),
                    np.mean(region_metrics[:, self._ENERGY_COST]),
                    np.mean(region_metrics[:, self._ACTIVE]),
                    np.mean(region_metrics[:, self._DISCONNECTED]),
                    progress,
                    stagnation,
                ],
                dtype=np.float32,
            )
        global_features = self._environment_features(self._observation_problem)
        repeated = np.repeat(global_features[None, :], self.n, axis=0)
        observations = np.concatenate((local, repeated), axis=1)
        return np.nan_to_num(
            observations.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0
        )

    def _crossover(self, searcher, good):
        difference = 2
        midpoint = self.code_length // 2
        if midpoint <= difference or self.code_length - difference <= midpoint:
            return searcher.copy(), good.copy()
        left = self.random.randint(difference, midpoint - difference)
        right = self.random.randint(
            midpoint - difference,
            self.code_length - difference,
        )
        return swap_segment(searcher, good, left, right)

    def _make_children(self, problem, searcher, good, action):
        children = self._crossover(searcher, good)
        for child in children:
            if self.random.random() >= self.mutation_rate:
                continue
            if int(action) == 0:
                self.mutate_candidate(problem, child)
            elif int(action) == 1:
                self._levy_mutate(problem, child)
            else:
                raise ValueError(f"unknown RL-SETSv3 action: {action}")
        return children

    def _levy_sensor_count(self, sensor_count):
        """Sample a bounded discrete power-law step, usually small, rarely large."""
        maximum = max(
            1,
            min(
                int(sensor_count),
                int(np.ceil(sensor_count * self.LEVY_MAX_SENSOR_FRACTION)),
            ),
        )
        uniform = max(self.random.random(), np.finfo(float).tiny)
        count = int(np.floor(uniform ** (-1.0 / self.LEVY_EXPONENT)))
        return min(maximum, max(1, count))

    def _levy_mutate(self, problem, candidate):
        sensor_count = int(problem.SENSOR_NUMBER)
        count = self._levy_sensor_count(sensor_count)
        sensor_ids = self.random.sample(range(sensor_count), count)
        for sensor_id in sensor_ids:
            level_id = sensor_id * 2
            priority_id = level_id + 1
            option_count = int(problem.sensing_option_count(sensor_id))
            if option_count > 1:
                current = int(candidate.code[level_id]) % option_count
                candidate.code[level_id] = self._other_value(
                    current, option_count
                )
            current_priority = (
                int(candidate.code[priority_id]) % SensorEncoding.RANK_PRECISION
            )
            candidate.code[priority_id] = self._other_value(
                current_priority, SensorEncoding.RANK_PRECISION
            )
        return candidate

    def _accepted_gain(self, parent_metrics, child_metrics):
        """Score a transition that passed the corresponding elitist update."""
        parent_feasible = bool(parent_metrics[self._FEASIBLE] > 0.5)
        child_feasible = bool(child_metrics[self._FEASIBLE] > 0.5)
        if not parent_feasible and child_feasible:
            return 1.0
        if not parent_feasible:
            parent_violation = float(parent_metrics[self._CONSTRAINT_VIOLATION])
            child_violation = float(child_metrics[self._CONSTRAINT_VIOLATION])
            return float(
                np.clip(
                    (parent_violation - child_violation)
                    / max(parent_violation, self.REWARD_EPSILON),
                    0.0,
                    1.0,
                )
            )
        if not child_feasible:
            return 0.0

        parent_depletion = (
            0.3 * float(parent_metrics[self._GLOBAL_DEPLETION])
            + 0.7 * float(parent_metrics[self._WORST_TARGET_DEPLETION])
        )
        child_depletion = (
            0.3 * float(child_metrics[self._GLOBAL_DEPLETION])
            + 0.7 * float(child_metrics[self._WORST_TARGET_DEPLETION])
        )
        return float(
            np.clip(
                (parent_depletion - child_depletion)
                / max(parent_depletion, self.REWARD_EPSILON),
                0.0,
                1.0,
            )
        )

    def _constraint_damage(self, parent_metrics, child_metrics):
        """Measure only newly introduced or worsened constraint violations."""
        parent_feasible = bool(parent_metrics[self._FEASIBLE] > 0.5)
        child_feasible = bool(child_metrics[self._FEASIBLE] > 0.5)
        if parent_feasible:
            return float(not child_feasible)
        if child_feasible:
            return 0.0
        parent_violation = float(parent_metrics[self._CONSTRAINT_VIOLATION])
        child_violation = float(child_metrics[self._CONSTRAINT_VIOLATION])
        return float(
            np.clip(
                (child_violation - parent_violation)
                / max(parent_violation, self.REWARD_EPSILON),
                0.0,
                1.0,
            )
        )

    def vision_search(self, problem):
        """Apply one operator per searcher and credit only accepted offspring."""
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
        actions = self.agent.select_actions(observations, masks)

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

        child_scores, child_metrics = self._evaluate_child_rows(problem, child_rows)
        child1_scores = child_scores[:, 0::2]
        child2_scores = child_scores[:, 1::2]
        child1_metrics = child_metrics[:, 0::2]
        child2_metrics = child_metrics[:, 1::2]

        for searcher_id, region in enumerate(active_regions):
            self.investment_quality[int(region), searcher_id] = float(
                np.mean(child1_scores[searcher_id])
            )
        probabilities = self.region_probabilities(
            goods_fitness_before, self.investment_quality
        )
        selected = self.select_regions(probabilities)

        searcher_gains = np.zeros(self.n, dtype=float)
        good_gains = np.zeros((self.n, self.w), dtype=float)
        successful_proposals = np.zeros(self.n, dtype=int)

        for searcher_id in range(self.n):
            good_id = int(np.argmax(child1_scores[searcher_id]))
            score = float(child1_scores[searcher_id, good_id])
            if (
                score
                <= searcher_fitness_before[searcher_id]
                + self.IMPROVEMENT_TOLERANCE
            ):
                continue
            self.searchers[searcher_id] = child_rows[searcher_id][
                good_id * 2
            ].copy()
            self.searcher_fitness[searcher_id] = score
            self.searcher_metrics[searcher_id] = child1_metrics[
                searcher_id, good_id
            ]
            searcher_gains[searcher_id] = self._accepted_gain(
                searcher_metrics_before[searcher_id],
                child1_metrics[searcher_id, good_id],
            )
            successful_proposals[searcher_id] += 1

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
                good_gains[winner, good_id] = self._accepted_gain(
                    goods_metrics_before[region, good_id],
                    child2_metrics[winner, good_id],
                )
                successful_proposals[winner] += 1

        rewards = np.zeros(self.n, dtype=float)
        for searcher_id in range(self.n):
            goods_gain = float(np.mean(good_gains[searcher_id]))
            reward = 0.5 * searcher_gains[searcher_id] + 0.5 * goods_gain
            damages = []
            region = int(active_regions[searcher_id])
            for good_id in range(self.w):
                damages.append(
                    self._constraint_damage(
                        searcher_metrics_before[searcher_id],
                        child1_metrics[searcher_id, good_id],
                    )
                )
                damages.append(
                    self._constraint_damage(
                        goods_metrics_before[region, good_id],
                        child2_metrics[searcher_id, good_id],
                    )
                )
            reward -= self.INFEASIBLE_PROPOSAL_PENALTY * float(
                np.mean(damages)
            )
            rewards[searcher_id] = float(np.clip(reward, -0.05, 1.0))

            action = int(actions[searcher_id])
            self.action_uses[action] += self.w + 1
            self.action_successes[action] += successful_proposals[searcher_id]
            self.reward_records.append(
                {
                    "action": action,
                    "progress": progress,
                    "reward": rewards[searcher_id],
                    "improved_goods": int(successful_proposals[searcher_id]),
                    "degraded_goods": 0,
                }
            )

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


__all__ = ["RL_SETSv3"]
