"""Role-separated selective-investment SETS with energy-based D3QN rewards."""

from __future__ import annotations

import numpy as np

from Algorithm.se.RL_SETS import RL_SETS


class RL_SETSv2(RL_SETS):
    """RL-SETSv2 with explicit investor and good offspring roles.

    Child 1 keeps the searcher as its base and competes only for that
    searcher. Child 2 keeps the good as its base and competes only for that
    good. Both children are evaluated, so every action has the same cost.
    """

    CHECKPOINT_SCHEMA = "rl_setsv4/1"
    REWARD_VERSION = "constraint_first_energy_improvement/1"
    ENERGY_REWARD_LIMIT = 0.5
    CONSTRAINT_REWARD = 1.0
    REWARD_EPSILON = 1e-15

    _GLOBAL_DEPLETION = 5
    _WORST_TARGET_DEPLETION = 6
    _CONSTRAINT_VIOLATION = 7
    _FEASIBLE = 8

    def __init__(self, problem, *args, **kwargs):
        super().__init__(problem, *args, **kwargs)
        self.name = (
            f"RL_SETSv2_{self.n}_{self.h}_{self.w}_{self.mutation_rate}"
        )
        # RL_SETS creates the agent before this subclass regains control.
        # Replace its metadata with the v4 schema before saving a checkpoint.
        self.agent.checkpoint_metadata = self._checkpoint_metadata(problem)

    def _check_checkpoint(self, problem):
        metadata = self.agent.checkpoint_metadata
        if metadata.get("schema") != self.CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint is not an RL_SETSv2 model")
        if tuple(metadata.get("observation_names", ())) != self.OBSERVATION_NAMES:
            raise ValueError("RL_SETSv2 checkpoint observation schema changed")
        if tuple(metadata.get("action_names", ())) != self.ACTION_NAMES:
            raise ValueError("RL_SETSv2 checkpoint action schema changed")
        if metadata.get("reward_version") != self.REWARD_VERSION:
            raise ValueError("RL_SETSv2 checkpoint reward version changed")
        from Algorithm.se.RL_SETS import check_environment_contract

        check_environment_contract(metadata.get("environment"), problem)

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

    def _transition_reward(self, parent_metrics, child_metrics):
        """Reward one role-specific parent-to-child transition."""
        parent_feasible = bool(parent_metrics[self._FEASIBLE] > 0.5)
        child_feasible = bool(child_metrics[self._FEASIBLE] > 0.5)
        if parent_feasible and not child_feasible:
            return -self.CONSTRAINT_REWARD
        if not parent_feasible and child_feasible:
            return self.CONSTRAINT_REWARD

        if not parent_feasible:
            parent_violation = float(
                parent_metrics[self._CONSTRAINT_VIOLATION]
            )
            child_violation = float(
                child_metrics[self._CONSTRAINT_VIOLATION]
            )
            improvement = (parent_violation - child_violation) / max(
                parent_violation, self.REWARD_EPSILON
            )
            return float(
                np.clip(
                    improvement,
                    -self.ENERGY_REWARD_LIMIT,
                    self.ENERGY_REWARD_LIMIT,
                )
            )

        global_parent = float(parent_metrics[self._GLOBAL_DEPLETION])
        global_child = float(child_metrics[self._GLOBAL_DEPLETION])
        target_parent = float(parent_metrics[self._WORST_TARGET_DEPLETION])
        target_child = float(child_metrics[self._WORST_TARGET_DEPLETION])
        global_improvement = (global_parent - global_child) / max(
            global_parent, self.REWARD_EPSILON
        )
        target_improvement = (target_parent - target_child) / max(
            target_parent, self.REWARD_EPSILON
        )
        reward = 0.5 * global_improvement + 0.5 * target_improvement
        return float(
            np.clip(
                reward,
                -self.ENERGY_REWARD_LIMIT,
                self.ENERGY_REWARD_LIMIT,
            )
        )

    def vision_search(self, problem):
        """Apply one action per searcher and update the two offspring roles."""
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
                # child1 is searcher-based; child2 is good-based.
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
        rewards = np.empty(self.n, dtype=float)

        for searcher_id, region in enumerate(active_regions):
            region = int(region)
            pair_rewards = []
            improved = 0
            degraded = 0
            for good_id in range(self.w):
                reward1 = self._transition_reward(
                    searcher_metrics_before[searcher_id],
                    child1_metrics[searcher_id, good_id],
                )
                reward2 = self._transition_reward(
                    goods_metrics_before[region, good_id],
                    child2_metrics[searcher_id, good_id],
                )
                pair_reward = 0.5 * (reward1 + reward2)
                pair_rewards.append(pair_reward)
                improved += int(pair_reward > self.IMPROVEMENT_TOLERANCE)
                degraded += int(pair_reward < -self.IMPROVEMENT_TOLERANCE)

            reward = float(np.mean(pair_rewards))
            rewards[searcher_id] = reward
            action = int(actions[searcher_id])
            self.action_uses[action] += self.w
            self.action_successes[action] += improved
            self.reward_records.append(
                {
                    "action": action,
                    "progress": progress,
                    "reward": reward,
                    "improved_goods": improved,
                    "degraded_goods": degraded,
                }
            )
            # Region choice concerns where this searcher can improve itself.
            self.investment_quality[region, searcher_id] = float(
                np.mean(child1_scores[searcher_id])
            )

        probabilities = self.region_probabilities(
            goods_fitness_before, self.investment_quality
        )
        selected = self.select_regions(probabilities)

        # Each searcher keeps the best child1 it produced this round, but only
        # when that child improves the searcher itself.
        for searcher_id in range(self.n):
            good_id = int(np.argmax(child1_scores[searcher_id]))
            score = float(child1_scores[searcher_id, good_id])
            if score > searcher_fitness_before[searcher_id]:
                self.searchers[searcher_id] = child_rows[searcher_id][
                    good_id * 2
                ].copy()
                self.searcher_fitness[searcher_id] = score
                self.searcher_metrics[searcher_id] = child1_metrics[
                    searcher_id, good_id
                ]

        # A visited good is always replaced by the best child2 proposed for
        # it this round, matching SA-SETS' non-elitist goods update.
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
                self.goods[region][good_id] = child_rows[winner][
                    good_id * 2 + 1
                ].copy()
                self.goods_fitness[region, good_id] = child2_scores[
                    winner, good_id
                ]
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


__all__ = ["RL_SETSv2"]
