"""RL-SETSv5: factorized-variation D3QN hyper-heuristic on SI-SETSv2.

Formerly ``RL_SETSv61``. The original identity-crossover design (plain
``RL_SETS``) and the independent-replay ``RL_SETSv2`` were removed outright
(nothing inherited from them), so the surviving lineage -- what used to be
v3/v4/v5/v6/v6.1 -- was renumbered down by two to close the gap:
old v3 -> new RL_SETS, old v4 -> new RL_SETSv2, old v5 -> new RL_SETSv3,
old v6 -> new RL_SETSv4, old v6.1 -> new RL_SETSv5 (this class). The
inheritance chain (RL_SETSv5 -> RL_SETSv4 -> ... -> RL_SETS -> SI_SETS) is
unchanged, only every class's exposed name shifted.
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.RL_SETSv4 import RL_SETSv4
from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding
from environment_defaults import check_environment_contract


class RL_SETSv5(RL_SETSv4):
    """Choose crossover and mutation strength independently with D3QN.

    The centre action is the original SA-SETS perturbation.  Rewards follow
    the actual role-separated elitist update: accepted proposals receive full
    fitness/lifetime credit, ordinary rejected proposals receive only a small
    negative diagnostic credit, and destroyed feasibility remains strongly
    penalised.
    """

    OBSERVATION_NAMES = (
        "searcher_feasible",
        "searcher_constraint_violation",
        "searcher_fitness",
        "searcher_predicted_lifetime",
        "searcher_global_depletion",
        "searcher_worst_target_depletion",
        "searcher_coverage_ratio",
        "searcher_active_sensor_ratio",
        "searcher_disconnected_ratio",
        "region_goods_fitness_mean",
        "region_goods_fitness_best",
        "region_goods_fitness_std",
        "region_goods_predicted_lifetime_mean",
        "region_goods_predicted_lifetime_best",
        "region_goods_feasible_ratio",
        "evaluation_progress",
        "stagnation_ratio",
        "global_remaining_energy_ratio",
        "sensor_remaining_energy_std",
        "minimum_remaining_energy_ratio",
        "low_energy_sensor_ratio",
        "vulnerable_target_energy_ratio",
    )
    OBSERVATION_SIZE = len(OBSERVATION_NAMES)

    CROSSOVER_NAMES = ("local", "sa", "global")
    MUTATION_NAMES = ("local", "sa", "heavy")
    ACTION_NAMES = (
        "local_crossover_local_mutation",
        "local_crossover_sa_mutation",
        "local_crossover_heavy_mutation",
        "sa_crossover_local_mutation",
        "sa_crossover_sa_mutation",
        "sa_crossover_heavy_mutation",
        "global_crossover_local_mutation",
        "global_crossover_sa_mutation",
        "global_crossover_heavy_mutation",
    )
    SA_BASELINE_ACTION = 4
    # Kept as the original schema string so checkpoints trained before this
    # rename still load and validate correctly.
    CHECKPOINT_SCHEMA = "rl_setsv61/1"
    REWARD_VERSION = "accepted_fitness_lifetime/1"

    LOCAL_CROSSOVER_FRACTION = (0.05, 0.10)
    GLOBAL_CROSSOVER_FRACTION = (0.45, 0.60)
    HEAVY_MUTATION_FRACTION = (0.05, 0.08)
    REJECTED_REWARD_SCALE = 0.1
    FITNESS_WEIGHT = 0.7
    LIFETIME_WEIGHT = 0.3
    REWARD_SENSITIVITY = 4.0

    def __init__(
        self,
        problem,
        *args,
        lifetime_normalization=5000.0,
        fixed_action=None,
        random_policy=False,
        **kwargs,
    ):
        self.lifetime_normalization = max(1.0, float(lifetime_normalization))
        self.fixed_action = None if fixed_action is None else int(fixed_action)
        if self.fixed_action is not None and not (
            0 <= self.fixed_action < len(self.ACTION_NAMES)
        ):
            raise ValueError("fixed_action must be between 0 and 8")
        super().__init__(
            problem,
            *args,
            random_policy=random_policy,
            **kwargs,
        )
        self.name = (
            f"RL_SETSv5_{self.n}_{self.h}_{self.w}_{self.mutation_rate}"
        )
        self.agent.checkpoint_metadata = self._checkpoint_metadata(problem)

    def _checkpoint_metadata(self, problem):
        metadata = super()._checkpoint_metadata(problem)
        metadata["lifetime_normalization"] = self.lifetime_normalization
        return metadata

    def _check_checkpoint(self, problem):
        metadata = self.agent.checkpoint_metadata
        if metadata.get("schema") != self.CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint is not an RL_SETSv5 model")
        if tuple(metadata.get("observation_names", ())) != self.OBSERVATION_NAMES:
            raise ValueError("RL_SETSv5 checkpoint observation schema changed")
        if tuple(metadata.get("action_names", ())) != self.ACTION_NAMES:
            raise ValueError("RL_SETSv5 checkpoint action schema changed")
        if metadata.get("reward_version") != self.REWARD_VERSION:
            raise ValueError("RL_SETSv5 checkpoint reward version changed")
        saved_normalization = float(
            metadata.get("lifetime_normalization", self.lifetime_normalization)
        )
        if not np.isclose(saved_normalization, self.lifetime_normalization):
            raise ValueError("RL_SETSv5 lifetime normalization changed")
        check_environment_contract(metadata.get("environment"), problem)

    def _lifetime_value(self, value):
        value = max(0.0, float(value))
        return float(
            np.clip(
                np.log1p(value) / np.log1p(self.lifetime_normalization),
                0.0,
                1.0,
            )
        )

    def _environment_features(self, problem):
        """Return five energy summaries, including the missing minimum."""
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
                float(np.mean(sensor_ratios)) if len(sensor_ratios) else 0.0,
                float(np.clip(np.std(sensor_ratios), 0.0, 1.0)),
                float(np.min(sensor_ratios)) if len(sensor_ratios) else 0.0,
                float(np.mean(sensor_ratios < 0.2)) if len(sensor_ratios) else 0.0,
                vulnerable_target,
            ],
            dtype=np.float32,
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
            violation = float(searcher[self._CONSTRAINT_VIOLATION])
            observations[searcher_id] = np.asarray(
                [
                    searcher[self._FEASIBLE],
                    violation / (1.0 + max(0.0, violation)),
                    self._fitness_value(searcher[self._FITNESS]),
                    self._lifetime_value(searcher[self._PREDICTED_LIFETIME]),
                    searcher[self._GLOBAL_DEPLETION],
                    searcher[self._WORST_TARGET_DEPLETION],
                    searcher[self._COVERAGE],
                    searcher[self._ACTIVE],
                    searcher[self._DISCONNECTED],
                    self._fitness_value(np.mean(region_metrics[:, self._FITNESS])),
                    self._fitness_value(np.max(region_metrics[:, self._FITNESS])),
                    self._fitness_spread(np.std(region_metrics[:, self._FITNESS])),
                    self._lifetime_value(
                        np.mean(region_metrics[:, self._PREDICTED_LIFETIME])
                    ),
                    self._lifetime_value(
                        np.max(region_metrics[:, self._PREDICTED_LIFETIME])
                    ),
                    float(np.mean(region_metrics[:, self._FEASIBLE])),
                    progress,
                    stagnation,
                    *environment,
                ],
                dtype=np.float32,
            )
        return np.clip(
            np.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=0.0),
            0.0,
            1.0,
        )

    def _pair_aligned_crossover(self, searcher, good, bounds):
        sensor_count = self.code_length // 2
        count = self._sensor_count_for_fraction(sensor_count, bounds)
        start = self.random.randrange(sensor_count - count + 1)
        return swap_segment(searcher, good, start * 2, (start + count) * 2)

    def _sa_crossover(self, searcher, good):
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

    def _crossover_for_mode(self, searcher, good, mode):
        if mode == 0:
            return self._pair_aligned_crossover(
                searcher, good, self.LOCAL_CROSSOVER_FRACTION
            )
        if mode == 1:
            return self._sa_crossover(searcher, good)
        if mode == 2:
            return self._pair_aligned_crossover(
                searcher, good, self.GLOBAL_CROSSOVER_FRACTION
            )
        raise ValueError(f"unknown crossover mode: {mode}")

    def _local_mutation(self, problem, candidate):
        sensor_id = self.random.randrange(int(problem.SENSOR_NUMBER))
        level_id = sensor_id * 2
        priority_id = level_id + 1
        option_count = int(problem.sensing_option_count(sensor_id))
        mutate_level = self.random.random() < 0.5 and option_count > 1
        if mutate_level:
            current = int(candidate.code[level_id]) % option_count
            candidate.code[level_id] = self._adjacent_value(current, option_count)
        else:
            current = int(candidate.code[priority_id]) % SensorEncoding.RANK_PRECISION
            candidate.code[priority_id] = self._adjacent_value(
                current, SensorEncoding.RANK_PRECISION
            )
        return candidate

    def _heavy_mutation(self, problem, candidate):
        sensor_count = int(problem.SENSOR_NUMBER)
        count = self._sensor_count_for_fraction(
            sensor_count, self.HEAVY_MUTATION_FRACTION
        )
        for sensor_id in self.random.sample(range(sensor_count), count):
            level_id = sensor_id * 2
            priority_id = level_id + 1
            option_count = int(problem.sensing_option_count(sensor_id))
            if option_count > 1:
                current = int(candidate.code[level_id]) % option_count
                candidate.code[level_id] = self._other_value(current, option_count)
            current = int(candidate.code[priority_id]) % SensorEncoding.RANK_PRECISION
            candidate.code[priority_id] = self._other_value(
                current, SensorEncoding.RANK_PRECISION
            )
        return candidate

    def _mutate_for_mode(self, problem, candidate, mode):
        if mode == 0:
            return self._local_mutation(problem, candidate)
        if mode == 1:
            if self.random.random() < self.mutation_rate:
                self.mutate_candidate(problem, candidate)
            return candidate
        if mode == 2:
            return self._heavy_mutation(problem, candidate)
        raise ValueError(f"unknown mutation mode: {mode}")

    def _make_children(self, problem, searcher, good, action):
        action = int(action)
        if not 0 <= action < len(self.ACTION_NAMES):
            raise ValueError(f"unknown RL-SETS action: {action}")
        crossover_mode, mutation_mode = divmod(action, 3)
        children = self._crossover_for_mode(searcher, good, crossover_mode)
        for child in children:
            self._mutate_for_mode(problem, child, mutation_mode)
        return children

    def _transition_components(self, parent_metrics, child_metrics):
        """Return bounded total, constraint, fitness and lifetime signals."""
        parent_feasible = bool(parent_metrics[self._FEASIBLE] > 0.5)
        child_feasible = bool(child_metrics[self._FEASIBLE] > 0.5)
        if not parent_feasible and child_feasible:
            return 1.0, 1.0, 0.0, 0.0
        if parent_feasible and not child_feasible:
            return -1.0, -1.0, 0.0, 0.0
        if not parent_feasible:
            parent_violation = float(parent_metrics[self._CONSTRAINT_VIOLATION])
            child_violation = float(child_metrics[self._CONSTRAINT_VIOLATION])
            constraint = float(
                np.clip(
                    (parent_violation - child_violation)
                    / max(parent_violation, self.REWARD_EPSILON),
                    -1.0,
                    1.0,
                )
            )
            return constraint, constraint, 0.0, 0.0

        parent_fitness = float(parent_metrics[self._FITNESS])
        child_fitness = float(child_metrics[self._FITNESS])
        fitness_gain = float(
            np.clip(
                (child_fitness - parent_fitness)
                / max(1.0 - parent_fitness, 1e-3),
                -1.0,
                1.0,
            )
        )
        parent_lifetime = float(parent_metrics[self._PREDICTED_LIFETIME])
        child_lifetime = float(child_metrics[self._PREDICTED_LIFETIME])
        lifetime_gain = float(
            np.clip(
                (child_lifetime - parent_lifetime) / max(parent_lifetime, 1.0),
                -1.0,
                1.0,
            )
        )
        combined = (
            self.FITNESS_WEIGHT * fitness_gain
            + self.LIFETIME_WEIGHT * lifetime_gain
        )
        total = float(np.tanh(self.REWARD_SENSITIVITY * combined))
        return total, 0.0, fitness_gain, lifetime_gain

    def _rejected_credit(self, components):
        total, constraint, _, _ = components
        if constraint < -self.IMPROVEMENT_TOLERANCE:
            return float(total)
        return self.REJECTED_REWARD_SCALE * min(0.0, float(total))

    def _select_actions(self, observations, masks):
        if self.fixed_action is not None:
            return np.full(len(observations), self.fixed_action, dtype=int)
        return super()._select_actions(observations, masks)

    def policy_statistics(self):
        statistics = super().policy_statistics()
        statistics["fixed_action"] = self.fixed_action
        component_names = (
            "accepted_reward",
            "rejected_penalty",
            "constraint_signal",
            "fitness_signal",
            "lifetime_signal",
        )
        statistics["reward_components"] = {
            name: (
                float(np.mean([row[name] for row in self.reward_records]))
                if self.reward_records
                else 0.0
            )
            for name in component_names
        }
        return statistics

    def vision_search(self, problem):
        """Evaluate nine factorized operators and credit accepted role updates."""
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

        child_scores, child_metrics = self._evaluate_child_rows(problem, child_rows)
        child1_scores = child_scores[:, 0::2]
        child2_scores = child_scores[:, 1::2]
        child1_metrics = child_metrics[:, 0::2]
        child2_metrics = child_metrics[:, 1::2]

        first_components = [[None] * self.w for _ in range(self.n)]
        second_components = [[None] * self.w for _ in range(self.n)]
        for searcher_id, region in enumerate(active_regions):
            region = int(region)
            self.investment_quality[region, searcher_id] = float(
                np.mean(child1_scores[searcher_id])
            )
            for good_id in range(self.w):
                first_components[searcher_id][good_id] = self._transition_components(
                    searcher_metrics_before[searcher_id],
                    child1_metrics[searcher_id, good_id],
                )
                second_components[searcher_id][good_id] = self._transition_components(
                    goods_metrics_before[region, good_id],
                    child2_metrics[searcher_id, good_id],
                )

        probabilities = self.region_probabilities(
            goods_fitness_before, self.investment_quality
        )
        selected = self.select_regions(probabilities)

        searcher_credits = np.asarray(
            [
                np.mean([self._rejected_credit(row) for row in component_row])
                for component_row in first_components
            ],
            dtype=float,
        )
        good_credits = np.asarray(
            [
                [self._rejected_credit(value) for value in component_row]
                for component_row in second_components
            ],
            dtype=float,
        )
        accepted_counts = np.zeros(self.n, dtype=int)
        accepted_searcher = np.zeros(self.n, dtype=float)
        accepted_goods = np.zeros((self.n, self.w), dtype=float)

        for searcher_id in range(self.n):
            good_id = int(np.argmax(child1_scores[searcher_id]))
            score = float(child1_scores[searcher_id, good_id])
            if score <= searcher_fitness_before[searcher_id] + self.IMPROVEMENT_TOLERANCE:
                continue
            self.searchers[searcher_id] = child_rows[searcher_id][good_id * 2].copy()
            self.searcher_fitness[searcher_id] = score
            self.searcher_metrics[searcher_id] = child1_metrics[searcher_id, good_id]
            credit = float(first_components[searcher_id][good_id][0])
            searcher_credits[searcher_id] = credit
            accepted_searcher[searcher_id] = credit
            accepted_counts[searcher_id] += 1

        for region in range(self.h):
            visitors = np.flatnonzero(active_regions == region)
            if not len(visitors):
                continue
            for good_id in range(self.w):
                winner = int(
                    max(visitors, key=lambda index: child2_scores[int(index), good_id])
                )
                winner_score = float(child2_scores[winner, good_id])
                if winner_score <= goods_fitness_before[region, good_id] + self.IMPROVEMENT_TOLERANCE:
                    continue
                self.goods[region][good_id] = child_rows[winner][good_id * 2 + 1].copy()
                self.goods_fitness[region, good_id] = winner_score
                self.goods_metrics[region, good_id] = child2_metrics[winner, good_id]
                credit = float(second_components[winner][good_id][0])
                good_credits[winner, good_id] = credit
                accepted_goods[winner, good_id] = credit
                accepted_counts[winner] += 1

        rewards = np.clip(
            0.5 * searcher_credits + 0.5 * np.mean(good_credits, axis=1),
            -1.0,
            1.0,
        )
        for searcher_id in range(self.n):
            raw_components = (
                first_components[searcher_id] + second_components[searcher_id]
            )
            accepted_reward = 0.5 * accepted_searcher[searcher_id] + 0.5 * float(
                np.mean(accepted_goods[searcher_id])
            )
            rejected_penalty = float(rewards[searcher_id]) - accepted_reward
            action = int(actions[searcher_id])
            self.action_uses[action] += self.w + 1
            self.action_successes[action] += accepted_counts[searcher_id]
            self.reward_records.append(
                {
                    "action": action,
                    "progress": progress,
                    "reward": float(rewards[searcher_id]),
                    "improved_goods": int(accepted_counts[searcher_id]),
                    "degraded_goods": int(
                        sum(value[0] < -self.IMPROVEMENT_TOLERANCE for value in raw_components)
                    ),
                    "accepted_reward": float(accepted_reward),
                    "rejected_penalty": rejected_penalty,
                    "constraint_signal": float(np.mean([value[1] for value in raw_components])),
                    "fitness_signal": float(np.mean([value[2] for value in raw_components])),
                    "lifetime_signal": float(np.mean([value[3] for value in raw_components])),
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


__all__ = ["RL_SETSv5"]
