"""Selective-investment SETS with one D3QN operator decision per searcher."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from Algorithm.se.SI_SETS import SI_SETS
from Algorithm.se.d3qn import D3QNAgent
from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding
from environment_defaults import environment_contract, check_environment_contract


class RL_SETS(SI_SETS):
    """Let each SI-SETS searcher choose an operator for its selected region.

    Region selection, regional goods, initialization and market memory are the
    SI-SETS rules.  D3QN receives one fixed-size observation per searcher and
    applies the selected operator to every good in that searcher's region.
    Every searcher/good pair creates and evaluates two children so all four
    actions consume the same fitness budget.
    """

    OBSERVATION_NAMES = (
        "searcher_fitness",
        "searcher_coverage_ratio",
        "searcher_energy_cost_ratio",
        "searcher_active_sensor_ratio",
        "searcher_disconnected_ratio",
        "region_goods_fitness_mean",
        "region_goods_fitness_best",
        "region_goods_fitness_std",
        "region_goods_coverage_mean",
        "region_goods_energy_cost_mean",
        "region_goods_active_sensor_mean",
        "region_goods_disconnected_mean",
        "evaluation_progress",
        "stagnation_ratio",
    )
    OBSERVATION_SIZE = len(OBSERVATION_NAMES)
    ACTION_NAMES = (
        "segment_exchange",
        "level_and_priority_mutation",
        "level_mutation",
        "priority_mutation",
    )
    CHECKPOINT_SCHEMA = "rl_setsv3/1"
    REWARD_VERSION = "best_of_two_parent_improvement/1"
    IMPROVEMENT_TOLERANCE = 1e-12

    # Candidate metric columns. Fitness is kept in its original scale and is
    # transformed only when it enters the observation.
    _FITNESS = 0
    _COVERAGE = 1
    _ENERGY_COST = 2
    _ACTIVE = 3
    _DISCONNECTED = 4

    def __init__(
        self,
        problem,
        n=8,
        h=4,
        w=2,
        mu=0.4,
        *,
        training=False,
        model_path=None,
        hidden_size=64,
        replay_capacity=20000,
        replay_group_capacity=None,
        replay_warmup=256,
        batch_size=64,
        gamma=0.95,
        learning_rate=3e-4,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9995,
        target_update_interval=500,
        gradient_steps=1,
        stagnation_window=50,
        seed=None,
    ):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        self.name = f"RL_SETS_{self.n}_{self.h}_{self.w}_{mu}"
        self.training = bool(training)
        self.model_path = None if model_path is None else Path(model_path)
        self.gradient_steps = max(1, int(gradient_steps))
        self.stagnation_window = max(1, int(stagnation_window))
        self.agent = D3QNAgent(
            self.OBSERVATION_SIZE,
            len(self.ACTION_NAMES),
            hidden_size=hidden_size,
            replay_capacity=replay_capacity,
            replay_group_capacity=replay_group_capacity,
            replay_warmup=replay_warmup,
            batch_size=batch_size,
            gamma=gamma,
            learning_rate=learning_rate,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update_interval=target_update_interval,
            training=self.training,
            seed=self.seed ^ 0xD3A10003,
        )
        self.agent.checkpoint_metadata = self._checkpoint_metadata(problem)
        if self.model_path is not None and self.model_path.is_file():
            self.agent.load(self.model_path)
            self._check_checkpoint(problem)
        elif self.model_path is not None and not self.training:
            raise FileNotFoundError(
                f"RL-SETS model does not exist: {self.model_path}"
            )

        self.searcher_metrics = np.zeros((self.n, 5), dtype=float)
        self.goods_metrics = np.zeros((self.h, self.w, 5), dtype=float)
        self.stagnation_rounds = 0
        self.reward_history = []
        self.reward_records = []
        self.loss_history = []
        self.action_uses = np.zeros(len(self.ACTION_NAMES), dtype=int)
        self.action_successes = np.zeros(len(self.ACTION_NAMES), dtype=int)

    def _checkpoint_metadata(self, problem):
        return {
            "schema": self.CHECKPOINT_SCHEMA,
            "reward_version": self.REWARD_VERSION,
            "observation_names": list(self.OBSERVATION_NAMES),
            "action_names": list(self.ACTION_NAMES),
            "environment": environment_contract(problem),
        }

    def _check_checkpoint(self, problem):
        metadata = self.agent.checkpoint_metadata
        if metadata.get("schema") != self.CHECKPOINT_SCHEMA:
            raise ValueError("checkpoint is not an RL_SETS model")
        if tuple(metadata.get("observation_names", ())) != self.OBSERVATION_NAMES:
            raise ValueError("RL_SETS checkpoint observation schema changed")
        if tuple(metadata.get("action_names", ())) != self.ACTION_NAMES:
            raise ValueError("RL_SETS checkpoint action schema changed")
        if metadata.get("reward_version") != self.REWARD_VERSION:
            raise ValueError("RL_SETS checkpoint reward version changed")
        check_environment_contract(metadata.get("environment"), problem)

    def initialize_market(self, problem, initial_state=None):
        """Initialize SI-SETS, then cache the 14-input candidate summaries."""
        self.stagnation_rounds = 0
        self.reward_history = []
        self.reward_records = []
        self.loss_history = []
        self.action_uses.fill(0)
        self.action_successes.fill(0)
        super().initialize_market(problem, initial_state)

        self.searcher_metrics = np.asarray(
            [
                self._summarize_candidate(
                    problem, candidate, self.searcher_fitness[index]
                )
                for index, candidate in enumerate(self.searchers)
            ],
            dtype=float,
        )
        self.goods_metrics = np.asarray(
            [
                [
                    self._summarize_candidate(
                        problem, candidate, self.goods_fitness[region, good_id]
                    )
                    for good_id, candidate in enumerate(region_goods)
                ]
                for region, region_goods in enumerate(self.goods)
            ],
            dtype=float,
        )

    def _summarize_candidate(self, problem, candidate, fitness):
        state = candidate.decode(problem)
        return self._summarize_state(problem, state, fitness)

    def _summarize_state(self, problem, state, fitness):
        sensor_count = max(1, int(problem.SENSOR_NUMBER))
        target_count = max(1, int(problem.TARGET_NUMBER))
        active_count = int(np.count_nonzero(np.asarray(state.levels) > 0))
        uncovered = len(problem.find_uncovered_targets(state.levels))
        disconnected = len(problem.find_disconnected_sensors(state))
        cost = np.asarray(problem.calculate_total_cost(state), dtype=float)
        available_energy = float(
            np.sum(np.maximum(np.asarray(problem.energy, dtype=float), 0.0))
        )
        energy_cost_ratio = float(
            np.sum(np.maximum(cost, 0.0)) / max(available_energy, 1e-15)
        )
        return np.asarray(
            [
                float(fitness),
                1.0 - uncovered / target_count,
                np.clip(energy_cost_ratio, 0.0, 1.0),
                active_count / sensor_count,
                disconnected / max(1, active_count),
            ],
            dtype=float,
        )

    @staticmethod
    def _fitness_value(value):
        """Map V5's useful [-3, 1] range to [0, 1]."""
        return float(np.clip((float(value) + 3.0) / 4.0, 0.0, 1.0))

    @staticmethod
    def _fitness_spread(value):
        return float(np.clip(float(value) / 4.0, 0.0, 1.0))

    def _build_observations(self):
        observations = np.zeros((self.n, self.OBSERVATION_SIZE), dtype=np.float32)
        progress = np.clip(
            self.evatime / max(1, self.evaluation_limit), 0.0, 1.0
        )
        stagnation = np.clip(
            self.stagnation_rounds / self.stagnation_window, 0.0, 1.0
        )
        for searcher_id, region in enumerate(self.selected_regions):
            region_metrics = self.goods_metrics[int(region)]
            searcher = self.searcher_metrics[searcher_id]
            observations[searcher_id] = np.asarray(
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
        return np.nan_to_num(observations, nan=0.0, posinf=1.0, neginf=0.0)

    def _make_children(self, problem, searcher, good, action):
        if int(action) == 0:
            difference = 2
            midpoint = self.code_length // 2
            if (
                midpoint <= difference
                or self.code_length - difference <= midpoint
            ):
                children = (searcher.copy(), good.copy())
            else:
                left = self.random.randint(difference, midpoint - difference)
                right = self.random.randint(
                    midpoint - difference, self.code_length - difference
                )
                children = swap_segment(searcher, good, left, right)
        else:
            children = (searcher.copy(), good.copy())
            for child in children:
                self._mutate_for_action(problem, child, int(action))
        return children

    def _mutate_for_action(self, problem, candidate, action):
        mutation_count = self.random.choice([1] * 90 + [2] * 5 + [3] * 5)
        for _ in range(mutation_count):
            sensor_id = self.random.randrange(problem.SENSOR_NUMBER)
            level_id = sensor_id * 2
            priority_id = level_id + 1
            if action in (1, 2):
                option_count = problem.sensing_option_count(sensor_id)
                if option_count > 1:
                    current = int(candidate.code[level_id]) % option_count
                    candidate.code[level_id] = self._other_value(
                        current, option_count
                    )
            if action in (1, 3):
                current = (
                    int(candidate.code[priority_id])
                    % SensorEncoding.RANK_PRECISION
                )
                candidate.code[priority_id] = self._other_value(
                    current, SensorEncoding.RANK_PRECISION
                )
        return candidate

    def _evaluate_child_rows(self, problem, rows):
        score_rows = []
        metric_rows = []
        results_per_row = self._route_to_pool(rows)
        for row_id, results in enumerate(results_per_row):
            scores = []
            metrics = []
            for child_id, (candidate, state, objectives, target_red) in enumerate(results):
                rows[row_id][child_id] = candidate
                fitness = float(np.sum(objectives))
                self.evatime += 1
                self.update_best(
                    problem, state, objectives, fitness, candidate=candidate
                )
                problem.target_red = target_red
                scores.append(fitness)
                metrics.append(self._summarize_state(problem, state, fitness))
            score_rows.append(scores)
            metric_rows.append(metrics)
        return np.asarray(score_rows, dtype=float), np.asarray(metric_rows, dtype=float)

    def vision_search(self, problem):
        """Choose one action per searcher and invest only in its SI region."""
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        self.current_adaptive_step = self.adaptive_step * (1.0 - progress)
        best_before = float(self.fitness)
        active_regions = self.selected_regions.copy()
        goods_fitness_before = self.goods_fitness.copy()
        goods_metrics_before = self.goods_metrics.copy()

        observations = self._build_observations()
        masks = np.ones((self.n, len(self.ACTION_NAMES)), dtype=bool)
        actions = self.agent.select_actions(observations, masks)

        child_rows = []
        for searcher_id, region in enumerate(active_regions):
            row = []
            for good in self.goods[int(region)]:
                children = self._make_children(
                    problem, self.searchers[searcher_id], good,
                    int(actions[searcher_id]),
                )
                row.extend(
                    self.align_region(problem, child, int(region))
                    for child in children
                )
            child_rows.append(row)

        child_scores, child_metrics = self._evaluate_child_rows(
            problem, child_rows
        )
        pair_scores = np.empty((self.n, self.w), dtype=float)
        pair_candidates = [[None] * self.w for _ in range(self.n)]
        pair_metrics = np.empty((self.n, self.w, 5), dtype=float)
        rewards = np.empty(self.n, dtype=float)

        parent_values = np.concatenate(
            (self.searcher_fitness, goods_fitness_before.ravel())
        )
        center = float(np.median(parent_values))
        reward_scale = max(
            0.01,
            float(np.quantile(np.abs(parent_values - center), 0.75)),
        )
        for searcher_id, region in enumerate(active_regions):
            region = int(region)
            deltas = []
            improved = 0
            degraded = 0
            for good_id in range(self.w):
                offset = good_id * 2
                local = child_scores[searcher_id, offset : offset + 2]
                winner = offset + int(np.argmax(local))
                score = float(child_scores[searcher_id, winner])
                pair_scores[searcher_id, good_id] = score
                pair_candidates[searcher_id][good_id] = child_rows[
                    searcher_id
                ][winner]
                pair_metrics[searcher_id, good_id] = child_metrics[
                    searcher_id, winner
                ]
                baseline = max(
                    float(self.searcher_fitness[searcher_id]),
                    float(goods_fitness_before[region, good_id]),
                )
                delta = score - baseline
                deltas.append(delta)
                improved += int(delta > self.IMPROVEMENT_TOLERANCE)
                degraded += int(delta < -self.IMPROVEMENT_TOLERANCE)

            reward = float(np.mean(np.tanh(np.asarray(deltas) / reward_scale)))
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
            self.investment_quality[region, searcher_id] = float(
                np.mean(pair_scores[searcher_id])
            )

        probabilities = self.region_probabilities(
            goods_fitness_before, self.investment_quality
        )
        selected = self.select_regions(probabilities)

        # Keep SI-SETS' searcher update order and pre-investment goods source.
        for searcher_id, region in enumerate(selected):
            region = int(region)
            good_id = int(np.argmax(goods_fitness_before[region]))
            good_score = float(goods_fitness_before[region, good_id])
            if good_score > self.searcher_fitness[searcher_id]:
                self.searchers[searcher_id] = self.goods[region][good_id].copy()
                self.searcher_fitness[searcher_id] = good_score
                self.searcher_metrics[searcher_id] = goods_metrics_before[
                    region, good_id
                ]

        # For every visited good, retain the best of the equally evaluated
        # two-child proposals made by all searchers visiting that region.
        for region in range(self.h):
            visitors = np.flatnonzero(active_regions == region)
            if not len(visitors):
                continue
            for good_id in range(self.w):
                winner = int(
                    max(
                        visitors,
                        key=lambda index: pair_scores[int(index), good_id],
                    )
                )
                self.goods[region][good_id] = pair_candidates[winner][
                    good_id
                ].copy()
                self.goods_fitness[region, good_id] = pair_scores[
                    winner, good_id
                ]
                self.goods_metrics[region, good_id] = pair_metrics[
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

    def search(self, problem, budget, state=None):
        best_state = super().search(problem, budget, state)
        if self.training and self.model_path is not None:
            self.agent.save(self.model_path)
        return best_state

    def run(self, problem, *, budget=10000, max_iteration=800, state=None):
        result = super().run(
            problem, budget=budget, max_iteration=max_iteration, state=state
        )
        result.metadata.update(
            {
                "rl_mode": "train" if self.training else "eval",
                "rl_model": None if self.model_path is None else str(self.model_path),
                "rl_epsilon": float(self.agent.epsilon),
                "rl_replay_size": len(self.agent.replay),
                "rl_training_steps": int(self.agent.training_steps),
                "rl_policy": self.policy_statistics(),
            }
        )
        return result

    def save_model(self, path=None):
        target = self.model_path if path is None else Path(path)
        if target is None:
            raise ValueError("save_model requires a model path")
        self.model_path = target
        return self.agent.save(target)

    def policy_statistics(self):
        actions = {}
        for action, name in enumerate(self.ACTION_NAMES):
            records = [row for row in self.reward_records if row["action"] == action]
            actions[name] = {
                "uses": int(self.action_uses[action]),
                "successes": int(self.action_successes[action]),
                "success_rate": (
                    0.0
                    if self.action_uses[action] == 0
                    else float(
                        self.action_successes[action] / self.action_uses[action]
                    )
                ),
                **self._summarize_rewards(records),
            }
        return {
            "training": self.training,
            "epsilon": float(self.agent.epsilon),
            "replay_size": len(self.agent.replay),
            "training_steps": int(self.agent.training_steps),
            "last_loss": self.agent.last_loss,
            "actions": actions,
            "reward_version": self.REWARD_VERSION,
            "reward": self._summarize_rewards(self.reward_records),
            "reward_by_progress": {
                "first_20_percent": self._summarize_rewards(
                    [row for row in self.reward_records if row["progress"] < 0.2]
                ),
                "remaining_80_percent": self._summarize_rewards(
                    [row for row in self.reward_records if row["progress"] >= 0.2]
                ),
            },
        }

    @staticmethod
    def _summarize_rewards(records):
        values = np.asarray([row["reward"] for row in records], dtype=float)
        count = len(records)
        positive = int(np.sum(values > 1e-12))
        negative = int(np.sum(values < -1e-12))
        zero = count - positive - negative
        return {
            "decisions": count,
            "reward_sum": float(np.sum(values)),
            "mean_reward": float(np.mean(values)) if count else 0.0,
            "positive_rewards": positive,
            "zero_rewards": zero,
            "negative_rewards": negative,
            "positive_rate": positive / max(1, count),
            "zero_rate": zero / max(1, count),
            "negative_rate": negative / max(1, count),
            "improved_goods": sum(row["improved_goods"] for row in records),
            "degraded_goods": sum(row["degraded_goods"] for row in records),
        }


__all__ = ["RL_SETS"]
