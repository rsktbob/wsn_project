"""Discrete adaptive CPO for the project's ``SensorEncoding``."""

from __future__ import annotations

import math

import numpy as np

from Algorithm.misc.CPO import CPO
from State.SensorEncoding import SensorEncoding


class CPOv2(CPO):
    """CPO whose four defence mechanisms act on discrete sensor pairs.

    The initial population is exactly CPO's SA-SETS-style coverage seeding,
    without region alignment.  CPOv2 changes only the search stage: operators
    make valid, observable ``[schedule, priority]`` edits rather than moving
    a continuous vector which may round back to the same chromosome.
    """

    BASE_PROBABILITIES = np.asarray((0.25, 0.25, 0.40, 0.10), dtype=float)

    def __init__(
        self,
        problem,
        n=30,
        min_population=10,
        cycles=2,
        adapt_interval=300,
        adaptive_weight=0.5,
        minimum_probability=0.05,
        stagnation_limit=1000,
        restart_fraction=0.2,
        seed=None,
    ):
        super().__init__(
            problem,
            n=n,
            min_population=min_population,
            cycles=cycles,
            seed=seed,
        )
        if int(adapt_interval) <= 0:
            raise ValueError("adapt_interval must be positive")
        if not 0.0 <= float(adaptive_weight) <= 1.0:
            raise ValueError("adaptive_weight must be in [0, 1]")
        if not 0.0 <= float(minimum_probability) < 0.25:
            raise ValueError("minimum_probability must be in [0, 0.25)")
        if int(stagnation_limit) <= 0:
            raise ValueError("stagnation_limit must be positive")
        if not 0.0 < float(restart_fraction) < 1.0:
            raise ValueError("restart_fraction must be in (0, 1)")

        self.name = f"CPOv2_{self.n}"
        self.adapt_interval = int(adapt_interval)
        self.adaptive_weight = float(adaptive_weight)
        self.minimum_probability = float(minimum_probability)
        self.stagnation_limit = int(stagnation_limit)
        self.restart_fraction = float(restart_fraction)
        self.operator_probabilities = self.BASE_PROBABILITIES.copy()
        self.operator_uses = np.zeros(4, dtype=int)
        self.operator_successes = np.zeros(4, dtype=int)
        self.window_uses = np.zeros(4, dtype=int)
        self.window_successes = np.zeros(4, dtype=int)
        self.restart_count = 0
        self.forced_change_count = 0
        self.last_best_evaluation = 0
        self.restart_queue = []

    @staticmethod
    def _sensor_pair(code, sensor_id):
        start = int(sensor_id) * 2
        return np.asarray(code[start : start + 2], dtype=int)

    @staticmethod
    def _write_sensor_pair(code, sensor_id, values):
        start = int(sensor_id) * 2
        code[start : start + 2] = np.asarray(values, dtype=int)

    def _sample_sensor_ids(self, count, candidate_ids=None):
        if candidate_ids is None:
            candidate_ids = np.arange(len(self.gene_domains) // 2)
        candidate_ids = np.asarray(candidate_ids, dtype=int)
        if not len(candidate_ids):
            return np.empty(0, dtype=int)
        count = min(max(1, int(count)), len(candidate_ids))
        return np.asarray(
            self.rng.choice(candidate_ids, size=count, replace=False),
            dtype=int,
        )

    def _random_pair(self, problem, sensor_id):
        return np.asarray(
            (
                int(self.rng.integers(problem.sensing_option_count(sensor_id))),
                int(self.rng.integers(SensorEncoding.RANK_PRECISION)),
            ),
            dtype=int,
        )

    def _different_sensors(self, first, second):
        pairs_first = np.asarray(first.code, dtype=int).reshape(-1, 2)
        pairs_second = np.asarray(second.code, dtype=int).reshape(-1, 2)
        return np.flatnonzero(np.any(pairs_first != pairs_second, axis=1))

    def _neighbor_value(self, current, domain):
        current = int(current)
        domain = int(domain)
        if domain <= 1:
            return current
        if current <= 0:
            return 1
        if current >= domain - 1:
            return domain - 2
        return current + (1 if self.rng.random() < 0.5 else -1)

    def _force_effective_change(self, problem, candidate):
        """Guarantee a legal chromosome edit before consuming an evaluation."""
        sensor_id = int(self.rng.integers(problem.SENSOR_NUMBER))
        gene_id = sensor_id * 2
        sensing_domain = problem.sensing_option_count(sensor_id)
        if sensing_domain > 1 and self.rng.random() < 0.5:
            candidate.code[gene_id] = self._neighbor_value(
                candidate.code[gene_id], sensing_domain
            )
        else:
            candidate.code[gene_id + 1] = self._neighbor_value(
                candidate.code[gene_id + 1], SensorEncoding.RANK_PRECISION
            )
        return candidate

    def _sight(self, problem, parent_id, active_ids):
        """Large exploration through peer inheritance and random sensor pairs."""
        child = self.population[parent_id].copy()
        peer_id = int(self.rng.choice(active_ids))
        peer = self.population[peer_id]
        ratio = self.rng.uniform(0.15, 0.30)
        sensor_ids = self._sample_sensor_ids(
            math.ceil(problem.SENSOR_NUMBER * ratio)
        )
        for sensor_id in sensor_ids:
            if self.rng.random() < 0.5:
                self._write_sensor_pair(
                    child.code, sensor_id, self._sensor_pair(peer.code, sensor_id)
                )
            else:
                self._write_sensor_pair(
                    child.code, sensor_id, self._random_pair(problem, sensor_id)
                )
        return child

    def _sound(self, problem, parent_id, active_ids):
        """Explore the discrete difference between two population members."""
        child = self.population[parent_id].copy()
        first = self.population[int(self.rng.choice(active_ids))]
        second = self.population[int(self.rng.choice(active_ids))]
        differing = self._different_sensors(first, second)
        if not len(differing):
            return child
        ratio = self.rng.uniform(0.15, 0.30)
        sensor_ids = self._sample_sensor_ids(
            math.ceil(len(differing) * ratio), differing
        )
        for sensor_id in sensor_ids:
            donor = first if self.rng.random() < 0.5 else second
            self._write_sensor_pair(
                child.code, sensor_id, self._sensor_pair(donor.code, sensor_id)
            )
        return child

    def _odor(self, problem, parent_id, active_ids):
        """Exploit by selectively inheriting the global-best sensor pairs."""
        child = self.population[parent_id].copy()
        best = self.best_candidate
        differing = self._different_sensors(child, best)
        if not len(differing):
            return child
        ratio = self.rng.uniform(0.05, 0.15)
        sensor_ids = self._sample_sensor_ids(
            math.ceil(len(differing) * ratio), differing
        )
        for sensor_id in sensor_ids:
            self._write_sensor_pair(
                child.code, sensor_id, self._sensor_pair(best.code, sensor_id)
            )
        return child

    def _physical_attack(self, problem, parent_id, active_ids):
        """Perform one to three legal local schedule/priority edits."""
        child = self.population[parent_id].copy()
        count = int(self.rng.integers(1, 4))
        for sensor_id in self._sample_sensor_ids(count):
            schedule_id = int(sensor_id) * 2
            if self.rng.random() < 0.5:
                child.code[schedule_id] = self._neighbor_value(
                    child.code[schedule_id],
                    problem.sensing_option_count(sensor_id),
                )
            else:
                child.code[schedule_id + 1] = self._neighbor_value(
                    child.code[schedule_id + 1],
                    SensorEncoding.RANK_PRECISION,
                )
        return child

    def _choose_operator(self):
        return int(self.rng.choice(4, p=self.operator_probabilities))

    def _propose_discrete(self, problem, parent_id, active_ids):
        operator_id = self._choose_operator()
        generators = (
            self._sight,
            self._sound,
            self._odor,
            self._physical_attack,
        )
        parent_code = np.asarray(self.population[parent_id].code, dtype=int)
        child = generators[operator_id](problem, parent_id, active_ids)
        if np.array_equal(parent_code, child.code):
            child = self._force_effective_change(problem, child)
            self.forced_change_count += 1
        return operator_id, child

    def _evaluate_encoding(self, problem, encoding):
        state = encoding.decode(problem)
        objectives = np.asarray(problem.evaluate_state(state), dtype=float)
        objectives = np.nan_to_num(
            objectives, nan=-5.0, neginf=-5.0, posinf=5.0
        )
        score = float(np.sum(objectives))
        self.evatime += 1
        global_improved = self.update_best(
            problem, state, objectives, score, candidate=encoding
        )
        if global_improved:
            self.last_best_evaluation = self.evatime
        self.history[self.evatime - 1] = self.fitness
        return score, global_improved

    def _adapt_probabilities(self):
        rates = (self.window_successes + 1.0) / (self.window_uses + 2.0)
        learned = rates / np.sum(rates)
        probability = (
            (1.0 - self.adaptive_weight) * self.BASE_PROBABILITIES
            + self.adaptive_weight * learned
        )
        probability = np.maximum(probability, self.minimum_probability)
        self.operator_probabilities = probability / np.sum(probability)
        self.window_uses.fill(0)
        self.window_successes.fill(0)

    def _queue_restart(self):
        count = max(1, int(math.ceil(self.restart_fraction * len(self.population))))
        worst = np.argsort(self.population_scores)[:count].astype(int).tolist()
        self.random.shuffle(worst)
        self.restart_queue = worst

    def _maybe_start_restart(self):
        if self.restart_queue:
            return
        if self.evatime - self.last_best_evaluation < self.stagnation_limit:
            return
        self._queue_restart()
        # Prevent the same plateau from continuously creating restart queues.
        self.last_best_evaluation = self.evatime

    def _restart_parent(self, problem, parent_id):
        child = self._coverage_seeded_encoding(problem)
        score, _ = self._evaluate_encoding(problem, child)
        self.population[parent_id] = child
        self.population_scores[parent_id] = score
        self.restart_count += 1

    def _reset_statistics(self):
        self.active_population_history = []
        self.defence_counts = {name: 0 for name in self.DEFENCE_NAMES}
        self.operator_probabilities = self.BASE_PROBABILITIES.copy()
        self.operator_uses.fill(0)
        self.operator_successes.fill(0)
        self.window_uses.fill(0)
        self.window_successes.fill(0)
        self.restart_count = 0
        self.forced_change_count = 0
        self.proposal_count = 0
        self.encoding_change_count = 0
        self.accepted_count = 0
        self.last_best_evaluation = 0
        self.restart_queue = []

    def search(self, problem, budget, state=None):
        """Run the discrete CPOv2 under an exact evaluation budget."""
        evaluation_limit = max(1, int(budget))
        population_size = min(self.n, evaluation_limit)
        original_n, original_minimum = self.n, self.min_population
        self.n = population_size
        self.min_population = min(original_minimum, population_size)

        self.history = np.full(evaluation_limit, -np.inf, dtype=float)
        self.evatime = 0
        self.iterations_completed = 0
        self.population = []
        self.initial_population = []
        self._reset_statistics()

        initial_encodings = [
            self._coverage_seeded_encoding(problem)
            for _ in range(population_size)
        ]
        self.initial_population = [item.copy() for item in initial_encodings]
        self.population_scores = np.empty(population_size, dtype=float)
        for agent_id, encoding in enumerate(initial_encodings):
            score, _ = self._evaluate_encoding(problem, encoding)
            self.population.append(encoding)
            self.population_scores[agent_id] = score
        self.initial_population_scores = self.population_scores.copy()

        remaining = evaluation_limit - self.evatime
        total_iterations = self._iterations_for_budget(remaining)
        while self.evatime < evaluation_limit:
            active_size = self._active_population_size(
                self.iterations_completed, total_iterations
            )
            evaluation_count = min(active_size, evaluation_limit - self.evatime)
            active_ids = self.rng.choice(
                self.n, size=active_size, replace=False
            ).astype(int).tolist()
            self.active_population_history.append(active_size)

            for _ in range(evaluation_count):
                self._maybe_start_restart()
                if self.restart_queue:
                    parent_id = int(self.restart_queue.pop())
                    if parent_id in active_ids:
                        active_ids.remove(parent_id)
                    self._restart_parent(problem, parent_id)
                    continue

                parent_id = int(active_ids.pop())
                operator_id, child = self._propose_discrete(
                    problem, parent_id, active_ids + [parent_id]
                )
                child_score, _ = self._evaluate_encoding(problem, child)
                self.proposal_count += 1
                self.encoding_change_count += 1
                self.operator_uses[operator_id] += 1
                self.window_uses[operator_id] += 1
                self.defence_counts[self.DEFENCE_NAMES[operator_id]] += 1
                if child_score > self.population_scores[parent_id]:
                    self.population[parent_id] = child
                    self.population_scores[parent_id] = child_score
                    self.accepted_count += 1
                    self.operator_successes[operator_id] += 1
                    self.window_successes[operator_id] += 1

                if self.proposal_count % self.adapt_interval == 0:
                    self._adapt_probabilities()

            self.on_iteration_finish(
                problem=problem,
                state=self.best_state,
                iteration=self.iterations_completed,
                run=0,
                best_value=self.best_objectives.copy(),
                fitness=self.fitness,
                active_population=active_size,
            )
            self.iterations_completed += 1

        self.n, self.min_population = original_n, original_minimum
        return None if self.best_state is None else self.best_state.copy()


__all__ = ["CPOv2"]
