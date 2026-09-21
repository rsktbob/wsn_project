"""Crested Porcupine Optimizer adapted to ``SensorEncoding``.

The four CPO defence equations run in the normalized continuous space
``[0, 1]``.  Every coordinate is mapped back to its own legal discrete
domain before it consumes one WSN fitness evaluation.
"""

from __future__ import annotations

import math

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from State.SensorEncoding import SensorEncoding


class CPO(Algorithm):
    """Continuous CPO with a coverage-seeded SensorEncoding population."""

    DEFENCE_NAMES = ("sight", "sound", "odor", "physical_attack")

    def __init__(
        self,
        problem,
        n=30,
        min_population=10,
        cycles=2,
        alpha=0.2,
        tradeoff=0.8,
        seed=None,
    ):
        super().__init__(seed=seed)
        if int(n) <= 0:
            raise ValueError("n must be positive")
        if int(min_population) <= 0:
            raise ValueError("min_population must be positive")
        if int(cycles) <= 0:
            raise ValueError("cycles must be positive")
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= float(tradeoff) <= 1.0:
            raise ValueError("tradeoff must be in [0, 1]")

        problem.prepare_coding_cache()
        self.n = int(n)
        self.min_population = min(int(min_population), self.n)
        self.cycles = int(cycles)
        self.alpha = float(alpha)
        self.tradeoff = float(tradeoff)
        self.length = int(problem.SENSOR_NUMBER) * 2
        self.gene_domains = np.empty(self.length, dtype=int)
        self.gene_domains[0::2] = np.asarray(
            problem.radius_option_counts,
            dtype=int,
        )
        self.gene_domains[1::2] = SensorEncoding.RANK_PRECISION
        if np.any(self.gene_domains <= 0):
            raise ValueError("all SensorEncoding gene domains must be positive")

        self.name = f"CPO_{self.n}"
        self.population = []
        self.positions = np.empty((0, self.length), dtype=float)
        self.population_scores = np.empty(0, dtype=float)
        self.initial_population = []
        self.initial_population_scores = np.empty(0, dtype=float)
        self.iterations_completed = 0
        self.active_population_history = []
        self.defence_counts = {name: 0 for name in self.DEFENCE_NAMES}
        self.proposal_count = 0
        self.encoding_change_count = 0
        self.accepted_count = 0

    def _coverage_seeded_encoding(self, problem):
        """Use SA-SETS coverage seeding without any region alignment."""
        candidate = SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
            rng=self.rng,
        )
        for target_candidates in problem.cover_candidates:
            if len(target_candidates) == 0:
                continue
            choice = int(self.rng.integers(len(target_candidates)))
            sensor_id, level = target_candidates[choice]
            candidate.code[int(sensor_id) * 2] = int(level)
        return candidate

    def _encoding_to_position(self, encoding):
        # The interval centre makes decoding the initial seed lossless.
        return (
            np.asarray(encoding.code, dtype=float) + 0.5
        ) / self.gene_domains

    def position_to_encoding(self, position):
        clipped = np.clip(
            np.asarray(position, dtype=float),
            0.0,
            np.nextafter(1.0, 0.0),
        )
        code = np.floor(clipped * self.gene_domains).astype(int)
        return SensorEncoding(np.minimum(code, self.gene_domains - 1))

    def _evaluate_position(self, problem, position):
        encoding = self.position_to_encoding(position)
        state = encoding.decode(problem)
        objectives = np.asarray(problem.evaluate_state(state), dtype=float)
        objectives = np.nan_to_num(
            objectives,
            nan=-5.0,
            neginf=-5.0,
            posinf=5.0,
        )
        score = float(np.sum(objectives))
        self.evatime += 1
        improved = self.update_best(
            problem,
            state,
            objectives,
            score,
            candidate=encoding,
        )
        if improved:
            self.best_position = np.asarray(position, dtype=float).copy()
        self.history[self.evatime - 1] = self.fitness
        return encoding, score

    def _active_population_size(self, zero_based_t, total_iterations):
        """Cyclic population reduction used by the reference CPO."""
        cycle_length = max(1, int(total_iterations) // self.cycles)
        cycle_progress = (int(zero_based_t) % cycle_length) / cycle_length
        size = int(
            self.min_population
            + (self.n - self.min_population) * (1.0 - cycle_progress)
        )
        return max(1, min(self.n, size))

    def _iterations_for_budget(self, remaining):
        """Find a CPO horizon whose CPR schedule covers the FE budget."""
        if remaining <= 0:
            return 1
        estimate = max(1, int(math.ceil(remaining / self.n)))
        while True:
            evaluations = sum(
                self._active_population_size(t, estimate)
                for t in range(estimate)
            )
            if evaluations >= remaining:
                return estimate
            estimate += 1

    def _fitness_scale(self, agent_id, active_size):
        """Stable minimization-equivalent scale for project maximization."""
        scores = np.asarray(self.population_scores[:active_size], dtype=float)
        losses = np.max(scores) - scores
        denominator = float(np.sum(losses)) + np.finfo(float).eps
        return float(np.exp(losses[int(agent_id)] / denominator))

    def _propose(self, agent_id, active_size, zero_based_t, total_iterations):
        """Apply one of the four original CPO defence mechanisms."""
        current = self.positions[agent_id]
        u1 = self.rng.random(self.length) > self.rng.random()

        if self.rng.random() < self.rng.random():
            random_id = int(self.rng.integers(active_size))
            midpoint = (current + self.positions[random_id]) / 2.0
            if self.rng.random() < self.rng.random():
                defence = "sight"
                proposal = current + self.rng.normal() * np.abs(
                    2.0 * self.rng.random() * self.best_position - midpoint
                )
            else:
                defence = "sound"
                first = int(self.rng.integers(active_size))
                second = int(self.rng.integers(active_size))
                difference = self.positions[first] - self.positions[second]
                proposal = u1 * current + (1 - u1) * (
                    midpoint + self.rng.random() * difference
                )
        else:
            progress = int(zero_based_t) / max(1, int(total_iterations))
            yt = 2.0 * self.rng.random() * (1.0 - progress) ** progress
            u2 = np.where(self.rng.random(self.length) < 0.5, -1.0, 1.0)
            scale = self._fitness_scale(agent_id, active_size)

            if self.rng.random() < self.tradeoff:
                defence = "odor"
                first = int(self.rng.integers(active_size))
                second = int(self.rng.integers(active_size))
                third = int(self.rng.integers(active_size))
                step = self.rng.random() * u2 * yt * scale
                proposal = (1 - u1) * current + u1 * (
                    self.positions[first]
                    + scale * (self.positions[second] - self.positions[third])
                    - step
                )
            else:
                defence = "physical_attack"
                peer = self.positions[int(self.rng.integers(active_size))]
                force = self.rng.random(self.length) * (
                    scale * (-current + peer)
                )
                step = self.rng.random() * u2 * yt * force
                r2 = self.rng.random()
                proposal = (
                    self.best_position
                    + self.alpha * (1.0 - r2)
                    + r2 * (u2 * self.best_position - current)
                    - step
                )

        self.defence_counts[defence] += 1
        return np.clip(proposal, 0.0, 1.0)

    def search(self, problem, budget, state=None):
        """Run CPO under an exact fitness-evaluation budget."""
        evaluation_limit = max(1, int(budget))
        population_size = min(self.n, evaluation_limit)
        # CPR parameters must follow the actually available population.
        original_n = self.n
        original_minimum = self.min_population
        self.n = population_size
        self.min_population = min(original_minimum, population_size)

        self.history = np.full(evaluation_limit, -np.inf, dtype=float)
        self.evatime = 0
        self.iterations_completed = 0
        self.population = []
        self.initial_population = []
        self.best_position = None
        self.active_population_history = []
        self.defence_counts = {name: 0 for name in self.DEFENCE_NAMES}
        self.proposal_count = 0
        self.encoding_change_count = 0
        self.accepted_count = 0

        initial_encodings = [
            self._coverage_seeded_encoding(problem)
            for _ in range(population_size)
        ]
        self.initial_population = [item.copy() for item in initial_encodings]
        self.positions = np.asarray(
            [self._encoding_to_position(item) for item in initial_encodings],
            dtype=float,
        )
        self.population_scores = np.empty(population_size, dtype=float)

        for agent_id, position in enumerate(self.positions):
            encoding, score = self._evaluate_position(problem, position)
            self.population.append(encoding)
            self.population_scores[agent_id] = score
        self.initial_population_scores = self.population_scores.copy()

        remaining = evaluation_limit - self.evatime
        total_iterations = self._iterations_for_budget(remaining)

        while self.evatime < evaluation_limit:
            zero_based_t = self.iterations_completed
            active_size = self._active_population_size(
                zero_based_t,
                total_iterations,
            )
            evaluation_count = min(
                active_size,
                evaluation_limit - self.evatime,
            )
            self.active_population_history.append(active_size)

            for agent_id in range(evaluation_count):
                parent_code = np.asarray(
                    self.population[agent_id].code,
                    dtype=int,
                ).copy()
                proposal = self._propose(
                    agent_id,
                    active_size,
                    zero_based_t,
                    total_iterations,
                )
                child, child_score = self._evaluate_position(problem, proposal)
                self.proposal_count += 1
                if not np.array_equal(parent_code, child.code):
                    self.encoding_change_count += 1
                if child_score > self.population_scores[agent_id]:
                    self.positions[agent_id] = proposal
                    self.population[agent_id] = child
                    self.population_scores[agent_id] = child_score
                    self.accepted_count += 1

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

        self.n = original_n
        self.min_population = original_minimum
        return None if self.best_state is None else self.best_state.copy()


__all__ = ["CPO"]
