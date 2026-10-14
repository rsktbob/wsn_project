"""Paper-faithful RIME adapted to the project's sensor encoding."""

from __future__ import annotations

import math

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from State.SensorEncoding import SensorEncoding


class RIME(Algorithm):
    """Rime optimization over sensing indexes and routing priorities.

    RIME operates in the paper's continuous search space ``[0, 1]``.  Before
    evaluation, every coordinate is mapped into its own discrete domain:
    sensing coordinates use each sensor's current ``radius_option_counts`` and
    routing-priority coordinates use ``SensorEncoding.RANK_PRECISION``.
    """

    def __init__(self, problem, n=50, w=5, seed=None):
        super().__init__(seed=seed)
        if int(n) <= 0:
            raise ValueError("n must be positive")
        if int(w) <= 0:
            raise ValueError("w must be positive")

        problem.prepare_coding_cache()
        self.n = int(n)
        self.w = int(w)
        self.length = int(problem.SENSOR_NUMBER) * 2
        self.gene_domains = np.empty(self.length, dtype=int)
        self.gene_domains[0::2] = np.asarray(
            problem.radius_option_counts,
            dtype=int,
        )
        self.gene_domains[1::2] = SensorEncoding.RANK_PRECISION
        if np.any(self.gene_domains <= 0):
            raise ValueError("all SensorEncoding gene domains must be positive")

        self.name = f"RIME_{self.n}"
        self.population = []
        self.positions = np.empty((0, self.length), dtype=float)
        self.population_scores = np.empty(0, dtype=float)
        self.initial_population = []
        self.initial_population_scores = np.empty(0, dtype=float)
        self.iterations_completed = 0

    def _random_encoding(self, problem):
        """Create one fully random, domain-valid SensorEncoding."""
        return SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
            rng=self.rng,
        )

    def _encoding_to_position(self, encoding):
        """Place every discrete index at the centre of its unit interval."""
        return (
            np.asarray(encoding.code, dtype=float) + 0.5
        ) / self.gene_domains

    def position_to_encoding(self, position):
        """Map a normalized RIME position into every gene's legal domain."""
        clipped = np.clip(
            np.asarray(position, dtype=float),
            0.0,
            np.nextafter(1.0, 0.0),
        )
        code = np.floor(clipped * self.gene_domains).astype(int)
        code = np.minimum(code, self.gene_domains - 1)
        return SensorEncoding(code)

    def _evaluate_position(self, problem, position):
        """Decode one RIME position and record the global best solution."""
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

    @staticmethod
    def _hard_rime_rates(scores):
        """Normalize maximization losses for hard-rime puncture.

        The reference MATLAB implementation applies row normalization to a
        minimization objective, so worse (larger) values have a higher chance
        of being punctured by the best agent.  ``max(scores) - scores`` is the
        corresponding loss for this project's maximization convention.
        """
        scores = np.asarray(scores, dtype=float)
        losses = np.max(scores) - scores
        norm = float(np.linalg.norm(losses))
        if norm <= np.finfo(float).eps:
            return np.zeros(len(scores), dtype=float)
        return losses / norm

    def _rime_factor(self, iteration, total_iterations):
        """Return the paper's soft-rime factor for one iteration."""
        progress = iteration / max(1, total_iterations)
        rounded_stage = math.floor(progress * self.w + 0.5)
        return (
            (self.rng.random() - 0.5)
            * 2.0
            * math.cos(math.pi * iteration / (total_iterations / 10.0))
            * (1.0 - rounded_stage / self.w)
        )

    def _propose_population(
        self,
        positions,
        best_position,
        hard_rates,
        iteration,
        total_iterations,
    ):
        """Apply soft-rime first and hard-rime second, as in the paper."""
        proposals = np.asarray(positions, dtype=float).copy()
        attachment = math.sqrt(iteration / max(1, total_iterations))
        rime_factor = self._rime_factor(iteration, total_iterations)

        for agent_id in range(len(proposals)):
            soft_mask = self.rng.random(self.length) < attachment
            random_span = self.rng.random(self.length)
            proposals[agent_id, soft_mask] = (
                best_position[soft_mask]
                + rime_factor * random_span[soft_mask]
            )

            hard_mask = self.rng.random(self.length) < hard_rates[agent_id]
            proposals[agent_id, hard_mask] = best_position[hard_mask]

        # Boundary absorption from the reference RIME implementation.
        return np.clip(proposals, 0.0, 1.0)

    def search(self, problem, budget, state=None):
        """Run RIME with an exact fitness-evaluation budget."""
        evaluation_limit = max(1, int(budget))
        population_size = min(self.n, evaluation_limit)
        self.history = np.full(evaluation_limit, -np.inf, dtype=float)
        self.evatime = 0
        self.iterations_completed = 0
        self.population = []
        self.initial_population = []
        self.best_position = None

        initial_encodings = [
            self._random_encoding(problem) for _ in range(population_size)
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
        total_iterations = max(
            1,
            int(math.ceil(remaining / population_size)),
        )

        while self.evatime < evaluation_limit:
            iteration = self.iterations_completed + 1
            iteration_best = self.best_position.copy()
            hard_rates = self._hard_rime_rates(self.population_scores)
            proposals = self._propose_population(
                self.positions,
                iteration_best,
                hard_rates,
                iteration,
                total_iterations,
            )

            evaluation_count = min(
                population_size,
                evaluation_limit - self.evatime,
            )
            for agent_id in range(evaluation_count):
                child, child_score = self._evaluate_position(
                    problem,
                    proposals[agent_id],
                )
                # Positive greedy selection: a child replaces only its own
                # parent, and only when it is strictly better.
                if child_score > self.population_scores[agent_id]:
                    self.positions[agent_id] = proposals[agent_id]
                    self.population[agent_id] = child
                    self.population_scores[agent_id] = child_score

            self.on_iteration_finish(
                problem=problem,
                state=self.best_state,
                iteration=self.iterations_completed,
                run=0,
                best_value=self.best_objectives.copy(),
                fitness=self.fitness,
            )
            self.iterations_completed += 1

        return None if self.best_state is None else self.best_state.copy()


__all__ = ["RIME"]
