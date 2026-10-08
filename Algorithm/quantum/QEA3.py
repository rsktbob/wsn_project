"""WSN QEA3 variant: local reference sharing and periodic global migration.

Uses the existing conditional sampler and masked first-quadrant rotation, NOT
Han & Kim's complete binary rotation lookup table. Fixed disjoint pairs are an
explicit implementation choice. Initialization is generation zero, without a
rotation or migration. Each later generation rotates against OLD references,
then updates references and migrates. Migration copies solutions, never angles.
"""
import numpy as np

from Algorithm.quantum.QEA import QEA
from State.QuantumSensorEncoding import QuantumSensorEncoding


class QEA3(QEA):
    def __init__(self, DP, n=30, rotation=0.01 * np.pi,
                 probability_floor=0.01, global_period=100, seed=None):
        super().__init__(DP, n=n, rotation=rotation,
                         probability_floor=probability_floor, seed=seed)
        if not np.isfinite(global_period) or int(global_period) != global_period or global_period < 1:
            raise ValueError('global_period must be a positive integer')
        self.global_period = int(global_period)
        self.name = 'QEA3_WSN'

    def _rotate_references(self, observations):
        size = len(observations)
        paths = [self.encoding.elite_path(ref.code) for ref in self.references[:size]]
        targets = np.asarray([target for target, _ in paths])
        active = np.asarray([path for _, path in paths])
        mask = (observations != targets) & active
        angles = self.angles[:size]
        angles += mask * np.where(targets, self.rotation, -self.rotation)
        np.clip(angles, np.arcsin(np.sqrt(self.gate_floor)),
                np.arcsin(np.sqrt(1 - self.gate_floor)), out=angles)
        self.probabilities[:size] = np.sin(angles) ** 2

    def _migrate(self, generation):
        if generation % self.global_period == 0:
            self.references = [self.best_candidate.copy() for _ in range(self.n)]
            self.reference_scores[:] = self.fitness
            self.global_migrations += 1
        else:
            # Snapshot within each disjoint pair; equal scores choose left.
            # With odd population size, the last member remains a singleton.
            for start in range(0, self.n - 1, 2):
                winner = start + int(self.reference_scores[start + 1] > self.reference_scores[start])
                reference = self.references[winner].copy()
                score = self.reference_scores[winner]
                for member in (start, start + 1):
                    self.references[member] = reference.copy()
                    self.reference_scores[member] = score
            self.local_migrations += 1

    def search(self, problem, budget, state=None):
        self.encoding = QuantumSensorEncoding(problem.radius_option_counts)
        problem.prepare_coding_cache()
        self.reset_best()
        self.evatime = self.iteration = 0
        self.local_migrations = self.global_migrations = 0
        self.references = [None] * self.n
        self.reference_scores = np.full(self.n, -np.inf)
        self.angles = np.tile(self.encoding.initial_angles, (self.n, 1))
        self.probabilities = np.tile(self.encoding.initial_probabilities, (self.n, 1))
        self.gate_floor = np.minimum(self.probability_floor,
                                     1 / self.encoding.domains[self.encoding.gene_ids])
        history = []
        # Initialization is evaluated and charged to the same fitness budget.
        for member in range(min(self.n, budget)):
            candidate, _ = self.encoding.observe(self.probabilities[member], self.rng)
            self.reference_scores[member] = self.score(problem, candidate)
            self.references[member] = candidate.copy()
            history.append(self.fitness)
        while self.evatime < budget and self.iteration < self.max_iterations:
            size = min(self.n, budget - self.evatime)
            candidates, observations, scores = [], [], []
            for member in range(size):
                candidate, bits = self.encoding.observe(self.probabilities[member], self.rng)
                scores.append(self.score(problem, candidate))
                candidates.append(candidate)
                observations.append(bits)
                history.append(self.fitness)
            # score() tracks the global best, but rotation reads only the old
            # references. New discoveries cannot change this generation's target.
            self._rotate_references(np.asarray(observations))
            for member, (candidate, score) in enumerate(zip(candidates, scores)):
                if score > self.reference_scores[member]:
                    self.references[member] = candidate.copy()
                    self.reference_scores[member] = score
            generation = self.iteration + 1
            if size == self.n:
                self._migrate(generation)
            self.on_iteration_finish(problem=problem, state=self.best_state.copy(),
                                     iteration=generation, best_value=self.best_objectives.copy(),
                                     fitness=self.fitness)
        self.history = np.asarray(history)
        self.iterations_completed = self.iteration
        self.early_stopped = self.evatime < budget
        return self.best_state.copy()
