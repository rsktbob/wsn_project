"""Global QEA with conditional thermometer observation and elite rotation.

Adaptation of the Q-bit / rotation idea of Han & Kim (2002), DOI
10.1109/TEVC.2002.804320; not a reproduction of their binary lookup table.
The whole network's sensing and routing-priority genes are searched jointly.
Candidates are ordinary SensorEncoding objects using the shared decode().
QuantumSensorEncoding supplies the probability representation and sampling;
SensorEncoding supplies the observed chromosome and shared WSN decoder.
"""

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from State.QuantumSensorEncoding import QuantumSensorEncoding


class QEA(Algorithm):
    def __init__(self, DP, n=30, rotation=0.01 * np.pi, probability_floor=0.01,
                 update_rule="rotation", learning_rate=0.05, seed=None):
        super().__init__(seed=seed)
        if int(n) != n or n < 1:
            raise ValueError("n must be a positive integer")
        if not np.isfinite(rotation) or not 0 <= rotation <= np.pi / 2:
            raise ValueError("rotation must be in [0, pi/2]")
        if not 0 < probability_floor < 0.5:
            raise ValueError("probability_floor must be in (0, 0.5)")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if update_rule not in ("rotation", "classical", "none"):
            raise ValueError("unknown update_rule")
        self.n = int(n)
        self.rotation = float(rotation)
        self.probability_floor = float(probability_floor)
        self.update_rule = update_rule
        self.learning_rate = float(learning_rate)
        self.encoding = QuantumSensorEncoding(DP.radius_option_counts)
        self.name = f"QEA_{update_rule}"

    def _update(self, observations):
        target, path = self.encoding.elite_path(self.best_candidate.code)
        mask = (observations != target) & path
        size = len(observations)
        angles = self.angles[:size]
        probabilities = self.probabilities[:size]
        # Preserve an exactly uniform initial distribution even for large K.
        # A smaller gate floor is necessary when 1/K < configured floor.
        floor = self.gate_floor
        if self.update_rule == "rotation":
            angles += mask * np.where(target, self.rotation, -self.rotation)
            np.clip(angles, np.arcsin(np.sqrt(floor)),
                    np.arcsin(np.sqrt(1 - floor)), out=angles)
            probabilities[:] = np.sin(angles) ** 2
        elif self.update_rule == "classical":
            probabilities += self.learning_rate * mask * (target - probabilities)
            np.clip(probabilities, floor, 1 - floor, out=probabilities)
            angles[:] = np.arcsin(np.sqrt(probabilities))

    def search(self, problem, budget, state=None):
        # Rebuild domains for lifetime calls after problem updates. Warm-start
        # state is deliberately unused: every call starts from the same prior.
        self.encoding = QuantumSensorEncoding(problem.radius_option_counts)
        problem.prepare_coding_cache()
        self.reset_best()
        self.evatime = 0
        self.iteration = 0
        self.angles = np.tile(self.encoding.initial_angles, (self.n, 1))
        self.probabilities = np.tile(self.encoding.initial_probabilities, (self.n, 1))
        self.gate_floor = np.minimum(self.probability_floor,
                                     1 / self.encoding.domains[self.encoding.gene_ids])
        history = []
        while self.evatime < budget and self.iteration < self.max_iterations:
            size = min(self.n, budget - self.evatime)
            observations = []
            for member in range(size):
                candidate, bits = self.encoding.observe(self.probabilities[member], self.rng)
                self.score(problem, candidate)
                observations.append(bits)
                history.append(self.fitness)
            self._update(np.asarray(observations))
            self.on_iteration_finish(problem=problem, state=self.best_state.copy(),
                                     iteration=self.iteration, best_value=self.best_objectives.copy(),
                                     fitness=self.fitness)
        self.history = np.asarray(history, dtype=float)
        self.iterations_completed = self.iteration
        self.early_stopped = self.evatime < budget
        return self.best_state.copy()
