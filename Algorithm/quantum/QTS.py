"""Quantum-inspired Tabu Search (QTS) on the whole SensorEncoding.

Adaptation of Chiang et al. (2014), Soft Computing 18, "A quantum-inspired
Tabu search algorithm for solving combinatorial optimization problems":
one Q-bit vector; each iteration observes n neighbours, takes the iteration's
best and worst, and rotates every gate toward the best except the "tabu"
gates where best and worst already agree.

Each gene is a conditional thermometer (QuantumSensorEncoding): a sensing
level with 6 options uses 5 gates (0xxxx->0, 10xxx->1, ..., 11111->5) and a
routing-priority rank uses 9 gates, so 100 sensors give 100*5 + 100*9 = 1400
Q-bits.  Gates after the first 0 are "x" (they do not change the decoded
value), so only gates on the decoded path are compared and rotated.

init="uniform" starts every gene with equal option probabilities (as QEA);
init="half" is the paper's |alpha|=|beta|=1/sqrt(2), which on thermometer
gates gives P(value=k) = 2^-(k+1) (half of the genes start at 0).

The loop stops on the evaluation budget only: with the runner's default
max_iteration=800, n=10 would otherwise stop at 8000 of 10000 evaluations.
"""

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from State.QuantumSensorEncoding import QuantumSensorEncoding


class QTS(Algorithm):
    def __init__(self, DP, n=10, rotation=0.01 * np.pi, probability_floor=0.01,
                 init="uniform", seed=None):
        super().__init__(seed=seed)
        if int(n) != n or n < 2:
            raise ValueError("n must be an integer >= 2")
        if not np.isfinite(rotation) or not 0 <= rotation <= np.pi / 2:
            raise ValueError("rotation must be in [0, pi/2]")
        if not 0 < probability_floor < 0.5:
            raise ValueError("probability_floor must be in (0, 0.5)")
        if init not in ("uniform", "half"):
            raise ValueError("init must be 'uniform' or 'half'")
        self.n = int(n)
        self.rotation = float(rotation)
        self.probability_floor = float(probability_floor)
        self.init = init
        self.encoding = QuantumSensorEncoding(DP.radius_option_counts)
        self.name = "QTS"

    def _update(self, best_code, worst_code):
        best_target, best_path = self.encoding.elite_path(best_code)
        worst_target, worst_path = self.encoding.elite_path(worst_code)
        tabu = best_path & worst_path & (best_target == worst_target)
        mask = best_path & ~tabu
        self.angles += mask * np.where(best_target, self.rotation, -self.rotation)
        np.clip(self.angles, np.arcsin(np.sqrt(self.gate_floor)),
                np.arcsin(np.sqrt(1 - self.gate_floor)), out=self.angles)
        self.probabilities = np.sin(self.angles) ** 2

    def search(self, problem, budget, state=None):
        # Warm-start state is unused: every call starts from the same prior.
        self.encoding = QuantumSensorEncoding(problem.radius_option_counts)
        problem.prepare_coding_cache()
        self.reset_best()
        self.evatime = 0
        self.iteration = 0
        if self.init == "uniform":
            self.angles = self.encoding.initial_angles.copy()
        else:
            self.angles = np.full(self.encoding.length, np.pi / 4)
        self.probabilities = np.sin(self.angles) ** 2
        # Same floor as QEA: keep an exactly uniform prior representable.
        self.gate_floor = np.minimum(self.probability_floor,
                                     1 / self.encoding.domains[self.encoding.gene_ids])
        history = []
        while self.evatime < budget:
            size = min(self.n, budget - self.evatime)
            codes, scores = [], []
            for _ in range(size):
                candidate, _ = self.encoding.observe(self.probabilities, self.rng)
                scores.append(self.score(problem, candidate))
                codes.append(candidate.code)
                history.append(self.fitness)
            best, worst = int(np.argmax(scores)), int(np.argmin(scores))
            if scores[best] > scores[worst]:
                self._update(codes[best], codes[worst])
            self.on_iteration_finish(problem=problem, state=self.best_state.copy(),
                                     iteration=self.iteration, best_value=self.best_objectives.copy(),
                                     fitness=self.fitness)
        self.history = np.asarray(history, dtype=float)
        self.iterations_completed = self.iteration
        self.early_stopped = False
        return self.best_state.copy()
