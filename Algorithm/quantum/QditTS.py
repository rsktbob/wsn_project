"""Quantum-inspired Tabu Search on multi-state Q-dits (Qdit-TS).

Same search loop as QTS (Chiang et al. 2014): one quantum register; each
iteration observes n candidates, takes the iteration's best and worst, and
moves the register toward the best except where best and worst agree (tabu).

Instead of conditional-thermometer Q-bits, every gene of SensorEncoding is one
Q-dit with one amplitude per option: a sensing level with d options holds
(a_0, ..., a_{d-1}) and a routing-priority rank holds 10 amplitudes, with
sum a_k^2 = 1 and P(option k) = a_k^2.  100 sensors give 100*6 + 100*10 = 1600
amplitudes, stored row by row in chromosome order [s0, p0, s1, p1, ...] and
padded with zeros up to the widest domain.  Observation draws one option per
row, so there are no invalid patterns and no "x" gates.

Update (per gene, best option b and worst option w): b == w is tabu; otherwise
a Givens rotation in the (a_b, a_w) plane moves probability from w to b and
leaves the other options unchanged:
    a_b' = a_b cos t + a_w sin t,   a_w' = a_w cos t - a_b sin t,
with t = min(rotation, the angle that keeps a_w^2 at the probability floor).
Without the floor an option could reach zero and never be observed again.
The options are unordered: no smoothing toward neighbouring levels or ranks.
Priority genes of sensors observed at level 0 are updated like any other gene.

The loop stops on the evaluation budget only, as QTS.
"""

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from State.SensorEncoding import SensorEncoding


class QditTS(Algorithm):
    def __init__(self, DP, n=10, rotation=0.01 * np.pi, probability_floor=0.01,
                 seed=None):
        super().__init__(seed=seed)
        if int(n) != n or n < 2:
            raise ValueError("n must be an integer >= 2")
        if not np.isfinite(rotation) or not 0 <= rotation <= np.pi / 2:
            raise ValueError("rotation must be in [0, pi/2]")
        if not 0 < probability_floor < 0.5:
            raise ValueError("probability_floor must be in (0, 0.5)")
        self.n = int(n)
        self.rotation = float(rotation)
        self.probability_floor = float(probability_floor)
        self._configure(DP.radius_option_counts)
        self.name = "QditTS"

    def _configure(self, sensing_domains):
        sensing_domains = np.asarray(sensing_domains, dtype=int)
        if sensing_domains.ndim != 1 or not len(sensing_domains) or np.any(sensing_domains < 1):
            raise ValueError("sensing_domains must be a nonempty positive vector")
        self.domains = np.empty(2 * len(sensing_domains), dtype=int)
        self.domains[0::2] = sensing_domains
        self.domains[1::2] = SensorEncoding.RANK_PRECISION
        self.valid = np.arange(self.domains.max()) < self.domains[:, None]
        # Same rule as QTS: never above the uniform probability 1/d.
        self.amplitude_floor = np.sqrt(np.minimum(self.probability_floor, 1 / self.domains))

    def observe(self):
        """Draw one option per Q-dit and return the SensorEncoding candidate."""
        cumulative = np.cumsum(self.amplitudes ** 2, axis=1)
        # Scaling by the row total keeps the draw inside the row's own domain.
        draw = self.rng.random((len(self.domains), 1)) * cumulative[:, -1:]
        return SensorEncoding((cumulative < draw).sum(axis=1))

    def _update(self, best_code, worst_code):
        best_code = np.asarray(best_code, dtype=int)
        worst_code = np.asarray(worst_code, dtype=int)
        rows = np.flatnonzero(best_code != worst_code)
        if not len(rows):
            return
        best, worst = best_code[rows], worst_code[rows]
        a_best = self.amplitudes[rows, best]
        a_worst = self.amplitudes[rows, worst]
        radius = np.hypot(a_best, a_worst)
        limit = np.arctan2(a_worst, a_best) - np.arcsin(
            np.minimum(1.0, self.amplitude_floor[rows] / radius))
        angle = np.clip(limit, 0.0, self.rotation)
        cos, sin = np.cos(angle), np.sin(angle)
        self.amplitudes[rows, best] = a_best * cos + a_worst * sin
        self.amplitudes[rows, worst] = a_worst * cos - a_best * sin

    def _reset_register(self):
        self.amplitudes = np.where(self.valid, 1 / np.sqrt(self.domains)[:, None], 0.0)

    @property
    def probabilities(self):
        return self.amplitudes ** 2

    def search(self, problem, budget, state=None):
        # Warm-start state is unused: every call starts from the same prior.
        self._configure(problem.radius_option_counts)
        problem.prepare_coding_cache()
        self.reset_best()
        self.evatime = 0
        self.iteration = 0
        self._reset_register()
        history = []
        while self.evatime < budget:
            size = min(self.n, budget - self.evatime)
            codes, scores = [], []
            for _ in range(size):
                candidate = self.observe()
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
