"""Particle swarm optimization over the shared sensor chromosome."""

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from State.SensorEncoding import SensorEncoding


class PSO(Algorithm):
    """Optimize sensing choices and routing priorities with a bounded swarm.

    Positions live in [0, 1]. Each coordinate is mapped to its own integer
    domain before evaluation, so sensors may have different sensing options.
    Personal and global bests retain the continuous position that produced the
    evaluated chromosome.
    """

    def __init__(self, DP, n=30, w=0.7, c1=1.5, c2=1.5, vmax=0.2, seed=None):
        super().__init__(seed=seed)
        if int(n) != n or n < 1:
            raise ValueError("n must be a positive integer")
        for name, value in (("w", w), ("c1", c1), ("c2", c2)):
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(vmax) or vmax <= 0:
            raise ValueError("vmax must be finite and positive")
        self.N = int(n)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.vmax = float(vmax)
        self.domains = np.full(DP.SENSOR_NUMBER * 2, SensorEncoding.RANK_PRECISION, dtype=int)
        self.domains[0::2] = DP.radius_option_counts
        if np.any(self.domains <= 0):
            raise ValueError("all sensor gene domains must be positive")
        DP.prepare_coding_cache()

    def _candidate(self, position):
        bounded = np.clip(position, 0.0, np.nextafter(1.0, 0.0))
        return SensorEncoding(np.floor(bounded * self.domains).astype(int))

    def search(self, P, budget, state=None):
        if budget < 1:
            raise ValueError("budget must be positive")
        self.evatime = 0
        self.history = np.empty(budget, dtype=float)
        swarm_size = min(self.N, budget)
        positions = self.rng.random((swarm_size, len(self.domains)))
        velocities = self.rng.uniform(-self.vmax, self.vmax, positions.shape)
        if isinstance(state, SensorEncoding):
            code = np.asarray(state.code, dtype=int)
            if code.shape != self.domains.shape or np.any(code < 0) or np.any(code >= self.domains):
                raise ValueError("initial SensorEncoding has invalid gene values")
            positions[0] = (code + 0.5) / self.domains

        personal_positions = positions.copy()
        personal_scores = np.full(swarm_size, -np.inf)
        global_position = None
        global_score = -np.inf
        generation = 0

        while self.evatime < budget:
            evaluated = min(swarm_size, budget - self.evatime)
            for i in range(evaluated):
                score = self.score(P, self._candidate(positions[i]))
                if score > personal_scores[i]:
                    personal_scores[i] = score
                    personal_positions[i] = positions[i].copy()
                if global_position is None or score > global_score:
                    global_score = score
                    global_position = positions[i].copy()
                self.history[self.evatime - 1] = self.fitness

            self.on_iteration_finish(
                problem=P, state=self.best_state, iteration=generation, run=0,
                best_value=self.best_objectives.copy(), fitness=self.fitness,
            )
            generation += 1
            if self.evatime >= budget:
                break

            r1 = self.rng.random(positions.shape)
            r2 = self.rng.random(positions.shape)
            velocities = np.clip(
                self.w * velocities
                + self.c1 * r1 * (personal_positions - positions)
                + self.c2 * r2 * (global_position - positions),
                -self.vmax, self.vmax,
            )
            positions = np.clip(positions + velocities, 0.0, 1.0)

        return self.best_state.copy()
