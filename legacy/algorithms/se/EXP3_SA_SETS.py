"""EXP3-based region allocation for SA-SETS."""

from __future__ import annotations

import numpy as np

from Algorithm.se.RegionSelectionSA_SETS import RegionSelectionSA_SETS


class EXP3_SA_SETS(RegionSelectionSA_SETS):
    """Sample SA-SETS regions with an adversarial-bandit EXP3 policy."""

    def __init__(self, problem, n=8, h=4, w=2, mu=0.4, *, gamma=0.08, seed=None):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        self.exp3_gamma = float(gamma)
        if not 0.0 < self.exp3_gamma <= 1.0:
            raise ValueError("gamma must be in (0, 1]")
        self.name = f"EXP3_SA_SETS_{self.n}_{self.h}_{self.w}_{mu}"
        self.exp3_weights = np.ones((self.h, self.n), dtype=float)
        self.exp3_probabilities = np.full((self.h, self.n), 1.0 / self.h)

    def _reset_region_policy(self):
        self.exp3_weights.fill(1.0)
        self.exp3_probabilities.fill(1.0 / self.h)

    def region_probabilities(self, goods_fitness, investment_fitness):
        del goods_fitness, investment_fitness
        totals = np.sum(self.exp3_weights, axis=0, keepdims=True)
        exploitation = self.exp3_weights / totals
        self.exp3_probabilities = (
            (1.0 - self.exp3_gamma) * exploitation
            + self.exp3_gamma / self.h
        )
        return self.exp3_probabilities.copy()

    def select_regions(self, probabilities):
        selected = np.empty(self.n, dtype=int)
        for searcher_id in range(self.n):
            selected[searcher_id] = self.random.choices(
                range(self.h), weights=probabilities[:, searcher_id], k=1
            )[0]
        return selected

    def _update_region_policy(self, selected, rewards):
        for searcher_id, region in enumerate(selected):
            probability = max(
                float(self.exp3_probabilities[region, searcher_id]), 1e-12
            )
            estimated_reward = float(rewards[region, searcher_id]) / probability
            exponent = np.clip(
                self.exp3_gamma * estimated_reward / self.h, -50.0, 50.0
            )
            self.exp3_weights[region, searcher_id] *= float(np.exp(exponent))
            self.exp3_weights[:, searcher_id] /= np.max(
                self.exp3_weights[:, searcher_id]
            )


__all__ = ["EXP3_SA_SETS"]

