"""Bernoulli Thompson-sampling region allocation for SA-SETS."""

from __future__ import annotations

import numpy as np

from Algorithm.se.RegionSelectionSA_SETS import RegionSelectionSA_SETS


class Thompson_SA_SETS(RegionSelectionSA_SETS):
    """Treat a region as successful when it improves its visiting searcher.

    Each searcher keeps an independent Beta posterior for every SA-SETS
    region.  The policy samples from those posteriors, rather than applying a
    Beta CDF to a hand-built region score as the original algorithm does.
    """

    IMPROVEMENT_TOLERANCE = 1e-12

    def __init__(self, problem, n=8, h=4, w=2, mu=0.4, seed=None):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        self.name = f"Thompson_SA_SETS_{self.n}_{self.h}_{self.w}_{mu}"
        self.thompson_alpha = np.ones((self.h, self.n), dtype=float)
        self.thompson_beta = np.ones((self.h, self.n), dtype=float)
        self.thompson_samples = np.full((self.h, self.n), 0.5)

    def _reset_region_policy(self):
        self.thompson_alpha.fill(1.0)
        self.thompson_beta.fill(1.0)
        self.thompson_samples.fill(0.5)

    def region_probabilities(self, goods_fitness, investment_fitness):
        del goods_fitness, investment_fitness
        # NumPy's generator is seeded independently from SA-SETS' own random
        # stream, preserving deterministic crossover and mutation sequences.
        samples = self.rng.beta(self.thompson_alpha, self.thompson_beta)
        self.thompson_samples = np.asarray(samples, dtype=float)
        return self.thompson_samples.copy()

    def _update_region_policy(self, selected, rewards):
        del rewards
        previous = self._searcher_fitness_before_selection
        # ``investment_fitness`` is not retained by the common callback, so
        # success is supplied through the temporary matrix built below.
        successes = self._thompson_successes
        for searcher_id, region in enumerate(selected):
            if successes[region, searcher_id]:
                self.thompson_alpha[region, searcher_id] += 1.0
            else:
                self.thompson_beta[region, searcher_id] += 1.0
        del previous

    def on_market_round_finish(
        self,
        problem,
        goods,
        goods_fitness,
        investment_fitness,
        selected,
    ):
        del problem, goods, goods_fitness
        best_investment = np.max(
            np.asarray(investment_fitness, dtype=float), axis=2
        )
        parent = self._searcher_fitness_before_selection
        self._thompson_successes = (
            best_investment > parent[None, :] + self.IMPROVEMENT_TOLERANCE
        )
        self._update_region_policy(np.asarray(selected, dtype=int), None)


__all__ = ["Thompson_SA_SETS"]
