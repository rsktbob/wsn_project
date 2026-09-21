"""Shared online region-selection policies for the SA-SETS market."""

from __future__ import annotations

import numpy as np

from Algorithm.se.SA_SETS import SA_SETS


class RegionSelectionSA_SETS(SA_SETS):
    """SA-SETS base class whose policy learns from selected-region feedback.

    The original market already evaluates every region/searcher/good investment.
    Consequently these policies do *not* claim to reduce evaluations: they only
    replace the rule used to select the region for the next market round.
    """

    REWARD_EPSILON = 1e-12

    def vision_search(self, problem):
        """Retain parent fitness so policies can credit real improvement."""
        self._searcher_fitness_before_selection = np.asarray(
            self.searcher_fitness, dtype=float
        ).copy()
        super().vision_search(problem)

    def initialize_market(self, problem, initial_state=None):
        super().initialize_market(problem, initial_state)
        self._reset_region_policy()

    def _reset_region_policy(self):
        """Reset one search episode's policy state."""
        raise NotImplementedError

    @staticmethod
    def normalized_investment_quality(investment_fitness):
        """Map each region/searcher mean investment to a stable [0, 1] reward."""
        quality = np.mean(np.asarray(investment_fitness, dtype=float), axis=2)
        finite = quality[np.isfinite(quality)]
        if not len(finite):
            return np.full_like(quality, 0.5, dtype=float)
        lower = float(np.min(finite))
        span = float(np.max(finite) - lower)
        if span <= RegionSelectionSA_SETS.REWARD_EPSILON:
            return np.full_like(quality, 0.5, dtype=float)
        return np.clip((quality - lower) / span, 0.0, 1.0)

    def on_market_round_finish(
        self,
        problem,
        goods,
        goods_fitness,
        investment_fitness,
        selected,
    ):
        """Credit only the selected region, using fitness already evaluated."""
        del problem, goods, goods_fitness
        rewards = self.normalized_investment_quality(investment_fitness)
        self._update_region_policy(np.asarray(selected, dtype=int), rewards)

    def _update_region_policy(self, selected, rewards):
        raise NotImplementedError


def normalized_region_features(goods_fitness, searcher_fitness, progress):
    """Return compact, bounded LinUCB context vectors for region choices."""
    goods_fitness = np.asarray(goods_fitness, dtype=float)
    searcher_fitness = np.asarray(searcher_fitness, dtype=float)
    h = goods_fitness.shape[0]
    n = len(searcher_fitness)

    goods_mean = np.mean(goods_fitness, axis=1)
    goods_best = np.max(goods_fitness, axis=1)
    all_values = np.concatenate((goods_mean, goods_best, searcher_fitness))
    lower = float(np.min(all_values))
    span = float(np.max(all_values) - lower)
    if span <= RegionSelectionSA_SETS.REWARD_EPSILON:
        goods_mean = np.full(h, 0.5)
        goods_best = np.full(h, 0.5)
        searcher = np.full(n, 0.5)
    else:
        goods_mean = np.clip((goods_mean - lower) / span, 0.0, 1.0)
        goods_best = np.clip((goods_best - lower) / span, 0.0, 1.0)
        searcher = np.clip((searcher_fitness - lower) / span, 0.0, 1.0)

    features = np.empty((h, n, 5), dtype=float)
    for region in range(h):
        for searcher_id in range(n):
            features[region, searcher_id] = (
                1.0,
                float(progress),
                float(searcher[searcher_id]),
                float(goods_mean[region]),
                float(goods_best[region]),
            )
    return features
