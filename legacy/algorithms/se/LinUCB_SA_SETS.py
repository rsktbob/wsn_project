"""Contextual-LinUCB region allocation for SA-SETS."""

from __future__ import annotations

import numpy as np

from Algorithm.se.RegionSelectionSA_SETS import (
    RegionSelectionSA_SETS,
    normalized_region_features,
)


class LinUCB_SA_SETS(RegionSelectionSA_SETS):
    """Use current market context to choose SA-SETS regions online."""

    FEATURE_COUNT = 5

    def __init__(
        self, problem, n=8, h=4, w=2, mu=0.4, *, alpha=0.5,
        ridge=1.0, seed=None,
    ):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        self.linucb_alpha = float(alpha)
        self.linucb_ridge = float(ridge)
        if self.linucb_alpha < 0.0 or self.linucb_ridge <= 0.0:
            raise ValueError("alpha must be non-negative and ridge positive")
        self.name = f"LinUCB_SA_SETS_{self.n}_{self.h}_{self.w}_{mu}"
        self.linucb_a = np.empty((self.h, self.FEATURE_COUNT, self.FEATURE_COUNT))
        self.linucb_b = np.zeros((self.h, self.FEATURE_COUNT))
        self._linucb_context = None

    def _reset_region_policy(self):
        self.linucb_a[:] = np.eye(self.FEATURE_COUNT) * self.linucb_ridge
        self.linucb_b.fill(0.0)
        self._linucb_context = None

    def region_probabilities(self, goods_fitness, investment_fitness):
        del investment_fitness
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        contexts = normalized_region_features(
            goods_fitness, self.searcher_fitness, progress
        )
        scores = np.empty((self.h, self.n), dtype=float)
        for region in range(self.h):
            inverse = np.linalg.inv(self.linucb_a[region])
            theta = inverse @ self.linucb_b[region]
            for searcher_id in range(self.n):
                context = contexts[region, searcher_id]
                uncertainty = float(np.sqrt(context @ inverse @ context))
                scores[region, searcher_id] = float(
                    context @ theta + self.linucb_alpha * uncertainty
                )
        self._linucb_context = contexts
        return scores

    def _update_region_policy(self, selected, rewards):
        if self._linucb_context is None:
            return
        for searcher_id, region in enumerate(selected):
            context = self._linucb_context[region, searcher_id]
            self.linucb_a[region] += np.outer(context, context)
            self.linucb_b[region] += rewards[region, searcher_id] * context


__all__ = ["LinUCB_SA_SETS"]

