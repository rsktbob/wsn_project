"""Contextual Thompson sampling fused with SA-SETS' immediate I-times-G value."""

from __future__ import annotations

import numpy as np

from Algorithm.se.RegionSelectionSA_SETS import (
    RegionSelectionSA_SETS,
    normalized_region_features,
)


class ContextualThompsonIG_SA_SETS(RegionSelectionSA_SETS):
    """Allocate SA-SETS regions using immediate value and Bayesian context.

    ``immediate_weight`` preserves the original SA-SETS signal
    ``best_good * mean_investment * region_share``.  The complementary term
    is a Thompson sample from a per-region Bayesian linear reward model.
    """

    BASE_FEATURE_COUNT = 5
    FEATURE_COUNT = 7
    IMPROVEMENT_TOLERANCE = 1e-12

    def __init__(
        self, problem, n=8, h=4, w=2, mu=0.4, *, immediate_weight=0.5,
        ridge=1.0, reward_noise=0.10, seed=None,
    ):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        self.immediate_weight = float(immediate_weight)
        self.context_ridge = float(ridge)
        self.reward_noise = float(reward_noise)
        if not 0.0 <= self.immediate_weight <= 1.0:
            raise ValueError("immediate_weight must be in [0, 1]")
        if self.context_ridge <= 0.0 or self.reward_noise <= 0.0:
            raise ValueError("ridge and reward_noise must be positive")
        self.name = f"ContextualThompsonIG_SA_SETS_{self.n}_{self.h}_{self.w}_{mu}"
        self.context_a = np.empty((self.h, self.FEATURE_COUNT, self.FEATURE_COUNT))
        self.context_b = np.zeros((self.h, self.FEATURE_COUNT))
        self._contexts = None
        self.immediate_values = np.zeros((self.h, self.n))
        self.contextual_samples = np.zeros((self.h, self.n))

    def _reset_region_policy(self):
        self.context_a[:] = np.eye(self.FEATURE_COUNT) * self.context_ridge
        self.context_b.fill(0.0)
        self._contexts = None
        self.immediate_values.fill(0.0)
        self.contextual_samples.fill(0.0)

    @staticmethod
    def _normalize_by_searcher(values):
        values = np.asarray(values, dtype=float)
        lower = np.min(values, axis=0, keepdims=True)
        span = np.max(values, axis=0, keepdims=True) - lower
        normalized = np.full_like(values, 0.5, dtype=float)
        valid = span[0] > 1e-12
        normalized[:, valid] = (
            (values[:, valid] - lower[:, valid]) / span[:, valid]
        )
        return np.clip(normalized, 0.0, 1.0)

    def region_probabilities(self, goods_fitness, investment_fitness):
        average_investment = np.mean(investment_fitness, axis=2)
        average_goods = np.mean(goods_fitness, axis=1)
        best_goods = np.max(goods_fitness, axis=1)
        total_goods = float(np.sum(average_goods))
        if abs(total_goods) <= 1e-12:
            region_share = np.full(self.h, 1.0 / self.h)
        else:
            region_share = average_goods / total_goods
        immediate_raw = (
            best_goods[:, None]
            * average_investment
            * region_share[:, None]
        )
        immediate = self._normalize_by_searcher(immediate_raw)

        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        base = normalized_region_features(
            goods_fitness, self.searcher_fitness, progress
        )
        contexts = np.empty((self.h, self.n, self.FEATURE_COUNT), dtype=float)
        contexts[:, :, : self.BASE_FEATURE_COUNT] = base
        contexts[:, :, self.BASE_FEATURE_COUNT] = self._normalize_by_searcher(
            average_investment
        )
        contexts[:, :, self.BASE_FEATURE_COUNT + 1] = immediate

        samples = np.empty((self.h, self.n), dtype=float)
        for region in range(self.h):
            inverse = np.linalg.inv(self.context_a[region])
            mean = inverse @ self.context_b[region]
            theta = self.rng.multivariate_normal(
                mean, self.reward_noise * inverse
            )
            samples[region] = contexts[region] @ theta

        contextual = self._normalize_by_searcher(samples)
        self._contexts = contexts
        self.immediate_values = immediate
        self.contextual_samples = contextual
        return (
            self.immediate_weight * immediate
            + (1.0 - self.immediate_weight) * contextual
        )

    def on_market_round_finish(
        self,
        problem,
        goods,
        goods_fitness,
        investment_fitness,
        selected,
    ):
        """Learn whether a context actually improves its pre-selection parent."""
        del problem, goods, goods_fitness
        best_child = np.max(np.asarray(investment_fitness, dtype=float), axis=2)
        gains = best_child - self._searcher_fitness_before_selection[None, :]
        rewards = self._normalize_by_searcher(gains)
        self._update_region_policy(np.asarray(selected, dtype=int), rewards)

    def _update_region_policy(self, selected, rewards):
        if self._contexts is None:
            return
        for searcher_id, region in enumerate(selected):
            context = self._contexts[region, searcher_id]
            self.context_a[region] += np.outer(context, context)
            self.context_b[region] += rewards[region, searcher_id] * context


__all__ = ["ContextualThompsonIG_SA_SETS"]
