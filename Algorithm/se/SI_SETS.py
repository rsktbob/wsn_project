"""Single-offspring, selected-region SETS; no inheritance from SA_SETS."""
import numpy as np
from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.population_updates import goods_from_children
from Algorithm.se.sensor_operators import SensorOperators, AdaptiveSensorInitialization
from Algorithm.se.market_components import (
    beta_cdf, PooledEvaluation, AdaptiveBetaMemory, create_selected_investments,
)


class SI_SETS(AdaptiveSensorInitialization, SensorOperators, PooledEvaluation, AdaptiveBetaMemory, BaseSE):
    def __init__(self, problem, n=5, h=4, w=8, mu=0.2, seed=None):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        self._pool = None
        self.investment_quality = np.empty((self.h, self.n), dtype=float)

    def initialize_market(self, problem, initial_state=None):
        super().initialize_market(problem, initial_state)
        self.goods = [[self.create_candidate(problem, region=region)
                       for _ in range(self.w)] for region in range(self.h)]
        flat = [good for group in self.goods for good in group]
        self.goods_fitness = self.evaluate_many(problem, flat).reshape(self.h, self.w)
        self.investment_quality = np.repeat(
            np.mean(self.goods_fitness, axis=1)[:, None], self.n, axis=1)

    def make_offspring(self, problem, searcher, good, region):
        return (self.align_region(problem, self.invest(problem, searcher, good), region),)

    def goods_for_region(self, region, goods):
        return goods[int(region)]

    def create_investments(self, problem, goods, active_regions=None):
        return create_selected_investments(self, problem, goods, active_regions)

    def summarize_round(self, active_regions, scores):
        for searcher_id, region in enumerate(active_regions):
            self.investment_quality[int(region), searcher_id] = float(
                np.mean(scores[searcher_id]))
        return self.investment_quality

    def update_goods(self, children, scores, active_regions):
        for region in range(self.h):
            goods_from_children(self.goods[region], self.goods_fitness[region],
                                children, scores, np.flatnonzero(active_regions == region))

    def region_probabilities(self, goods_fitness, investment_quality):
        """Use current observations and cached values to score every pool."""
        average_goods = np.mean(goods_fitness, axis=1)
        best_goods = np.max(goods_fitness, axis=1)
        total = float(np.sum(average_goods))
        if abs(total) <= np.finfo(float).eps:
            region_share = np.full(self.h, 1.0 / self.h)
        else:
            region_share = average_goods / total

        expected_value = (
            best_goods[:, None]
            * investment_quality
            * region_share[:, None]
        )
        probability = np.empty((self.h, self.n), dtype=float)
        for region in range(self.h):
            for searcher_id in range(self.n):
                probability[region, searcher_id] = beta_cdf(
                    self.ta[region],
                    self.tb[region],
                    expected_value[region, searcher_id],
                )
        return probability

__all__ = ["SI_SETS"]
