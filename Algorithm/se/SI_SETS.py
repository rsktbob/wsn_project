"""SI-SETS：分區與 SA-SETS 相同，但每個 searcher 只和自己選到的區域商品投資。

SE 三個家族的第二個。沒去的區域沿用上次觀察到的投資品質；投資由
``PooledEvaluation`` 的 h 個無狀態子程序平行評估。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.market_components import PooledEvaluation, beta_cdf
from Algorithm.se.SA_SETS import SA_SETS


class SI_SETS(PooledEvaluation, SA_SETS):
    """Choose a goods pool first, then invest only in that pool.

    Each region owns a persistent goods pool and the same identity-sensor
    partition used by SA-SETS.  The most recent observed investment quality
    is kept for searcher/region pairs that are not evaluated in the current
    round.
    """

    def __init__(self, problem, n=5, h=4, w=8, mu=0.2, seed=None):
        # Reuse SA-SETS initialization, partition encoding, crossover and
        # mutation verbatim.  SI-SETS only changes which pools are invested in.
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        self.goods = []
        self._pool = None
        self.goods_fitness = np.empty((self.h, self.w), dtype=float)
        self.investment_quality = np.empty((self.h, self.n), dtype=float)

    def initialize_market(self, problem, initial_state=None):
        """Initialize searchers, goods pools, and stale-value estimates."""
        # This inherited call performs exactly the same searcher initialization
        # and region assignment as SA-SETS.
        super().initialize_market(problem, initial_state)
        self.goods = [
            [self.create_candidate(problem, region=region) for _ in range(self.w)]
            for region in range(self.h)
        ]
        flat_goods = [good for region_goods in self.goods for good in region_goods]
        self.goods_fitness = self.evaluate_many(problem, flat_goods).reshape(
            self.h,
            self.w,
        )

        # Before a searcher has visited a pool, its estimated investment value
        # starts from that pool's existing average goods quality.
        initial_quality = np.mean(self.goods_fitness, axis=1)
        self.investment_quality = np.repeat(
            initial_quality[:, None],
            self.n,
            axis=1,
        )

    def vision_search(self, problem):
        """Invest only in each searcher's already-selected goods pool."""
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        self.current_adaptive_step = self.adaptive_step * (1.0 - progress)

        active_regions = self.selected_regions.copy()
        goods_fitness_before = self.goods_fitness.copy()
        investments = [[None for _ in range(self.w)] for _ in range(self.n)]
        investment_fitness = np.empty((self.n, self.w), dtype=float)

        for searcher_id, region in enumerate(active_regions):
            region = int(region)
            candidates = [
                self.align_region(
                    problem,
                    self.invest(
                        problem,
                        self.searchers[searcher_id],
                        self.goods[region][good_id],
                    ),
                    region,
                )
                for good_id in range(self.w)
            ]
            investments[searcher_id] = candidates
        investment_fitness = self.evaluate_investments(problem, investments)
        for searcher_id, region in enumerate(active_regions):
            scores = investment_fitness[searcher_id]
            self.investment_quality[region, searcher_id] = float(
                np.mean(scores)
            )

        # Choose the next region with the same tournament and Beta rule as
        # SA-SETS.  Unvisited-region qualities are necessarily cached because
        # SI-SETS deliberately did not evaluate those investments this round.
        probabilities = self.region_probabilities(
            goods_fitness_before,
            self.investment_quality,
        )
        selected = self.select_regions(probabilities)

        # Match SA-SETS' update order: after region selection, compare the
        # searcher with the best pre-update good in that selected region.
        for searcher_id, region in enumerate(selected):
            region = int(region)
            good_id = int(np.argmax(goods_fitness_before[region]))
            good_score = float(goods_fitness_before[region, good_id])
            if good_score > self.searcher_fitness[searcher_id]:
                self.searchers[searcher_id] = self.goods[region][good_id].copy()
                self.searcher_fitness[searcher_id] = good_score

        # Preserve the current goods rule: each visited good is replaced by
        # the best investment made for that good during this round.
        for region in range(self.h):
            visitors = np.flatnonzero(active_regions == region)
            if not len(visitors):
                continue
            for good_id in range(self.w):
                best_searcher = int(
                    max(
                        visitors,
                        key=lambda searcher_id: investment_fitness[
                            int(searcher_id), good_id
                        ],
                    )
                )
                self.goods[region][good_id] = investments[best_searcher][
                    good_id
                ].copy()
                self.goods_fitness[region, good_id] = investment_fitness[
                    best_searcher, good_id
                ]

        self.selected_regions = selected

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
