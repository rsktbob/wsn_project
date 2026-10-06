"""SI-SETS：分區與 SA-SETS 相同，但每個 searcher 只和自己選到的區域商品投資。

SE 三個家族的第二個。每回合流程和 SA_SETS 不同（先選區再投資、只評估
n × w 筆交易），因此直接繼承 :class:`BaseSE`，不掛在 SA_SETS 下面；
分區、初始解、交配突變透過和 SA_SETS 相同的 mixin 共用。

每組 searcher-good 交易產生兩個子代（原 SI_SETSv2 的角色分離菁英更新）：

* ``child1`` 以 searcher 為底，只能取代該 searcher，且必須嚴格變好。
* ``child2`` 以 good 為底，造訪該區的 searcher 中最好的一個 child2
  只有嚴格變好時才取代該 good。
* 區域吸引力只看 child1；沒去的區域沿用上次觀察到的投資品質。

投資由 ``PooledEvaluation`` 的 h 個無狀態子程序平行評估。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.market_components import (
    AdaptiveBetaMemory,
    PooledEvaluation,
    accept_good_children,
    accept_searcher_children,
)
from Algorithm.se.sensor_operators import (
    CCSInitialization,
    NearBaseIdentitySensors,
    SensorOperators,
)


class SI_SETS(
    NearBaseIdentitySensors,
    CCSInitialization,
    SensorOperators,
    PooledEvaluation,
    AdaptiveBetaMemory,
    BaseSE,
):
    """Choose a goods pool first, then trade only with that pool.

    Each region owns a persistent goods pool and the same identity-sensor
    partition used by SA-SETS.  The most recent observed child1 quality is
    kept for searcher/region pairs that are not evaluated in the current
    round.
    """

    def __init__(self, problem, n=8, h=4, w=8, mu=0.4, seed=None):
        # Same partition encoding, initialization, crossover and mutation as
        # SA-SETS (shared mixins); SI-SETS only changes the market round.
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        self.goods = []
        self.goods_fitness = np.empty((self.h, self.w), dtype=float)
        self.investment_quality = np.empty((self.h, self.n), dtype=float)

    def initialize_market(self, problem, initial_state=None):
        """Initialize searchers, goods pools, and stale-value estimates."""
        # Same identity-sensor choice, searcher initialization and region
        # assignment as SA-SETS.
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

    def make_children(self, problem, searcher, good, region=None):
        """Crossover both directions, then mutate each child with rate mu."""
        child1, child2 = self.crossover(searcher, good)

        for child in (child1, child2):
            if self.random.random() < self.mutation_rate:
                self.mutate_candidate(problem, child)
        return child1, child2

    def vision_search(self, problem):
        """Trade with the selected pool and apply the role-separated update."""
        self.update_adaptive_step()

        active_regions = self.selected_regions.copy()
        goods_fitness_before = self.goods_fitness.copy()
        searcher_fitness_before = self.searcher_fitness.copy()
        children = []

        for searcher_id, region in enumerate(active_regions):
            region = int(region)
            row = []
            for good_id in range(self.w):
                child1, child2 = self.make_children(
                    problem,
                    self.searchers[searcher_id],
                    self.goods[region][good_id],
                    region=region,
                )
                row.append(self.align_region(problem, child1, region))
                row.append(self.align_region(problem, child2, region))
            children.append(row)

        child_fitness = self.evaluate_investments(problem, children)
        child1_fitness = child_fitness[:, 0::2]
        child2_fitness = child_fitness[:, 1::2]

        # Region attractiveness measures how well child1 improves the
        # visiting searcher side of the interaction.
        for searcher_id, region in enumerate(active_regions):
            self.investment_quality[int(region), searcher_id] = float(
                np.mean(child1_fitness[searcher_id])
            )

        probabilities = self.region_probabilities(
            goods_fitness_before,
            self.investment_quality,
        )
        selected = self.select_regions(probabilities)

        # A searcher considers only its own child1 proposals and remains
        # unchanged unless the best proposal improves its previous fitness.
        accept_searcher_children(
            self, children, child1_fitness, searcher_fitness_before
        )

        # A visited good accepts the best child2 only when it improves the
        # pre-investment good; otherwise the old good remains in the market.
        for region in range(self.h):
            accept_good_children(
                self.goods[region],
                self.goods_fitness[region],
                goods_fitness_before[region],
                children,
                child2_fitness,
                np.flatnonzero(active_regions == region),
                self.IMPROVEMENT_TOLERANCE,
            )

        self.selected_regions = selected

    def region_probabilities(self, goods_fitness, investment_quality):
        """Use current observations and cached values to score every pool."""
        return self.market_probabilities(goods_fitness, investment_quality)


__all__ = ["SI_SETS"]
