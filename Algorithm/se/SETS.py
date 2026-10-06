"""SETS：以 identity sensor 分區、跟所有區域商品投資的原始 Search Economics。

SE 三個家族的第一個（另外兩個是 SI_SETS 與 Ring_SETS）：

* 分區：取 log2(h) 顆 identity sensor，其開關組合對應 h 個區域，
  所有候選解都會對齊到自己區域的開關組合。
* 投資：每個 searcher 每回合都和所有區域的所有商品投資（h × n × w），
  由區域的 worker 平行評估（``ParallelSEMarket``）。
* 區域選擇：Beta 記憶加上市場期望值，以錦標賽選出下一回合的區域。

SA_SETS 繼承本類別，只改 identity sensor 的選法與初始解。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.market_components import AdaptiveBetaMemory
from Algorithm.se.sensor_operators import SensorOperators


class SETS(SensorOperators, AdaptiveBetaMemory, BaseSE):
    """Search Economics with identity-sensor regions and full investment.

    ``BaseSE`` owns the four stable SE phases.  ``SensorOperators`` owns the
    SETS region definition, sensor chromosome and transition operators;
    ``AdaptiveBetaMemory`` owns the Beta-based region probabilities.
    """

    def vision_search(self, problem):
        """Update the adaptive learning step before the normal SE search."""
        self.update_adaptive_step()
        super().vision_search(problem)

    def region_probabilities(self, goods_fitness, investment_fitness):
        """Calculate the SA-SETS Beta probability for each region/searcher."""
        return self.market_probabilities(
            goods_fitness,
            np.mean(investment_fitness, axis=2),
        )


__all__ = ["SETS"]
