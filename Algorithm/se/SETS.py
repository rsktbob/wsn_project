"""Identity-sensor SETS with the full-investment BaseSE flow."""
import numpy as np
from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.sensor_operators import SensorOperators
from Algorithm.se.market_components import AdaptiveBetaMemory, beta_cdf


class SETS(SensorOperators, AdaptiveBetaMemory, BaseSE):
    def region_probabilities(self, goods_fitness, investment_fitness):
        """Calculate the SA-SETS Beta probability for each region/searcher."""
        average_investment = np.mean(investment_fitness, axis=2)
        average_goods = np.mean(goods_fitness, axis=1)
        best_goods = np.max(goods_fitness, axis=1)
        total = float(np.sum(average_goods))
        if abs(total) <= np.finfo(float).eps:
            region_share = np.full(self.h, 1.0 / self.h)
        else:
            region_share = average_goods / total

        expected_value = (
            best_goods[:, None]
            * average_investment
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



__all__ = ["SETS"]
