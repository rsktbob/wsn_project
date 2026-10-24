"""LinUCB-Ring-SEpri：LinUCB_Ring_SE 的流程，染色體換成 PriorityEncoding。

初始解與突變和 Ring_SEpri 相同（2026-10-13 起用同一個 PriorityEncoding）。
"""

from __future__ import annotations

from Algorithm.se.ring_se.LinUCB_Ring_SE import LinUCB_Ring_SE
from State.PriorityEncoding import PriorityEncoding
from State.initialization import random_priority_coding


class LinUCB_Ring_SEpri(LinUCB_Ring_SE):
    """LinUCB_Ring_SE searching over priority chromosomes."""

    # 路由基因突變顆數 = 開啟基因突變顆數 × 3
    ROUTING_MUTATION_MULTIPLIER = 3

    def create_candidate(self, problem, region=None):
        """Uniform random priorities; the decoder already reaches full coverage."""
        return random_priority_coding(problem, self.rng)

    def mutate_candidate(self, problem, candidate, sensor_span):
        """Redraw k sensors' activation genes and 3k sensors' routing genes.

        k is one to three; both gene kinds always mutate. The 3k routing
        sensors are drawn without replacement from the same ring scope (all of
        it when the scope is smaller).
        """
        _, left_sensor, right_sensor = sensor_span
        mutation_count = self.random.choice(self.MUTATION_COUNTS)
        for _ in range(mutation_count):
            sensor_id = self.random.randrange(left_sensor, right_sensor)
            candidate.code[sensor_id * 2] = self.random.randrange(
                PriorityEncoding.RANK_PRECISION
            )
        scope = range(left_sensor, right_sensor)
        routing_count = min(self.ROUTING_MUTATION_MULTIPLIER * mutation_count, len(scope))
        for sensor_id in self.random.sample(scope, routing_count):
            candidate.code[sensor_id * 2 + 1] = self.random.randrange(
                PriorityEncoding.RANK_PRECISION
            )
        return candidate


__all__ = ["LinUCB_Ring_SEpri"]
