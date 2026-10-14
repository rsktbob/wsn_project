"""Ring_SE' market with the PriorityEncoding chromosome.

The market, regions and operators are exactly those of :class:`Ring_SE`:
a region only decides which ring segment a searcher's crossover/mutation may
touch this round (investment behaviour), not a separate part of the solution
space. Only the chromosome changes: each sensor carries two priority genes
(energy activation order + energy-plus-farness routing order), and the decoder
chooses sensing levels greedily (see :class:`State.PriorityEncoding.PriorityEncoding`).

Other priority bases moved to ``legacy/priority_encoding/`` on 2026-10-13.
"""

from __future__ import annotations

from Algorithm.se.ring_se.Ring_SE import Ring_SE
from Algorithm.se.sensor_operators import draw_mutation_count
from State.PriorityEncoding import PriorityEncoding


class Ring_SEpri(Ring_SE):
    """Ring_SE searching over priority chromosomes.

    The two genes of a sensor stay adjacent, so the ring-segment crossover
    (sensor span × 2) swaps whole sensors exactly as in Ring_SE.
    """

    # 路由基因突變顆數 = 開啟基因突變顆數 × 3
    ROUTING_MUTATION_MULTIPLIER = 3

    def create_candidate(self, problem, region=None):
        """Uniform random priorities; the decoder already reaches full coverage."""
        return PriorityEncoding.random(problem.SENSOR_NUMBER, rng=self.rng)

    def mutate_candidate(self, problem, candidate, sensor_span):
        """Redraw k sensors' activation genes and 3k sensors' routing genes.

        k is one to three; both gene kinds always mutate. The 3k routing
        sensors are drawn without replacement from the same ring scope (all of
        it when the scope is smaller).
        """
        _, left_sensor, right_sensor = sensor_span
        mutation_count = draw_mutation_count(self.random)
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


__all__ = ["Ring_SEpri"]
