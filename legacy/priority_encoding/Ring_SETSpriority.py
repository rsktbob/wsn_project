"""Ring_SETS' market with the EF PriorityEncoding.

The market, regions and operators are exactly those of :class:`Ring_SETS`:
a region only decides which ring segment a searcher's crossover/mutation may
touch this round (investment behaviour), not a separate part of the solution
space. Only the chromosome changes: each sensor carries two priority genes
``[a, b]`` in ``[0, RANK_PRECISION)`` instead of ``[level, priority]``.
Sensing levels are chosen greedily by the decoder.

EF = energy-only activation order + energy-plus-farness routing order
(2026-10-12 lifetime experiments, tmp/route_dir and tmp/open_dir): routing
far sensors first beat C4's near-first routing by 2.8–4.9 % at every budget,
while activation was indifferent to distance, so it uses energy alone.
C4 = cost-based activation order + energy-based routing order, the earlier
GA-tuned setting; it stays defined for LinUCB_Ring_SETSpriority and old runs
(the RL variants moved to legacy/rl_ring/ on 2026-10-13).
"""

from __future__ import annotations

from Algorithm.se.Ring_SETS import Ring_SETS
from Algorithm.se.sensor_operators import draw_mutation_count
from State.PriorityEncoding import PriorityEncoding


class PriorityEncodingC4(PriorityEncoding):
    """PriorityEncoding with the C4 activation/routing bases."""

    ACTIVATION_BASE = "cost"
    ROUTING_BASE = "energy"


class PriorityEncodingEF(PriorityEncoding):
    """PriorityEncoding with energy activation and energy + farness routing.

    activation base = E_s / E_max；routing base = E_s / E_max + d_s / d_max。
    """

    ACTIVATION_BASE = "energy"
    ROUTING_BASE = "far"


class Ring_SETSpriority(Ring_SETS):
    """Ring_SETS searching over priority chromosomes (EF by default).

    The two genes of a sensor stay adjacent, so the ring-segment crossover
    (sensor span × 2) swaps whole sensors exactly as in Ring_SETS.
    """

    ENCODING_TYPE = PriorityEncodingEF
    # 路由基因突變顆數 = 開啟基因突變顆數 × 3
    ROUTING_MUTATION_MULTIPLIER = 3

    def create_candidate(self, problem, region=None):
        """Uniform random priorities; the decoder already reaches full coverage."""
        return self.ENCODING_TYPE.random(problem.SENSOR_NUMBER, rng=self.rng)

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
                self.ENCODING_TYPE.RANK_PRECISION
            )
        scope = range(left_sensor, right_sensor)
        routing_count = min(self.ROUTING_MUTATION_MULTIPLIER * mutation_count, len(scope))
        for sensor_id in self.random.sample(scope, routing_count):
            candidate.code[sensor_id * 2 + 1] = self.random.randrange(
                self.ENCODING_TYPE.RANK_PRECISION
            )
        return candidate


__all__ = ["PriorityEncodingC4", "PriorityEncodingEF", "Ring_SETSpriority"]
