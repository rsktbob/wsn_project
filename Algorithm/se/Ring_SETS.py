"""Ring-SETS：沒有固定分區，區域只決定交配與突變能動的 sensor 範圍。

SE 三個家族的第三個（另外兩個是 SETS 與 SI_SETS）：

* 分區：sensor 依環狀順序（ring-major）切成 ``h`` 段連續範圍。區域不綁定
  任何解的特徵，只決定這一回合的投資範圍：80% 只動該段，20% 動整段
  再加上相鄰段的一部分。突變使用同一個範圍。
* 投資：所有 searcher 每回合都和同一組共用的 ``w`` 個商品交換。
* 更新：角色分離的菁英更新。``child1`` 以 searcher 為底，只能取代
  該 searcher；``child2`` 以 good 為底，只有最好的一個能取代該 good，
  兩者都必須嚴格變好。
* 區域選擇：只看最近造訪該段時 ``child1`` 的平均表現（Beta CDF 修正）。

``region`` 沿用 ``self.h``、``self.selected_regions`` 等名稱，以配合 SE 共用流程。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.population_updates import searchers_from_children, goods_from_children
from Algorithm.se.market_components import (
    beta_cdf, PooledEvaluation, AdaptiveBetaMemory, create_selected_investments,
)
from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding


class Ring_SETS(PooledEvaluation, AdaptiveBetaMemory, BaseSE):
    """SE with ring-segment operator scopes and one shared elitist goods pool.

    For ``h=4`` and 100 ring-ordered sensors, segments are ``[0:25)``,
    ``[25:50)``, ``[50:75)``, and ``[75:100)``. On every visit, both
    directions of crossover are retained:

    * ``child1`` is searcher ``a`` with the selected segment from good ``b``;
      only ``a`` may accept it.
    * ``child2`` is good ``b`` with the selected segment from searcher ``a``;
      only ``b`` may accept it.

    Both updates are elitist. Mutation follows the same chosen sensor span.
    Unlike SETS, ``h`` need not be a power of two.
    """

    LOCAL_OPERATOR_PROBABILITY = 0.80
    IMPROVEMENT_TOLERANCE = 1e-12

    def __init__(self, problem, n=8, h=4, w=2, mu=0.4, seed=None):
        super().__init__(
            problem,
            n=n,
            h=h,
            w=w,
            mu=mu,
            code_length=problem.SENSOR_NUMBER * 2,
            seed=seed,
        )
        if problem.SENSOR_NUMBER < self.h:
            raise ValueError("Ring_SETS requires at least one sensor per region")
        self.adaptive_step = 0.001
        self.current_adaptive_step = self.adaptive_step
        self._pool = None
        self.goods = []
        self.goods_fitness = np.empty(self.w, dtype=float)
        self.region_sensor_bounds = self._build_region_sensor_bounds(
            problem.SENSOR_NUMBER
        )
        self.segment_quality = np.empty((self.h, self.n), dtype=float)

    def _build_region_sensor_bounds(self, sensor_count):
        """Split ring-major sensor ids into contiguous, near-equal regions."""
        base_size, remainder = divmod(int(sensor_count), self.h)
        bounds = []
        start = 0
        for region in range(self.h):
            end = start + base_size + (1 if region < remainder else 0)
            bounds.append((start, end))
            start = end
        return tuple(bounds)

    def initialize_market(self, problem, initial_state=None):
        """Create searchers and one shared goods pool (no per-segment pools)."""
        BaseSE.initialize_market(self, problem, initial_state)
        self.goods = [self.create_candidate(problem) for _ in range(self.w)]
        self.goods_fitness = self.evaluate_many(problem, self.goods)

        # No segment has a track record yet, so every segment starts from the
        # same neutral guess: the shared pool's current average quality.
        initial_quality = float(np.mean(self.goods_fitness))
        self.segment_quality = np.full(
            (self.h, self.n), initial_quality, dtype=float
        )

    def create_candidate(self, problem, region=None):
        """Create the V2 coverage/routing chromosome without hard alignment."""
        candidate = SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
            rng=self.rng,
        )
        for target_candidates in problem.cover_candidates:
            if target_candidates:
                sensor_id, level = self.random.choice(target_candidates)
                candidate.code[int(sensor_id) * 2] = int(level)
        return candidate

    def align_region(self, problem, candidate, region):
        """Ring membership guides operators only; never force sensing levels."""
        return candidate

    def _resolve_region(self, region):
        region = int(region)
        if not 0 <= region < self.h:
            raise ValueError("region index is outside the configured market")
        return region

    def _local_sensor_span(self, region):
        start, end = self.region_sensor_bounds[region]
        left = self.random.randrange(start, end)
        right = self.random.randrange(left + 1, end + 1)
        return left, right

    def _cross_ring_sensor_span(self, region):
        """Include all of ``region`` plus a non-empty adjacent fragment."""
        start, end = self.region_sensor_bounds[region]
        neighbours = []
        if region > 0:
            neighbours.append(region - 1)
        if region + 1 < self.h:
            neighbours.append(region + 1)
        neighbour = self.random.choice(neighbours)
        neighbour_start, neighbour_end = self.region_sensor_bounds[neighbour]
        if neighbour < region:
            return self.random.randrange(neighbour_start, start), end
        return start, self.random.randrange(end + 1, neighbour_end + 1)

    def select_sensor_span(self, region):
        """Choose the 80/20 crossover-and-mutation scope for one visit."""
        region = self._resolve_region(region)
        if self.random.random() < self.LOCAL_OPERATOR_PROBABILITY:
            left, right = self._local_sensor_span(region)
            return "local", left, right
        left, right = self._cross_ring_sensor_span(region)
        return "cross_ring", left, right

    def make_children(self, problem, searcher, good, region=None):
        """Return role-preserving children for one selected regional visit."""
        region = self._resolve_region(region)
        sensor_span = self.select_sensor_span(region)
        _, left_sensor, right_sensor = sensor_span
        child1, child2 = swap_segment(
            searcher,
            good,
            left_sensor * 2,
            right_sensor * 2,
        )

        # Each child independently mutates with probability mu, but both use
        # the exact ring scope selected for this investor-good transaction.
        for child in (child1, child2):
            if self.random.random() < self.mutation_rate:
                self.mutate_candidate(problem, child, sensor_span=sensor_span)
        return child1, child2

    def mutate_candidate(self, problem, candidate, sensor_span):
        """Mutate one to three sensor pairs inside the chosen ring scope."""
        _, left_sensor, right_sensor = sensor_span
        mutation_count = self.random.choice([1] * 90 + [2] * 5 + [3] * 5)
        for _ in range(mutation_count):
            sensor_id = self.random.randrange(left_sensor, right_sensor)
            sensing_id = sensor_id * 2
            priority_id = sensing_id + 1
            operation = self.random.randint(0, 2)
            if operation in (0, 1):
                sensing_bound = problem.sensing_option_count(sensor_id)
                candidate.code[sensing_id] = self.random.randrange(sensing_bound)
            if operation in (0, 2):
                candidate.code[priority_id] = self.random.randrange(
                    SensorEncoding.RANK_PRECISION
                )
        return candidate

    def goods_for_region(self, region, goods):
        return goods

    def create_investments(self, problem, goods, active_regions=None):
        return create_selected_investments(self, problem, goods, active_regions)

    def make_offspring(self, problem, searcher, good, region):
        return self.make_children(problem, searcher, good, region)

    def summarize_round(self, active_regions, scores):
        for searcher_id, region in enumerate(active_regions):
            self.segment_quality[int(region), searcher_id] = float(
                np.mean(scores[searcher_id, 0::2]))
        return self.segment_quality

    def region_probabilities(self, goods_fitness, quality):
        return self.segment_probabilities(quality)

    def update_searchers(self, children, scores, selected, goods_before):
        searchers_from_children(self, [row[0::2] for row in children], scores[:, 0::2])

    def update_goods(self, children, scores, active_regions):
        goods_from_children(self.goods, self.goods_fitness,
            [row[1::2] for row in children], scores[:, 1::2], range(self.n),
            tolerance=self.IMPROVEMENT_TOLERANCE)

    def segment_probabilities(self, segment_quality):
        """Score each ring segment purely by its recent investment payoff.

        Unlike :meth:`SI_SETS.region_probabilities`, there is no
        ``best_goods`` or ``region_share`` term here: those measured a
        per-region goods pool that no longer exists. The Beta-CDF fairness
        correction (``ta``/``tb``) is unchanged, so segments that have not
        been visited recently still get a rising exploration bonus.
        """
        probability = np.empty((self.h, self.n), dtype=float)
        for segment in range(self.h):
            for searcher_id in range(self.n):
                probability[segment, searcher_id] = beta_cdf(
                    self.ta[segment],
                    self.tb[segment],
                    segment_quality[segment, searcher_id],
                )
        return probability



__all__ = ["Ring_SETS"]
