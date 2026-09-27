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
from Algorithm.se.market_components import (
    AdaptiveBetaMemory,
    PooledEvaluation,
    beta_cdf,
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
        super().initialize_market(problem, initial_state)
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

    def vision_search(self, problem):
        """Trade every searcher against the shared pool; score segments only
        by how well operating on them has paid off for the visiting searcher.
        """
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        self.current_adaptive_step = self.adaptive_step * (1.0 - progress)

        active_segments = self.selected_regions.copy()
        goods_fitness_before = self.goods_fitness.copy()
        searcher_fitness_before = self.searcher_fitness.copy()
        children = []

        for searcher_id, segment in enumerate(active_segments):
            segment = int(segment)
            row = []
            for good_id in range(self.w):
                child1, child2 = self.make_children(
                    problem,
                    self.searchers[searcher_id],
                    self.goods[good_id],
                    region=segment,
                )
                row.append(child1)
                row.append(child2)
            children.append(row)

        child_fitness = self.evaluate_investments(problem, children)
        child1_fitness = child_fitness[:, 0::2]
        child2_fitness = child_fitness[:, 1::2]

        # A segment's attractiveness is how well it improved the searcher who
        # just visited it -- there is no goods-pool quality or market-share
        # term to blend in, because goods are no longer partitioned by
        # segment.
        for searcher_id, segment in enumerate(active_segments):
            self.segment_quality[segment, searcher_id] = float(
                np.mean(child1_fitness[searcher_id])
            )

        probabilities = self.segment_probabilities(self.segment_quality)
        selected = self.select_regions(probabilities)

        # A searcher considers only its own child1 proposals and remains
        # unchanged unless the best proposal improves its previous fitness.
        for searcher_id in range(self.n):
            good_id = int(np.argmax(child1_fitness[searcher_id]))
            score = float(child1_fitness[searcher_id, good_id])
            if score > searcher_fitness_before[searcher_id]:
                self.searchers[searcher_id] = children[searcher_id][
                    good_id * 2
                ].copy()
                self.searcher_fitness[searcher_id] = score

        # The pool is shared: every searcher this round is a candidate
        # visitor for every good slot, regardless of which segment it
        # operated on. A good accepts the best child2 only when it improves
        # the pre-investment good by more than the elitist tolerance.
        for good_id in range(self.w):
            best_searcher = int(np.argmax(child2_fitness[:, good_id]))
            winner_score = float(child2_fitness[best_searcher, good_id])
            if (
                winner_score
                <= goods_fitness_before[good_id] + self.IMPROVEMENT_TOLERANCE
            ):
                continue
            self.goods[good_id] = children[best_searcher][good_id * 2 + 1].copy()
            self.goods_fitness[good_id] = winner_score

        self.selected_regions = selected

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
