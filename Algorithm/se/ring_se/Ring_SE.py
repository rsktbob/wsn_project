"""Ring-SE：沒有固定分區，區域只決定交配與突變能動的 sensor 範圍。

SE 三個家族的第三個（另外兩個是 SETS 與 SI_SETS）：

* 分區：sensor 依環狀順序（ring-major）切成 ``h`` 段連續範圍。區域不綁定
  任何解的特徵，只決定這一回合的投資範圍：80% 只動該段，20% 動整段
  再加上相鄰段的一部分。突變使用同一個範圍。
* 投資：所有 searcher 每回合都和同一組共用的 ``w`` 個商品交換。
* 更新：角色分離的 EN 更新。``child1`` 以 searcher 為底，只能取代
  該 searcher，而且必須嚴格變好（E，菁英）；``child2`` 以 good 為底，
  每回合該 good 一律換成拜訪者中最好的 child2，即使變差（N，不菁英）。
  E／N 由 BaseSE 的 ``SEARCHER_ELITIST``／``GOOD_ELITIST`` 決定。
  歷史最佳解由 ``update_best`` 在每次評估時保留，不會遺失。
  （2026-10-12 由 EE 改成 EN：搭配學長突變 GA、fitness v3 的 lifetime
  實驗中，EN 在 1000／3000／10000 都顯著贏同編碼 GA。）
* 區域選擇：每組 searcher-good 交易各自均勻隨機選一段（2026-10-13 起；
  之前是每回合每個 searcher 選一段、它的 w 筆交易共用）。MAP01–06 × seed 7–11
  兩者 lifetime 在雜訊內（1000：+0.7%，3000：+0.4%，10000：+0.1%，都 ns；tmp/pair_random）。
  原本的 Beta CDF 選區在後期主要反映造訪次數而非 fitness，消融也測不出差異，所以拿掉。

``region`` 沿用 ``self.h`` 等名稱，以配合 SE 共用流程。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.pooled_evaluator import PooledEvaluator
from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding
from State.initialization import random_sensor_coding


class Ring_SE(BaseSE):
    """SE with ring-segment operator scopes and one shared elitist goods pool.

    For ``h=4`` and 100 ring-ordered sensors, segments are ``[0:25)``,
    ``[25:50)``, ``[50:75)``, and ``[75:100)``. On every visit, both
    directions of crossover are retained:

    * ``child1`` is searcher ``a`` with the selected segment from good ``b``;
      only ``a`` may accept it.
    * ``child2`` is good ``b`` with the selected segment from searcher ``a``;
      only ``b`` may accept it.

    Searchers are elitist (E); goods always take their best visitor's
    child2 (N). Mutation follows the same chosen sensor span.
    Unlike SETS, ``h`` need not be a power of two.
    """

    LOCAL_OPERATOR_PROBABILITY = 0.80
    # 每次突變改動的 sensor 數：90% 一顆、5% 兩顆、5% 三顆。
    MUTATION_COUNTS = [1] * 90 + [2] * 5 + [3] * 5

    def __init__(self, problem, n=8, h=4, w=8, mu=0.4, seed=None):
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
            raise ValueError("Ring_SE requires at least one sensor per region")
        self.goods = []
        self.goods_fitness = np.empty(self.w, dtype=float)
        self.region_sensor_bounds = self._build_region_sensor_bounds(
            problem.SENSOR_NUMBER
        )
        self.evaluator = PooledEvaluator()

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

    def create_candidate(self, problem, region=None):
        """CCS 初始解；Ring 的區域不限制解，所以不對齊。"""
        return random_sensor_coding(problem, self.rng, self.random)

    def align_region(self, problem, candidate, region):
        """Ring membership guides operators only; never force sensing levels."""
        return candidate

    def arrange_resources(self, problem):
        """開啟 h 個評估子程序。"""
        self.evaluator.start(problem, self.h, self.next_seed)

    def close_market(self):
        """搜尋結束或發生例外時關閉評估子程序。"""
        self.evaluator.close()

    def evaluate_investments(self, problem, investments):
        """評估 ``investments[searcher][child]``，回傳同形狀的 fitness。"""
        return self.evaluator.evaluate(self, problem, investments)

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
        mutation_count = self.random.choice(self.MUTATION_COUNTS)
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

    def select_pair_segment(self):
        """Draw one searcher-good pair's segment uniformly at random."""
        return self.random.randrange(self.h)

    def select_regions(self, probabilities=None):
        """Draw one segment per searcher uniformly at random.

        Ring_SE itself draws per pair (``select_pair_segment``); this
        per-searcher draw is kept for QRing_SE, whose Q-bit goods still use
        one segment per searcher and round.
        """
        return np.array(
            [self.random.randrange(self.h) for _ in range(self.n)],
            dtype=int,
        )

    def vision_search(self, problem):
        """Trade every searcher against the shared pool; each pair draws its own segment."""
        goods_fitness_before = self.goods_fitness.copy()
        searcher_fitness_before = self.searcher_fitness.copy()
        children = []

        for searcher_id in range(self.n):
            row = []
            for good_id in range(self.w):
                child1, child2 = self.make_children(
                    problem,
                    self.searchers[searcher_id],
                    self.goods[good_id],
                    region=self.select_pair_segment(),
                )
                row.append(child1)
                row.append(child2)
            children.append(row)

        child_fitness = self.evaluate_investments(problem, children)
        child1_fitness = child_fitness[:, 0::2]
        child2_fitness = child_fitness[:, 1::2]

        # A searcher considers only its own child1 proposals and remains
        # unchanged unless the best proposal improves its previous fitness.
        self.accept_searcher_children(
            children, child1_fitness, searcher_fitness_before
        )

        # The pool is shared: every searcher this round is a candidate
        # visitor for every good slot, regardless of which segment it
        # operated on. Each good is replaced by its best child2 even when
        # that child is worse (N; see GOOD_ELITIST).
        self.accept_good_children(
            self.goods,
            self.goods_fitness,
            goods_fitness_before,
            children,
            child2_fitness,
            range(self.n),
        )

    def accept_searcher_children(self, children, child1_fitness,
                                 searcher_fitness_before):
        """每個 searcher 取自己最好的 child1（E：嚴格變好才換）。

        ``children[s]`` 依 good 交錯排列 ``[child1, child2, ...]``。
        """
        for searcher_id in range(self.n):
            good_id = int(np.argmax(child1_fitness[searcher_id]))
            score = float(child1_fitness[searcher_id, good_id])
            if self.SEARCHER_ELITIST and score <= searcher_fitness_before[searcher_id]:
                continue
            self.searchers[searcher_id] = children[searcher_id][good_id * 2].copy()
            self.searcher_fitness[searcher_id] = score

    def accept_good_children(self, goods, goods_fitness, goods_fitness_before,
                             children, child2_fitness, visitors):
        """每個 good 取造訪者中最好的 child2（N：一律換）；同分取第一個造訪者。"""
        if not len(visitors):
            return
        for good_id in range(len(goods)):
            winner = int(
                max(visitors, key=lambda index: child2_fitness[int(index), good_id])
            )
            winner_score = float(child2_fitness[winner, good_id])
            if self.GOOD_ELITIST and winner_score <= goods_fitness_before[good_id]:
                continue
            goods[good_id] = children[winner][good_id * 2 + 1].copy()
            goods_fitness[good_id] = winner_score

    def update_search_memory(self, problem, evaluation_start):
        """No region memory to update; only record this round's history."""
        history_end = min(int(self.evatime), len(self.history))
        if history_end > evaluation_start:
            self.history[evaluation_start:history_end] = self.fitness


__all__ = ["Ring_SE"]
