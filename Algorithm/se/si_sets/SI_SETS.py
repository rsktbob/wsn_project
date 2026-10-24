"""SI-SETS：分區與 SA-SETS 相同，但每個 searcher 只和自己選到的區域商品投資。

SE 三個家族的第二個。每回合流程和 SA_SETS 不同（先選區再投資、只評估
n × w × 2 筆交易），因此直接繼承 :class:`BaseSE`，分區、初始解、交配突變、
Beta 記憶都寫在本類別裡。

* 分區：在靠近 BS 的 sensor 裡選電量最高的 log2(h) 顆當 identity sensor，
  其開關組合對應 h 個區域，每區有自己的 goods pool。
* 初始解：CCS（先全部隨機，再讓每個 target 放入一組覆蓋得到它的 (sensor, level)）。
* 投資：searcher 只和上一回合選到的區域的 w 個 goods 交易，每筆交易交配出
  兩個子代（角色分離，EN 更新：searcher 菁英、good 不菁英，同 SA_SETS 的學長規則）：

  * ``child1`` 以 searcher 為底，只能取代該 searcher，且必須嚴格變好
    （``SEARCHER_ELITIST``）。
  * ``child2`` 以 good 為底，造訪該區的 searcher 中最好的一個 child2
    不論好壞都取代該 good（``GOOD_ELITIST`` 為 True 時要變好才換）。

* 選區：區域吸引力只看 child1；沒去的區域沿用上次觀察到的投資品質。
  吸引力過 Beta CDF 後以錦標賽選出下一回合的區域。

投資由 ``PooledEvaluator`` 的 h 個無狀態子程序平行評估。
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import betainc

from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.pooled_evaluator import PooledEvaluator
from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding
from State.initialization import random_sensor_coding


class SI_SETS(BaseSE):
    """Choose a goods pool first, then trade only with that pool."""

    # 每次突變改動的基因組數：90% 一組、5% 兩組、5% 三組。
    MUTATION_COUNTS = [1] * 90 + [2] * 5 + [3] * 5
    # Beta 記憶的初始步長，隨搜尋進度線性遞減到 0。
    ADAPTIVE_STEP = 0.001
    # 選區錦標賽每次隨機抽幾個區域比較。
    TOURNAMENT_SIZE = 2

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
        # Beta 記憶：每區一組 (ta, tb)，每次搜尋開始時重設為 1。
        self.ta = np.ones(self.h, dtype=float)
        self.tb = np.ones(self.h, dtype=float)
        self.current_adaptive_step = self.ADAPTIVE_STEP
        self.identity_bit_count = int(math.log2(self.h))
        if 2**self.identity_bit_count != self.h:
            raise ValueError("SETS requires h to be a power of two")
        if not 1 <= self.TOURNAMENT_SIZE <= self.h:
            raise ValueError("tournament size must be between 1 and h")
        self.identity_sensors = []
        self.goods = []
        self.goods_fitness = np.empty((self.h, self.w), dtype=float)
        self.investment_quality = np.empty((self.h, self.n), dtype=float)
        self.evaluator = PooledEvaluator()

    # ---- 初始化 -------------------------------------------------------------
    def select_identity_sensors(self, problem):
        """在靠近 BS 的 sensor 裡選電量最高的 log2(h) 顆。

        距離上限 = 第 2 近的 sensor 到 BS 的距離 + 15；上限內依電量由高到低挑，
        電量要達 liveJ。挑不滿時改取離 BS 最近的幾顆活著的 sensor。
        """
        distances = np.asarray(
            problem.distances[: problem.SENSOR_NUMBER, problem.BSID],
            dtype=float,
        )
        distance_order = getattr(
            problem,
            "sensor_ids_by_bs_distance",
            np.argsort(distances, kind="stable"),
        )
        boundary_id = int(distance_order[min(1, len(distance_order) - 1)])
        distance_limit = distances[boundary_id] + 15
        candidate_ids = distance_order[
            distances[distance_order] <= distance_limit
        ]

        selected = []
        energy = np.asarray(problem.energy[candidate_ids], dtype=float).copy()
        for _ in range(self.identity_bit_count):
            if not len(energy):
                break
            candidate_index = int(np.argmax(energy))
            if energy[candidate_index] < problem.liveJ:
                break
            selected.append(int(candidate_ids[candidate_index]))
            energy[candidate_index] = -np.inf

        if len(selected) == self.identity_bit_count:
            return selected

        return [
            int(sensor_id)
            for sensor_id in distance_order
            if problem.energy[sensor_id] >= problem.liveJ
        ][: self.identity_bit_count]

    def initialize_market(self, problem, initial_state=None):
        """選 identity sensor、產生 searcher，再為每區建立 goods pool。"""
        self.identity_sensors = self.select_identity_sensors(problem)
        if len(self.identity_sensors) != self.identity_bit_count:
            raise RuntimeError("not enough live sensors for SETS regions")
        self.ta = np.ones(self.h, dtype=float)
        self.tb = np.ones(self.h, dtype=float)
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

    def create_candidate(self, problem, region=None):
        """CCS 初始解，再對齊到 ``region``。"""
        candidate = random_sensor_coding(problem, self.rng, self.random)
        if region is not None:
            self.align_region(problem, candidate, region)
        return candidate

    def align_region(self, problem, candidate, region):
        """依區域編號的每個 bit，強制對應的 identity sensor 開或關。"""
        bits = int(region)
        for bit_id, sensor_id in enumerate(self.identity_sensors):
            must_open = (bits >> bit_id) & 1
            gene_id = int(sensor_id) * 2
            option_count = problem.sensing_option_count(sensor_id)
            if must_open:
                if int(candidate.code[gene_id]) % option_count == 0:
                    candidate.code[gene_id] = self.random.randrange(
                        1, option_count
                    )
            else:
                candidate.code[gene_id] = 0
        return candidate

    # ---- 平行評估 -----------------------------------------------------------
    def arrange_resources(self, problem):
        """開啟 h 個評估子程序。"""
        self.evaluator.start(problem, self.h, self.next_seed)

    def close_market(self):
        """搜尋結束或發生例外時關閉評估子程序。"""
        self.evaluator.close()

    def evaluate_investments(self, problem, investments):
        """評估 ``investments[searcher][child]``，回傳同形狀的 fitness。"""
        return self.evaluator.evaluate(self, problem, investments)

    # ---- 投資：交配＋突變 ----------------------------------------------------
    def crossover_points(self):
        """一刀在中點左邊、一刀在中點右邊，離中點與兩端至少 2 格。

        chromosome 太短切不出來時回傳 ``None``。
        """
        difference = 2
        midpoint = self.code_length // 2
        if midpoint <= difference or self.code_length - difference <= midpoint:
            return None
        left = self.random.randint(difference, midpoint - difference)
        right = self.random.randint(
            midpoint - difference,
            self.code_length - difference,
        )
        return left, right

    def crossover(self, searcher, good):
        """交換兩刀之間的片段，回傳兩個子代。"""
        points = self.crossover_points()
        if points is None:
            return searcher.copy(), good.copy()
        return swap_segment(searcher, good, *points)

    def make_children(self, problem, searcher, good, region=None):
        """雙向交配，兩個子代各自以機率 mu 突變。"""
        child1, child2 = self.crossover(searcher, good)

        for child in (child1, child2):
            if self.random.random() < self.mutation_rate:
                self.mutate_candidate(problem, child)
        return child1, child2

    def mutate_candidate(self, problem, candidate):
        """隨機改 1–3 組基因，每組改 sensing level、priority 或兩者。"""
        mutation_count = self.random.choice(self.MUTATION_COUNTS)
        for _ in range(mutation_count):
            gene_id = self.random.randrange(self.code_length)
            sensing_id = gene_id if gene_id % 2 == 0 else gene_id - 1
            priority_id = sensing_id + 1
            operation = self.random.randint(0, 2)
            if operation in (0, 1):
                sensing_bound = problem.sensing_option_count(
                    sensing_id // 2
                )
                candidate.code[sensing_id] = self.random.randrange(
                    sensing_bound
                )
            if operation in (0, 2):
                candidate.code[priority_id] = self.random.randrange(
                    SensorEncoding.RANK_PRECISION
                )
        return candidate

    # ---- 一回合：投資、選區、更新 -------------------------------------------
    def vision_search(self, problem):
        """只和選到的區域交易，再做角色分離的更新。"""
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        self.current_adaptive_step = self.ADAPTIVE_STEP * (1.0 - progress)

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

        # searcher 只看自己的 child1；good 只看造訪該區的 searcher 給它的 child2。
        # 沒被造訪的 good 保持不變。
        self.accept_searcher_children(
            children, child1_fitness, searcher_fitness_before
        )
        for region in range(self.h):
            self.accept_good_children(
                self.goods[region],
                self.goods_fitness[region],
                goods_fitness_before[region],
                children,
                child2_fitness,
                np.flatnonzero(active_regions == region),
            )

        self.selected_regions = selected

    def select_regions(self, probabilities):
        """每個 searcher 隨機抽 TOURNAMENT_SIZE 個區域，選機率最高的。"""
        selected = np.empty(self.n, dtype=int)
        for searcher_id in range(self.n):
            players = self.random.sample(range(self.h), self.TOURNAMENT_SIZE)
            selected[searcher_id] = max(
                players,
                key=lambda region: probabilities[region, searcher_id],
            )
        return selected

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

    def region_probabilities(self, goods_fitness, investment_quality):
        """區域期望值 = 最好的 good × 投資品質 × 市占，再過該區的 Beta CDF。"""
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
                probability[region, searcher_id] = float(betainc(
                    float(self.ta[region]),
                    float(self.tb[region]),
                    float(expected_value[region, searcher_id]),
                ))
        return probability

    def update_search_memory(self, problem, evaluation_start):
        """被選到的區域 ta 加步長，其餘區域 tb 加步長。"""
        step = self.current_adaptive_step
        for selected_region in self.selected_regions:
            selected_region = int(selected_region)
            self.ta[selected_region] += step
            for region in range(self.h):
                if region != selected_region:
                    self.tb[region] += step

        history_end = min(int(self.evatime), len(self.history))
        if history_end > evaluation_start:
            self.history[evaluation_start:history_end] = self.fitness


__all__ = ["SI_SETS"]
