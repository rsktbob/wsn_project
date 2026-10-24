"""SETS：以 identity sensor 分區、跟所有區域商品投資的原始 Search Economics。

SE 三個家族的第一個（另外兩個是 SI_SETS 與 Ring_SE）：

* 分區：取 log2(h) 顆 identity sensor，其開關組合對應 h 個區域，
  所有候選解都會對齊到自己區域的開關組合。
* 投資：每個 searcher 每回合都和所有區域的所有商品投資（h × n × w），
  交給 ``PooledEvaluator`` 平行評估。投資完，每個 good 換成投資它的最好
  結果（N；``GOOD_ELITIST`` 時要變好才換）。
* 區域選擇：Beta 記憶加上市場期望值，以錦標賽選出區域；searcher 換成
  該區最好的 good（E：變好才換；``SEARCHER_ELITIST`` 為 False 時一律換）。
* 初始解：``random_sensor_coding``（全部隨機再加 CCS），再對齊到所屬區域。

SA_SETS 繼承本類別，只改 identity sensor 的選法。
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


class SETS(BaseSE):
    """Search Economics with identity-sensor regions and full investment."""

    # 每次突變改動的基因組數：90% 一組、5% 兩組、5% 三組。
    MUTATION_COUNTS = [1] * 90 + [2] * 5 + [3] * 5
    # Beta 記憶的初始步長，隨搜尋進度線性遞減到 0。
    ADAPTIVE_STEP = 0.001
    # 選區錦標賽每次隨機抽幾個區域比較。
    TOURNAMENT_SIZE = 2

    def __init__(self, problem, n=5, h=4, w=2, mu=0.2, seed=None):
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
        # goods[region][good_id] 與同形狀的 goods_fitness 在 initialize_market 建立。
        self.goods = []
        self.goods_fitness = np.empty((self.h, self.w), dtype=float)
        self.evaluator = PooledEvaluator()

    # ---- 初始化 -------------------------------------------------------------
    def select_identity_sensors(self, problem):
        """取編號最小的 log2(h) 顆活著的 sensor 當 identity sensor。"""
        return [
            sensor_id
            for sensor_id in range(problem.SENSOR_NUMBER)
            if problem.energy[sensor_id] >= problem.liveJ
        ][: self.identity_bit_count]

    def initialize_market(self, problem, initial_state=None):
        """先選 identity sensor，再產生 searcher 與每區 w 個 goods。"""
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

    def create_candidate(self, problem, region=None):
        """隨機解加 CCS，再對齊到 ``region``。"""
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

    def invest(self, problem, searcher, good):
        """交配後兩個子代 50/50 留一個，再以機率 mu 突變。"""
        points = self.crossover_points()
        if points is None:
            investment = searcher.copy()
        else:
            first, second = swap_segment(searcher, good, *points)
            investment = (
                first if self.random.random() < 0.5 else second
            )
        if self.random.random() < self.mutation_rate:
            self.mutate_candidate(problem, investment)
        return investment

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

    # ---- 選區 ---------------------------------------------------------------
    def vision_search(self, problem):
        """和所有區域投資、選區，再讓 searcher 換成所選區最好的 good。"""
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        self.current_adaptive_step = self.ADAPTIVE_STEP * (1.0 - progress)

        # 每個 searcher 和每區每個 good 投資一次並對齊到該區；一區一列送去評估。
        investments = [
            [
                self.align_region(
                    problem,
                    self.invest(problem, searcher, self.goods[region][good_id]),
                    region,
                )
                for searcher in self.searchers
                for good_id in range(self.w)
            ]
            for region in range(self.h)
        ]
        investment_fitness = self.evaluator.evaluate(
            self, problem, investments
        ).reshape(self.h, self.n, self.w)

        # 選區和 searcher 更新都看投資前的 goods。
        goods = [list(region_goods) for region_goods in self.goods]
        goods_fitness = self.goods_fitness.copy()
        self.update_goods(investments, investment_fitness)

        probabilities = self.region_probabilities(
            goods_fitness,
            investment_fitness,
        )
        selected = self.select_regions(probabilities)
        for searcher_id, region in enumerate(selected):
            good_id = int(np.argmax(goods_fitness[region]))
            if (
                self.SEARCHER_ELITIST
                and goods_fitness[region, good_id]
                <= self.searcher_fitness[searcher_id]
            ):
                continue
            self.searchers[searcher_id] = goods[region][good_id].copy()
            self.searcher_fitness[searcher_id] = goods_fitness[
                region, good_id
            ]
        self.selected_regions = selected

    def update_goods(self, investments, investment_fitness):
        """每個 good 換成這回合投資它的最好結果；GOOD_ELITIST 時要嚴格變好才換。"""
        for region in range(self.h):
            for good_id in range(self.w):
                searcher_id = int(np.argmax(investment_fitness[region, :, good_id]))
                fitness = investment_fitness[region, searcher_id, good_id]
                if (
                    self.GOOD_ELITIST
                    and fitness <= self.goods_fitness[region, good_id]
                ):
                    continue
                self.goods[region][good_id] = investments[region][
                    searcher_id * self.w + good_id
                ].copy()
                self.goods_fitness[region, good_id] = fitness

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

    def region_probabilities(self, goods_fitness, investment_fitness):
        """區域期望值 = 最好的 good × 投資平均 × 市占，再過該區的 Beta CDF。"""
        investment_quality = np.mean(investment_fitness, axis=2)
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

    # ---- 更新記憶 -----------------------------------------------------------
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


__all__ = ["SETS"]
