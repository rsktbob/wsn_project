"""Sensor-encoded baseline Search Economics algorithm."""

from __future__ import annotations

import math

import numpy as np

from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.pooled_evaluator import PooledEvaluator
from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding
from State.initialization import random_sensor_coding


class CodingSE(BaseSE):
    """Standard SE market flow with sensor schedule/routing chromosomes."""

    # 每次突變改動的基因數：90% 一個、5% 兩個、5% 三個。
    MUTATION_COUNTS = [1] * 90 + [2] * 5 + [3] * 5
    # 選區錦標賽每次隨機抽幾個區域比較。
    TOURNAMENT_SIZE = 2

    def __init__(self, problem, n=5, h=4, w=2, mu=0.2, seed=None):
        code_length = problem.SENSOR_NUMBER * 2
        super().__init__(
            problem,
            n=n,
            h=h,
            w=w,
            mu=mu,
            code_length=code_length,
            seed=seed,
        )
        self.identity_bit_count = int(math.log2(self.h))
        if 2**self.identity_bit_count != self.h:
            raise ValueError("CodingSE requires h to be a power of two")
        if not 1 <= self.TOURNAMENT_SIZE <= self.h:
            raise ValueError("tournament size must be between 1 and h")
        self.identity_sensors = []
        # goods[region][good_id] 與同形狀的 goods_fitness 在 initialize_market 建立。
        self.goods = []
        self.goods_fitness = np.empty((self.h, self.w), dtype=float)
        self.evaluator = PooledEvaluator()
        # 原版 SE 的區域記憶：ta 記被選次數，tb 記多久沒被選。
        self.ta = np.ones(self.h, dtype=float)
        self.tb = np.ones(self.h, dtype=float)
        problem.prepare_coding_cache()

    def select_identity_sensors(self, problem):
        return [
            sensor_id
            for sensor_id in range(problem.SENSOR_NUMBER)
            if problem.energy[sensor_id] >= problem.liveJ
        ][: self.identity_bit_count]

    def initialize_market(self, problem, initial_state=None):
        self.identity_sensors = self.select_identity_sensors(problem)
        if len(self.identity_sensors) != self.identity_bit_count:
            raise RuntimeError("not enough live sensors for CodingSE regions")
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
        candidate = random_sensor_coding(problem, self.rng, self.random)
        if region is not None:
            self.align_region(problem, candidate, region)
        return candidate

    def align_region(self, problem, candidate, region):
        bits = int(region)
        for bit_id, sensor_id in enumerate(self.identity_sensors):
            gene_id = sensor_id * 2
            must_open = (bits >> bit_id) & 1
            if must_open:
                option_count = problem.sensing_option_count(sensor_id)
                if int(candidate.code[gene_id]) % option_count == 0:
                    candidate.code[gene_id] = self.random.randrange(
                        1, option_count
                    )
            else:
                candidate.code[gene_id] = 0
        return candidate

    def invest(self, problem, searcher, good):
        difference = 2
        midpoint = self.code_length // 2
        left = self.random.randint(difference, midpoint - difference)
        right = self.random.randint(
            midpoint - difference,
            self.code_length - difference,
        )
        first, second = swap_segment(searcher, good, left, right)
        investment = first if self.random.random() < 0.5 else second

        if self.random.random() < self.mutation_rate:
            mutation_count = self.random.choice(self.MUTATION_COUNTS)
            upper_bound = max(
                problem.LEVEL,
                SensorEncoding.RANK_PRECISION,
            )
            for _ in range(mutation_count):
                gene_id = self.random.randrange(self.code_length)
                investment.code[gene_id] = self.random.randrange(upper_bound)
        return investment


    def arrange_resources(self, problem):
        """開啟 h 個評估子程序。"""
        self.evaluator.start(problem, self.h, self.next_seed)

    def close_market(self):
        """搜尋結束或發生例外時關閉評估子程序。"""
        self.evaluator.close()

    def vision_search(self, problem):
        """和所有區域投資、選區，再讓 searcher 換成所選區最好的 good。"""
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
        """原版 SE 的投資潛力：(tb / ta) × 投資平均 × 市占。"""
        investment_quality = np.mean(investment_fitness, axis=2)
        market_quality = np.mean(goods_fitness, axis=1)
        total_quality = float(np.sum(market_quality))
        if abs(total_quality) <= np.finfo(float).eps:
            market_share = np.full(self.h, 1.0 / self.h)
        else:
            market_share = market_quality / total_quality
        history_weight = self.tb / self.ta
        return (
            history_weight[:, None]
            * investment_quality
            * market_share[:, None]
        )

    def update_search_memory(self, problem, evaluation_start):
        """被選到的區域 ta 加 1、tb 歸 1；其他區域 tb 加 1。"""
        for region in self.selected_regions:
            self.ta[int(region)] += 1.0
        self.tb += 1.0
        for region in self.selected_regions:
            self.tb[int(region)] = 1.0
        super().update_search_memory(problem, evaluation_start)


__all__ = ["CodingSE"]
