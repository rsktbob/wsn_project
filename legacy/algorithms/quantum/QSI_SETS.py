"""Q-SI-SETS：SI_SETS 的 goods 換成量子位元（Q-bit）機率模型。

和 SI_SETS 相同的部分：identity sensor 分區、每區 w 個 goods、每個 searcher
依 Beta 記憶選一區、只和那一區的 goods 交易（n x w x 2 個子代）、
角色分離的 EN 更新（searcher 菁英、good 一律換成造訪者最好的 child2）、
區域吸引力只看 child1。

不同的部分（量子 goods，作法同 QRing_SE）：

* 每個 good ``(r, k)`` 有一組 Q-bit 機率 ``Q_rk``（QuantumSensorEncoding 的
  條件 thermometer 閘）和一個吸引子解 ``b_rk``（就是原本的 good）。
* ``child1`` = searcher，交配點 ``[left, right)`` 那段改成從 ``Q_rk`` 觀測
  出來的基因（取代原本複製 good 的那段），不另外突變，再對齊區域。
* ``child2`` = ``b_rk`` 換上 searcher 的那段，同 SI_SETS（含突變）。
* 旋轉門：每個 ``child1`` 只在自己觀測的那段內更新 ``Q_rk``。比較
  child1 與 ``b_rk``：較好的一方當目標，只轉兩者觀測結果不同的閘，
  同分不轉。機率夾在 ``[floor, 1 - floor]``，避免塌縮。

註：QRing_SE 的 rotate 用觀測 bits 比對，「觀測解較好」那一支永遠不轉
（2026-10-12 發現）；這裡改成比對兩個解的閘目標，兩支都會轉。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.si_sets.SI_SETS import SI_SETS
from Algorithm.se.market_components import (
    accept_good_children,
    accept_searcher_children,
)
from State.QuantumSensorEncoding import QuantumSensorEncoding


class QSI_SETS(SI_SETS):
    def __init__(self, problem, n=8, h=4, w=8, mu=0.4, rotation=0.05 * np.pi,
                 probability_floor=0.01, init_confidence=0.0, seed=None):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        if not np.isfinite(rotation) or not 0 <= rotation <= np.pi / 2:
            raise ValueError("rotation must be in [0, pi/2]")
        if not 0 < probability_floor < 0.5:
            raise ValueError("probability_floor must be in (0, 0.5)")
        if not 0 <= init_confidence < 1:
            raise ValueError("init_confidence must be in [0, 1)")
        self.rotation = float(rotation)
        self.probability_floor = float(probability_floor)
        self.init_confidence = float(init_confidence)

    def initialize_market(self, problem, initial_state=None):
        """SI 的 searchers／goods pools，再為每個 good 建一組 Q-bit。"""
        super().initialize_market(problem, initial_state)
        # lifetime 每段的可用等級會變，每次 search 都重建。
        self.q_encoding = QuantumSensorEncoding(problem.radius_option_counts)
        enc = self.q_encoding
        floor = np.minimum(self.probability_floor,
                           1 / enc.domains[enc.gene_ids])
        self.angle_low = np.arcsin(np.sqrt(floor))
        self.angle_high = np.arcsin(np.sqrt(1 - floor))
        self.q_angles = np.tile(enc.initial_angles, (self.h, self.w, 1))
        if self.init_confidence > 0:
            for region in range(self.h):
                for good_id, good in enumerate(self.goods[region]):
                    self._bias_toward(region, good_id, good.code)
        self.q_probabilities = np.sin(self.q_angles) ** 2

    def _bias_toward(self, region, good_id, code):
        """把路徑上的閘設成以 init_confidence 機率觀測到 ``code``。"""
        target, path = self.q_encoding.elite_path(code)
        p = np.where(target, self.init_confidence, 1 - self.init_confidence)
        biased = np.clip(np.arcsin(np.sqrt(p)), self.angle_low, self.angle_high)
        angles = self.q_angles[region, good_id]
        self.q_angles[region, good_id] = np.where(path, biased, angles)

    def _gate_mask(self, left, right):
        """基因位置 ``[left, right)`` 對應的閘。"""
        genes = self.q_encoding.gene_ids
        return (genes >= left) & (genes < right)

    def make_quantum_children(self, problem, searcher, region, good_id):
        """child1 從 Q_rk 觀測交配段；child2 同 SI_SETS。回傳觀測段的閘遮罩。"""
        points = self.crossover_points()
        left, right = points if points is not None else (0, 0)
        observed, _ = self.q_encoding.observe(
            self.q_probabilities[region, good_id], self.rng
        )
        child1 = searcher.copy()
        child1.code[left:right] = observed.code[left:right]

        child2 = self.goods[region][good_id].copy()
        child2.code[left:right] = searcher.code[left:right]
        if self.random.random() < self.mutation_rate:
            self.mutate_candidate(problem, child2)

        self.align_region(problem, child1, region)
        self.align_region(problem, child2, region)
        return child1, child2, self._gate_mask(left, right)

    def rotate(self, region, good_id, masks, child1_codes, child1_fitness,
               good_code, good_fitness):
        """在觀測過的段內，把 Q_rk 轉向每對（child1, b_rk）中較好的一方。"""
        good_target, good_path = self.q_encoding.elite_path(good_code)
        delta = np.zeros(self.q_encoding.length)
        for mask, code, fitness in zip(masks, child1_codes, child1_fitness):
            child_target, child_path = self.q_encoding.elite_path(code)
            if fitness > good_fitness + self.IMPROVEMENT_TOLERANCE:
                target, path = child_target, child_path
            elif fitness < good_fitness - self.IMPROVEMENT_TOLERANCE:
                target, path = good_target, good_path
            else:
                continue
            move = mask & path & (child_target != good_target)
            delta += move * np.where(target, self.rotation, -self.rotation)
        angles = self.q_angles[region, good_id]
        angles += delta
        np.clip(angles, self.angle_low, self.angle_high, out=angles)
        self.q_probabilities[region, good_id] = np.sin(angles) ** 2

    def vision_search(self, problem):
        self.update_adaptive_step()

        active_regions = self.selected_regions.copy()
        goods_fitness_before = self.goods_fitness.copy()
        goods_before = [[good.code.copy() for good in pool]
                        for pool in self.goods]
        searcher_fitness_before = self.searcher_fitness.copy()
        children = []
        masks = []

        for searcher_id, region in enumerate(active_regions):
            region = int(region)
            row = []
            row_masks = []
            for good_id in range(self.w):
                child1, child2, mask = self.make_quantum_children(
                    problem, self.searchers[searcher_id], region, good_id
                )
                row.append(child1)
                row.append(child2)
                row_masks.append(mask)
            children.append(row)
            masks.append(row_masks)

        child_fitness = self.evaluate_investments(problem, children)
        child1_fitness = child_fitness[:, 0::2]
        child2_fitness = child_fitness[:, 1::2]

        # evaluate_investments 可能把 children 換成評估後的物件，所以評估後再取 code。
        for region in range(self.h):
            visitors = np.flatnonzero(active_regions == region)
            if not len(visitors):
                continue
            for good_id in range(self.w):
                self.rotate(
                    region,
                    good_id,
                    [masks[s][good_id] for s in visitors],
                    [children[s][good_id * 2].code for s in visitors],
                    child1_fitness[visitors, good_id],
                    goods_before[region][good_id],
                    goods_fitness_before[region, good_id],
                )

        for searcher_id, region in enumerate(active_regions):
            self.investment_quality[int(region), searcher_id] = float(
                np.mean(child1_fitness[searcher_id])
            )
        probabilities = self.region_probabilities(
            goods_fitness_before,
            self.investment_quality,
        )
        selected = self.select_regions(probabilities)

        accept_searcher_children(
            self, children, child1_fitness, searcher_fitness_before
        )
        # EN 更新：被造訪的 good 不論好壞都換成最好的 child2（門檻 −∞）。
        for region in range(self.h):
            accept_good_children(
                self.goods[region],
                self.goods_fitness[region],
                goods_fitness_before[region],
                children,
                child2_fitness,
                np.flatnonzero(active_regions == region),
                -np.inf,
            )

        self.selected_regions = selected


__all__ = ["QSI_SETS"]
