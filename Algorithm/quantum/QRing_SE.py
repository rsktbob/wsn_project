"""Q-Ring-SE：Ring_SE 的 goods 換成量子位元（Q-bit）機率模型。

和 Ring_SE 相同的部分：環狀分段、每回合每個 searcher 均勻隨機選一段、
80/20 範圍、n x w x 2 個子代、角色分離的更新。

接受規則是 EE（searcher、good 都要嚴格變好才換），不跟正式 Ring_SE 的 EN：
``b_k`` 是 ``Q_k`` 旋轉的目標，必須是菁英。2026-10-12 試過 EN（good 一律換成
最好的 child2），三種預算都不比 EE 好，10000 次時 seed 間變異也變大。

不同的部分（量子 goods）：

* 每個 good ``k`` 有一組 Q-bit 機率 ``Q_k``（QuantumSensorEncoding 的
  條件 thermometer 閘）和一個吸引子解 ``b_k``（就是原本的 good）。
* ``child1`` = searcher，選到的那段改成從 ``Q_k`` 觀測出來的基因
  （取代原本整段複製 good），不另外突變，探索來自觀測本身。
* ``child2`` = ``b_k`` 換上 searcher 的那段，同 Ring_SE（含突變）。
* 旋轉門：每個 ``child1`` 只在自己觀測的那段內更新 ``Q_k``；比 ``b_k``
  好就轉向觀測解，比較差就轉向 ``b_k``，同分不轉。機率夾在
  ``[floor, 1 - floor]``，避免塌縮。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.ring_se.Ring_SE import Ring_SE
from Algorithm.se.market_components import (
    accept_good_children,
    accept_searcher_children,
)
from State.QuantumSensorEncoding import QuantumSensorEncoding


class QRing_SE(Ring_SE):
    # EE：覆寫 Ring_SE 的 EN 門檻（-inf），good 也必須嚴格變好。
    GOOD_ACCEPT_TOLERANCE = Ring_SE.IMPROVEMENT_TOLERANCE

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
        """Ring 的 searchers／goods，再為每個 good 建一組 Q-bit。"""
        super().initialize_market(problem, initial_state)
        # lifetime 每段的可用等級會變，每次 search 都重建。
        self.q_encoding = QuantumSensorEncoding(problem.radius_option_counts)
        enc = self.q_encoding
        floor = np.minimum(self.probability_floor,
                           1 / enc.domains[enc.gene_ids])
        self.angle_low = np.arcsin(np.sqrt(floor))
        self.angle_high = np.arcsin(np.sqrt(1 - floor))
        self.q_angles = np.tile(enc.initial_angles, (self.w, 1))
        if self.init_confidence > 0:
            for good_id, good in enumerate(self.goods):
                self._bias_toward(good_id, good.code)
        self.q_probabilities = np.sin(self.q_angles) ** 2

    def _bias_toward(self, good_id, code):
        """把路徑上的閘設成以 init_confidence 機率觀測到 ``code``。"""
        target, path = self.q_encoding.elite_path(code)
        p = np.where(target, self.init_confidence, 1 - self.init_confidence)
        biased = np.clip(np.arcsin(np.sqrt(p)), self.angle_low, self.angle_high)
        self.q_angles[good_id] = np.where(path, biased, self.q_angles[good_id])

    def _gate_mask(self, left_sensor, right_sensor):
        genes = self.q_encoding.gene_ids
        return (genes >= left_sensor * 2) & (genes < right_sensor * 2)

    def make_quantum_children(self, problem, searcher, good_id, region):
        """child1 從 Q_k 觀測該段；child2 同 Ring_SE。回傳觀測紀錄。"""
        region = self._resolve_region(region)
        sensor_span = self.select_sensor_span(region)
        _, left, right = sensor_span
        observed, bits = self.q_encoding.observe(
            self.q_probabilities[good_id], self.rng
        )
        child1 = searcher.copy()
        child1.code[left * 2:right * 2] = observed.code[left * 2:right * 2]

        child2 = self.goods[good_id].copy()
        child2.code[left * 2:right * 2] = searcher.code[left * 2:right * 2]
        if self.random.random() < self.mutation_rate:
            self.mutate_candidate(problem, child2, sensor_span=sensor_span)
        return child1, child2, (bits, self._gate_mask(left, right))

    def rotate(self, good_id, records, child1_codes, child1_fitness, good_code,
               good_fitness):
        """在觀測過的段內，把 Q_k 轉向每對（觀測解, b_k）中較好的一方。"""
        delta = np.zeros(self.q_encoding.length)
        for (bits, mask), code, fitness in zip(records, child1_codes,
                                               child1_fitness):
            if fitness > good_fitness + self.IMPROVEMENT_TOLERANCE:
                target, path = self.q_encoding.elite_path(code)
            elif fitness < good_fitness - self.IMPROVEMENT_TOLERANCE:
                target, path = self.q_encoding.elite_path(good_code)
            else:
                continue
            move = mask & path & (bits != target)
            delta += move * np.where(target, self.rotation, -self.rotation)
        angles = self.q_angles[good_id]
        angles += delta
        np.clip(angles, self.angle_low, self.angle_high, out=angles)
        self.q_probabilities[good_id] = np.sin(angles) ** 2

    def vision_search(self, problem):
        active_segments = self.selected_regions.copy()
        goods_fitness_before = self.goods_fitness.copy()
        goods_before = [good.code.copy() for good in self.goods]
        searcher_fitness_before = self.searcher_fitness.copy()
        children = []
        records = [[] for _ in range(self.w)]

        for searcher_id, segment in enumerate(active_segments):
            row = []
            for good_id in range(self.w):
                child1, child2, record = self.make_quantum_children(
                    problem, self.searchers[searcher_id], good_id, int(segment)
                )
                row.append(child1)
                row.append(child2)
                records[good_id].append(record)
            children.append(row)

        child_fitness = self.evaluate_investments(problem, children)
        child1_fitness = child_fitness[:, 0::2]
        child2_fitness = child_fitness[:, 1::2]

        # evaluate_investments 可能把 children 換成評估後的物件，所以評估後再取 code。
        for good_id in range(self.w):
            self.rotate(
                good_id,
                records[good_id],
                [children[s][good_id * 2].code for s in range(self.n)],
                child1_fitness[:, good_id],
                goods_before[good_id],
                goods_fitness_before[good_id],
            )

        accept_searcher_children(
            self, children, child1_fitness, searcher_fitness_before
        )
        accept_good_children(
            self.goods,
            self.goods_fitness,
            goods_fitness_before,
            children,
            child2_fitness,
            range(self.n),
            self.GOOD_ACCEPT_TOLERANCE,
        )
        self.selected_regions = self.select_regions()


__all__ = ["QRing_SE"]
