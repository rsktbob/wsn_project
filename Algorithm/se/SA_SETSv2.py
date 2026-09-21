"""嚴格依照 2023 論文流程的 SA-SETSv2。"""

import numpy as np

from Algorithm.se.SETSv2 import SETSv2


class SA_SETSv2(SETSv2):
    """嚴格沿用 SETSv2 論文市場流程的 2023 自適應分類版本。

    本類別只改變識別感測器的選擇方式；不繼承舊 ``SA_SETS``，因為兩者的
    市場搜尋、評估預算與搜尋者更新流程並不相同。
    """
    """2023 SA-SETS：依電量與基地台距離自適應選擇識別感測器。"""

    def __init__(
        self,
        problem,
        n=8,
        h=4,
        w=2,
        player=2,
        crossover_rate=1.0,
        mutation_rate=0.4,
        adaptive_constant=0.001,
        energy_weight=0.5,
        distance_weight=0.5,
        seed=None,
    ):
        """設定公式 (44) 的能量/距離權重，其餘流程交由 SETSv2。"""
        self.energy_weight = float(energy_weight)
        self.distance_weight = float(distance_weight)
        if self.energy_weight < 0.0 or self.distance_weight < 0.0:
            raise ValueError("自適應分類權重不可為負數")
        if self.energy_weight + self.distance_weight <= 0.0:
            raise ValueError("自適應分類至少需要一個正權重")
        super().__init__(
            problem,
            n=n,
            h=h,
            w=w,
            player=player,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            adaptive_constant=adaptive_constant,
            seed=seed,
        )

    def select_identity_sensors(self, problem):
        """依論文自適應分類範圍與加權分數選出識別感測器。

        先用第 ``I_bits`` 近感測器距離加最大感測半徑界定候選集合，再以
        剩餘能量及正規化反距離排序；同分時優先選擇距離較近者。
        """
        distances = np.asarray(
            problem.distances[: problem.SENSOR_NUMBER, problem.BSID],
            dtype=float,
        )
        order = np.argsort(distances, kind="stable")
        boundary_sensor = order[self.identity_bit_count - 1]
        maximum_sensing_radius = problem.maximum_radius()
        candidate_limit = (
            distances[boundary_sensor] + maximum_sensing_radius
        )
        # 對應論文式 (43) 的基地台最大範圍候選集合 gamma。
        candidates = np.where(distances <= candidate_limit)[0]

        if len(candidates) < self.identity_bit_count:
            candidates = order[: self.identity_bit_count]

        inverse_distance = 1.0 / np.maximum(
            distances[candidates], np.finfo(float).eps
        )
        normalized_inverse_distance = inverse_distance / np.max(
            inverse_distance
        )
        # 對應論文式 (44)；反距離越大代表越靠近基地台。
        scores = (
            self.energy_weight
            * np.asarray(problem.energy[candidates], dtype=float)
            + self.distance_weight * normalized_inverse_distance
        )
        ranked = np.lexsort((distances[candidates], -scores))
        selected = candidates[ranked[: self.identity_bit_count]]
        return selected.astype(int).tolist()
