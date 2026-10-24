"""SA-SETS：SETS 的流程，只改 identity sensor 的選法。

identity sensor：在靠近 BS 的 sensor 裡選電量最高的 log2(h) 顆（SETS 取編號最小的）。

市場流程、初始解（隨機＋CCS）、分區對齊、交配突變、Beta 記憶都沿用 :class:`SETS`。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.sets.SETS import SETS


class SA_SETS(SETS):
    """SETS with near-BS high-energy identity sensors."""

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


__all__ = ["SA_SETS"]
