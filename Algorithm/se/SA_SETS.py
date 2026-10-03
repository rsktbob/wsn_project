"""SA-SETS：在 SETS 上改用感測器感知的 identity sensor 與 CCS 初始解。

市場流程、分區對齊、交配突變、Beta 記憶都沿用 :class:`SETS`；
這裡只覆寫 identity sensor 的選法與初始 chromosome。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.SETS import SETS
from State.SensorEncoding import SensorEncoding


class SA_SETS(SETS):
    """SETS with near-BS high-energy identity sensors and CCS seeding."""

    def select_identity_sensors(self, problem):
        """Select the two high-energy sensors near the base station."""
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

        # Preserve the original fallback: use the first live sensors when the
        # near-base classification cannot provide enough identities.
        return [
            int(sensor_id)
            for sensor_id in distance_order
            if problem.energy[sensor_id] >= problem.liveJ
        ][: self.identity_bit_count]

    def create_candidate(self, problem, region=None):
        """以隨機 chromosome 為底，依 CCS 為每個 target 放入一組覆蓋。"""
        candidate = SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
            rng=self.rng,
        )

        # 原始 SA-SETS 初始化：每個 target 從其 CCS 候選表隨機抽一組
        # (sensor, level)，直接寫入該 sensor 的 sensing gene。若後面的
        # target 再選到同一 sensor，則以後一次抽樣覆寫，保留原本的行為。
        # 路由 priority 仍由 SensorEncoding.random() 隨機產生。
        for target_candidates in problem.cover_candidates:
            if target_candidates:
                sensor_id, level = self.random.choice(target_candidates)
                candidate.code[int(sensor_id) * 2] = int(level)

        if region is not None:
            self.align_region(problem, candidate, region)
        return candidate


__all__ = ["SA_SETS"]
