"""以 target 為單位的決策編碼與決定性解碼器。"""

from __future__ import annotations

import numpy as np

from State.Encoding import Encoding
from State.State import State


class TargetEncoding(Encoding):
    """每個 target 使用一個基因，同時選擇候選感測器與處理優先值。

    本類別保存染色體並提供共用均勻初始化；需要特殊初始化策略的演算法，
    仍可自行建立 code 後傳入 ``TargetEncoding(code)``。
    """

    RANK_PRECISION = 100
    GENE_DOMAIN = 10000

    def __init__(self, code):
        """建立 target 編碼；路由選擇器預設使用容量貪婪規則。"""
        super().__init__(code)
        self.route_selector = None

    @classmethod
    def random(cls, target_count, rng=None, *, route_selector=None):
        """均勻建立 target 染色體，並可附加自訂下一跳選擇器。"""
        target_count = int(target_count)
        if target_count <= 0:
            raise ValueError("target_count must be positive")

        if rng is None:
            code = np.random.randint(0, cls.GENE_DOMAIN, target_count)
        else:
            code = rng.integers(0, cls.GENE_DOMAIN, target_count)
        encoding = cls(code)
        encoding.route_selector = route_selector
        return encoding

    def copy(self):
        """複製染色體並沿用同一個路由選擇政策。"""
        copied = self.__class__(self.code.copy())
        copied.route_selector = self.route_selector
        return copied

    def decode(self, problem):
        """依 target 順序解碼 assignments，刪除冗餘 sensor 後建立路由。"""
        if self.code is None:
            raise ValueError("TargetEncoding.code has not been initialized")

        # TargetEncoding 額外記錄每個 target 最後由哪顆 sensor 覆蓋。
        state = State.empty(problem, with_target_assignment=True)

        # 先完成所有 target assignments，再由整體覆蓋關係刪除多餘
        # sensor；回傳的順序同時也是 RoutingService 的處理順序。
        open_sensors = self._decode_coverage(problem, state)

        # 先同步排程對應的物理半徑，讓路由容量使用正確的感測耗能。
        problem.resolve_state_radius(state)

        # RoutingService 依序把 sensor 接入目前已建立的 routing tree。
        problem.routing_service.build_routes(
            state,
            open_sensors,
            route_selector=self.route_selector,
        )
        # 路由可能關閉容量不足的感測器，因此完成後再同步一次半徑。
        problem.resolve_state_radius(state)
        return state

    def _decode_coverage(self, problem, state):
        """先選完所有 target assignments，再移除覆蓋冗餘 sensor。"""
        priorities = self._target_priorities(problem)
        open_order = []
        seen_sensors = set()
        # 排序值由小到大處理；感測器第一次被 target 選中時的順序，
        # 會成為後續建立路由樹的處理順序。
        for target_id in np.argsort(priorities):
            candidates = problem.cover_candidates[target_id]
            if not candidates:
                continue

            raw_gene = max(0, int(self.code[target_id]))
            # 高位部分選候選感測器，低位部分保留給 target 排序。
            schedule_gene = raw_gene // self.RANK_PRECISION
            board = self.RANK_PRECISION // len(candidates) + 1
            candidate_index = min(
                schedule_gene // board,
                len(candidates) - 1,
            )
            sensor_id, level = candidates[candidate_index]
            state.levels[sensor_id] = max(
                state.levels[sensor_id], level
            )
            state.target_assignment[target_id] = sensor_id
            if sensor_id not in seen_sensors:
                seen_sensors.add(sensor_id)
                open_order.append(sensor_id)

        # 一次建立所有已選 sensor 的覆蓋矩陣。冗餘移除仍依照
        # open_order 逐顆執行，因此只減少重複索引，不改變舊版結果。
        if open_order:
            sensor_ids = np.asarray(open_order, dtype=int)
            levels = state.levels[sensor_ids].astype(int)
            coverage_matrix = (
                problem.coverage_table[:, sensor_ids, levels] > 0
            )
            coverage_count = np.sum(
                coverage_matrix,
                axis=1,
                dtype=int,
            )
        else:
            coverage_matrix = np.empty(
                (problem.TARGET_NUMBER, 0),
                dtype=bool,
            )
            coverage_count = np.zeros(
                problem.TARGET_NUMBER,
                dtype=int,
            )
        # 統計每個 target 目前被多少啟用感測器實際覆蓋，而非只看指派關係。

        retained_reversed = []
        # 從後加入、較低優先的 sensor 開始嘗試刪除；只有其所有
        # 覆蓋 target 都有其他 sensor 接手時才能刪除。每次刪除後
        # 立即更新 coverage_count，避免後續又刪掉最後一個覆蓋來源。
        for column in range(len(open_order) - 1, -1, -1):
            sensor_id = open_order[column]
            covered = coverage_matrix[:, column]
            if np.any(covered) and np.all(coverage_count[covered] >= 2):
                coverage_count[covered] -= 1
                state.levels[sensor_id] = 0
            else:
                retained_reversed.append(sensor_id)

        # 刪除判斷由後往前，但建立路由仍保留原本的高優先順序。
        retained = list(reversed(retained_reversed))

        return [
            [int(sensor_id), int(state.levels[sensor_id])]
            for sensor_id in retained
        ]

    def _target_priorities(self, problem):
        """結合 target 距離、候選區域能量與基因低位計算排序值。"""
        target_distance = problem.G.CalDistance(
            problem.target,
            np.array([problem.BS]),
        )[:, 0]
        max_distance = (
            float(np.max(target_distance)) if len(target_distance) else 0.0
        )
        distance_rank = (
            target_distance / max_distance
            if max_distance > 0
            else np.zeros(problem.TARGET_NUMBER, dtype=float)
        )

        target_energy = problem.calculate_target_remaining_energy(
            problem.energy
        )
        max_energy = (
            float(np.max(target_energy)) if len(target_energy) else 0.0
        )
        energy_rank = (
            (max_energy - target_energy) / max_energy
            if max_energy > 0
            else np.zeros(problem.TARGET_NUMBER, dtype=float)
        )
        gene_rank = (
            np.mod(self.code, self.RANK_PRECISION) / self.RANK_PRECISION
        )
        # _decode_coverage 使用 argsort 由小到大處理，故此處數值越小越早。
        return distance_rank + energy_rank + gene_rank

__all__ = ["TargetEncoding"]
