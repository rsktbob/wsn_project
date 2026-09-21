"""保留舊演算法介面的 target 編碼狀態，路由改由共用服務建立。"""

import random

import numpy as np


class TargetCodingState:
    """供舊演算法使用的 target 優先權編碼與解碼狀態。

    為維持相容性，覆蓋解碼仍保留在本類別；多跳轉送樹則完全委派給
    ``Problem.routing_service``，以便與新版編碼共用相同路由規則。
    """

    RANK_PRECISION = 100

    def __init__(self, P, route_selector=None):
        """依問題規模建立空白欄位，並可注入自訂下一跳選擇器。"""
        self.P = P
        self.route_selector = route_selector
        self.length = P.TARGET_NUMBER
        self.code = None
        self.levels = None
        self.next_hops = None
        self.tx_load = None
        self.remaining_capacity = None
        self.paths = None
        self.sensing_radii = None
        self.use_continuous_radius = False
        self.target_assignment = None
        self.target_red = None
        self.value = None
        self.objectives = None
        self.constraint_violation = 0
        self.rank = 0
        self.crowding_distance = 0

    def CreateCode_Zero(self, P):
        """建立每個 target 基因皆為 0 的染色體。"""
        self.code = np.zeros(P.TARGET_NUMBER, dtype=int)

    def CreateCode(self, P):
        """在完整 target 基因範圍內均勻隨機初始化染色體。"""
        self.code = np.random.randint(0, 10000, P.TARGET_NUMBER)

    def RandomValue(self, P, target_id):
        """回傳舊版突變操作使用的隨機 target 基因值。"""
        return random.randint(0, (P.SENSOR_NUMBER + 100) * 4 - 1)

    def Evaluate(self, P):
        """透過 ``Problem`` 共用評估器計算目前解碼狀態。"""
        return P.evaluate_state(self)

    def Decode(self, P):
        """重建狀態後依序解碼 target 覆蓋與多跳路由。"""
        # 每次解碼都清除舊排程、路由、容量與負載，避免候選解互相污染。
        self.levels = np.zeros(P.SENSOR_NUMBER, dtype=int)
        self.next_hops = np.full(P.SENSOR_NUMBER, -1, dtype=int)
        self.remaining_capacity = np.full(
            P.DEVICE_NUMBER, -1.0, dtype=float
        )
        self.tx_load = np.zeros(P.SENSOR_NUMBER, dtype=int)
        self.sensing_radii = None
        self.use_continuous_radius = False
        self.target_assignment = np.full(P.TARGET_NUMBER, -1, dtype=int)

        open_sensors = self.CoverageDecoding(P)
        # 路由容量會受到感測半徑耗能影響，建立路由前先解析一次半徑。
        P.resolve_state_radius(self)
        self.RouDecoding(P, open_sensors)
        # 路由失敗可能關閉感測器，完成後再次同步最終半徑。
        P.resolve_state_radius(self)

    def CoverageDecoding(self, P):
        """依 target 基因選擇感測器，並移除不影響覆蓋的冗餘節點。"""
        priorities = self._target_priority(P)
        open_order = []
        seen_sensors = set()
        # 排序值較小的 target 先處理；感測器第一次出現的先後，會直接
        # 成為路由解碼順序，因此 target 優先值也間接影響路由拓樸。
        for target_id in np.argsort(priorities):
            candidates = P.cover_candidates[target_id]
            if len(candidates) == 0:
                continue

            raw_gene = max(0, int(self.code[target_id]))
            # 高位映射到候選感測器，低兩位則用於 target 排序。
            schedule_code = raw_gene // self.RANK_PRECISION
            board = self.RANK_PRECISION // len(candidates) + 1
            candidate_index = min(
                schedule_code // board,
                len(candidates) - 1,
            )
            sensor_id, level = candidates[candidate_index]
            self.levels[sensor_id] = max(self.levels[sensor_id], level)
            self.target_assignment[target_id] = sensor_id
            if sensor_id not in seen_sensors:
                seen_sensors.add(sensor_id)
                open_order.append(sensor_id)

        coverage_count = np.zeros(P.TARGET_NUMBER, dtype=int)
        # 依實際感測範圍統計覆蓋次數，可能包含非指派但順帶覆蓋的 target。
        for sensor_id in open_order:
            level = int(self.levels[sensor_id])
            coverage_count += (
                P.coverage_table[:, sensor_id, level] > 0
            ).astype(int)

        retained = []
        # 從 open_order 前端開始嘗試刪除；刪除後立即遞減覆蓋次數，
        # 因此先前的決定會限制後續感測器是否仍可刪除。
        for sensor_id in open_order:
            level = int(self.levels[sensor_id])
            covered = P.coverage_table[:, sensor_id, level] > 0
            if np.any(covered) and np.all(coverage_count[covered] >= 2):
                coverage_count[covered] -= 1
                self.levels[sensor_id] = 0
            else:
                retained.append(sensor_id)

        return [
            [int(sensor_id), int(self.levels[sensor_id])]
            for sensor_id in retained
        ]

    def _target_priority(self, P):
        """計算 target 的距離、能量風險與基因排序值總和。"""
        target_distance = P.G.CalDistance(
            P.target,
            np.array([P.BS]),
        )[:, 0]
        max_distance = (
            float(np.max(target_distance)) if len(target_distance) else 0.0
        )
        distance_rank = (
            target_distance / max_distance
            if max_distance > 0
            else np.zeros(P.TARGET_NUMBER, dtype=float)
        )

        target_energy = P.calculate_target_remaining_energy(P.energy)
        max_energy = (
            float(np.max(target_energy)) if len(target_energy) else 0.0
        )
        energy_rank = (
            (max_energy - target_energy) / max_energy
            if max_energy > 0
            else np.zeros(P.TARGET_NUMBER, dtype=float)
        )

        gene_rank = (
            np.mod(self.code, self.RANK_PRECISION)
            / self.RANK_PRECISION
        )
        # CoverageDecoding 使用 argsort 由小到大處理，數值越小越早解碼。
        return distance_rank + energy_rank + gene_rank

    def RouDecoding(self, P, open_sensors, test=False):
        """按感測器第一次出現順序，使用共用服務建立多跳路由樹。"""
        return P.routing_service.build_routes(
            self,
            open_sensors,
            route_selector=self.route_selector,
            test=test,
        )

    def Copy(self):
        """複製染色體、解碼狀態與演算法評估欄位。"""
        copied_state = self.__class__(
            self.P,
            route_selector=self.route_selector,
        )
        copied_state.length = self.length
        copied_state.constraint_violation = self.constraint_violation
        copied_state.rank = self.rank
        copied_state.crowding_distance = self.crowding_distance
        copied_state.use_continuous_radius = self.use_continuous_radius

        for field_name in (
            "code",
            "levels",
            "next_hops",
            "tx_load",
            "remaining_capacity",
            "sensing_radii",
            "target_assignment",
            "target_red",
            "objectives",
        ):
            # 陣列欄位逐一複製，避免父代與子代共享可變資料。
            value = getattr(self, field_name)
            if value is not None:
                setattr(copied_state, field_name, value.copy())
        if self.paths is not None:
            copied_state.paths = {
                key: value.copy() for key, value in self.paths.items()
            }
        copied_state.value = self.value
        return copied_state

    def create_zero_code(self, P):
        """新命名介面：建立全零 target 染色體。"""
        return self.CreateCode_Zero(P)

    def create_random_code(self, P):
        """新命名介面：建立隨機 target 染色體。"""
        return self.CreateCode(P)

    def random_gene_value(self, P, target_id):
        """新命名介面：產生一個隨機 target 基因值。"""
        return self.RandomValue(P, target_id)

    def evaluate(self, P):
        """新命名介面：評估目前狀態。"""
        return self.Evaluate(P)

    def decode(self, P):
        """新命名介面：解碼目前 target 染色體。"""
        return self.Decode(P)

    def decode_coverage(self, P):
        """新命名介面：解碼 target 覆蓋與感測器列表。"""
        return self.CoverageDecoding(P)

    def decode_routes(self, P, open_sensors, test=False):
        """新命名介面：替已保留的感測器建立路由。"""
        return self.RouDecoding(P, open_sensors, test)

    def copy(self):
        """新命名介面：複製此舊式 target 狀態。"""
        return self.Copy()

    # 舊名稱僅留在類別尾端當過渡入口；類別內部只用新名稱。
    sch = property(
        lambda self: self.levels,
        lambda self, value: setattr(self, "levels", value),
    )
    rou = property(
        lambda self: self.next_hops,
        lambda self, value: setattr(self, "next_hops", value),
    )
    use = property(
        lambda self: self.tx_load,
        lambda self, value: setattr(self, "tx_load", value),
    )
    cap = property(
        lambda self: self.remaining_capacity,
        lambda self, value: setattr(self, "remaining_capacity", value),
    )
    path = property(
        lambda self: self.paths,
        lambda self, value: setattr(self, "paths", value),
    )
    radius = property(
        lambda self: self.sensing_radii,
        lambda self, value: setattr(self, "sensing_radii", value),
    )


__all__ = ["TargetCodingState"]
