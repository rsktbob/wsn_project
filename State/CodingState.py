"""保留舊演算法介面的感測器編碼狀態，路由改由共用服務建立。"""

import random

import numpy as np


class CodingState:
    """供 Algorithm2 舊呼叫端使用的「染色體與狀態合一」物件。

    排程與優先權解碼維持歷史行為；轉送樹的實際建立則全部委派給
    ``Problem.routing_service``，避免不同編碼各自維護一份路由公式。
    """

    def __init__(self, P):
        """依問題規模建立空白欄位，尚不產生染色體或解碼結果。"""
        self.P = P
        self.rank_prec = 10
        self.FMAX = max(P.LEVEL, self.rank_prec)
        self.length = P.SENSOR_NUMBER * 2
        self.levels = None
        self.next_hops = None
        self.tx_load = None
        self.remaining_capacity = None
        self.paths = None
        self.sensing_radii = None
        self.use_continuous_radius = False
        self.code = None
        self.target_red = None
        self.value = None

    def CreateCode_Zero(self, P):
        """建立所有基因皆為 0 的感測器染色體。"""
        self.code = np.zeros(P.SENSOR_NUMBER * 2, dtype=int)

    def CreateCode(self, P):
        """隨機初始化染色體，並為每個可覆蓋 target 植入一個候選感測器。"""
        self.code = np.random.randint(
            0,
            self.FMAX,
            P.SENSOR_NUMBER * 2,
        )
        # 偶數位置是感測選項。逐 target 隨機挑一個可行候選，可降低
        # 完全隨機初始化時大量 target 沒有任何感測器覆蓋的機率。
        for candidates in P.cover_candidates:
            if candidates:
                sensor_id, level = random.choice(candidates)
                self.code[int(sensor_id) * 2] = int(level)

    def RandomValue(self, P, value):
        """回傳舊版突變操作使用的隨機基因值。"""
        return random.randint(0, self.FMAX - 1)

    def Evaluate(self, P):
        """透過 ``Problem`` 共用評估器計算目前已解碼狀態。"""
        return P.evaluate_state(self)

    def Decode(self, P):
        """依序解碼路由優先權、感測排程與多跳路由。"""
        # 每次解碼都重建狀態陣列，避免沿用上一次候選解的路由或負載。
        self.levels = np.zeros(P.SENSOR_NUMBER, dtype=int)
        self.next_hops = np.full(P.SENSOR_NUMBER, -1, dtype=int)
        self.remaining_capacity = np.full(
            P.DEVICE_NUMBER, -1.0, dtype=float
        )
        self.tx_load = np.zeros(P.SENSOR_NUMBER, dtype=int)

        schedule_code = self.code[0::2]
        priority_code = self.code[1::2]
        # 奇數基因先決定感測器處理順序；偶數基因再決定感測範圍。
        routing_priority = self.decode_routing_priority(
            P,
            priority_code,
        )
        open_sensors = self.decode_schedule(
            P,
            schedule_code,
            routing_priority,
        )
        self.decode_routes(P, open_sensors)
        # 路由可能關閉容量不足的感測器，因此最後重新同步實際半徑。
        P.resolve_state_radius(self)

    def RankDecoding(self, P, rank_code):
        """結合基因、基地台距離與剩餘電量，產生感測器處理順序。"""
        normalized_rank = (
            (rank_code % self.rank_prec)
            / (self.rank_prec - 1)
        )
        priority = (P.proximity_score + P.energy_score) * normalized_rank
        # 高分感測器先覆蓋 target，也會先加入可供後續節點使用的路由樹。
        return np.argsort(priority)[::-1]

    def SchDecoding(self, P, sch_code, routing_priority, test=False):
        """依優先順序開啟能增加新覆蓋的感測器，並回傳開啟列表。"""
        # 每顆 sensor 在 bucketed／exact 模式下可能有不同的感測選項數；
        # discrete 模式的 option_count 仍等於 P.LEVEL，因此舊行為不變。
        schedule = np.asarray(sch_code, dtype=int).copy()
        for sensor_id in range(P.SENSOR_NUMBER):
            schedule[sensor_id] %= P.sensing_option_count(sensor_id)
        uncovered = np.ones(P.TARGET_NUMBER, dtype=int)
        open_sensors = []
        for sensor_id in routing_priority:
            covered_targets = np.where(
                P.coverage_table[:, sensor_id, schedule[sensor_id]]
            )[0]
            # 只有至少涵蓋一個尚未覆蓋的 target 才啟用此感測器，
            # 因此列表順序同時成為後續路由樹的插入順序。
            if np.sum(uncovered[covered_targets]) > 0:
                uncovered[covered_targets] = 0
                self.levels[sensor_id] = schedule[sensor_id]
                open_sensors.append(
                    [int(sensor_id), int(self.levels[sensor_id])]
                )
        return open_sensors

    def RouDecoding(self, P, open_sensors, test=False):
        """使用共用路由服務建立轉送樹，並停用容量不足的排程基因。"""
        routing_result = P.routing_service.build_routes(
            self,
            open_sensors,
            test=test,
        )
        for sensor_id in (
            routing_result.capacity_failed_ids
        ):
            # 保留歷史副作用：路由失敗時連染色體中的感測基因也歸零。
            self.code[int(sensor_id) * 2] = 0
        return routing_result

    def Copy(self):
        """複製舊式狀態中已建立的染色體、排程、路由與半徑。"""
        copied_state = CodingState(self.P)
        if self.levels is not None:
            copied_state.levels = self.levels.copy()
        if self.next_hops is not None:
            copied_state.next_hops = self.next_hops.copy()
        if self.code is not None:
            copied_state.code = self.code.copy()
        if self.sensing_radii is not None:
            copied_state.sensing_radii = self.sensing_radii.copy()
        copied_state.use_continuous_radius = self.use_continuous_radius
        return copied_state

    def create_zero_code(self, P):
        """新命名介面：建立全零染色體。"""
        return self.CreateCode_Zero(P)

    def create_random_code(self, P):
        """新命名介面：建立具覆蓋種子的隨機染色體。"""
        return self.CreateCode(P)

    def random_gene_value(self, P, value):
        """新命名介面：產生一個隨機基因值。"""
        return self.RandomValue(P, value)

    def evaluate(self, P):
        """新命名介面：評估目前狀態。"""
        return self.Evaluate(P)

    def decode(self, P):
        """新命名介面：解碼目前染色體。"""
        return self.Decode(P)

    def decode_routing_priority(self, P, rank_code):
        """新命名介面：將優先權基因解碼成感測器順序。"""
        return self.RankDecoding(P, rank_code)

    def decode_schedule(
        self,
        P,
        schedule_code,
        routing_priority,
        test=False,
    ):
        """新命名介面：依指定順序解碼感測排程。"""
        return self.SchDecoding(
            P,
            schedule_code,
            routing_priority,
            test,
        )

    def decode_routes(self, P, open_sensors, test=False):
        """新命名介面：替已開啟的感測器建立路由。"""
        return self.RouDecoding(P, open_sensors, test)

    def copy(self):
        """新命名介面：複製此舊式狀態。"""
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


__all__ = ["CodingState"]
