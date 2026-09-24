"""只編碼優先序的感測器編碼：level 由貪婪規則決定。"""

from __future__ import annotations

import numpy as np

from State.Encoding import Encoding
from State.State import State


class PriorityEncoding(Encoding):
    """每顆感測器使用 priority 基因，不含 level。

    ``split_priority=True``：兩個基因 [a0, b0, a1, b1, ...]，``a`` 決定排程
    時被考慮的順序，``b`` 決定入選 sensor 接上路由樹的順序。
    ``split_priority=False``：一個基因 [a0, a1, ...]，兩個順序相同，用來
    和拆開的版本做對照。

    level 不在染色體中：依排程順序處理每顆 sensor，選「新增覆蓋最多的
    最小 level」；沒有新增覆蓋就不開。之後照 SensorEncoding v3 的規則移除
    冗餘並建立路由，路由失敗時保留 level，也不修改染色體。
    """

    RANK_PRECISION = 10
    # 開啟順序的 base：
    #   "sum"  = energy_score + proximity_score（與 SensorEncoding 相同）
    #   "cost" = (E_s / E_max) × (C_min / C_s)，C_s 為 s 傳一個封包回 BS 的
    #            最小路徑耗能；電量相對開啟代價越高越先開
    ACTIVATION_BASE = "sum"
    # 路由順序的 base（只在 split_priority 時使用）：
    #   "sum"    = energy_score + proximity_score
    #   "linear" = (1 - d_s / d_max) + ROUTING_ENERGY_WEIGHT × (E_s / E_max)
    #   "energy" = (E_s / E_max) + ROUTING_DISTANCE_WEIGHT × (1 - d_s / d_max)
    ROUTING_BASE = "sum"
    ROUTING_ENERGY_WEIGHT = 0.2
    ROUTING_DISTANCE_WEIGHT = 0.2

    def __init__(self, code, *, split_priority=True):
        super().__init__(code)
        self.split_priority = bool(split_priority)
        if self.split_priority and self.length % 2 != 0:
            raise ValueError("priority genes must be [a0, b0, a1, b1, ...]")

    @classmethod
    def random(cls, sensor_count, rng=None, *, split_priority=True):
        """均勻建立 priority 基因。"""
        size = int(sensor_count) * (2 if split_priority else 1)
        if rng is None:
            code = np.random.randint(0, cls.RANK_PRECISION, size)
        else:
            code = rng.integers(0, cls.RANK_PRECISION, size)
        return cls(code, split_priority=split_priority)

    def copy(self):
        return self.__class__(
            self.code.copy(), split_priority=self.split_priority
        )

    def _order(self, base, genes):
        """priority = base × (0.5 + gene / 9)，由高到低排序。"""
        genes = np.asarray(genes, dtype=float)
        priority = base * (0.5 + genes / (self.RANK_PRECISION - 1))
        return np.argsort(priority)[::-1]

    @staticmethod
    def _bs_distance(problem):
        return np.asarray(problem.distances)[
            : problem.SENSOR_NUMBER, problem.BSID
        ]

    @classmethod
    def _delivery_cost(cls, problem):
        """每顆 sensor 傳一個封包回 BS 的最小路徑耗能 C_s。

        每跳耗能沿用容量公式的 amp_cost + 2 × circuit_cost。實際路由允許
        任何未封鎖的 link（d(s, p) <= d(s, BS)），這裡另外限制只經過更靠近
        BS 的節點，才能依 BS 距離由近到遠做一次動態規劃。6 張測試地圖上
        與不加此限制的 Dijkstra 結果完全相同（繞遠路在 d²/d⁴ 模型下不省電）。
        只和位置有關，快取在 problem 上（移動後 problem 會重建）。
        """
        cached = getattr(problem, "_priority_delivery_cost", None)
        if cached is not None:
            return cached
        sensor_count = problem.SENSOR_NUMBER
        bs_id = problem.BSID
        distances = np.asarray(problem.distances)
        hop = np.asarray(problem.amp_cost) + 2 * problem.circuit_cost
        to_bs = cls._bs_distance(problem)
        allowed = distances[:sensor_count] < problem.blocked_distance / 100
        cost = np.full(sensor_count, np.inf)
        for sensor_id in np.argsort(to_bs):
            best = hop[sensor_id, bs_id] if allowed[sensor_id, bs_id] else np.inf
            nearer = np.where(
                (to_bs < to_bs[sensor_id]) & allowed[sensor_id, :sensor_count]
            )[0]
            if len(nearer):
                best = min(
                    best, float(np.min(hop[sensor_id, nearer] + cost[nearer]))
                )
            cost[sensor_id] = best
        problem._priority_delivery_cost = cost
        return cost

    def _activation_base(self, problem):
        if self.ACTIVATION_BASE == "sum":
            return problem.energy_score + problem.proximity_score
        if self.ACTIVATION_BASE == "cost":
            cost = self._delivery_cost(problem)
            finite = np.isfinite(cost)
            efficiency = np.zeros_like(cost)
            if np.any(finite):
                efficiency[finite] = np.min(cost[finite]) / cost[finite]
            return problem.energy_score * efficiency
        raise ValueError(f"unknown ACTIVATION_BASE {self.ACTIVATION_BASE!r}")

    def _routing_base(self, problem):
        if self.ROUTING_BASE == "sum":
            return problem.energy_score + problem.proximity_score
        if self.ROUTING_BASE in ("linear", "energy"):
            to_bs = self._bs_distance(problem)
            max_distance = float(np.max(to_bs))
            closeness = (
                1.0 - to_bs / max_distance
                if max_distance > 0
                else np.zeros_like(to_bs)
            )
            if self.ROUTING_BASE == "linear":
                return (
                    closeness
                    + self.ROUTING_ENERGY_WEIGHT * problem.energy_score
                )
            return (
                problem.energy_score
                + self.ROUTING_DISTANCE_WEIGHT * closeness
            )
        raise ValueError(f"unknown ROUTING_BASE {self.ROUTING_BASE!r}")

    def decode(self, problem):
        if self.code is None:
            raise ValueError("PriorityEncoding.code has not been initialized")
        code = np.asarray(self.code, dtype=int)
        genes_per_sensor = 2 if self.split_priority else 1
        expected_shape = (problem.SENSOR_NUMBER * genes_per_sensor,)
        if code.shape != expected_shape:
            raise ValueError(
                f"priority code must have shape {expected_shape}, "
                f"got {code.shape}"
            )
        schedule_order = self._order(
            self._activation_base(problem), code[::genes_per_sensor]
        )
        state = State.empty(problem)
        coverage = np.asarray(problem.coverage_table) > 0
        option_counts = np.asarray(problem.radius_option_counts, dtype=int)
        uncovered = np.ones(problem.TARGET_NUMBER, dtype=bool)

        opened = []
        for sensor_id in schedule_order:
            sensor_id = int(sensor_id)
            # 每個 level 能新增覆蓋的 target 數；level 0 是關閉。
            gains = uncovered.astype(int) @ coverage[
                :, sensor_id, 1:option_counts[sensor_id]
            ].astype(int)
            if gains.size == 0 or gains.max() == 0:
                continue
            level = int(np.argmax(gains)) + 1  # argmax 取第一個 → 最小 level
            uncovered &= ~coverage[:, sensor_id, level]
            state.levels[sensor_id] = level
            opened.append(sensor_id)

        # 與 SensorEncoding v2/v3 相同：從排程順序尾端移除冗餘。
        coverage_count = np.zeros(problem.TARGET_NUMBER, dtype=int)
        for sensor_id in opened:
            coverage_count += coverage[:, sensor_id, state.levels[sensor_id]]
        retained = []
        for sensor_id in reversed(opened):
            covered = coverage[:, sensor_id, state.levels[sensor_id]]
            if np.any(covered) and np.all(coverage_count[covered] >= 2):
                coverage_count[covered] -= 1
                state.levels[sensor_id] = 0
            else:
                retained.append(sensor_id)

        if self.split_priority:
            rank = np.empty(problem.SENSOR_NUMBER, dtype=int)
            rank[self._order(self._routing_base(problem), code[1::2])] = np.arange(
                problem.SENSOR_NUMBER
            )
            routing_ids = sorted(retained, key=lambda s: rank[s])
        else:
            routing_ids = list(reversed(retained))  # 排程順序

        problem.routing_service.build_routes(
            state, routing_ids, disable_failed=False
        )
        problem.resolve_state_radius(state)
        return state


__all__ = ["PriorityEncoding"]
