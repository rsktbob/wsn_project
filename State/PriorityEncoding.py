"""只編碼優先序的感測器編碼：level 由貪婪規則決定。"""

from __future__ import annotations

import numpy as np

from Problem.services.evaluation_kernels import (
    NUMBA_AVAILABLE,
    decode_priority_schedule_kernel,
)
from State.Encoding import Encoding
from State.State import State


class PriorityEncoding(Encoding):
    """每顆感測器兩個 priority 基因 [a0, b0, a1, b1, ...]，不含 level。

    * ``a``（開啟）：priority = (E_s / E_max) × (0.5 + a / 9)，只看電量。
    * ``b``（路由）：priority = (E_s / E_max + d_s / d_max) × (0.5 + b / 9)，
      電量多、離 BS 遠的先接上路由樹，讓靠近 BS 的晚接、少當父節點。

    level 不在染色體中：依開啟順序處理每顆 sensor，選「新增覆蓋最多的最小
    level」；沒有新增覆蓋就不開。之後照 SensorEncoding v3 的規則從開啟順序
    尾端移除冗餘，再依路由順序建立路由；路由失敗時保留 level，也不修改染色體。

    這組開啟／路由 base 是 2026-10-12 lifetime 實驗選定的（tmp/route_dir、tmp/open_dir）。
    其他 base、level 規則與單基因版本在 2026-10-13 移到 ``legacy/priority_encoding/``。

    歷史版本（開啟 base／路由 base；priority 都是 base × (0.5 + gene / 9)）：

    * C0：E + P／E + P（P = 離 BS 近的分數，同 SensorEncoding）。
    * C1：E × C_min / C_s／E + P（C_s = 傳一個封包回 BS 的最小路徑耗能）。
    * C4：E × C_min / C_s／E + 0.2 × (1 − d / d_max)。2026-09-23 GA 實驗最好，
      2026-10-12 前 Ring_SEpri 用這版。
    * C5（現行，又稱 EF）：E／E + d / d_max。路由「遠先接」三個預算都贏 C4
      +2.8～4.9%；開啟加上距離或 C_min / C_s 都沒差，所以只看電量。
    * 試過但沒用的：路由 (1 − d / d_max) + 0.2 × E（linear，比 C0 差）、
      level 規則 efficiency（−44%）／total（−2.1%，ns）。

    E = E_s / E_max。
    """

    RANK_PRECISION = 10

    def __init__(self, code):
        super().__init__(code)
        if self.length % 2 != 0:
            raise ValueError("priority genes must be [a0, b0, a1, b1, ...]")

    @classmethod
    def random(cls, sensor_count, rng=None):
        """均勻建立 priority 基因。"""
        size = int(sensor_count) * 2
        if rng is None:
            code = np.random.randint(0, cls.RANK_PRECISION, size)
        else:
            code = rng.integers(0, cls.RANK_PRECISION, size)
        return cls(code)

    def copy(self):
        return self.__class__(self.code.copy())

    def _order(self, base, genes):
        """priority = base × (0.5 + gene / 9)，由高到低排序。"""
        genes = np.asarray(genes, dtype=float)
        priority = base * (0.5 + genes / (self.RANK_PRECISION - 1))
        return np.argsort(priority)[::-1]

    @staticmethod
    def _bs_farness(problem):
        """d_s / d_max。只和位置有關，快取在 problem 上（移動後 problem 會重建）。"""
        cached = getattr(problem, "_priority_bs_farness", None)
        if cached is not None:
            return cached
        to_bs = np.asarray(problem.distances)[
            : problem.SENSOR_NUMBER, problem.BSID
        ]
        max_distance = float(np.max(to_bs))
        farness = (
            to_bs / max_distance if max_distance > 0 else np.zeros_like(to_bs)
        )
        problem._priority_bs_farness = farness
        return farness

    def _activation_base(self, problem):
        return problem.energy_score

    def _routing_base(self, problem):
        return problem.energy_score + self._bs_farness(problem)

    def _schedule_python(self, problem, state, schedule_order):
        """選 level 並移除冗餘；回傳保留的 sensor id（依開啟順序）。

        有 numba 時改走 ``decode_priority_schedule_kernel``，兩者結果相同。
        """
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

        # 與 SensorEncoding v2/v3 相同：從開啟順序尾端移除冗餘。
        coverage_count = np.zeros(problem.TARGET_NUMBER, dtype=int)
        for sensor_id in opened:
            coverage_count += coverage[:, sensor_id, state.levels[sensor_id]]
        retained_reversed = []
        for sensor_id in reversed(opened):
            covered = coverage[:, sensor_id, state.levels[sensor_id]]
            if np.any(covered) and np.all(coverage_count[covered] >= 2):
                coverage_count[covered] -= 1
                state.levels[sensor_id] = 0
            else:
                retained_reversed.append(sensor_id)
        return np.asarray(retained_reversed[::-1], dtype=np.int64)

    def decode(self, problem):
        if self.code is None:
            raise ValueError("PriorityEncoding.code has not been initialized")
        code = np.asarray(self.code, dtype=int)
        expected_shape = (problem.SENSOR_NUMBER * 2,)
        if code.shape != expected_shape:
            raise ValueError(
                f"priority code must have shape {expected_shape}, "
                f"got {code.shape}"
            )
        schedule_order = self._order(self._activation_base(problem), code[0::2])
        state = State.empty(problem)
        if NUMBA_AVAILABLE:
            retained = decode_priority_schedule_kernel(
                np.asarray(problem.coverage_table),
                np.asarray(problem.radius_option_counts, dtype=np.int64),
                np.asarray(schedule_order, dtype=np.int64),
                state.levels,
            )
        else:
            retained = self._schedule_python(problem, state, schedule_order)

        rank = np.empty(problem.SENSOR_NUMBER, dtype=int)
        rank[self._order(self._routing_base(problem), code[1::2])] = np.arange(
            problem.SENSOR_NUMBER
        )
        # rank 各不相同，排序結果唯一。
        routing_ids = retained[np.argsort(rank[retained])]

        problem.routing_service.build_routes(
            state, routing_ids, disable_failed=False
        )
        problem.resolve_state_radius(state)
        return state


__all__ = ["PriorityEncoding"]
