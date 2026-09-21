"""以感測器為單位的決策編碼與解碼器。"""

from __future__ import annotations

import numpy as np

from State.Encoding import Encoding
from State.State import State
from environment_defaults import DEFAULT_SENSOR_ENCODING
from Problem.services.evaluation_kernels import (
    NUMBA_AVAILABLE,
    decode_sensor_schedule_kernel,
)


class SensorEncoding(Encoding):
    """每顆感測器使用兩個基因：感測選項與路由優先值。

    本類別保存染色體，並提供所有演算法共用的均勻初始化。具有非均勻
    初始化策略的演算法仍可自行建立 code，再傳入 ``SensorEncoding(code)``。
    """

    RANK_PRECISION = 10

    def __init__(self, code):
        """建立感測器編碼，並驗證每顆感測器都有一對基因。"""
        super().__init__(code)
        if self.length % 2 != 0:
            raise ValueError(
                "sensor genes must be [s0, p0, s1, p1, ...]"
            )

    @classmethod
    def random(cls, sensor_count, sensing_domains=None, rng=None):
        """均勻建立一條合法的 sensor chromosome。

        ``sensing_domains`` 決定每顆感測器排程基因的選項數量，可以是單一
        整數或長度等於 ``sensor_count`` 的陣列；路由基因不受它影響，永遠
        從 ``[0, RANK_PRECISION)``（目前即 0～9）產生。

        ``SensorEncoding`` 只接收 domain，不依賴整個 Problem。演算法可將
        ``problem.radius_option_counts`` 傳入，以支援每顆感測器不同的範圍。
        """
        sensor_count = int(sensor_count)
        if sensor_count <= 0:
            raise ValueError("sensor_count must be positive")

        if sensing_domains is None:
            domains = np.full(
                sensor_count,
                cls.RANK_PRECISION,
                dtype=int,
            )
        elif np.isscalar(sensing_domains):
            domains = np.full(
                sensor_count,
                int(sensing_domains),
                dtype=int,
            )
        else:
            domains = np.asarray(sensing_domains, dtype=int)
            if domains.shape != (sensor_count,):
                raise ValueError(
                    "sensing_domains must have one value per sensor"
                )
        if np.any(domains <= 0):
            raise ValueError("sensing domains must be positive")

        code = np.empty(sensor_count * 2, dtype=int)

        if rng is None:
            # 所有 domain 相同時保留原本的向量化 randint 呼叫；不同時
            # NumPy 會依陣列中的每個上限分別產生合法排程基因。
            sensing_high = (
                int(domains[0])
                if np.all(domains == domains[0])
                else domains
            )
            code[0::2] = np.random.randint(
                0,
                sensing_high,
                sensor_count,
            )
            code[1::2] = np.random.randint(
                0,
                cls.RANK_PRECISION,
                sensor_count,
            )
        else:
            code[0::2] = rng.integers(
                0,
                domains,
                sensor_count,
            )
            code[1::2] = rng.integers(
                0,
                cls.RANK_PRECISION,
                sensor_count,
            )
        return cls(code)        
    
    def copy(self):
        """複製染色體，讓子代可在不影響父代下修改。"""
        return self.__class__(self.code.copy())

    def decode(self, problem):
        """Use the shared training/experiment decoder version."""
        return self._decode_version(problem, version=DEFAULT_SENSOR_ENCODING)

    def decodev1(self, problem):
        """只開啟能增加尚未覆蓋 target 的 sensor。"""
        return self._decode_version(problem, version="v1")

    def decodev2(self, problem):
        """先依 v1 開啟 sensor，再反向移除可被取代的冗餘 sensor。"""
        return self._decode_version(problem, version="v2")

    def decodev3(self, problem):
        """沿用 v2 排程，但保留無法加入路由的 sensor 與 chromosome。"""
        return self._decode_version(problem, version="v3")

    def _decode_version(self, problem, version):
        """執行共用的基因準備、版本排程規則及路由建構。"""
        if self.code is None:
            raise ValueError("SensorEncoding.code has not been initialized")
        code = np.asarray(self.code, dtype=int)
        expected_shape = (problem.SENSOR_NUMBER * 2,)
        if code.shape != expected_shape:
            raise ValueError(
                f"sensor code must have shape {expected_shape}, "
                f"got {code.shape}"
            )

        schedule_genes = code[0::2].copy()
        priority_genes = code[1::2]

        # 三個版本都只在 State 內將 sensing gene 正規化；是否回寫
        # chromosome 只由後面的路由失敗處理決定。
        np.mod(
            schedule_genes,
            np.asarray(problem.radius_option_counts, dtype=int),
            out=schedule_genes,
        )

        # ``proximity_score`` 是目前的路由分數：越接近基地台，越容易
        # 優先接入既有路由樹。priority gene 仍為 0～9，但所有 sensor
        # 至少保留 0.5 倍的能源／路由基礎排序，避免 gene=0 完全消失。
        priority = (problem.energy_score + problem.proximity_score) * (
            0.5 + priority_genes / (self.RANK_PRECISION - 1)
        )
        # 優先值越高者越早處理；它同時影響覆蓋選擇與路由加入順序。
        sensor_order = np.argsort(priority)[::-1]
        state = State.empty(problem)

        if NUMBA_AVAILABLE:
            routing_ids = decode_sensor_schedule_kernel(
                np.asarray(problem.coverage_table),
                np.asarray(schedule_genes, dtype=np.int64),
                np.asarray(sensor_order, dtype=np.int64),
                version in ("v2", "v3"),
                state.levels,
            )
        else:
            uncovered = np.ones(problem.TARGET_NUMBER, dtype=int)
            routing_ids = []
            for sensor_id in sensor_order:
                covered_targets = np.where(
                    problem.coverage_table[
                        :, sensor_id, schedule_genes[sensor_id]
                    ]
                )[0]
                # 只保留能增加新覆蓋的 sensor。
                if np.sum(uncovered[covered_targets]) > 0:
                    uncovered[covered_targets] = 0
                    state.levels[sensor_id] = schedule_genes[sensor_id]
                    routing_ids.append(int(sensor_id))

            if version in ("v2", "v3"):
                # v2／v3：從低 priority 端移除仍可由其他 sensor 接手的冗餘。
                coverage_count = np.zeros(
                    problem.TARGET_NUMBER,
                    dtype=int,
                )
                for sensor_id in routing_ids:
                    level = int(state.levels[sensor_id])
                    coverage_count += (
                        problem.coverage_table[:, sensor_id, level] > 0
                    ).astype(int)

                retained_reversed = []
                for sensor_id in reversed(routing_ids):
                    level = int(state.levels[sensor_id])
                    covered = (
                        problem.coverage_table[:, sensor_id, level] > 0
                    )
                    if (
                        np.any(covered)
                        and np.all(coverage_count[covered] >= 2)
                    ):
                        coverage_count[covered] -= 1
                        state.levels[sensor_id] = 0
                    else:
                        retained_reversed.append(sensor_id)
                routing_ids = list(reversed(retained_reversed))

        # 路由服務只允許連接到已成功接上基地台的節點，逐步形成無環轉送樹。
        # V3 失敗時直接保留 level，不需要先關閉再恢復排程。
        routing_result = problem.routing_service.build_routes(
            state,
            routing_ids,
            disable_failed=version != "v3",
        )
        if version != "v3":
            # v1／v2 保留歷史副作用：容量失敗會同步關閉 chromosome；
            # v3 則不修改 State 排程或 encoding。
            for sensor_id in routing_result.capacity_failed_ids:
                self._disable_sensor(sensor_id)
        # 同步最終排程對應的物理感測半徑。
        problem.resolve_state_radius(state)
        return state

    def _disable_sensor(self, sensor_id):
        """將無法建立可行路由之感測器的排程基因設為關閉。"""
        # 保留歷史解碼副作用；是否移除應視為另一項演算法行為修正。
        self.code[int(sensor_id) * 2] = 0


__all__ = ["SensorEncoding"]
