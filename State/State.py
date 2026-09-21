"""保存已解碼的 WSN 感測排程、路由與評估暫存資料。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class State:
    """表示一個已解碼的感測排程與路由解。

    ``State`` 刻意不保存染色體，也不負責解碼或評估。編碼類別產生
    ``State``，再由 ``Problem`` 的服務計算覆蓋、耗能、路由可行性與
    fitness，藉此分離「搜尋表示」與「物理解」。
    """

    # levels[i]：感測器 i 使用的 sensing option；0 代表關閉。
    levels: np.ndarray
    # next_hops[i]：感測器 i 的下一跳節點 id；-1 代表尚未建立路由。
    next_hops: np.ndarray
    # tx_load[i]：感測器 i 自產加轉送後，實際送往下一跳的總負載。
    tx_load: np.ndarray
    # remaining_capacity[i]：節點 i 到基地台整條路徑的剩餘容量瓶頸。
    remaining_capacity: np.ndarray
    # paths[i]：從感測器 i 一路到基地台的完整節點序列。
    paths: dict[int, list[int]] = field(default_factory=dict)
    # sensing_radii 保存 levels 解析後的實際物理半徑，不是 option id。
    sensing_radii: np.ndarray | None = None
    use_continuous_radius: bool = False
    # target_assignment[t]：target t 目前指派的負責感測器 id。
    target_assignment: np.ndarray | None = None
    target_red: np.ndarray | None = None
    value: Any = None
    objectives: np.ndarray | None = None
    constraint_violation: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0

    @classmethod
    def empty(cls, problem, *, with_target_assignment=False):
        """依 ``problem`` 尺寸建立尚未啟用任何感測器的空白狀態。"""
        assignment = None
        if with_target_assignment:
            # -1 表示該 target 尚未指派給任何負責感測器。
            assignment = np.full(problem.TARGET_NUMBER, -1, dtype=int)
        return cls(
            levels=np.zeros(problem.SENSOR_NUMBER, dtype=int),
            next_hops=np.full(problem.SENSOR_NUMBER, -1, dtype=int),
            tx_load=np.zeros(problem.SENSOR_NUMBER, dtype=int),
            remaining_capacity=np.full(
                problem.DEVICE_NUMBER, -1.0, dtype=float
            ),
            target_assignment=assignment,
        )

    def copy(self):
        """複製所有可變欄位，讓新狀態可獨立進行模擬更新。"""
        # NumPy 陣列與 path 內的列表皆需複製，否則演算法修改子代時
        # 可能同步污染父代；純量與不可變值則可直接沿用。
        return State(
            levels=self.levels.copy(),
            next_hops=self.next_hops.copy(),
            tx_load=self.tx_load.copy(),
            remaining_capacity=self.remaining_capacity.copy(),
            paths={
                int(key): list(value) for key, value in self.paths.items()
            },
            sensing_radii=(
                None
                if self.sensing_radii is None
                else self.sensing_radii.copy()
            ),
            use_continuous_radius=self.use_continuous_radius,
            target_assignment=(
                None
                if self.target_assignment is None
                else self.target_assignment.copy()
            ),
            target_red=(
                None if self.target_red is None else self.target_red.copy()
            ),
            value=self.value,
            objectives=(
                None if self.objectives is None else self.objectives.copy()
            ),
            constraint_violation=self.constraint_violation,
            rank=self.rank,
            crowding_distance=self.crowding_distance,
        )

    # 舊名稱僅保留為相容入口；新程式一律使用上方具體名稱。
    @property
    def sch(self):
        return self.levels

    @sch.setter
    def sch(self, value):
        self.levels = value

    @property
    def rou(self):
        return self.next_hops

    @rou.setter
    def rou(self, value):
        self.next_hops = value

    @property
    def use(self):
        return self.tx_load

    @use.setter
    def use(self, value):
        self.tx_load = value

    @property
    def cap(self):
        return self.remaining_capacity

    @cap.setter
    def cap(self, value):
        self.remaining_capacity = value

    @property
    def path(self):
        return self.paths

    @path.setter
    def path(self, value):
        self.paths = value

    @property
    def radius(self):
        return self.sensing_radii

    @radius.setter
    def radius(self, value):
        self.sensing_radii = value


__all__ = ["State"]
