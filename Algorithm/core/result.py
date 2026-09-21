"""所有最佳化演算法共用的執行結果格式。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class AlgorithmResult:
    """一次演算法執行的摘要，不參與演算法內部的編碼搜尋。

    編碼是各演算法的實作細節；``best_state`` 可直接交給 Problem
    檢查可行性、計算 fitness 與扣除能量。
    """

    best_state: Any
    best_fitness: np.ndarray
    evaluations: int
    iterations: int
    history: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def best_score(self) -> float:
        """回傳所有 fitness 分量的總和。"""
        return float(np.sum(self.best_fitness))


__all__ = ["AlgorithmResult"]
