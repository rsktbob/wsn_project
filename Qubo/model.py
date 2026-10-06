"""QUBO 模型：常數、一次項與二次項，供量子或經典求解器共用。"""

from __future__ import annotations

import numpy as np


class QuboModel:
    """最小化 ``offset + Σ h_i x_i + Σ_{i<j} q_ij x_i x_j``，x 為 0/1。

    變數以整數索引保存，``labels`` 只用於除錯與報表。同一對變數重複加入
    的係數會累加；``x_i · x_i`` 因為 x² = x，直接併入一次項。
    """

    def __init__(self):
        self.labels = []
        self.linear = {}
        self.quadratic = {}
        self.offset = 0.0

    @property
    def num_variables(self):
        return len(self.labels)

    def add_variable(self, label):
        """新增一個 0/1 變數並回傳其索引。"""
        self.labels.append(label)
        return len(self.labels) - 1

    def add_linear(self, index, value):
        self.linear[index] = self.linear.get(index, 0.0) + float(value)

    def add_quadratic(self, index1, index2, value):
        if index1 == index2:
            self.add_linear(index1, value)
            return
        key = (min(index1, index2), max(index1, index2))
        self.quadratic[key] = self.quadratic.get(key, 0.0) + float(value)

    def add_offset(self, value):
        self.offset += float(value)

    def add_squared(self, terms, constant, weight):
        """加入 ``weight · (Σ a_k x_k + constant)²``。

        ``terms`` 是 ``[(index, a_k), ...]``；同一索引可出現多次。展開後
        a_k² x_k² 化為一次項，交叉項成為二次項。
        """
        merged = {}
        for index, coefficient in terms:
            merged[index] = merged.get(index, 0.0) + float(coefficient)
        items = list(merged.items())
        constant = float(constant)
        weight = float(weight)
        for position, (index, coefficient) in enumerate(items):
            self.add_linear(
                index,
                weight * (coefficient * coefficient + 2.0 * constant * coefficient),
            )
            for other_index, other_coefficient in items[position + 1:]:
                self.add_quadratic(
                    index,
                    other_index,
                    weight * 2.0 * coefficient * other_coefficient,
                )
        self.add_offset(weight * constant * constant)

    def to_arrays(self):
        """回傳 ``(h, J, offset)``；J 為零對角的對稱矩陣。

        E(x) = offset + h·x + ½ xᵀJx，翻轉 x_j 的能量變化為
        ``(1 − 2x_j)(h_j + Σ_k J_jk x_k)``。
        """
        n = self.num_variables
        h = np.zeros(n, dtype=float)
        coupling = np.zeros((n, n), dtype=float)
        for index, value in self.linear.items():
            h[index] = value
        for (index1, index2), value in self.quadratic.items():
            coupling[index1, index2] = value
            coupling[index2, index1] = value
        return h, coupling, self.offset

    def energy(self, x):
        """計算單一 0/1 向量的能量。"""
        h, coupling, offset = self.to_arrays()
        x = np.asarray(x, dtype=float)
        return float(offset + h @ x + 0.5 * x @ coupling @ x)

    def term_counts(self):
        """非零一次項與二次項的數量，用來比較編碼的算式規模。"""
        linear = sum(1 for value in self.linear.values() if abs(value) > 1e-12)
        quadratic = sum(
            1 for value in self.quadratic.values() if abs(value) > 1e-12
        )
        return {1: linear, 2: quadratic}


__all__ = ["QuboModel"]
