"""讓 SETS 系列（SA_SETS、SI_SETS）改搜尋 PriorityEncoding 染色體的 mixin。

PriorityEncoding 每顆 sensor 兩個基因 [開啟 a, 路由 b]，值都在 0–9，level 由
解碼器貪婪決定，所以和 SensorEncoding 版有三處不同：

* 初始解：均勻隨機。CCS 寫的是 level，這個編碼沒有 level；隨機解解碼後本來就
  盡量覆蓋所有 target。
* 不分區：沒有開關基因可以對齊，identity sensor 不選、``align_region`` 不動解；
  h 個區域只剩市場分組（SA：各區自己的 goods；SI：各區自己的 goods pool）。
* 突變：沿用 ``mutatev1`` 的結構（1–3 組、改開啟／路由／兩者），新值都是 0–9。

交配（跨中點兩刀）、市場流程、更新規則都沿用原演算法。
"""

from __future__ import annotations

from Algorithm.se.sensor_operators import draw_mutation_count
from State.PriorityEncoding import PriorityEncoding


class PriorityOperators:
    """PriorityEncoding chromosome, no identity-sensor partition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 不分區：不選 identity sensor（SensorOperators 會檢查選到的數量 == 0）。
        self.identity_bit_count = 0

    def create_candidate(self, problem, region=None):
        """均勻隨機的 priority 染色體；不做區域對齊。"""
        return PriorityEncoding.random(problem.SENSOR_NUMBER, rng=self.rng)

    def align_region(self, problem, candidate, region):
        return candidate

    def mutatev1(self, problem, candidate):
        """1–3 組（90/5/5），每組改開啟、路由或兩者，新值都是 0–9。"""
        precision = PriorityEncoding.RANK_PRECISION
        for _ in range(draw_mutation_count(self.random)):
            gene_id = self.random.randrange(self.code_length)
            activation_id = gene_id - gene_id % 2
            operation = self.random.randint(0, 2)
            if operation in (0, 1):
                candidate.code[activation_id] = self.random.randrange(precision)
            if operation in (0, 2):
                candidate.code[activation_id + 1] = self.random.randrange(precision)
        return candidate


__all__ = ["PriorityOperators"]
