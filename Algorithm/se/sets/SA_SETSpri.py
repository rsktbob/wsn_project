"""SA-SETSpri：SA_SETS 的市場流程，染色體換成 PriorityEncoding、不分區。

PriorityEncoding 每顆 sensor 兩個基因 [開啟 a, 路由 b]，值都在 0–9，level 由
解碼器貪婪決定，所以和 SA_SETS 有三處不同：

* 初始解：均勻隨機。CCS 寫的是 level，這個編碼沒有 level；隨機解解碼後本來就
  盡量覆蓋所有 target。
* 不分區：沒有開關基因可以對齊，identity sensor 不選、``align_region`` 不動解；
  h 個區域只剩市場分組（各區自己的 goods）。
* 突變：1–3 組，每組改開啟、路由或兩者，新值都是 0–9。

交配（跨中點兩刀）、市場流程、更新規則都沿用 SA_SETS。2026-10-13 由 tmp/sa_pri 的
SA_SETSpriority 正式化（A2、預算 10000、5 場：fitness v3 和 SensorEncoding 版打平，
v1 +39%）。
"""

from __future__ import annotations

from Algorithm.se.sets.SA_SETS import SA_SETS
from State.PriorityEncoding import PriorityEncoding
from State.initialization import random_priority_coding


class SA_SETSpri(SA_SETS):
    """SA_SETS over priority chromosomes without region alignment."""

    def __init__(self, problem, n=5, h=4, w=2, mu=0.2, seed=None):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        # 不分區：不選 identity sensor。
        self.identity_bit_count = 0

    def create_candidate(self, problem, region=None):
        """均勻隨機的 priority 染色體；不做區域對齊。"""
        return random_priority_coding(problem, self.rng)

    def align_region(self, problem, candidate, region):
        return candidate

    def mutate_candidate(self, problem, candidate):
        """1–3 組（90/5/5），每組改開啟、路由或兩者，新值都是 0–9。"""
        precision = PriorityEncoding.RANK_PRECISION
        for _ in range(self.random.choice(self.MUTATION_COUNTS)):
            gene_id = self.random.randrange(self.code_length)
            activation_id = gene_id - gene_id % 2
            operation = self.random.randint(0, 2)
            if operation in (0, 1):
                candidate.code[activation_id] = self.random.randrange(precision)
            if operation in (0, 2):
                candidate.code[activation_id + 1] = self.random.randrange(precision)
        return candidate


__all__ = ["SA_SETSpri"]
