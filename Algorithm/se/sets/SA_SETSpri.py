"""SA-SETSpri：SA_SETS 的市場流程，染色體換成 PriorityEncoding、不分區。

細節見 :mod:`Algorithm.se.priority_operators`。2026-10-13 由 tmp/sa_pri 的
SA_SETSpriority 正式化（A2、預算 10000、5 場：fitness v3 和 SensorEncoding 版打平，
v1 +39%）。
"""

from __future__ import annotations

from Algorithm.se.sets.SA_SETS import SA_SETS
from Algorithm.se.priority_operators import PriorityOperators


class SA_SETSpri(PriorityOperators, SA_SETS):
    """SA_SETS over priority chromosomes without region alignment."""


__all__ = ["SA_SETSpri"]
