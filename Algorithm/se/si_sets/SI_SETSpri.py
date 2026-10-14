"""SI-SETSpri：SI_SETS 的市場流程（選池、EN 更新），染色體換成 PriorityEncoding、不分區。

細節見 :mod:`Algorithm.se.priority_operators`。
"""

from __future__ import annotations

from Algorithm.se.si_sets.SI_SETS import SI_SETS
from Algorithm.se.priority_operators import PriorityOperators


class SI_SETSpri(PriorityOperators, SI_SETS):
    """SI_SETS over priority chromosomes without region alignment."""


__all__ = ["SI_SETSpri"]
