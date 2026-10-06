"""SA-SETS：在 SETS 上改用感測器感知的 identity sensor 與 CCS 初始解。

市場流程、分區對齊、交配突變、Beta 記憶都沿用 :class:`SETS`；
identity sensor 的選法與初始 chromosome 由兩個 mixin 換掉
（同樣的 mixin 也給 SI_SETS 使用）。
"""

from __future__ import annotations

from Algorithm.se.SETS import SETS
from Algorithm.se.sensor_operators import (
    CCSInitialization,
    NearBaseIdentitySensors,
)


class SA_SETS(NearBaseIdentitySensors, CCSInitialization, SETS):
    """SETS with near-BS high-energy identity sensors and CCS seeding."""


__all__ = ["SA_SETS"]
