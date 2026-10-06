"""感測等級子問題的 QUBO 建模與求解。

``sensor_levels`` 建立區域子問題並提供溫度計／one-hot 編碼；``solvers``
提供 SA 與窮舉。量子求解器（QAOA、量子退火）之後可直接吃同一個
``QuboModel``。
"""

from Qubo.model import QuboModel
from Qubo.sensor_levels import (
    DEFAULT_FOCUS_POWER,
    DEFAULT_PENALTY,
    ENCODINGS,
    OBJECTIVES,
    LevelSubproblem,
    OneHotEncoding,
    ThermometerEncoding,
    apply_with_fixed_routes,
    build_level_subproblem,
    repair_and_trim,
    slack_weights,
    solve_exact,
)
from Qubo.solvers import brute_force, simulated_annealing

__all__ = [
    "DEFAULT_FOCUS_POWER",
    "DEFAULT_PENALTY",
    "ENCODINGS",
    "LevelSubproblem",
    "OBJECTIVES",
    "OneHotEncoding",
    "QuboModel",
    "ThermometerEncoding",
    "apply_with_fixed_routes",
    "brute_force",
    "build_level_subproblem",
    "repair_and_trim",
    "simulated_annealing",
    "slack_weights",
    "solve_exact",
]
