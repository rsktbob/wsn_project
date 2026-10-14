"""註冊 ring_se_oracle（參數沿用 ring_se preset：n8 h4 w8 μ0.4）後執行 experiment_algorithms。"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "../../../../../.."))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import experiments.registry as registry  # noqa: E402

VARIANTS = {"ring_se_oracle": "Ring_SE_Oracle"}
for name, cls in VARIANTS.items():
    registry.ALGORITHM_IMPORTS[name] = ("variants", cls)
registry.ALGORITHM_NAMES = tuple(registry.ALGORITHM_IMPORTS)
registry.ALGORITHM_CHOICES = ("all",) + registry.ALGORITHM_NAMES
registry.ALL_RUNNABLE_ALGORITHMS = tuple(
    n for n in registry.ALGORITHM_NAMES if n not in registry.RL_SETS_ALGORITHMS
)

from experiments.presets import ALGORITHM_PRESETS  # noqa: E402

for name, cls in VARIANTS.items():
    ALGORITHM_PRESETS[name] = copy.deepcopy(ALGORITHM_PRESETS["ring_se"])
    ALGORITHM_PRESETS[name]["label"] = cls

import experiment_algorithms  # noqa: E402

if __name__ == "__main__":
    experiment_algorithms.main()
