"""註冊 sa_sets2023（參數沿用 sa_sets preset：n8 h4 w2 mu0.4）後執行 experiment_algorithms。

用法（在 2027WSN 根目錄）：
    python legacy/sa_sets2023/run.py --algorithm sa_sets2023 --fitness-service v1 ...
其餘參數和 experiment_algorithms.py 相同。
"""
import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import experiments.registry as registry  # noqa: E402

registry.ALGORITHM_IMPORTS["sa_sets2023"] = ("SA_SETS2023", "SA_SETS2023")
registry.ALGORITHM_NAMES = tuple(registry.ALGORITHM_IMPORTS)
registry.ALGORITHM_CHOICES = ("all",) + registry.ALGORITHM_NAMES
registry.ALL_RUNNABLE_ALGORITHMS = tuple(
    name for name in registry.ALGORITHM_NAMES
    if name not in registry.RL_SETS_ALGORITHMS
)

from experiments.presets import ALGORITHM_PRESETS  # noqa: E402

ALGORITHM_PRESETS["sa_sets2023"] = copy.deepcopy(ALGORITHM_PRESETS["sa_sets"])
ALGORITHM_PRESETS["sa_sets2023"]["label"] = "SA-SETS 2023 (original code)"

import experiment_algorithms  # noqa: E402

if __name__ == "__main__":
    experiment_algorithms.main()
