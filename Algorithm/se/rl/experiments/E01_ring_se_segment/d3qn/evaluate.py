"""E01 / D3QN lifetime 評估：凍結的策略（checkpoint）在 Ring_SE_E01 上選段，跑正式的 experiment_algorithms。

用法（在 2027WSN 根目錄）：
    E01_CKPT=.../best.pt python Algorithm/se/rl/experiments/E01_ring_se_segment/d3qn/evaluate.py \
        --algorithm e01_d3qn --runs 5 --seed 7 --evaluate 1000 --maps maps/maps_100100100.json ...
其餘參數同 experiment_algorithms.py。checkpoint 由環境變數 E01_CKPT 指定（子程序也讀得到）。
"""
import copy
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E01 = HERE.parent
ROOT = E01.parents[4]
for p in (str(HERE), str(E01), str(ROOT)):
    sys.path.insert(0, p)
os.chdir(ROOT)

import experiments.registry as registry  # noqa: E402

registry.ALGORITHM_IMPORTS["e01_d3qn"] = ("evaluate", "Ring_SE_E01_D3QN")
registry.ALGORITHM_NAMES = tuple(registry.ALGORITHM_IMPORTS)
registry.ALGORITHM_CHOICES = ("all",) + registry.ALGORITHM_NAMES
registry.ALL_RUNNABLE_ALGORITHMS = tuple(n for n in registry.ALGORITHM_NAMES if n not in registry.RL_SETS_ALGORITHMS)

from experiments.presets import ALGORITHM_PRESETS  # noqa: E402

ALGORITHM_PRESETS["e01_d3qn"] = copy.deepcopy(ALGORITHM_PRESETS["ring_se"])
ALGORITHM_PRESETS["e01_d3qn"]["label"] = "Ring_SE_E01_D3QN"

from agent import load_policy  # noqa: E402
from env import Ring_SE_E01  # noqa: E402


class Ring_SE_E01_D3QN(Ring_SE_E01):
    """Greedy frozen D3QN policy from $E01_CKPT."""

    def __init__(self, problem, **kwargs):
        super().__init__(problem, policy=load_policy(os.environ["E01_CKPT"]), **kwargs)


if __name__ == "__main__":
    import experiment_algorithms
    experiment_algorithms.main()
