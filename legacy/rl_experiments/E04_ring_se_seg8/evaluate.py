"""E04 lifetime 評估：e04_random（8 段整段隨機）、e04_bottleneck（瓶頸規則）、
e04_concat／e04_attention（凍結的 D3QN 網路，greedy；checkpoint 由環境變數 E04_CKPT 給）。

用法（在 2027WSN 根目錄）：
    E04_CKPT=.../best.pt python Algorithm/se/rl/experiments/E04_ring_se_seg8/evaluate.py \
        --algorithm e04_concat --runs 30 --seed 7 --evaluate 1000 --maps maps/maps_100100100.json ...
參數同 ring_se preset，h 改 8。
"""
import copy
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
for p in (str(HERE), str(ROOT)):
    sys.path.insert(0, p)
os.chdir(ROOT)
# 註冊必須在 experiment_algorithms 第一次被 import 之前（common → E01 snapshots 會 import 它），所以 env／train 放後面

import torch  # noqa: E402

import experiments.registry as registry  # noqa: E402

VARIANTS = {"e04_random": "Ring_SE_E04_Random", "e04_bottleneck": "Ring_SE_E04_Bottleneck",
            "e04_concat": "Ring_SE_E04_Net", "e04_attention": "Ring_SE_E04_Net"}
for name, cls in VARIANTS.items():
    registry.ALGORITHM_IMPORTS[name] = ("evaluate", cls)
registry.ALGORITHM_NAMES = tuple(registry.ALGORITHM_IMPORTS)
registry.ALGORITHM_CHOICES = ("all",) + registry.ALGORITHM_NAMES
registry.ALL_RUNNABLE_ALGORITHMS = tuple(n for n in registry.ALGORITHM_NAMES if n not in registry.RL_SETS_ALGORITHMS)

from experiments.presets import ALGORITHM_PRESETS  # noqa: E402

for name, cls in VARIANTS.items():
    ALGORITHM_PRESETS[name] = copy.deepcopy(ALGORITHM_PRESETS["ring_se"])
    ALGORITHM_PRESETS[name]["label"] = cls
    ALGORITHM_PRESETS[name]["params"]["h"] = 8

from env import Ring_SE_E04  # noqa: E402
from train import load_policy  # noqa: E402


class Ring_SE_E04_Random(Ring_SE_E04):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, policy="random", **kwargs)


class Ring_SE_E04_Bottleneck(Ring_SE_E04):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, policy="bottleneck", **kwargs)


class Ring_SE_E04_Net(Ring_SE_E04):
    def __init__(self, problem, **kwargs):
        torch.set_num_threads(1)
        super().__init__(problem, policy=load_policy(os.environ["E04_CKPT"]), **kwargs)


if __name__ == "__main__":
    import experiment_algorithms
    experiment_algorithms.main()
