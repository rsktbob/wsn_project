"""E05 lifetime 評估：e05_{concat,attention}_{sample,argmax}（凍結的監督式網路；checkpoint 由環境變數 E05_CKPT 給）。

sample：依 softmax(s) 的機率抽段（亂數 seed＝演算法 seed，可重現）；argmax：選分數最大的段。
參數同 ring_se preset，h 改 8；環境同 E04（Ring_SE_E04）。

用法（在 2027WSN 根目錄）：
    E05_CKPT=.../best.pt python Algorithm/se/rl/experiments/E05_ring_se_seg8_supervised/evaluate.py \
        --algorithm e05_attention_sample --runs 30 --seed 7 --evaluate 1000 --maps maps/maps_100100100.json ...
"""
import copy
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E04 = HERE.parent / "E04_ring_se_seg8"
ROOT = HERE.parents[4]
for p in (str(ROOT), str(E04), str(HERE)):   # 最後插入的在最前面：E05 的 train.py 優先於 E04 的
    sys.path.insert(0, p)
os.chdir(ROOT)
# 註冊必須在 experiment_algorithms 第一次被 import 之前（E04 common → E01 snapshots 會 import 它）

import torch  # noqa: E402

import experiments.registry as registry  # noqa: E402

VARIANTS = {f"e05_{net}_{mode}": f"Ring_SE_E05_{mode.capitalize()}"
            for net in ("concat", "attention") for mode in ("sample", "argmax")}
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
from train import load_policy  # noqa: E402  （E05 的 train.py；HERE 在 sys.path 最前面）


class Ring_SE_E05_Sample(Ring_SE_E04):
    def __init__(self, problem, seed=None, **kwargs):
        torch.set_num_threads(1)
        policy = load_policy(os.environ["E05_CKPT"], "sample", seed=0 if seed is None else seed)
        super().__init__(problem, policy=policy, seed=seed, **kwargs)


class Ring_SE_E05_Argmax(Ring_SE_E04):
    def __init__(self, problem, **kwargs):
        torch.set_num_threads(1)
        super().__init__(problem, policy=load_policy(os.environ["E05_CKPT"], "argmax"), **kwargs)


if __name__ == "__main__":
    import experiment_algorithms
    experiment_algorithms.main()
