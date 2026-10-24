"""E06 lifetime 評估：e06_random（4 候選均勻）、e06_bottleneck（瓶頸規則）、
e06_net（凍結的網路，greedy；checkpoint 由環境變數 E06_CKPT 給，裝置 E06_DEVICE，預設有 GPU 就用）。

用法（在 2027WSN 根目錄）：
    E06_CKPT=.../best.pt python Algorithm/se/rl/experiments/E06_ring_se_span_context/evaluate.py \
        --algorithm e06_net --runs 30 --seed 7 --evaluate 1000 --maps maps/maps_100100100.json ...
參數同 ring_se preset（n8 h4 w8 μ0.4）。
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
# 註冊必須在 experiment_algorithms 第一次被 import 之前，所以 env／model 放後面

import torch  # noqa: E402

import experiments.registry as registry  # noqa: E402

VARIANTS = {"e06_random": "Ring_SE_E06_Random", "e06_bottleneck": "Ring_SE_E06_Bottleneck",
            "e06_net": "Ring_SE_E06_Net"}
for name, cls in VARIANTS.items():
    registry.ALGORITHM_IMPORTS[name] = ("evaluate", cls)
registry.ALGORITHM_NAMES = tuple(registry.ALGORITHM_IMPORTS)
registry.ALGORITHM_CHOICES = ("all",) + registry.ALGORITHM_NAMES
registry.ALL_RUNNABLE_ALGORITHMS = tuple(n for n in registry.ALGORITHM_NAMES if n not in registry.RL_SETS_ALGORITHMS)

from experiments.presets import ALGORITHM_PRESETS  # noqa: E402

for name, cls in VARIANTS.items():
    ALGORITHM_PRESETS[name] = copy.deepcopy(ALGORITHM_PRESETS["ring_se"])
    ALGORITHM_PRESETS[name]["label"] = cls

from env import Ring_SE_E06  # noqa: E402
from model import greedy_policy, load_net  # noqa: E402

_POLICY = {}


def _net_policy():
    """每個行程只載入一次網路。"""
    path = os.environ["E06_CKPT"]
    if path not in _POLICY:
        torch.set_num_threads(1)
        device = os.environ.get("E06_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        _POLICY[path] = greedy_policy(load_net(path, device), device)
    return _POLICY[path]


class Ring_SE_E06_Random(Ring_SE_E06):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, policy="random", **kwargs)


class Ring_SE_E06_Bottleneck(Ring_SE_E06):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, policy="bottleneck", **kwargs)


class Ring_SE_E06_Net(Ring_SE_E06):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, policy=_net_policy(), **kwargs)


if __name__ == "__main__":
    import experiment_algorithms
    experiment_algorithms.main()
