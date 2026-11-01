"""E11 lifetime 評估：e11_full ＝ E07 的 env（原版 ring_se、每組交易 4 個預抽候選），由 FullNet 選分數最大的候選。

輸入和 E07 相同（round_inputs），不解碼子代。checkpoint 由環境變數 E11_CKPT 給，裝置 E11_DEVICE（預設有 GPU 就用）。

用法（2027WSN 根目錄）：
    E11_CKPT=.../best.pt python Algorithm/se/rl/experiments/E11_ring_se_fullattn/evaluate_full.py \
        --algorithm e11_full --runs 5 --seed 7 --evaluate 1000 --fitness-service v3 --maps maps/maps_100100100.json ...
參數同 ring_se preset（n8 h4 w8 μ0.4）。
"""
import copy
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E07 = HERE.parent / "E07_ring_se_lifetime"
ROOT = HERE.parents[4]
for p in (str(HERE), str(E07), str(ROOT)):
    sys.path.insert(0, p)
os.chdir(ROOT)
# 註冊必須在 experiment_algorithms 第一次被 import 之前，所以 env／網路放後面

import torch  # noqa: E402

import experiments.registry as registry  # noqa: E402

VARIANTS = {"e11_full": "Ring_SE_E11_Full"}
for name, cls in VARIANTS.items():
    registry.ALGORITHM_IMPORTS[name] = ("evaluate_full", cls)
registry.ALGORITHM_NAMES = tuple(registry.ALGORITHM_IMPORTS)
registry.ALGORITHM_CHOICES = ("all",) + registry.ALGORITHM_NAMES
registry.ALL_RUNNABLE_ALGORITHMS = tuple(n for n in registry.ALGORITHM_NAMES if n not in registry.RL_SETS_ALGORITHMS)

from experiments.presets import ALGORITHM_PRESETS  # noqa: E402

for name, cls in VARIANTS.items():
    ALGORITHM_PRESETS[name] = copy.deepcopy(ALGORITHM_PRESETS["ring_se"])
    ALGORITHM_PRESETS[name]["label"] = cls

from env import Ring_SE_E07  # noqa: E402
from model import round_inputs  # noqa: E402
from fullnet import load_fullnet  # noqa: E402

_NET = {}


def _net():
    """每個行程只載入一次網路 → (net, device)。"""
    path = os.environ["E11_CKPT"]
    if path not in _NET:
        torch.set_num_threads(1)
        device = os.environ.get("E11_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        _NET[path] = (load_fullnet(path, device), device)
    return _NET[path]


def full_policy(net, device):
    """回合資料 → 每組交易選 FullNet 分數最大的候選。"""
    def policy(data):
        with torch.no_grad():
            out = net(*round_inputs(data, device))
        return out.argmax(dim=1).cpu().numpy()
    return policy


class Ring_SE_E11_Full(Ring_SE_E07):
    def __init__(self, problem, **kwargs):
        net, device = _net()
        super().__init__(problem, policy=full_policy(net, device), **kwargs)


if __name__ == "__main__":
    import experiment_algorithms
    experiment_algorithms.main()
