"""E12 lifetime 評估：e12_relflag ＝ E07 的 env（原版 ring_se、每組交易 4 個預抽候選），由 RelFlagNet 選分數最大的候選。

每回合在 E07 env 的回合資料上即時算 ②③⑤（features.round_extras，地圖覆蓋表取自 problem），不解碼子代。
checkpoint 由環境變數 E12_CKPT 給，裝置 E12_DEVICE（預設有 GPU 就用）。

用法（2027WSN 根目錄）：
    E12_CKPT=.../best.pt python Algorithm/se/rl/experiments/E12_ring_se_relflag/evaluate_relflag.py \
        --algorithm e12_relflag --runs 5 --seed 7 --evaluate 1000 --fitness-service v3 --maps maps/maps_100100100.json ...
參數同 ring_se preset（n8 h4 w8 μ0.4）。
"""
import copy
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E07 = HERE.parent / "E07_ring_se_lifetime"
E08 = HERE.parent / "E08_ring_se_relations"
ROOT = HERE.parents[4]
for p in (str(HERE), str(E07), str(E08), str(ROOT)):
    sys.path.insert(0, p)
os.chdir(ROOT)
# 註冊必須在 experiment_algorithms 第一次被 import 之前，所以 env／網路放後面

import numpy as np  # noqa: E402
import torch  # noqa: E402

import experiments.registry as registry  # noqa: E402

VARIANTS = {"e12_relflag": "Ring_SE_E12_RelFlag"}
for name, cls in VARIANTS.items():
    registry.ALGORITHM_IMPORTS[name] = ("evaluate_relflag", cls)
registry.ALGORITHM_NAMES = tuple(registry.ALGORITHM_IMPORTS)
registry.ALGORITHM_CHOICES = ("all",) + registry.ALGORITHM_NAMES
registry.ALL_RUNNABLE_ALGORITHMS = tuple(n for n in registry.ALGORITHM_NAMES if n not in registry.RL_SETS_ALGORITHMS)

from experiments.presets import ALGORITHM_PRESETS  # noqa: E402

for name, cls in VARIANTS.items():
    ALGORITHM_PRESETS[name] = copy.deepcopy(ALGORITHM_PRESETS["ring_se"])
    ALGORITHM_PRESETS[name]["label"] = cls

from env import Ring_SE_E07  # noqa: E402
from features import levels_of, round_extras  # noqa: E402
from model import round_inputs  # noqa: E402
from relflagnet import load_relflagnet  # noqa: E402

_NET = {}


def _net():
    """每個行程只載入一次網路 → (net, device)。"""
    path = os.environ["E12_CKPT"]
    if path not in _NET:
        torch.set_num_threads(1)
        device = os.environ.get("E12_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        _NET[path] = (load_relflagnet(path, device), device)
    return _NET[path]


def rel_policy(net, cover, device):
    """回合資料 → 每組交易選 RelFlagNet 分數最大的候選。"""
    def policy(data):
        route, covrel, swap = round_extras(levels_of(data["S"]), levels_of(data["G"]), data["S_par"], data["G_par"],
                                           cover, data["s_id"], data["g_id"], data["span"])
        tok, inside, cand, glob = round_inputs(data, device)
        t = lambda x, dt: torch.as_tensor(x, dtype=dt, device=device)  # noqa: E731
        with torch.no_grad():
            out = net(tok, inside, t(route, torch.bool), t(covrel, torch.bool), cand, t(swap, torch.float32), glob)
        return out.argmax(dim=1).cpu().numpy()
    return policy


class Ring_SE_E12_RelFlag(Ring_SE_E07):
    def __init__(self, problem, **kwargs):
        net, device = _net()
        cover = np.asarray(problem.coverage_table) > 0
        super().__init__(problem, policy=rel_policy(net, cover, device), **kwargs)


if __name__ == "__main__":
    import experiment_algorithms
    experiment_algorithms.main()
