"""E07 lifetime 評估：
    e07_net：凍結的網路，選分數／Q 最大（監督式、D3QN 都用這個，網路種類存在 checkpoint 裡）
    e07_net_sample：監督式依 softmax(分數) 的機率抽樣（亂數依演算法 seed，各場可重現）
    e07_oracle：作弊版天花板（偷看 4 個候選只交換的子代、選一般 Δ 最大；偷看不計預算；不需要 E07_CKPT）
    e07_oracle_diff：作弊版，跟 diff_cur 標籤一致（其他交易固定為原版隨機選擇算差異獎勵、選最大；全 0 留隨機）
    e07_oracle_plain：作弊版，跟 plain_delta 標籤一致（只看這筆交易、選一般 Δ 最大；全 0 留隨機）
    e07_random_pri、e07_oracle_plain_pri：染色體換成 PriorityEncoding（初始化、突變同 Ring_SEpri）；
        random_pri 應和 ring_sepri 逐場相同（驗證用），oracle_plain_pri 是 ring_sepri 的 plain 作弊版
checkpoint 由環境變數 E07_CKPT 給，裝置 E07_DEVICE（預設有 GPU 就用）。

用法（在 2027WSN 根目錄）：
    E07_CKPT=.../final.pt python Algorithm/se/rl/experiments/E07_ring_se_lifetime/evaluate.py \
        --algorithm e07_net --runs 30 --seed 7 --evaluate 10000 --maps maps/maps_100100100.json ...
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

import numpy as np  # noqa: E402
import torch  # noqa: E402

import experiments.registry as registry  # noqa: E402

VARIANTS = {"e07_net": "Ring_SE_E07_Net", "e07_net_sample": "Ring_SE_E07_NetSample", "e07_oracle": "Ring_SE_E07_Oracle",
            "e07_oracle_diff": "Ring_SE_E07_OracleDiff", "e07_oracle_plain": "Ring_SE_E07_OraclePlain",
            "e07_random_pri": "Ring_SE_E07_RandomPri", "e07_oracle_plain_pri": "Ring_SE_E07_OraclePlainPri"}
for name, cls in VARIANTS.items():
    registry.ALGORITHM_IMPORTS[name] = ("evaluate", cls)
registry.ALGORITHM_NAMES = tuple(registry.ALGORITHM_IMPORTS)
registry.ALGORITHM_CHOICES = ("all",) + registry.ALGORITHM_NAMES
registry.ALL_RUNNABLE_ALGORITHMS = tuple(n for n in registry.ALGORITHM_NAMES if n not in registry.RL_SETS_ALGORITHMS)

from experiments.presets import ALGORITHM_PRESETS  # noqa: E402

for name, cls in VARIANTS.items():
    ALGORITHM_PRESETS[name] = copy.deepcopy(ALGORITHM_PRESETS["ring_se"])
    ALGORITHM_PRESETS[name]["label"] = cls

from env import Ring_SE_E07  # noqa: E402
from Algorithm.se.ring_se.Ring_SEpri import Ring_SEpri  # noqa: E402
from model import greedy_policy, load_net, sample_policy  # noqa: E402

_NET = {}


def _net():
    """每個行程只載入一次網路 → (net, device)。"""
    path = os.environ["E07_CKPT"]
    if path not in _NET:
        torch.set_num_threads(1)
        device = os.environ.get("E07_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        _NET[path] = (load_net(path, device), device)
    return _NET[path]


class Ring_SE_E07_Net(Ring_SE_E07):
    def __init__(self, problem, **kwargs):
        net, device = _net()
        super().__init__(problem, policy=greedy_policy(net, device), **kwargs)


class Ring_SE_E07_NetSample(Ring_SE_E07):
    def __init__(self, problem, **kwargs):
        net, device = _net()
        rng = np.random.default_rng([20261019, int(kwargs.get("seed") or 0)])
        super().__init__(problem, policy=sample_policy(net, rng, device), **kwargs)


class Ring_SE_E07_Oracle(Ring_SE_E07):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, policy="oracle", **kwargs)


class Ring_SE_E07_OracleDiff(Ring_SE_E07):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, policy="oracle_diff", **kwargs)


class Ring_SE_E07_OraclePlain(Ring_SE_E07):
    def __init__(self, problem, **kwargs):
        super().__init__(problem, policy="oracle_plain", **kwargs)


class Ring_SE_E07_RandomPri(Ring_SE_E07):
    ROUTING_MUTATION_MULTIPLIER = Ring_SEpri.ROUTING_MUTATION_MULTIPLIER
    create_candidate = Ring_SEpri.create_candidate
    mutate_candidate = Ring_SEpri.mutate_candidate


class Ring_SE_E07_OraclePlainPri(Ring_SE_E07):
    ROUTING_MUTATION_MULTIPLIER = Ring_SEpri.ROUTING_MUTATION_MULTIPLIER
    create_candidate = Ring_SEpri.create_candidate
    mutate_candidate = Ring_SEpri.mutate_candidate

    def __init__(self, problem, **kwargs):
        super().__init__(problem, policy="oracle_plain", **kwargs)


if __name__ == "__main__":
    import experiment_algorithms
    experiment_algorithms.main()
