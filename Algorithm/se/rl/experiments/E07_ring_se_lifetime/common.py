"""E07 共用：路徑設定、訓練圖、完整 lifetime 推進。

lifetime 推進的流程同 experiments/runner.run_lifetime_case（固定預算、不移動）：
每段最佳化 → 解碼最佳解 → 不可行就結束 → 依最佳解逐時間槽耗電（最多 5000 槽）→ 下一段。
演算法物件跨段重用（同正式流程）；問題 seed 固定 7、fitness v3（同 E05 collect）。
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ROOT)

from experiments.evaluation import run_optimizer  # noqa: E402
from experiments.maps import load_map_specs  # noqa: E402
from experiments.problem_setup import build_problem  # noqa: E402
from experiments.runner import _advance_lifetime, _decode_and_measure  # noqa: E402

RESULTS = HERE / "results"          # symlink → runs/2027WSN/rl/E07_ring_se_lifetime
DATA = RESULTS / "data"

TRAIN_MAPS = "maps/maps_100100100_train.json"   # RLTRAIN001–200（10/22 起 E07 自己切：train 1–140、val 141–170、test 171–200）
N_MAPS = 30                         # 冒煙測試等預設張數
BUDGET = 10000
PROBLEM_SEED = 7
SLOT_LIMIT = 5000
MAX_SEGMENTS = 2000


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def train_maps(n=N_MAPS):
    """maps_100100100_train.json 的前 n 張（RLTRAIN001 起，共 200 張）。"""
    return load_map_specs(TRAIN_MAPS)[:n]


def build_ring(problem, policy, seed, recorder=None):
    """原版 ring_se 參數（n8 h4 w8 μ0.4，EN），每組交易 4 個預抽候選由 policy 選。"""
    from env import Ring_SE_E07
    return Ring_SE_E07(problem, policy=policy, recorder=recorder, n=8, h=4, w=8, mu=0.4, seed=seed)


def run_lifetime(spec, make_algorithm, on_segment=None, budget=BUDGET, max_segments=MAX_SEGMENTS):
    """一張圖從滿電跑完整 lifetime。

    make_algorithm(problem) → 演算法；on_segment(algorithm, segment, lifetime_before) 在每段最佳化完、
    耗電之前呼叫（最後那段也會呼叫）。回傳 (lifetime, 段數)。budget、max_segments 只在冒煙測試時改。
    """
    problem = build_problem(SimpleNamespace(_current_map=spec, fitness_service="v3"), PROBLEM_SEED)
    algorithm = make_algorithm(problem)
    lifetime = 0
    segments = 0
    for segment in range(max_segments):
        optimization = run_optimizer(algorithm, problem, budget)
        segments = segment + 1
        if on_segment is not None:
            on_segment(algorithm, segment, lifetime)
        if optimization["state"] is None:
            break
        state, cost, _ = _decode_and_measure(algorithm, problem, optimization)
        if not problem.LifeCheck(state, cost):
            break
        lifetime, length, _ = _advance_lifetime(problem, state, cost, lifetime, SLOT_LIMIT, None)
        if length > 0:
            problem.prepare_coding_cache(cost * length)
    return lifetime, segments


__all__ = ["HERE", "ROOT", "RESULTS", "DATA", "N_MAPS", "BUDGET", "load_module", "train_maps", "build_ring",
           "run_lifetime"]
