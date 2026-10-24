"""E05 標籤（第二版）：190 張訓練圖每張用 8 段整段隨機的 ring_se 從滿電跑完整 lifetime（不用 E01 的快照），
每 3 段標一次，起點依地圖錯開：第 i 張圖標段號 ≡ i (mod 3) 的段（圖 0 標 0、3、6…，圖 1 標 1、4、7…，
圖 2 標 2、5、8…，圖 3 又從 0 開始），所以各段號都有涵蓋。

* 推進：每組交易 8 段均勻隨機選段（＝E04 的 8 段隨機），整段交換、段內突變、EN 更新，預算 1000。
* 標答案：每組交易偷看 8 段只交換的結果（不計預算、不更新最佳），存 8 個 Δ。
  偷看不用演算法的亂數，所以有標和沒標的段，推進過程都和純 8 段隨機完全相同。
* lifetime 的流程同 E01 snapshots.py：問題 seed 7、fitness v3；每段最佳化後依最佳排程逐時間槽耗電
  （最多 5000 槽），找不到可行解或一槽都撐不了就結束，最多 MAX_SEGMENTS 段。
* 不每段都標（使用者 10/18 決定）：每段都標約 330 萬筆、18 GB、3–5 小時；錯開標約 110 萬筆、6 GB。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E05_ring_se_seg8_supervised/collect.py --shard 0 --shards 8
    （第 i 張訓練圖給 shard i % shards；每張圖一個檔，已存在就跳過，可中斷續跑）
輸出：data/labels_lifetime/{地圖}.npz
  states (P, 2770) float16；delta、f2 (P, 8)；choice（作弊版會選的段）、taken（實際隨機選的段）(P,)；
  segment（lifetime 第幾段）、round（該段第幾回合）(P,)；lifetime（到該段開始前的累積 lifetime）(P,)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e05_common import split_maps  # noqa: E402  （設好 sys.path 與 cwd）
from env import LabelRecorder, Ring_SE_E04  # noqa: E402
from experiments.evaluation import prepare_result_state, run_optimizer  # noqa: E402
from experiments.problem_setup import build_problem  # noqa: E402

BUDGET = 1000
PROBLEM_SEED = 7
SEED_BASE = 600000
LABEL_STRIDE = 3
MAX_SEGMENTS = 60
SLOT_LIMIT = 5000
OUT = HERE / "data" / "labels_lifetime"
KEYS = ("states", "delta", "f2", "choice", "taken", "segment", "round", "lifetime")


def run_map(spec, seed, offset):
    problem = build_problem(SimpleNamespace(_current_map=spec, fitness_service="v3"), PROBLEM_SEED)
    algorithm = Ring_SE_E04(problem, policy="random", n=8, w=8, mu=0.4, seed=seed)
    out = {k: [] for k in KEYS}
    lifetime = 0
    for segment in range(MAX_SEGMENTS):
        algorithm.recorder = LabelRecorder() if segment % LABEL_STRIDE == offset else None
        result = run_optimizer(algorithm, problem, BUDGET)
        if algorithm.recorder is not None:
            for r_index, r in enumerate(algorithm.recorder.rounds):
                P = len(r["choice"])
                for key in ("states", "delta", "f2", "choice", "taken"):
                    out[key].append(r[key])
                out["segment"].append(np.full(P, segment, dtype=np.int16))
                out["round"].append(np.full(P, r_index, dtype=np.int16))
                out["lifetime"].append(np.full(P, lifetime, dtype=np.int32))
        state = prepare_result_state(algorithm, problem, result["state"])
        if state is None:
            break
        cost = problem.calculate_total_cost(state)
        length = 0
        for _ in range(SLOT_LIMIT):
            if not problem.LifeCheck(state, cost):
                break
            problem.energy = problem.calculate_remaining_energy(cost)
            length += 1
        if length == 0:
            break
        lifetime += length
        problem.prepare_coding_cache(cost * length)
    return {k: np.concatenate(v) for k, v in out.items()}, lifetime, segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    train, _ = split_maps()
    OUT.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(train):
        if index % args.shards != args.shard:
            continue
        path = OUT / f"{spec['name']}.npz"
        if path.exists():
            continue
        data, lifetime, segments = run_map(spec, SEED_BASE + index, index % LABEL_STRIDE)
        tmp = path.with_suffix(".tmp.npz")
        np.savez(tmp, **data)
        tmp.rename(path)
        print(f"[collect] {spec['name']} lifetime {lifetime} segments {segments} rows {len(data['delta'])}", flush=True)


if __name__ == "__main__":
    main()
