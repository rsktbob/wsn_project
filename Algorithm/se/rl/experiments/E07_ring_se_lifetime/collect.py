"""E07 標籤：RLTRAIN 每張用原版 ring_se（4 候選隨機選）從滿電跑完整 lifetime，每段預算 --budget。
每 3 段標一段，起點依地圖錯開：第 i 張圖（0 起算）標段號 ≡ i (mod 3) 的段。
標的段作弊版只標答案（偷看不計預算、不用演算法亂數，推進和沒標時完全相同；見 env.py）。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E07_ring_se_lifetime/collect.py --budget 1000 --shard 0 --shards 8
    預設收 RLTRAIN01–200 到 results/data/b{預算}/：train ＝ 01–140、val ＝ 141–170、test ＝ 171–200（訓練時取 train 前 N 張；
    test 的標籤用來看網路猜得準不準，跑 lifetime 用地圖清單 maps_test.json）
    （第 i 張圖給 shard i % shards；每張圖一個檔，已存在就跳過，可中斷續跑）
輸出：results/data/b{預算}/{train,val,test}/{地圖}.npz（欄位見 env.py；說明見 results/data/README.md）
    回合層級 (R, ...)：S、G (8, N, 10) float16、S_prio、G_prio、S_par、G_par (8, N)、sensor (N, 2) float16、
        searcher_fit、good_fit (8,)
    交易層級 (R, 64, ...)：s_id、g_id、span (4, 2)、cand (4, 10)、glob (10)、action（實際隨機選的）、reward、
        標籤 diff_cur、diff_prev、oracle_choice、plain_delta，原始值 child1_fit、child2_fit、scale_cur、scale_prev
    segment（lifetime 第幾段）、round（該段第幾回合）、lifetime（該段開始前的累積 lifetime）、done（每段最後一回合 = 1）(R,)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import BUDGET, DATA, MAX_SEGMENTS, build_ring, run_lifetime, train_maps
from env import RoundRecorder

SEED_BASE = 610000
LABEL_STRIDE = 3
HALF = ("S", "G", "sensor")
SPLITS = (("train", 140), ("val", 170), ("test", 200))   # RLTRAIN 第 1–140／141–170／171–200 張


def split_of(index):
    """第 index 張圖（0 起算）屬於 train／val／test。"""
    return next(name for name, end in SPLITS if index < end)


def run_map(index, spec, budget, max_segments):
    offset = index % LABEL_STRIDE
    parts = []

    def recorder_for(segment):
        return RoundRecorder(spec["name"]) if segment % LABEL_STRIDE == offset else None

    def on_segment(algorithm, segment, lifetime):
        rec = algorithm.recorder
        if rec is not None and rec.rounds:
            data = rec.stacked()
            R = len(rec.rounds)
            data["segment"] = np.full(R, segment, dtype=np.int16)
            data["round"] = np.arange(R, dtype=np.int16)
            data["lifetime"] = np.full(R, lifetime, dtype=np.int32)
            parts.append(data)
        algorithm.recorder = recorder_for(segment + 1)

    lifetime, segments = run_lifetime(
        spec, lambda problem: build_ring(problem, "random", SEED_BASE + index, recorder=recorder_for(0)),
        on_segment, budget=budget, max_segments=max_segments)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for k in HALF:
        data[k] = data[k].astype(np.float16)
    return data, lifetime, segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--out", default=None, help="預設 results/data/b{預算}")
    parser.add_argument("--maps", type=int, default=200, help="收前幾張（預設 200＝train 140＋val 30＋test 30）")
    parser.add_argument("--start", type=int, default=0, help="從第幾張開始（0 起算）")
    parser.add_argument("--budget", type=int, default=BUDGET, help="每段預算（預設 10000）")
    parser.add_argument("--max-segments", type=int, default=MAX_SEGMENTS, help="冒煙測試才改")
    args = parser.parse_args()
    root = Path(args.out) if args.out else DATA / f"b{args.budget}"
    maps = train_maps(args.maps)
    for index, spec in enumerate(maps):
        if index < args.start or index % args.shards != args.shard:
            continue
        out = root / split_of(index)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{spec['name']}.npz"
        if path.exists():
            continue
        t0 = time.time()
        data, lifetime, segments = run_map(index, spec, args.budget, args.max_segments)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.rename(path)
        print(f"[E07 collect] {spec['name']} lifetime {lifetime} segments {segments} rounds {len(data['done'])} "
              f"pairs {data['oracle_choice'].size} minutes {(time.time() - t0) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
