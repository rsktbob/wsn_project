"""E13 輸入：用和 E07 collect.py 相同的地圖、seed、標註段重跑原版 ring_se，只收 E13 的網路輸入（不偷看子代）。

標籤（plain_delta 等）沿用 E07 results/data/b{預算}，②③⑤ 沿用 E08 results/data/b{預算}；這裡逐回合檢查和 E07 對齊
（span、s_id、g_id、父節點、priority、segment、round 相同，候選／全域裡和 E07 定義相同的欄位數值相同），不同就報錯。

用法（2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E13_ring_se_newinput/collect13.py --budget 1000 --shard 0 --shards 12
輸出：E13 results/data/b{預算}/{train,val,test}/{地圖}.npz：S、G (R, 8, N, 10)、sensor (R, N, 4)、S_prio、G_prio、S_par、G_par、
    s_id、g_id、span、cand (R, 64, 4, 9)、glob (R, 64, 7)、segment、round、done
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
E07 = HERE.parent / "E07_ring_se_lifetime"
sys.path.insert(0, str(E07))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from collect import LABEL_STRIDE, SEED_BASE, split_of  # noqa: E402
from common import DATA as E07_DATA, MAX_SEGMENTS, run_lifetime, train_maps  # noqa: E402
from env import RoundRecorder  # noqa: E402
from env13 import Ring_SE_E13  # noqa: E402

DATA = HERE / "results" / "data"
HALF = ("S", "G", "sensor")
SAME = ("span", "s_id", "g_id", "S_par", "G_par", "S_prio", "G_prio", "segment", "round")
CAND_COLS = ([0, 1, 2, 3, 4, 5, 6, 7, 8], [0, 1, 2, 4, 5, 6, 7, 8, 9])     # E13 欄 ↔ E07 欄（E07 第 3 欄是否跨段）
GLOB_COLS = ([0, 1, 2, 3, 4, 5, 6], [2, 4, 5, 6, 7, 8, 9])


def run_map(index, spec, budget, max_segments):
    offset = index % LABEL_STRIDE
    parts = []

    def recorder_for(segment):
        return RoundRecorder(spec["name"]) if segment % LABEL_STRIDE == offset else None

    def on_segment(algorithm, segment, lifetime):
        rec = algorithm.feat_recorder
        if rec is not None and rec.rounds:
            data = rec.stacked()
            R = len(rec.rounds)
            data["segment"] = np.full(R, segment, dtype=np.int16)
            data["round"] = np.arange(R, dtype=np.int16)
            parts.append(data)
        algorithm.feat_recorder = recorder_for(segment + 1)

    def build(problem):
        return Ring_SE_E13(problem, policy="random", feat_recorder=recorder_for(0),
                           n=8, h=4, w=8, mu=0.4, seed=SEED_BASE + index)

    lifetime, segments = run_lifetime(spec, build, on_segment, budget=budget, max_segments=max_segments)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for k in HALF:
        data[k] = data[k].astype(np.float16)
    return data, lifetime, segments


def check_aligned(data, ref_path):
    with np.load(ref_path) as ref:
        for k in SAME:
            if ref[k].shape != data[k].shape or not np.array_equal(ref[k], data[k]):
                raise RuntimeError(f"{ref_path.name}：{k} 和 E07 不同")
        for key, (mine, theirs) in (("cand", CAND_COLS), ("glob", GLOB_COLS)):
            if not np.allclose(data[key][..., mine], ref[key][..., theirs], atol=1e-5):
                raise RuntimeError(f"{ref_path.name}：{key} 和 E07 同定義的欄位數值不同")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--maps", type=int, default=200)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--max-segments", type=int, default=MAX_SEGMENTS, help="冒煙測試才改（不對齊檢查）")
    args = parser.parse_args()
    for index, spec in enumerate(train_maps(args.maps)):
        if index % args.shards != args.shard:
            continue
        split = split_of(index)
        out = DATA / f"b{args.budget}" / split
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{spec['name']}.npz"
        if path.exists():
            continue
        t0 = time.time()
        data, lifetime, segments = run_map(index, spec, args.budget, args.max_segments)
        if args.max_segments == MAX_SEGMENTS:
            check_aligned(data, E07_DATA / f"b{args.budget}" / split / f"{spec['name']}.npz")
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **data)
        tmp.rename(path)
        print(f"[E13 collect] {spec['name']} lifetime {lifetime} segments {segments} rounds {len(data['done'])} "
              f"minutes {(time.time() - t0) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
