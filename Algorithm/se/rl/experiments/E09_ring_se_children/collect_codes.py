"""E09 收資料：和 E07 collect 完全相同（同 seed、同地圖、同每 3 段標一段），另外每回合多存 8 個 searcher、
8 個 goods 的原始染色體 S_code、G_code (8, 2N) int8（等級基因、優先度基因交錯），用來直接拼出子代。

用法（2027WSN 根目錄）：BUDGET=1000 bash Algorithm/se/rl/experiments/E09_ring_se_children/run_collect.sh（呼叫 collect_codes.py）
輸出：E09 results/data/b{預算}/{train,val,test}/{地圖}.npz（E07 的全部欄位＋S_code、G_code）。
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
E07 = HERE.parent / "E07_ring_se_lifetime"
sys.path.insert(0, str(E07))

import numpy as np  # noqa: E402

from collect import HALF, LABEL_STRIDE, SEED_BASE, split_of  # noqa: E402
from common import MAX_SEGMENTS, build_ring, run_lifetime, train_maps  # noqa: E402
from env import RoundRecorder  # noqa: E402

DATA = HERE / "results" / "data"


class CodeRecorder(RoundRecorder):
    """E07 的回合紀錄＋這一回合開始時（交換前）所有 searcher、goods 的原始染色體。"""

    def __init__(self, map_name, algorithm):
        super().__init__(map_name)
        self.algorithm = algorithm

    def add_round(self, record):
        record["S_code"] = np.stack([np.asarray(x.code) for x in self.algorithm.searchers]).astype(np.int8)
        record["G_code"] = np.stack([np.asarray(x.code) for x in self.algorithm.goods]).astype(np.int8)
        super().add_round(record)


def run_map(index, spec, budget, max_segments):
    offset = index % LABEL_STRIDE
    parts = []

    def recorder_for(segment, algorithm):
        return CodeRecorder(spec["name"], algorithm) if segment % LABEL_STRIDE == offset else None

    def make(problem):
        algorithm = build_ring(problem, "random", SEED_BASE + index)
        algorithm.recorder = recorder_for(0, algorithm)
        return algorithm

    def on_segment(algorithm, segment, lifetime):
        rec = algorithm.recorder
        if rec is not None and rec.rounds:
            data = rec.stacked()
            R = len(rec.rounds)
            data["segment"] = np.full(R, segment, dtype=np.int16)
            data["round"] = np.arange(R, dtype=np.int16)
            data["lifetime"] = np.full(R, lifetime, dtype=np.int32)
            parts.append(data)
        algorithm.recorder = recorder_for(segment + 1, algorithm)

    lifetime, segments = run_lifetime(spec, make, on_segment, budget=budget, max_segments=max_segments)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    for k in HALF:
        data[k] = data[k].astype(np.float16)
    return data, lifetime, segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--maps", type=int, default=200)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--max-segments", type=int, default=MAX_SEGMENTS)
    args = parser.parse_args()
    root = DATA / f"b{args.budget}"
    for index, spec in enumerate(train_maps(args.maps)):
        if index % args.shards != args.shard:
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
        print(f"[E09 collect] {spec['name']} lifetime {lifetime} segments {segments} rounds {len(data['done'])} "
              f"minutes {(time.time() - t0) / 60:.1f}", flush=True)


if __name__ == "__main__":
    main()
