"""E02 標籤資料：在 E01 的訓練快照上跑作弊版，存下每組交易 4 個候選的特徵與只交換的結果。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E02_ring_se_candidate/collect_labels.py --shard 0 --shards 4 --episodes 200
輸出：data/labels/shard{i}.npz
    X (N, 4, 25)  4 個候選的特徵
    delta (N, 4)  4 個候選只交換後的 Δ = max(0, f(c1)−f(s)) + max(0, f(c2)−f(g))（≤ 1e-12 為 0）
    f2 (N, 4)     4 個候選只交換後的 f(child2)（作弊版同分時的比較依據）
    choice (N,)   作弊版的選擇
    map (N,)      訓練圖名稱（之後依地圖切訓練／驗證）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)

from env import Ring_SE_E02  # noqa: E402

from snapshot_pool import SnapshotPool, split_maps  # noqa: E402  （沿用 E01 的訓練快照）

BUDGET = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--offset", type=int, default=0, help="episode 編號起點（接著收資料時避免和舊資料重複）")
    args = parser.parse_args()
    train, _ = split_maps()
    pool = SnapshotPool(train)
    rng = np.random.default_rng([20261014, args.shard] + ([args.offset] if args.offset else []))
    out_dir = HERE / "data" / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    X, delta, f2, choice, maps = [], [], [], [], []
    mine = [e for e in range(args.episodes) if e % args.shards == args.shard]
    for count, episode in enumerate(mine, start=1):
        snap, problem = pool.sample(rng)
        sink = []
        Ring_SE_E02(problem, policy="oracle", label_sink=sink, n=8, h=4, w=8, mu=0.4,
                    seed=600000 + args.offset + episode).run(problem, budget=BUDGET)
        for x, d, c, g in sink:
            X.append(x); delta.append(d); choice.append(c); f2.append(g); maps.append(snap["map"])
        if count % 10 == 0:
            print(f"[labels] shard {args.shard} {count}/{len(mine)}", flush=True)
    name = f"shard{args.shard}.npz" if not args.offset else f"shard{args.shard}_o{args.offset}.npz"
    np.savez_compressed(out_dir / name, X=np.asarray(X, dtype=np.float32),
                        delta=np.asarray(delta), f2=np.asarray(f2), choice=np.asarray(choice),
                        map=np.asarray(maps))
    print(f"[labels] shard {args.shard} saved {len(X)} pairs", flush=True)


if __name__ == "__main__":
    main()
