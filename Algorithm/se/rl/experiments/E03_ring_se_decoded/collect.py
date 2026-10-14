"""E03 標籤資料：在 E01 的訓練快照上跑作弊版，存下每回合的解碼特徵與每組交易 4 個候選只交換的結果。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E03_ring_se_decoded/collect.py --shard 0 --shards 8 --episodes 400
輸出：data/labels/shard{i}.npz
  回合層級（R 回合）：S、G (R, 8, N, 10) 每顆 sensor 解碼特徵；S_genes、G_genes (R, 8, N, 2)；sensor (R, N, 2)
  交易層級（P 組）：round (P,) 屬於哪個回合；s_id、g_id；spans (P, 4, 3)；outside、position (P, 4, 4)；global (P, 10)
                    delta、f2 (P, 4)；choice (P,)；map (P,)
token 在訓練時才由 env.build_tokens 組出來（不重複存）。
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

from env import RoundRecorder, Ring_SE_E03  # noqa: E402
from snapshot_pool import SnapshotPool, split_maps  # noqa: E402

BUDGET = 1000
ROUND_KEYS = ("S", "G", "S_genes", "G_genes", "sensor")
PAIR_KEYS = ("s_id", "g_id", "spans", "outside", "position", "global", "delta", "f2", "choice")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=400)
    args = parser.parse_args()
    train, _ = split_maps()
    pool = SnapshotPool(train)
    rng = np.random.default_rng([20261014, 3, args.shard])
    out_dir = HERE / "data" / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    rounds = []
    mine = [e for e in range(args.episodes) if e % args.shards == args.shard]
    for count, episode in enumerate(mine, start=1):
        snap, problem = pool.sample(rng)
        rec = RoundRecorder(snap["map"])
        Ring_SE_E03(problem, policy="oracle", recorder=rec, n=8, h=4, w=8, mu=0.4,
                    seed=700000 + episode).run(problem, budget=BUDGET)
        rounds += rec.rounds
        if count % 10 == 0:
            print(f"[E03 labels] shard {args.shard} {count}/{len(mine)}", flush=True)
    out = {k: np.stack([r[k] for r in rounds]).astype(np.float16 if k in ("S", "G", "sensor") else None)
           for k in ROUND_KEYS}
    for k in PAIR_KEYS:
        out[k] = np.concatenate([r[k] for r in rounds])
    out["round"] = np.concatenate([np.full(len(r["choice"]), i, dtype=np.int32) for i, r in enumerate(rounds)])
    out["map"] = np.concatenate([np.full(len(r["choice"]), r["map"]) for r in rounds])
    np.savez_compressed(out_dir / f"shard{args.shard}.npz", **out)
    print(f"[E03 labels] shard {args.shard} saved {len(rounds)} rounds, {len(out['choice'])} pairs", flush=True)


if __name__ == "__main__":
    main()
