"""監督式診斷的標籤：在 E01 的訓練快照上跑 8 段作弊版（預算 1000），每組交易存狀態與 8 段只交換的 Δ。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E04_ring_se_seg8/supervised/collect.py --shard 0 --shards 8 --episodes 100
輸出：../data/labels/shard{i}.npz
  states (P, 2770) float16；delta、f2 (P, 8)；choice (P,)；round (P,) 全域回合編號；map (P,)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import DATA, SnapshotPool, run_episode, split_maps  # noqa: E402
from env import LabelRecorder  # noqa: E402

BUDGET = 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    train, _ = split_maps()
    pool = SnapshotPool(train)
    rng = np.random.default_rng(7000 + args.shard)
    out = {k: [] for k in ("states", "delta", "f2", "choice", "round", "map")}
    round_id = 0
    for episode in range(args.episodes):
        snap, problem = pool.sample(rng)
        recorder = LabelRecorder()
        run_episode(problem, "oracle", 500000 + 1000 * args.shard + episode, BUDGET, recorder)
        for r in recorder.rounds:
            P = len(r["choice"])
            for key in ("states", "delta", "f2", "choice"):
                out[key].append(r[key])
            out["round"].append(np.full(P, args.shard * 10 ** 6 + round_id, dtype=np.int64))
            out["map"].append(np.full(P, snap["map"]))
            round_id += 1
        print(f"[collect] shard {args.shard} episode {episode + 1}/{args.episodes} {snap['map']} rounds {round_id}",
              flush=True)
    path = DATA / "labels"
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / f"shard{args.shard}.npz", **{k: np.concatenate(v) for k, v in out.items()})


if __name__ == "__main__":
    main()
