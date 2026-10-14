"""E01 示範資料：在訓練快照上跑作弊版（偷看 4 段只交換的結果選段），存下每組交易的轉移。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E01_ring_se_segment/collect_demos.py --shard 0 --shards 4 --episodes 200
每個 episode：隨機一張訓練圖、隨機一個快照，跑一段預算 1000 的作弊版最佳化。
輸出：data/demos/shard{i}.npz（欄位 s, a, r, s1, d1, rn, sn, dn, gn；γ = 0.5、n = 3）
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)

from env import EpisodeRecorder, Ring_SE_E01  # noqa: E402
from snapshots import OUT as SNAPSHOT_DIR, split_maps  # noqa: E402

BUDGET = 1000
GAMMA, N_STEP = 0.5, 3


class SnapshotPool:
    """依需要讀進快照檔（每張圖一個檔），讀過的留在記憶體。"""

    def __init__(self, specs):
        self.names = [s["name"] for s in specs if (SNAPSHOT_DIR / f"{s['name']}.pkl.gz").exists()]
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            with gzip.open(SNAPSHOT_DIR / f"{name}.pkl.gz", "rb") as handle:
                self.cache[name] = pickle.load(handle)
        return self.cache[name]

    def sample(self, rng):
        name = self.names[int(rng.integers(len(self.names)))]
        snaps = self.get(name)
        snap = snaps[int(rng.integers(len(snaps)))]
        return snap, pickle.loads(snap["problem"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=200, help="總 episode 數（所有 shard 加起來）")
    args = parser.parse_args()
    train, _ = split_maps()
    pool = SnapshotPool(train)
    rng = np.random.default_rng([20261013, args.shard])
    recorder = EpisodeRecorder(GAMMA, N_STEP)
    out_dir = HERE / "data" / "demos"
    out_dir.mkdir(parents=True, exist_ok=True)
    mine = [e for e in range(args.episodes) if e % args.shards == args.shard]
    for count, episode in enumerate(mine, start=1):
        snap, problem = pool.sample(rng)
        algorithm = Ring_SE_E01(problem, policy="oracle", recorder=recorder,
                                n=8, h=4, w=8, mu=0.4, seed=500000 + episode)
        algorithm.run(problem, budget=BUDGET)
        if count % 10 == 0:
            print(f"[demos] shard {args.shard} {count}/{len(mine)} {snap['map']} seg {snap['segment']}", flush=True)
    parts = recorder.pop_all()
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(out_dir / f"shard{args.shard}.npz", **data)
    print(f"[demos] shard {args.shard} saved {len(data['s'])} transitions", flush=True)


if __name__ == "__main__":
    main()
