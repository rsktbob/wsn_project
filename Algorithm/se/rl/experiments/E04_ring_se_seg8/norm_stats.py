"""輸入標準化的平均／標準差：在訓練快照上用隨機選段跑 EPISODES 段最佳化，算一次後固定。

用法（在 2027WSN 根目錄）：python Algorithm/se/rl/experiments/E04_ring_se_seg8/norm_stats.py
輸出：data/norm.npz（token／seg／glob 各自的 mean、std；token 只算真的 sensor，不算補的空位）
"""
from __future__ import annotations

import numpy as np

from common import DATA, SnapshotPool, run_episode, split_maps
from env import split_state

EPISODES = 40
SEED = 12345
STD_FLOOR = 1e-3


def main():
    train, _ = split_maps()
    pool = SnapshotPool(train)
    rng = np.random.default_rng(SEED)
    states = []
    for episode in range(EPISODES):
        _, problem = pool.sample(rng)
        algorithm = run_episode(problem, "random", SEED + episode, 1000)
        states += [s for steps in algorithm.pop_trajectories().values() for s, _, _ in steps]
    tokens, mask, seg, glob = split_state(np.stack(states))
    valid = tokens[mask > 0]
    out = {}
    for name, x in (("token", valid), ("seg", seg.reshape(-1, seg.shape[-1])), ("glob", glob)):
        out[f"{name}_mean"] = x.mean(axis=0).astype(np.float32)
        out[f"{name}_std"] = np.maximum(x.std(axis=0), STD_FLOOR).astype(np.float32)
    DATA.mkdir(parents=True, exist_ok=True)
    np.savez(DATA / "norm.npz", **out)
    for key, value in out.items():
        print(key, np.round(value, 3))


if __name__ == "__main__":
    main()
