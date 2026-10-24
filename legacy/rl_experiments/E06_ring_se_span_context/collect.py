"""E06 資料：在 E01 的訓練快照上跑一段最佳化，存每回合的回合資料。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E06_ring_se_span_context/collect.py --policy oracle --shard 0 --shards 8
    python Algorithm/se/rl/experiments/E06_ring_se_span_context/collect.py --policy random --norm
--policy oracle：作弊版推進，存監督式標籤（results/data/labels/shard{i}.npz）。
--norm：隨機策略跑 norm_episodes 段，算 token、候選特徵、全域的平均／標準差（results/data/norm.npz）。
回合層級 (R, ...)、交易層級 (R, 64, ...)；episode (R,) 第幾段；map (R,)。
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from common import DATA, HERE, SnapshotPool, run_episode, split_maps
from env import RoundRecorder
from model import make_tokens

HALF = ("S", "G", "sensor")


def episodes_data(policy, episodes, seed_base, rng, pool, budget, log_prefix):
    parts = []
    for count, episode in enumerate(episodes, start=1):
        snap, problem = pool.sample(rng)
        rec = RoundRecorder(snap["map"])
        run_episode(problem, policy, seed_base + episode, budget, recorder=rec)
        data = rec.stacked()
        data["episode"] = np.full(len(rec.rounds), episode, dtype=np.int32)
        data["map"] = np.full(len(rec.rounds), snap["map"])
        parts.append(data)
        if count % 10 == 0:
            print(f"{log_prefix} {count}/{len(episodes)}", flush=True)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def norm_stats(data):
    t = lambda k: torch.as_tensor(data[k].astype(np.float32 if data[k].dtype.kind == "f" else np.int64))  # noqa: E731
    R, P = data["s_id"].shape
    r = np.repeat(np.arange(R), P)
    s, g = data["s_id"].reshape(-1).astype(int), data["g_id"].reshape(-1).astype(int)
    tok = make_tokens(t("S")[r, s], t("G")[r, g], t("S_prio")[r, s], t("G_prio")[r, g], t("sensor")[r]).numpy()
    tok = tok.reshape(-1, tok.shape[-1])
    cand = data["cand"].reshape(-1, data["cand"].shape[-1])
    glob = data["glob"].reshape(-1, data["glob"].shape[-1])
    return {"tok_mean": tok.mean(0), "tok_std": tok.std(0), "cand_mean": cand.mean(0), "cand_std": cand.std(0),
            "glob_mean": glob.mean(0), "glob_std": glob.std(0)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="oracle")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--norm", action="store_true")
    args = parser.parse_args()
    cfg = json.loads((HERE / "config.json").read_text())
    train, _ = split_maps()
    pool = SnapshotPool(train)
    if args.norm:
        rng = np.random.default_rng([20261019, 0])
        data = episodes_data(args.policy, range(cfg["norm_episodes"]), 600000, rng, pool, cfg["budget"], "[E06 norm]")
        DATA.mkdir(parents=True, exist_ok=True)
        np.savez(DATA / "norm.npz", **norm_stats(data))
        print("[E06 norm] saved", DATA / "norm.npz", flush=True)
        return
    rng = np.random.default_rng([20261019, 1, args.shard])
    total = int(cfg["collect_episodes"])
    mine = [e for e in range(total) if e % args.shards == args.shard]
    data = episodes_data(args.policy, mine, 700000, rng, pool, cfg["budget"], f"[E06 {args.policy}] shard {args.shard}")
    for k in HALF:
        data[k] = data[k].astype(np.float16)
    out = DATA / ("labels" if args.policy == "oracle" else args.policy)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / f"shard{args.shard}.npz", **data)
    print(f"[E06 {args.policy}] shard {args.shard} saved {len(data['done'])} rounds", flush=True)


if __name__ == "__main__":
    main()
