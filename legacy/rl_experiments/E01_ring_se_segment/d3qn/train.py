"""E01 / D3QN 訓練（DQfD 式）。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E01_ring_se_segment/d3qn/train.py --mode dqfd --seed 0
mode：dqfd（主方法）／d3qn（不用示範）／imitation（只用大邊界模仿）。超參數在 config.json。
輸出：results/{mode}_seed{seed}/（log.jsonl、config.json、best.pt、last.pt）
驗證：10 張驗證圖 × 快照 val_snapshot_index，同 seed 和隨機選段比較這段最佳化的最終 fitness。
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E01 = HERE.parent
ROOT = E01.parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(E01))
sys.path.insert(0, str(HERE))
os.chdir(ROOT)

from agent import DQfDAgent  # noqa: E402
from collect_demos import SnapshotPool  # noqa: E402
from env import EpisodeRecorder, Ring_SE_E01  # noqa: E402
from snapshots import split_maps  # noqa: E402

KEYS = ("s", "a", "r", "s1", "d1", "rn", "sn", "dn", "gn")


def load_demos():
    parts = [np.load(p) for p in sorted((E01 / "data" / "demos").glob("shard*.npz"))]
    if not parts:
        raise FileNotFoundError("no demos; run collect_demos.py first")
    return {k: np.concatenate([p[k] for p in parts]) for k in KEYS}


def run_episode(problem, policy, seed, budget, recorder=None):
    algorithm = Ring_SE_E01(problem, policy=policy, recorder=recorder, n=8, h=4, w=8, mu=0.4, seed=seed)
    algorithm.run(problem, budget=budget)
    return algorithm


def validation_set(cfg):
    _, val = split_maps()
    pool = SnapshotPool(val)
    items = []
    for name in pool.names:
        snaps = pool.get(name)
        for idx in cfg["val_snapshot_index"]:
            if idx < len(snaps):
                items.append((name, snaps[idx]["segment"], snaps[idx]["problem"]))
    return items


def validate(items, policy, budget):
    return np.array([run_episode(pickle.loads(blob), policy, 900000 + i, budget).fitness
                     for i, (_, _, blob) in enumerate(items)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--mode", choices=("dqfd", "d3qn", "imitation"), default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for key in ("mode", "seed", "episodes"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    torch.set_num_threads(int(cfg["torch_threads"]))

    out = HERE / "results" / f"{cfg['mode']}_seed{cfg['seed']}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log = open(out / "log.jsonl", "a")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    rng = np.random.default_rng(int(cfg["seed"]))
    agent = DQfDAgent(cfg, rng)
    recorder = EpisodeRecorder(cfg["gamma"], cfg["n_step"])
    train_specs, _ = split_maps()
    pool = SnapshotPool(train_specs)

    # ---- 驗證基準：同快照、同 seed 的隨機選段 ----
    val_items = validation_set(cfg)
    started = time.time()
    val_random = validate(val_items, "random", cfg["budget"])
    write({"event": "val_baseline", "n": len(val_items), "random_mean_fitness": float(val_random.mean()),
           "seconds": round(time.time() - started, 1)})

    # ---- 示範與預訓練 ----
    if cfg["mode"] in ("dqfd", "imitation"):
        demos = load_demos()
        agent.replay.set_demo(demos)
        write({"event": "demos", "transitions": len(demos["s"]), "reward_positive": float((demos["r"] > 0).mean())})
        started = time.time()
        losses = [agent.update(cfg["per_beta0"], demo_only=True) for _ in range(int(cfg["pretrain_steps"]))]
        write({"event": "pretrain", "steps": len(losses), "loss_last1k": float(np.mean(losses[-1000:])),
               "seconds": round(time.time() - started, 1)})

    def run_validation(episode):
        policy = lambda s: agent.act(s, 0.0)  # noqa: E731
        fit = validate(val_items, policy, cfg["budget"])
        diff = fit - val_random
        return {"event": "val", "episode": episode, "mean_diff": float(diff.mean()),
                "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "mean_fitness": float(fit.mean())}

    best = run_validation(0) if cfg["mode"] != "d3qn" else {"mean_diff": -np.inf}
    if cfg["mode"] != "d3qn":
        write(best)
        agent.save(out / "best.pt", best)

    episodes = int(cfg["episodes"])
    eps_start = cfg["eps_start_d3qn"] if cfg["mode"] == "d3qn" else cfg["eps_start"]
    decay = max(1, int(cfg["eps_decay_frac"] * episodes))
    for episode in range(1, episodes + 1):
        t0 = time.time()
        epsilon = eps_start + (cfg["eps_end"] - eps_start) * min(1.0, (episode - 1) / decay)
        beta = cfg["per_beta0"] + (1.0 - cfg["per_beta0"]) * min(1.0, episode / episodes)
        snap, problem = pool.sample(rng)
        if cfg["mode"] == "imitation":
            algorithm = run_episode(problem, lambda s: agent.act(s, epsilon), 100000 * (cfg["seed"] + 1) + episode,
                                    cfg["budget"])
            recorder.pop_all()
        else:
            algorithm = run_episode(problem, lambda s: agent.act(s, epsilon), 100000 * (cfg["seed"] + 1) + episode,
                                    cfg["budget"], recorder)
            for part in recorder.pop_all():
                agent.replay.add(part)
        losses = []
        if agent.replay.agent_size + agent.replay.demo_size >= cfg["warmup"]:
            for _ in range(int(cfg["updates_per_round"]) * len(algorithm.round_stats)):
                losses.append(agent.update(beta, demo_only=cfg["mode"] == "imitation"))
        counts = np.sum([r["segment_counts"] for r in algorithm.round_stats], axis=0).tolist()
        write({"event": "episode", "episode": episode, "map": snap["map"], "segment": snap["segment"],
               "fitness": float(algorithm.fitness), "mean_reward": float(np.mean([r["mean_reward"] for r in algorithm.round_stats])),
               "epsilon": round(epsilon, 4), "loss": float(np.mean(losses)) if losses else None,
               "segment_counts": counts, "replay": agent.replay.agent_size, "seconds": round(time.time() - t0, 2)})
        if episode % int(cfg["val_every"]) == 0 or episode == episodes:
            result = run_validation(episode)
            write(result)
            agent.save(out / "last.pt", result)
            if result["mean_diff"] > best["mean_diff"]:
                best = result
                agent.save(out / "best.pt", result)
    write({"event": "done", "best": best})


if __name__ == "__main__":
    main()
