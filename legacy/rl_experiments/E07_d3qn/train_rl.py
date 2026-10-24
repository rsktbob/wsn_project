"""E07 D3QN（從零、γ = 0）：RLTRAIN01–30 每張從滿電跑完整 lifetime，一段最佳化（每段預算 rl.budget）＝一個 episode。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E07_ring_se_lifetime/train_rl.py --seed 0
設定：config.json 的 rl。輸出：results/d3qn_b{budget}/seed{seed}/（log.jsonl、config.json、ep{k}.pt、final.pt、TRAIN_DONE）。

* 網路：model.DuelingNet（Q_c ＝ V ＋ A_c − mean(A)）；標準化統計用 rl.norm（預算 1000 標籤資料估的）。
* 行動：每回合 64 組交易，ε-greedy 選 Q 最大；ε 1.0 → 0.05，前 eps_decay_frac 的 expected_episodes 線性下降。
* 獎勵：環境給的 E06 差異獎勵（實際子代、含突變，÷ 上一回合一般 Δ 的正值中位數、上限 5）。
* γ = 0：目標就是 r（Huber），所以不需要下一個狀態、target network 和 Double DQN。
* Replay：FIFO，最多 replay_rounds 回合（每回合 64 組），全部放 GPU；比例式 PER（α、β 線性升到 1），在 GPU 上抽樣。
* 每個 episode 結束後更新 updates_per_round × 回合數 次（replay 不到 warmup 組時不更新）。
* 不驗證：每 save_every 個 episode 存一次，評估用 final.pt。另記錄每張訓練圖的 lifetime 當學習曲線。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import DATA, HERE, RESULTS, build_ring, run_lifetime, train_maps
from env import RoundRecorder
from model import DuelingNet, make_tokens, param_count, round_inputs, span_mask

SEED_BASE = 620000
PAIRS = 64


class Replay:
    """FIFO 回合緩衝：回合層級（S、G…）與交易層級（64 組）都放在裝置上；PER 優先度 (cap, 64)。"""

    def __init__(self, cap, n, device, alpha, eps):
        z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=device)  # noqa: E731
        self.S, self.G = z((cap, 8, n, 10), torch.float16), z((cap, 8, n, 10), torch.float16)
        self.S_prio, self.G_prio = z((cap, 8, n), torch.int8), z((cap, 8, n), torch.int8)
        self.sensor = z((cap, n, 2), torch.float16)
        self.s, self.g = z((cap, PAIRS), torch.long), z((cap, PAIRS), torch.long)
        self.span = z((cap, PAIRS, 4, 2), torch.long)
        self.cand = z((cap, PAIRS, 4, 10), torch.float32)
        self.glob = z((cap, PAIRS, 10), torch.float32)
        self.action = z((cap, PAIRS), torch.long)
        self.reward = z((cap, PAIRS), torch.float32)
        self.prio = z((cap, PAIRS), torch.float32)
        self.cap, self.n, self.device = cap, n, device
        self.alpha, self.eps = float(alpha), float(eps)
        self.ptr = self.rounds = 0
        self.max_prio = 1.0

    @property
    def size(self):
        return self.rounds * PAIRS

    def add_episode(self, data):
        R = len(data["done"])
        slots = torch.as_tensor((self.ptr + np.arange(R)) % self.cap, device=self.device)
        t = lambda k, dt: torch.as_tensor(np.asarray(data[k]), dtype=dt, device=self.device)  # noqa: E731
        for k, dt in (("S", torch.float16), ("G", torch.float16), ("S_prio", torch.int8), ("G_prio", torch.int8),
                      ("sensor", torch.float16)):
            getattr(self, k)[slots] = t(k, dt)
        for name, key, dt in (("s", "s_id", torch.long), ("g", "g_id", torch.long), ("span", "span", torch.long),
                              ("cand", "cand", torch.float32), ("glob", "glob", torch.float32),
                              ("action", "action", torch.long), ("reward", "reward", torch.float32)):
            getattr(self, name)[slots] = t(key, dt)
        self.prio[slots] = self.max_prio
        self.ptr = (self.ptr + R) % self.cap
        self.rounds = min(self.rounds + R, self.cap)

    def sample(self, batch, beta):
        p = self.prio[:self.rounds].reshape(-1) ** self.alpha
        idx = torch.multinomial(p, batch, replacement=True)
        weights = (len(p) * p[idx] / p.sum()) ** (-beta)
        return idx // PAIRS, idx % PAIRS, weights / weights.max()

    def inputs(self, r, p):
        s, g = self.s[r, p], self.g[r, p]
        tok = make_tokens(self.S[r, s].float(), self.G[r, g].float(), self.S_prio[r, s], self.G_prio[r, g],
                          self.sensor[r].float())
        return tok, span_mask(self.span[r, p], self.n), self.cand[r, p], self.glob[r, p]


class Agent:
    def __init__(self, cfg, net_cfg, stats, device, seed):
        self.cfg, self.device = cfg, device
        self.rng = np.random.default_rng(seed)
        self.online = DuelingNet(stats, **net_cfg).to(device)
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg["lr"], weight_decay=cfg["l2"])
        self.replay = Replay(int(cfg["replay_rounds"]), 100, device, cfg["per_alpha"], cfg["per_eps"])
        self.updates = 0

    def act(self, data, epsilon):
        self.online.eval()
        with torch.no_grad():
            actions = self.online(*round_inputs(data, self.device)).argmax(dim=1).cpu().numpy()
        explore = self.rng.random(len(actions)) < epsilon
        actions[explore] = self.rng.integers(0, 4, explore.sum())
        return actions

    def update(self, beta):
        cfg, rp = self.cfg, self.replay
        r, p, w = rp.sample(int(cfg["batch"]), beta)
        self.online.train()
        q = self.online(*rp.inputs(r, p)).gather(1, rp.action[r, p][:, None]).squeeze(1)
        y = rp.reward[r, p]
        loss = (w * F.smooth_l1_loss(q, y, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg["grad_clip"])
        self.opt.step()
        new = (q - y).detach().abs() + rp.eps
        rp.prio[r, p] = new
        rp.max_prio = max(rp.max_prio, float(new.max().item()))
        self.updates += 1
        return float(loss.item())

    def save(self, path, extra=None):
        torch.save({"kind": "d3qn", "model": self.online.state_dict(), "net_config": self.online.config,
                    "cfg": self.cfg, "updates": self.updates, "extra": extra or {}}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--maps", type=int, default=None, help="只用前幾張（冒煙測試）")
    parser.add_argument("--budget", type=int, default=None, help="冒煙測試才改")
    parser.add_argument("--max-segments", type=int, default=None, help="冒煙測試才改")
    parser.add_argument("--warmup", type=int, default=None, help="冒煙測試才改")
    args = parser.parse_args()
    raw = json.loads((HERE / "config.json").read_text())
    cfg = dict(raw["rl"])
    if args.budget is not None:
        cfg["budget"] = args.budget
    if args.warmup is not None:
        cfg["warmup"] = args.warmup
    cfg["seed"] = args.seed
    out = Path(args.out) if args.out else RESULTS / f"d3qn_b{cfg['budget']}" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"rl": cfg, "net": raw["net"]}, indent=2))
    log = open(out / "log.jsonl", "a")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    torch.set_num_threads(int(raw["torch_threads"]))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    stats = dict(np.load(DATA / cfg["norm"]))
    agent = Agent(cfg, raw["net"], stats, device, args.seed)
    maps = train_maps()[:args.maps]
    write({"event": "start", "params": param_count(agent.online), "net": agent.online.config, "maps": len(maps),
           "device": str(device)})

    expected = int(cfg["expected_episodes"])
    decay = max(1, int(cfg["eps_decay_frac"] * expected))
    eps_start, eps_end = cfg["eps"]
    state = {"episode": 0, "epsilon": eps_start, "t0": time.time()}
    run_kwargs = {"budget": cfg["budget"]}
    if args.max_segments is not None:
        run_kwargs["max_segments"] = args.max_segments
    order_rng = np.random.default_rng([args.seed, 1])

    for pass_id in range(int(cfg["passes"])):
        for map_index in order_rng.permutation(len(maps)):
            spec = maps[map_index]
            t_map = time.time()

            def on_segment(algorithm, segment, lifetime):
                rec = algorithm.recorder
                algorithm.recorder = RoundRecorder(spec["name"], peek=False)
                if not rec.rounds:
                    return
                state["episode"] += 1
                episode = state["episode"]
                agent.replay.add_episode(rec.stacked())
                t1 = time.time()
                losses = []
                if agent.replay.size >= cfg["warmup"]:
                    beta = cfg["per_beta0"] + (1.0 - cfg["per_beta0"]) * min(1.0, episode / expected)
                    for _ in range(int(cfg["updates_per_round"]) * len(rec.rounds)):
                        losses.append(agent.update(beta))
                stats_ep = algorithm.round_stats
                write({"event": "episode", "episode": episode, "pass": pass_id, "map": spec["name"],
                       "segment": segment, "lifetime_before": lifetime, "rounds": len(rec.rounds),
                       "fitness": float(algorithm.fitness),
                       "mean_reward": float(np.mean([x["mean_reward"] for x in stats_ep])),
                       "improve_rate": float(np.mean([x["improve_rate"] for x in stats_ep])),
                       "segment_counts": np.sum([x["segment_counts"] for x in stats_ep], axis=0).tolist(),
                       "distinct": float(np.mean([x["distinct_per_searcher"] for x in stats_ep])),
                       "epsilon": round(state["epsilon"], 4), "loss": float(np.mean(losses)) if losses else None,
                       "updates": agent.updates, "replay_pairs": agent.replay.size,
                       "update_s": round(time.time() - t1, 1)})
                if episode % int(cfg["save_every"]) == 0:
                    agent.save(out / f"ep{episode}.pt", {"episode": episode})
                state["epsilon"] = eps_start + (eps_end - eps_start) * min(1.0, episode / decay)

            def make(problem):
                return build_ring(problem, lambda d: agent.act(d, state["epsilon"]),
                                  SEED_BASE + 1000 * pass_id + int(map_index),
                                  recorder=RoundRecorder(spec["name"], peek=False))

            lifetime, segments = run_lifetime(spec, make, on_segment, **run_kwargs)
            write({"event": "map", "pass": pass_id, "map": spec["name"], "lifetime": lifetime, "segments": segments,
                   "episodes": state["episode"], "epsilon": round(state["epsilon"], 4),
                   "minutes": round((time.time() - t_map) / 60, 1),
                   "total_minutes": round((time.time() - state["t0"]) / 60, 1)})

    agent.save(out / "final.pt", {"episode": state["episode"]})
    write({"event": "done", "episodes": state["episode"], "updates": agent.updates})
    (out / "TRAIN_DONE").touch()


if __name__ == "__main__":
    main()
