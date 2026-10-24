"""E06 強化學習（D3QN；γ = 0 時等於 contextual bandit）。網路同監督式。

用法（在 2027WSN 根目錄）：
    python .../train_rl.py --gamma 0 --seed 0                                   # 從零
    python .../train_rl.py --gamma 0 --seed 0 --init results/supervised/seed0/best.pt   # 監督式暖啟動
    python .../train_rl.py --gamma 0.5 --seed 0 --init results/supervised/seed0/best.pt # γ 消融
輸出：results/d3qn/g{γ}_{scratch|warm}_seed{seed}/（log.jsonl、config.json、best.pt、last.pt）。

D3QN：Double DQN（γ > 0 時 1-step ＋ n-step，Huber）＋ Dueling ＋ target network ＋ 比例式 PER；γ = 0 時目標就是 r。
轉移：同一個 searcher、同一個 good 位置的下一回合；這段最佳化的最後一回合為終止。
每個 episode＝一段最佳化（預算 1000），從 190 張訓練圖的快照均勻抽一份；每回合更新 updates_per_round 次。
驗證：每 val_every 個 episode，20 份驗證快照，同 seed 和隨機比最終 fitness；best.pt 依對隨機的平均差選。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import DATA, HERE, RESULTS, SnapshotPool, run_episode, split_maps, validate, validation_baseline, \
    validation_items
from env import H, RoundRecorder
from model import E06Net, RoundStore, param_count, round_inputs

PAIRS = 64


class Replay(RoundStore):
    """回合層級放在裝置上（float16）；交易層級放 numpy，PER 優先度、1-step／n-step 轉移用索引。"""

    def __init__(self, cap_rounds, n, device, alpha, eps, rng):
        z = lambda shape, dt: torch.zeros(shape, dtype=dt, device=device)  # noqa: E731
        self.S, self.G = z((cap_rounds, 8, n, 10), torch.float16), z((cap_rounds, 8, n, 10), torch.float16)
        self.S_prio, self.G_prio = z((cap_rounds, 8, n), torch.int16), z((cap_rounds, 8, n), torch.int16)
        self.S_par, self.G_par = z((cap_rounds, 8, n), torch.int16), z((cap_rounds, 8, n), torch.int16)
        self.sensor = z((cap_rounds, n, 2), torch.float16)
        self.n, self.device, self.rng = n, device, rng
        self.alpha, self.eps = float(alpha), float(eps)
        cap = cap_rounds * PAIRS
        self.r, self.s, self.g = (np.zeros(cap, dtype=np.int64) for _ in range(3))
        self.span = np.zeros((cap, H, 2), dtype=np.int64)
        self.cand = np.zeros((cap, H, 10), dtype=np.float32)
        self.glob = np.zeros((cap, 10), dtype=np.float32)
        self.action = np.zeros(cap, dtype=np.int64)
        self.reward, self.reward_n = np.zeros(cap, dtype=np.float32), np.zeros(cap, dtype=np.float32)
        self.next1, self.next_n = np.zeros(cap, dtype=np.int64), np.zeros(cap, dtype=np.int64)
        self.done1, self.done_n = np.zeros(cap, dtype=np.float32), np.zeros(cap, dtype=np.float32)
        self.gamma_n = np.zeros(cap, dtype=np.float32)
        self.prio = np.zeros(cap)
        self.rounds = self.size = 0

    def add_episode(self, data, gamma, n_step):
        R = len(data["done"])
        r0, p0 = self.rounds, self.size
        if r0 + R > self.S.shape[0]:
            raise RuntimeError("replay full")
        t = lambda k, dt: torch.as_tensor(np.asarray(data[k]), dtype=dt, device=self.device)  # noqa: E731
        for k, dt in (("S", torch.float16), ("G", torch.float16), ("S_prio", torch.int16), ("G_prio", torch.int16),
                      ("S_par", torch.int16), ("G_par", torch.int16), ("sensor", torch.float16)):
            getattr(self, k)[r0:r0 + R] = t(k, dt)
        idx = p0 + np.arange(R * PAIRS)
        rnd = np.repeat(np.arange(R), PAIRS)
        self.r[idx] = r0 + rnd
        self.s[idx] = data["s_id"].reshape(-1)
        self.g[idx] = data["g_id"].reshape(-1)
        self.span[idx] = data["span"].reshape(-1, H, 2)
        self.cand[idx] = data["cand"].reshape(-1, H, 10)
        self.glob[idx] = data["glob"].reshape(-1, 10)
        self.action[idx] = data["action"].reshape(-1)
        rew = data["reward"].astype(np.float32)               # (R, 64)
        self.reward[idx] = rew.reshape(-1)
        last = rnd + 1 >= R
        self.done1[idx] = last
        self.next1[idx] = np.where(last, idx, idx + PAIRS)
        m = np.minimum(n_step, R - rnd)
        disc = np.array([[gamma ** j for j in range(n_step)]], dtype=np.float64)
        padded = np.concatenate([rew, np.zeros((n_step, PAIRS), dtype=np.float32)])
        windows = np.stack([padded[j:j + R] for j in range(n_step)], axis=-1)   # (R, 64, n)
        mask = np.arange(n_step)[None, None, :] < m.reshape(R, PAIRS)[..., None]
        self.reward_n[idx] = (windows * disc * mask).sum(-1).reshape(-1)
        self.done_n[idx] = rnd + m >= R
        self.next_n[idx] = np.where(rnd + m >= R, idx, idx + PAIRS * m)
        self.gamma_n[idx] = gamma ** m
        self.prio[idx] = max(1.0, float(self.prio[:p0].max()) if p0 else 1.0)
        self.rounds += R
        self.size += R * PAIRS

    def sample(self, batch, beta):
        p = self.prio[:self.size] ** self.alpha
        p /= p.sum()
        idx = self.rng.choice(self.size, size=batch, p=p)
        weights = (self.size * p[idx]) ** (-beta)
        return idx, (weights / weights.max()).astype(np.float32)

    def batch_inputs(self, idx):
        t = lambda x, dt: torch.as_tensor(x[idx], dtype=dt, device=self.device)  # noqa: E731
        return self.inputs(t(self.r, torch.long), t(self.s, torch.long), t(self.g, torch.long),
                           t(self.span, torch.long), t(self.cand, torch.float32), t(self.glob, torch.float32))


class Agent:
    def __init__(self, cfg, net_cfg, stats, init, device, rng):
        self.cfg, self.device, self.rng = cfg, device, rng
        self.online = E06Net(stats, **net_cfg).to(device)
        if init:
            self.online.load_state_dict(torch.load(init, map_location=device, weights_only=False)["model"])
        self.target = E06Net(stats, **net_cfg).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg["lr"], weight_decay=cfg["l2"])
        self.replay = Replay(int(cfg["episodes"]) * 10, net_cfg["n_sensor"], device, cfg["per_alpha"],
                             cfg["per_eps"], rng)
        self.updates = 0

    def act(self, data, epsilon):
        self.online.eval()
        with torch.no_grad():
            actions = self.online(*round_inputs(data, self.device)).argmax(dim=1).cpu().numpy()
        explore = self.rng.random(len(actions)) < epsilon
        actions[explore] = self.rng.integers(0, H, explore.sum())
        return actions

    def update(self, beta):
        cfg, rp = self.cfg, self.replay
        idx, weights = rp.sample(cfg["batch"], beta)
        dev = self.device
        w = torch.as_tensor(weights, device=dev)
        self.online.train()
        q_sa = self.online(*rp.batch_inputs(idx)).gather(1, torch.as_tensor(rp.action[idx], device=dev)[:, None])
        q_sa = q_sa.squeeze(1)
        r1 = torch.as_tensor(rp.reward[idx], device=dev)
        if cfg["gamma"] == 0:
            y1 = r1
            loss = (w * F.smooth_l1_loss(q_sa, y1, reduction="none")).mean()
        else:
            nxt = np.concatenate([rp.next1[idx], rp.next_n[idx]])
            with torch.no_grad():
                inputs = rp.batch_inputs(nxt)
                a_star = self.online(*inputs).argmax(dim=1, keepdim=True)
                q_next = self.target(*inputs).gather(1, a_star).squeeze(1)
            B = len(idx)
            t = lambda x: torch.as_tensor(x[idx], device=dev)  # noqa: E731
            y1 = r1 + cfg["gamma"] * (1.0 - t(rp.done1)) * q_next[:B]
            yn = t(rp.reward_n) + t(rp.gamma_n) * (1.0 - t(rp.done_n)) * q_next[B:]
            loss = (w * F.smooth_l1_loss(q_sa, y1, reduction="none")).mean() \
                + cfg["lambda_n"] * (w * F.smooth_l1_loss(q_sa, yn, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg["grad_clip"])
        self.opt.step()
        rp.prio[idx] = np.abs((q_sa - y1).detach().cpu().numpy()) + rp.eps
        self.updates += 1
        if self.updates % int(cfg["target_every"]) == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path, extra=None):
        torch.save({"model": self.online.state_dict(), "net_config": self.online.config, "cfg": self.cfg,
                    "updates": self.updates, "extra": extra or {}}, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init", default="")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    raw = json.loads((HERE / "config.json").read_text())
    cfg = dict(raw["rl"])
    cfg.update({"gamma": args.gamma, "seed": args.seed, "init": args.init, "budget": raw["budget"]})
    if args.episodes is not None:
        cfg["episodes"] = args.episodes
    if args.gamma == 0:
        cfg["n_step"], cfg["lambda_n"] = 1, 0.0
    eps_start, eps_end = cfg["eps_warm"] if args.init else cfg["eps_scratch"]
    torch.set_num_threads(int(raw["torch_threads"]))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gamma_tag = "g0" if args.gamma == 0 else f"g{str(args.gamma).replace('.', '')}"
    out = RESULTS / "d3qn" / f"{gamma_tag}_{'warm' if args.init else 'scratch'}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log = open(out / "log.jsonl", "a")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    stats = dict(np.load(DATA / "norm.npz"))
    net_cfg = dict(raw["net"])
    if args.init:
        net_cfg = torch.load(args.init, map_location="cpu", weights_only=False)["net_config"]
    net_cfg.setdefault("n_sensor", 100)
    rng = np.random.default_rng(args.seed)
    agent = Agent(cfg, net_cfg, stats, args.init, device, rng)
    write({"event": "start", "params": param_count(agent.online), "net": agent.online.config, "device": str(device)})
    pool = SnapshotPool(split_maps()[0])
    items = validation_items(tuple(raw["val_snapshot_index"]))
    base = validation_baseline(items, cfg["budget"])
    random_fit, rule_fit = np.array(base["random"]["fitness"]), np.array(base["bottleneck"]["fitness"])

    def run_validation(episode):
        fits, summary = validate(items, lambda d: agent.act(d, 0.0), cfg["budget"])
        diff = fits - random_fit
        return {"event": "val", "episode": episode, "mean_diff": float(diff.mean()), "wins": int((diff > 0).sum()),
                "losses": int((diff < 0).sum()), "vs_bottleneck": float((fits - rule_fit).mean()), **summary}

    if args.init:
        write(run_validation(0))
    best = {"mean_diff": -np.inf}
    episodes = int(cfg["episodes"])
    decay = max(1, int(cfg["eps_decay_frac"] * episodes))
    for episode in range(1, episodes + 1):
        t0 = time.time()
        epsilon = eps_start + (eps_end - eps_start) * min(1.0, (episode - 1) / decay)
        beta = cfg["per_beta0"] + (1.0 - cfg["per_beta0"]) * min(1.0, episode / episodes)
        snap, problem = pool.sample(rng)
        rec = RoundRecorder(snap["map"])
        algorithm = run_episode(problem, lambda d: agent.act(d, epsilon), 100000 * (args.seed + 1) + episode,
                                cfg["budget"], recorder=rec)
        agent.replay.add_episode(rec.stacked(), cfg["gamma"], int(cfg["n_step"]))
        t1 = time.time()
        losses = []
        if agent.replay.size >= cfg["warmup"]:
            for _ in range(int(cfg["updates_per_round"]) * len(rec.rounds)):
                losses.append(agent.update(beta))
        stats_ep = algorithm.round_stats
        write({"event": "episode", "episode": episode, "map": snap["map"], "fitness": float(algorithm.fitness),
               "mean_reward": float(np.mean([r["mean_reward"] for r in stats_ep])),
               "improve_rate": float(np.mean([r["improve_rate"] for r in stats_ep])),
               "segment_counts": np.sum([r["segment_counts"] for r in stats_ep], axis=0).tolist(),
               "distinct": float(np.mean([r["distinct_per_searcher"] for r in stats_ep])),
               "epsilon": round(epsilon, 4), "loss": float(np.mean(losses)) if losses else None,
               "replay": agent.replay.size, "env_s": round(t1 - t0, 1), "update_s": round(time.time() - t1, 1)})
        if episode % int(cfg["val_every"]) == 0 or episode == episodes:
            result = run_validation(episode)
            write(result)
            agent.save(out / "last.pt", result)
            if result["mean_diff"] > best["mean_diff"]:
                best = result
                agent.save(out / "best.pt", result)
    write({"event": "done", "best": best})
    (out / "TRAIN_DONE").touch()


if __name__ == "__main__":
    main()
