"""E04 D3QN 訓練（兩個網路共用；只換網路）。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E04_ring_se_seg8/train.py --net concat --seed 0
    python Algorithm/se/rl/experiments/E04_ring_se_seg8/train.py --net attention --seed 0
超參數在 config.json（net_overrides 裡是各網路自己的值，例如學習率）。
輸出：{dueling_concat|attention}/results/seed{seed}/（log.jsonl、config.json、best.pt、last.pt）

D3QN：Double DQN（1-step ＋ n-step，Huber）＋ Dueling ＋ target network ＋ 比例式 PER；不用示範資料。
每個 episode＝一段最佳化（預算 1000），從 190 張訓練圖的快照均勻抽一份。
驗證：10 張驗證圖 × 快照 val_snapshot_index，同 seed 和 8 段隨機選段比這段最佳化的最終 fitness；
      瓶頸規則同條件只算一次當參考。best.pt 依「對隨機的平均差」選。
"""
from __future__ import annotations

import argparse
import json
import pickle
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import DATA, HERE, NETS, SnapshotPool, net_class, run_episode, split_maps
from env import H, STATE_DIM


class StateReplay:
    """比例式 PER；每個 state 只存一份（float16），轉移用索引指到下一個 state。

    同一個 episode 的轉移連續存放，下一個 state 一定在後面；放不下時整段從 0 開始放，
    所以覆蓋從前面開始時，還留著的轉移指到的 state 都還在。
    """

    def __init__(self, capacity, alpha, eps, rng):
        self.capacity = int(capacity)
        self.alpha, self.eps = float(alpha), float(eps)
        self.rng = rng
        self.state = np.zeros((self.capacity, STATE_DIM), dtype=np.float16)
        self.action = np.zeros(self.capacity, dtype=np.int64)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.next1 = np.zeros(self.capacity, dtype=np.int64)
        self.done1 = np.zeros(self.capacity, dtype=np.float32)
        self.reward_n = np.zeros(self.capacity, dtype=np.float32)
        self.next_n = np.zeros(self.capacity, dtype=np.int64)
        self.done_n = np.zeros(self.capacity, dtype=np.float32)
        self.gamma_n = np.zeros(self.capacity, dtype=np.float32)
        self.prio = np.zeros(self.capacity)          # 0 ＝ 空位（不會被抽到）
        self.pos = 0
        self.size = 0

    def add_episode(self, trajectories, gamma, n_step):
        steps_all = list(trajectories.values())
        total = sum(len(s) for s in steps_all)
        if self.pos + total > self.capacity:
            self.prio[self.pos:] = 0.0
            self.pos = 0
        start_prio = max(1.0, float(self.prio.max()))
        base = self.pos
        for steps in steps_all:
            length = len(steps)
            for t, (state, action, reward) in enumerate(steps):
                i = base + t
                self.state[i] = state
                self.action[i] = action
                self.reward[i] = reward
                self.done1[i] = float(t + 1 >= length)
                self.next1[i] = base + t + 1 if t + 1 < length else i
                m = min(n_step, length - t)
                self.reward_n[i] = sum(gamma ** j * steps[t + j][2] for j in range(m))
                self.done_n[i] = float(t + m >= length)
                self.next_n[i] = base + t + m if t + m < length else i
                self.gamma_n[i] = gamma ** m
                self.prio[i] = start_prio
            base += length
        self.pos = base
        self.size = int((self.prio > 0).sum())

    def sample(self, batch, beta):
        p = self.prio ** self.alpha
        p /= p.sum()
        idx = self.rng.choice(self.capacity, size=batch, p=p)
        weights = (self.size * p[idx]) ** (-beta)
        return idx, (weights / weights.max()).astype(np.float32)

    def update(self, idx, td):
        self.prio[idx] = np.abs(td) + self.eps


class Agent:
    def __init__(self, cfg, stats, device, rng):
        torch.manual_seed(int(cfg["seed"]))
        cls = net_class(cfg["net"])
        self.cfg, self.device, self.rng = cfg, device, rng
        self.online = cls(stats).to(device)
        self.target = cls(stats).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg["lr"], weight_decay=cfg["l2"])
        self.replay = StateReplay(cfg["replay_capacity"], cfg["per_alpha"], cfg["per_eps"], rng)
        self.updates = 0

    def q_values(self, states):
        with torch.no_grad():
            return self.online(torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)).cpu().numpy()

    def act(self, states, epsilon):
        actions = self.q_values(states).argmax(axis=1)
        explore = self.rng.random(len(actions)) < epsilon
        actions[explore] = self.rng.integers(0, H, explore.sum())
        return actions

    def _states(self, idx):
        return torch.as_tensor(self.replay.state[idx], dtype=torch.float32, device=self.device)

    def _double_target(self, reward, next_idx, done, discount):
        s_next = self._states(next_idx)
        with torch.no_grad():
            a_star = self.online(s_next).argmax(dim=1, keepdim=True)
            q_next = self.target(s_next).gather(1, a_star).squeeze(1)
        return reward + discount * (1.0 - done) * q_next

    def update(self, beta):
        cfg, rp, dev = self.cfg, self.replay, self.device
        idx, weights = rp.sample(cfg["batch"], beta)
        t = lambda x: torch.as_tensor(x[idx], device=dev)  # noqa: E731
        w = torch.as_tensor(weights, device=dev)
        q = self.online(self._states(idx))
        q_sa = q.gather(1, t(rp.action).view(-1, 1)).squeeze(1)
        y1 = self._double_target(t(rp.reward), rp.next1[idx], t(rp.done1), cfg["gamma"])
        yn = self._double_target(t(rp.reward_n), rp.next_n[idx], t(rp.done_n), t(rp.gamma_n))
        loss = (w * F.smooth_l1_loss(q_sa, y1, reduction="none")).mean() \
            + cfg["lambda_n"] * (w * F.smooth_l1_loss(q_sa, yn, reduction="none")).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), cfg["grad_clip"])
        self.opt.step()
        rp.update(idx, (q_sa - y1).detach().cpu().numpy())
        self.updates += 1
        if self.updates % cfg["target_every"] == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path, extra=None):
        torch.save({"online": self.online.state_dict(), "cfg": self.cfg, "updates": self.updates,
                    "extra": extra or {}}, path)


def load_policy(path, device="cpu"):
    """讀 checkpoint，回傳 greedy 策略 callable(states) -> actions。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = net_class(ckpt["cfg"]["net"])().to(device)
    net.load_state_dict(ckpt["online"])
    net.eval()

    def policy(states):
        with torch.no_grad():
            return net(torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)).argmax(dim=1).cpu().numpy()
    return policy


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
    parser.add_argument("--net", choices=tuple(NETS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    raw = json.loads(open(args.config).read())
    cfg = {k: v for k, v in raw.items() if k != "net_overrides"}
    cfg.update(raw["net_overrides"][args.net])
    cfg.update({"net": args.net, "seed": args.seed})
    if args.episodes is not None:
        cfg["episodes"] = args.episodes
    torch.set_num_threads(int(cfg["torch_threads"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out = HERE / NETS[args.net][0] / "results" / f"seed{args.seed}{args.tag}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    log = open(out / "log.jsonl", "a")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    stats = dict(np.load(DATA / "norm.npz"))
    rng = np.random.default_rng(int(cfg["seed"]))
    agent = Agent(cfg, stats, device, rng)
    write({"event": "start", "net": args.net, "device": str(device),
           "params": sum(p.numel() for p in agent.online.parameters())})
    train_specs, _ = split_maps()
    pool = SnapshotPool(train_specs)

    val_items = validation_set(cfg)
    started = time.time()
    val_random = validate(val_items, "random", cfg["budget"])
    val_rule = validate(val_items, "bottleneck", cfg["budget"])
    write({"event": "val_baseline", "n": len(val_items), "random_mean_fitness": float(val_random.mean()),
           "bottleneck_mean_diff": float((val_rule - val_random).mean()),
           "bottleneck_wins": int((val_rule > val_random).sum()), "seconds": round(time.time() - started, 1)})

    def run_validation(episode):
        fit = validate(val_items, lambda s: agent.act(s, 0.0), cfg["budget"])
        diff = fit - val_random
        return {"event": "val", "episode": episode, "mean_diff": float(diff.mean()),
                "wins": int((diff > 0).sum()), "losses": int((diff < 0).sum()), "mean_fitness": float(fit.mean()),
                "vs_bottleneck": float((fit - val_rule).mean())}

    best = {"mean_diff": -np.inf}
    episodes = int(cfg["episodes"])
    decay = max(1, int(cfg["eps_decay_frac"] * episodes))
    for episode in range(1, episodes + 1):
        t0 = time.time()
        epsilon = cfg["eps_start"] + (cfg["eps_end"] - cfg["eps_start"]) * min(1.0, (episode - 1) / decay)
        beta = cfg["per_beta0"] + (1.0 - cfg["per_beta0"]) * min(1.0, episode / episodes)
        snap, problem = pool.sample(rng)
        algorithm = run_episode(problem, lambda s: agent.act(s, epsilon),
                                100000 * (cfg["seed"] + 1) + episode, cfg["budget"])
        agent.replay.add_episode(algorithm.pop_trajectories(), cfg["gamma"], cfg["n_step"])
        losses = []
        if agent.replay.size >= cfg["warmup"]:
            for _ in range(int(cfg["updates_per_round"]) * len(algorithm.round_stats)):
                losses.append(agent.update(beta))
        counts = np.sum([r["segment_counts"] for r in algorithm.round_stats], axis=0).tolist()
        write({"event": "episode", "episode": episode, "map": snap["map"], "segment": snap["segment"],
               "fitness": float(algorithm.fitness),
               "mean_reward": float(np.mean([r["mean_reward"] for r in algorithm.round_stats])),
               "improve_rate": float(np.mean([r["improve_rate"] for r in algorithm.round_stats])),
               "epsilon": round(epsilon, 4), "loss": float(np.mean(losses)) if losses else None,
               "segment_counts": counts, "replay": agent.replay.size, "seconds": round(time.time() - t0, 2)})
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
