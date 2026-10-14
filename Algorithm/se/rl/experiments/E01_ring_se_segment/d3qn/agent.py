"""E01 / D3QN：經典 Dueling 網路、優先回放（示範永久保留）、DQfD 損失。

損失（DQfD，Hester et al. 2018）：
    J = J_DQ(1-step Double DQN) + λ1·J_n(n-step Double DQN) + λ2·J_E(大邊界模仿，只算示範) + λ3·L2
三種模式：
    "dqfd"：上面全部（主方法）；"d3qn"：λ2 = 0、不用示範；"imitation"：只有 J_E（只抽示範）。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

KEYS = ("s", "a", "r", "s1", "d1", "rn", "sn", "dn", "gn")


class DuelingNet(nn.Module):
    """經典 Dueling（Wang et al., 2016）：60 → 128 → 128 → V(128→64→1)、A(128→64→4)。"""

    def __init__(self, state_dim=60, hidden=128, branch=64, n_actions=4):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(state_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(hidden, branch), nn.ReLU(), nn.Linear(branch, 1))
        self.advantage = nn.Sequential(nn.Linear(hidden, branch), nn.ReLU(), nn.Linear(branch, n_actions))

    def forward(self, x):
        h = self.shared(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


class PrioritizedReplay:
    """比例式 PER。示範資料放在前段、永久保留；agent 資料放在後段的環狀緩衝區。"""

    def __init__(self, capacity, state_dim, alpha, eps_agent, eps_demo, rng):
        self.capacity = int(capacity)
        self.alpha, self.eps_agent, self.eps_demo = float(alpha), float(eps_agent), float(eps_demo)
        self.rng = rng
        self.demo = None          # dict of arrays
        self.demo_prio = np.zeros(0)
        self.agent = {k: None for k in KEYS}
        self.agent_prio = np.zeros(self.capacity)
        self.agent_size = 0
        self.agent_pos = 0
        self.state_dim = state_dim

    def set_demo(self, data):
        self.demo = {k: np.asarray(data[k]) for k in KEYS}
        self.demo_prio = np.full(len(self.demo["s"]), 1.0 + self.eps_demo)

    @property
    def demo_size(self):
        return 0 if self.demo is None else len(self.demo["s"])

    def add(self, data):
        n = len(data["s"])
        if self.agent["s"] is None:
            for k in KEYS:
                shape = (self.capacity,) + np.asarray(data[k]).shape[1:]
                self.agent[k] = np.zeros(shape, dtype=np.asarray(data[k]).dtype)
        start_prio = max(1.0, float(self.agent_prio[:self.agent_size].max()) if self.agent_size else 1.0)
        for i in range(n):
            for k in KEYS:
                self.agent[k][self.agent_pos] = data[k][i]
            self.agent_prio[self.agent_pos] = start_prio
            self.agent_pos = (self.agent_pos + 1) % self.capacity
            self.agent_size = min(self.agent_size + 1, self.capacity)

    def sample(self, batch, beta, demo_only=False):
        prio = self.demo_prio if demo_only else np.concatenate([self.demo_prio, self.agent_prio[:self.agent_size]])
        p = prio ** self.alpha
        p /= p.sum()
        idx = self.rng.choice(len(p), size=batch, p=p)
        weights = (len(p) * p[idx]) ** (-beta)
        weights /= weights.max()
        out = {k: [] for k in KEYS}
        is_demo = idx < self.demo_size
        for i, j in enumerate(idx):
            src, pos = (self.demo, j) if is_demo[i] else (self.agent, j - self.demo_size)
            for k in KEYS:
                out[k].append(src[k][pos])
        batch_data = {k: np.asarray(v) for k, v in out.items()}
        return batch_data, idx, weights.astype(np.float32), is_demo

    def update(self, idx, td, is_demo):
        new = np.abs(td) + np.where(is_demo, self.eps_demo, self.eps_agent)
        for j, value, demo in zip(idx, new, is_demo):
            if demo:
                self.demo_prio[j] = value
            else:
                self.agent_prio[j - self.demo_size] = value


class DQfDAgent:
    def __init__(self, cfg, rng):
        torch.manual_seed(int(cfg["seed"]))
        self.cfg = cfg
        self.mode = cfg["mode"]
        self.online = DuelingNet(cfg["state_dim"], cfg["hidden"], cfg["branch"], cfg["n_actions"])
        self.target = DuelingNet(cfg["state_dim"], cfg["hidden"], cfg["branch"], cfg["n_actions"])
        self.target.load_state_dict(self.online.state_dict())
        self.opt = torch.optim.Adam(self.online.parameters(), lr=cfg["lr"], weight_decay=cfg["lambda_l2"])
        self.replay = PrioritizedReplay(cfg["replay_capacity"], cfg["state_dim"], cfg["per_alpha"],
                                        cfg["per_eps_agent"], cfg["per_eps_demo"], rng)
        self.rng = rng
        self.updates = 0

    # ---- 行動 ----
    def act(self, states, epsilon):
        with torch.no_grad():
            q = self.online(torch.as_tensor(states, dtype=torch.float32)).numpy()
        actions = q.argmax(axis=1)
        explore = self.rng.random(len(actions)) < epsilon
        actions[explore] = self.rng.integers(0, q.shape[1], explore.sum())
        return actions

    # ---- 學習 ----
    def _double_target(self, r, s_next, done, discount):
        with torch.no_grad():
            a_star = self.online(s_next).argmax(dim=1, keepdim=True)
            q_next = self.target(s_next).gather(1, a_star).squeeze(1)
        return r + discount * (1.0 - done) * q_next

    def update(self, beta, demo_only=False):
        cfg = self.cfg
        data, idx, weights, is_demo = self.replay.sample(cfg["batch"], beta, demo_only=demo_only)
        t = {k: torch.as_tensor(v) for k, v in data.items()}
        w = torch.as_tensor(weights)
        demo_mask = torch.as_tensor(is_demo, dtype=torch.float32)
        q = self.online(t["s"])
        q_sa = q.gather(1, t["a"].view(-1, 1)).squeeze(1)

        loss = torch.zeros(())
        td1 = torch.zeros_like(q_sa)
        if self.mode != "imitation":
            y1 = self._double_target(t["r"], t["s1"], t["d1"], cfg["gamma"])
            yn = self._double_target(t["rn"], t["sn"], t["dn"], t["gn"])
            td1 = q_sa - y1
            loss = loss + (w * F.smooth_l1_loss(q_sa, y1, reduction="none")).mean()
            loss = loss + cfg["lambda_n"] * (w * F.smooth_l1_loss(q_sa, yn, reduction="none")).mean()
        if self.mode in ("dqfd", "imitation") and demo_mask.sum() > 0:
            margins = cfg["margin"] * (1.0 - F.one_hot(t["a"], q.shape[1]).float())
            j_e = (q + margins).max(dim=1).values - q_sa
            loss = loss + cfg["lambda_e"] * (demo_mask * j_e).sum() / demo_mask.sum()
            if self.mode == "imitation":
                td1 = j_e.detach()

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.opt.step()
        self.replay.update(idx, td1.detach().numpy(), is_demo)
        self.updates += 1
        if self.updates % cfg["target_every"] == 0:
            self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path, extra=None):
        torch.save({"online": self.online.state_dict(), "cfg": self.cfg, "updates": self.updates,
                    "extra": extra or {}}, path)


def load_policy(path):
    """讀 checkpoint，回傳 greedy 策略 callable(states) -> actions。"""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    net = DuelingNet(cfg["state_dim"], cfg["hidden"], cfg["branch"], cfg["n_actions"])
    net.load_state_dict(ckpt["online"])
    net.eval()

    def policy(states):
        with torch.no_grad():
            return net(torch.as_tensor(states, dtype=torch.float32)).argmax(dim=1).numpy()
    return policy


__all__ = ["DQfDAgent", "DuelingNet", "PrioritizedReplay", "load_policy"]
