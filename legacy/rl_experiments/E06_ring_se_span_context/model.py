"""E06 網路（監督式與 RL 共用）：全網 sensor 前後文編碼 → 候選遮罩池化 → 候選間 attention → Dueling 4 個 Q。

sensor token (B, N, 25) ─ z-score ─ φ 25→64→64
  → sensor_layers 層 Transformer（d64、4 頭、FF 128、pre-LN），注意力偏置：
      環上距離 |i−j| 分桶的可學偏置 ＋「j 是 i 在 S 的父／子」「在 G 的父／子」各一個可學偏置（每層、每頭各自）
  → h_i (B, N, 64)
候選 c：遮罩池化 平均 ‖ 最大 ‖ 總和÷50 (192) ‖ 候選特徵 10 → 202→128→64 = u_c
全域 10 → 64→64 = g；[g, u_1..u_4] → cand_layers 層 Transformer（不加候選編號）
V = MLP(g′) 64→64→1；A_c = MLP(u_c′) 64→64→1；Q_c = V + A_c − mean(A)。
sensor_layers = 0 是消融（只看範圍內，≈ E03 DeepSets ＋ 候選 attention）。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from env import CAND_DIM, GLOBAL_DIM, H, TOKEN_DIM

POOL_SCALE = 50.0
# |i−j| 分桶的上界（含）：0,1,2,3,4,5–7,8–11,12–17,18–25,26–49,50+
RING_EDGES = (0, 1, 2, 3, 4, 7, 11, 17, 25, 49)


def ring_buckets(n):
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return torch.as_tensor(np.searchsorted(np.asarray(RING_EDGES), dist), dtype=torch.long)


class BiasedLayer(nn.Module):
    """pre-LN Transformer 層，注意力分數加上外部偏置（B, heads, L, L）。"""

    def __init__(self, d, heads, ff):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.ReLU(), nn.Linear(ff, d))

    def forward(self, x, bias=None):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = x + self.out(att.transpose(1, 2).reshape(B, L, d))
        return x + self.ff(self.ln2(x))


class E06Net(nn.Module):
    def __init__(self, stats=None, n_sensor=100, d=64, heads=4, ff=128, sensor_layers=2, cand_layers=1):
        super().__init__()
        self.config = {"n_sensor": n_sensor, "d": d, "heads": heads, "ff": ff,
                       "sensor_layers": sensor_layers, "cand_layers": cand_layers}
        for name, dim in (("tok", TOKEN_DIM), ("cand", CAND_DIM), ("glob", GLOBAL_DIM)):
            mean = np.zeros(dim, dtype=np.float32) if stats is None else np.asarray(stats[f"{name}_mean"], np.float32)
            std = np.ones(dim, dtype=np.float32) if stats is None else np.asarray(stats[f"{name}_std"], np.float32)
            self.register_buffer(f"{name}_mean", torch.as_tensor(mean))
            self.register_buffer(f"{name}_std", torch.as_tensor(np.where(std > 1e-6, std, 1.0)))
        self.register_buffer("buckets", ring_buckets(n_sensor), persistent=False)
        self.phi = nn.Sequential(nn.Linear(TOKEN_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.sensor_layers = nn.ModuleList(BiasedLayer(d, heads, ff) for _ in range(sensor_layers))
        self.ring_bias = nn.Parameter(torch.zeros(sensor_layers, heads, len(RING_EDGES) + 1))
        self.tree_bias = nn.Parameter(torch.zeros(sensor_layers, 4, heads))
        self.sensor_norm = nn.LayerNorm(d) if sensor_layers else nn.Identity()
        self.cand_mlp = nn.Sequential(nn.Linear(3 * d + CAND_DIM, 128), nn.ReLU(), nn.Linear(128, d))
        self.glob_mlp = nn.Sequential(nn.Linear(GLOBAL_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.cand_layers = nn.ModuleList(BiasedLayer(d, heads, ff) for _ in range(cand_layers))
        self.cand_norm = nn.LayerNorm(d)
        self.value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))
        self.advantage = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def _tree(self, par_s, par_g):
        """(B, 4, N, N)：j 是 i 在 S 的父、子，在 G 的父、子。"""
        n = par_s.shape[1]
        ids = torch.arange(n, device=par_s.device)
        ps = (par_s.long().unsqueeze(2) == ids).float()
        pg = (par_g.long().unsqueeze(2) == ids).float()
        return torch.stack([ps, ps.transpose(1, 2), pg, pg.transpose(1, 2)], dim=1)

    def forward(self, tok, par_s, par_g, mask, cand, glob):
        """tok (B, N, 25)、par_s/par_g (B, N)、mask (B, 4, N) bool、cand (B, 4, 10)、glob (B, 10) → Q (B, 4)。"""
        h = self.phi((tok - self.tok_mean) / self.tok_std)
        if len(self.sensor_layers):
            tree = self._tree(par_s, par_g)
            for layer, ring, tb in zip(self.sensor_layers, self.ring_bias, self.tree_bias):
                bias = ring[:, self.buckets].unsqueeze(0) + torch.einsum("btij,th->bhij", tree, tb)
                h = layer(h, bias)
            h = self.sensor_norm(h)
        m = mask.float().unsqueeze(-1)                                    # (B, 4, N, 1)
        total = torch.einsum("bcn,bnd->bcd", mask.float(), h)
        count = m.sum(2).clamp(min=1.0)
        peak = h.unsqueeze(1).masked_fill(~mask.unsqueeze(-1), -1e9).amax(2)
        cand_n = (cand - self.cand_mean) / self.cand_std
        u = self.cand_mlp(torch.cat([total / count, peak, total / POOL_SCALE, cand_n], dim=-1))
        g = self.glob_mlp((glob - self.glob_mean) / self.glob_std).unsqueeze(1)
        x = torch.cat([g, u], dim=1)
        for layer in self.cand_layers:
            x = layer(x)
        x = self.cand_norm(x)
        adv = self.advantage(x[:, 1:]).squeeze(-1)
        return self.value(x[:, 0]) + adv - adv.mean(dim=1, keepdim=True)


def make_tokens(S, G, S_prio, G_prio, sensor):
    """S、G (B, N, 10)、S_prio、G_prio (B, N)、sensor (B, N, 2) → token (B, N, 25)（torch）。"""
    on_s, on_g = S[..., 0] > 0, G[..., 0] > 0
    diff = torch.stack([(on_s != on_g), on_s & on_g & (S[..., 1] != G[..., 1]), S_prio != G_prio], dim=-1)
    return torch.cat([S, G, sensor, diff.float()], dim=-1)


def span_mask(span, n):
    """span (B, 4, 2) [left, right) → (B, 4, N) bool。"""
    ids = torch.arange(n, device=span.device)
    return (ids >= span[..., :1].long()) & (ids < span[..., 1:].long())


class RoundStore:
    """回合層級資料放在裝置上；用 (回合, searcher, good) 索引組出網路輸入。"""

    def __init__(self, rounds, device):
        t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=device)  # noqa: E731
        self.S, self.G = t(rounds["S"], torch.float32), t(rounds["G"], torch.float32)
        self.S_prio, self.G_prio = t(rounds["S_prio"], torch.int16), t(rounds["G_prio"], torch.int16)
        self.S_par, self.G_par = t(rounds["S_par"], torch.int16), t(rounds["G_par"], torch.int16)
        self.sensor = t(rounds["sensor"], torch.float32)
        self.n = self.S.shape[2]

    def inputs(self, r, s, g, span, cand, glob):
        """r、s、g (B,) long；span (B, 4, 2)、cand (B, 4, 10)、glob (B, 10) 張量 → forward 的參數。"""
        tok = make_tokens(self.S[r, s].float(), self.G[r, g].float(), self.S_prio[r, s], self.G_prio[r, g],
                          self.sensor[r].float())
        return tok, self.S_par[r, s], self.G_par[r, g], span_mask(span, self.n), cand, glob


def round_inputs(data, device):
    """單一回合的 numpy 回合資料（env._round_data）→ forward 的參數（64 組交易）。"""
    rounds = {k: np.asarray(data[k])[None] for k in ("S", "G", "S_prio", "G_prio", "S_par", "G_par", "sensor")}
    store = RoundStore(rounds, device)
    P = len(data["s_id"])
    t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=device)  # noqa: E731
    return store.inputs(torch.zeros(P, dtype=torch.long, device=device), t(data["s_id"], torch.long),
                        t(data["g_id"], torch.long), t(data["span"], torch.long),
                        t(data["cand"], torch.float32), t(data["glob"], torch.float32))


def load_net(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = E06Net(**ckpt["net_config"]).to(device)
    net.load_state_dict(ckpt["online"] if "online" in ckpt else ckpt["model"])
    net.eval()
    return net


def greedy_policy(net, device="cpu"):
    def policy(data):
        with torch.no_grad():
            return net(*round_inputs(data, device)).argmax(dim=1).cpu().numpy()
    return policy


def param_count(net):
    return sum(p.numel() for p in net.parameters())


__all__ = ["E06Net", "RoundStore", "round_inputs", "make_tokens", "span_mask", "load_net", "greedy_policy",
           "param_count", "H"]
