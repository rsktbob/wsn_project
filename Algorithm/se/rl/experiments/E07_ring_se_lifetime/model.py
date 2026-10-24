"""E07 網路（簡單版；attn > 0 時範圍內 sensor 之間加 Transformer）。4 個候選各走一條路，4 條路是同一個網路、同一組權重。

候選 c：範圍內每顆 sensor token 25 → φ 25→64→64 → 平均 ‖ 最大（128）‖ 候選特徵 10 ‖ 全域 10
        → 148→128→64 ＝ u_c
SupNet（監督式）：分數 s_c ＝ u_c → 1；softmax 在損失裡算，推論選分數最大。
DuelingNet（D3QN）：A_c ＝ u_c → 1；V ＝ 平均(u_1…u_4) → 64 → 1；Q_c ＝ V + A_c − mean(A)。
token、候選特徵、全域先 z-score（統計量存在網路的 buffer）。
實作上把全網 100 顆 token 都過 φ 再用遮罩池化：φ 逐顆獨立，結果和只算範圍內的 sensor 一樣。
attn 層（預設 0 ＝ 上面的簡單版）：φ 之後、池化之前，每個候選的範圍內 sensor 互相做注意力
（4 頭、前饋 2d、pre-norm、無 dropout；範圍外用 key padding 遮掉，不參與）。
topk 顆（預設 0）：範圍內依緊迫度 max(S, G)（token 第 7、17 欄，標準化前）排序，最緊迫的 topk 顆的 φ 輸出
直接接起來（不足補 0），和平均 ‖ 最大一起進 cand_mlp：148 → 148 + topk·d。
full 維（預設 0）：全網 N 顆 φ 輸出各壓成 full 維（線性），範圍外乘 0，照 sensor 編號接起來，
和平均 ‖ 最大一起進 cand_mlp：148 → 148 + N·full（N 固定 100）。
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from env import CAND_DIM, GLOBAL_DIM, TIGHT, TOKEN_DIM

ROUND_KEYS = ("S", "G", "S_prio", "G_prio", "sensor")


class Encoder(nn.Module):
    def __init__(self, stats=None, d=64, hidden=128, attn=0, topk=0, full=0, n=100):
        super().__init__()
        self.topk, self.full = topk, full
        for name, dim in (("tok", TOKEN_DIM), ("cand", CAND_DIM), ("glob", GLOBAL_DIM)):
            mean = np.zeros(dim, dtype=np.float32) if stats is None else np.asarray(stats[f"{name}_mean"], np.float32)
            std = np.ones(dim, dtype=np.float32) if stats is None else np.asarray(stats[f"{name}_std"], np.float32)
            self.register_buffer(f"{name}_mean", torch.as_tensor(mean))
            self.register_buffer(f"{name}_std", torch.as_tensor(np.where(std > 1e-6, std, 1.0)))
        self.phi = nn.Sequential(nn.Linear(TOKEN_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.attn = nn.ModuleList(nn.TransformerEncoderLayer(d, 4, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
                                  for _ in range(attn))
        self.squeeze = nn.Linear(d, full) if full else None
        self.cand_mlp = nn.Sequential(nn.Linear((2 + topk) * d + n * full + CAND_DIM + GLOBAL_DIM, hidden), nn.ReLU(),
                                      nn.Linear(hidden, d), nn.ReLU())

    def forward(self, tok, mask, cand, glob):
        """tok (B, N, 25)、mask (B, 4, N) bool、cand (B, 4, 10)、glob (B, 10) → u (B, 4, d)。"""
        h = self.phi((tok - self.tok_mean) / self.tok_std)                       # (B, N, d)
        B, C, N = mask.shape
        if self.attn:
            x = h.unsqueeze(1).expand(-1, C, -1, -1).reshape(B * C, N, -1)       # 每個候選各一條序列
            pad = ~mask.reshape(B * C, N)
            for layer in self.attn:
                x = layer(x, src_key_padding_mask=pad)
            h = x.reshape(B, C, N, -1)
        else:
            h = h.unsqueeze(1).expand(-1, C, -1, -1)                             # (B, 4, N, d)
        m = mask.float()
        mean = torch.einsum("bcn,bcnd->bcd", m, h) / m.sum(-1, keepdim=True).clamp(min=1.0)
        peak = h.masked_fill(~mask.unsqueeze(-1), -1e9).amax(2)
        pooled = [mean, peak]
        if self.topk:
            tight = torch.maximum(tok[..., TIGHT], tok[..., 10 + TIGHT])                  # (B, N)
            key = tight.unsqueeze(1).masked_fill(~mask, -1.0)                             # 範圍外排最後
            order = key.topk(self.topk, dim=-1).indices                                   # (B, 4, k)
            picked = h.gather(2, order.unsqueeze(-1).expand(-1, -1, -1, h.shape[-1]))     # (B, 4, k, d)
            picked = picked * mask.gather(2, order).unsqueeze(-1).float()                 # 不足 k 顆補 0
            pooled.append(picked.flatten(2))
        if self.full:
            pooled.append((self.squeeze(h) * mask.unsqueeze(-1).float()).flatten(2))       # (B, 4, N·full)
        cand_n = (cand - self.cand_mean) / self.cand_std
        glob_n = ((glob - self.glob_mean) / self.glob_std).unsqueeze(1).expand(-1, C, -1)
        return self.cand_mlp(torch.cat(pooled + [cand_n, glob_n], dim=-1))


class SupNet(nn.Module):
    kind = "sup"

    def __init__(self, stats=None, d=64, hidden=128, attn=0, topk=0, full=0, n=100):
        super().__init__()
        self.config = {"d": d, "hidden": hidden, "attn": attn, "topk": topk, "full": full, "n": n}
        self.encoder = Encoder(stats, d, hidden, attn, topk, full, n)
        self.score = nn.Linear(d, 1)

    def forward(self, tok, mask, cand, glob):
        """→ 分數 (B, 4)（logits）。"""
        return self.score(self.encoder(tok, mask, cand, glob)).squeeze(-1)


class DuelingNet(nn.Module):
    kind = "d3qn"

    def __init__(self, stats=None, d=64, hidden=128):
        super().__init__()
        self.config = {"d": d, "hidden": hidden}
        self.encoder = Encoder(stats, d, hidden)
        self.advantage = nn.Linear(d, 1)
        self.value = nn.Sequential(nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, tok, mask, cand, glob):
        """→ Q (B, 4)。"""
        u = self.encoder(tok, mask, cand, glob)
        adv = self.advantage(u).squeeze(-1)
        return self.value(u.mean(1)) + adv - adv.mean(dim=1, keepdim=True)


NETS = {"sup": SupNet, "d3qn": DuelingNet}


def make_tokens(S, G, S_prio, G_prio, sensor):
    """S、G (B, N, 10)、S_prio、G_prio (B, N)、sensor (B, N, 2) → token (B, N, 25)（同 E06／E03）。"""
    on_s, on_g = S[..., 0] > 0, G[..., 0] > 0
    diff = torch.stack([(on_s != on_g), on_s & on_g & (S[..., 1] != G[..., 1]), S_prio != G_prio], dim=-1)
    return torch.cat([S, G, sensor, diff.float()], dim=-1)


def span_mask(span, n):
    """span (B, 4, 2) [left, right) → (B, 4, N) bool。"""
    ids = torch.arange(n, device=span.device)
    return (ids >= span[..., :1].long()) & (ids < span[..., 1:].long())


class RoundStore:
    """回合層級資料（float16）放在裝置上；用 (回合, searcher, good) 索引組出網路輸入。"""

    def __init__(self, rounds, device):
        t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=device)  # noqa: E731
        self.S, self.G = t(rounds["S"], torch.float16), t(rounds["G"], torch.float16)
        self.S_prio, self.G_prio = t(rounds["S_prio"], torch.int8), t(rounds["G_prio"], torch.int8)
        self.sensor = t(rounds["sensor"], torch.float16)
        self.n = self.S.shape[2]

    def inputs(self, r, s, g, span, cand, glob):
        """r、s、g (B,) long；span (B, 4, 2)、cand (B, 4, 10)、glob (B, 10) 張量 → forward 的參數。"""
        tok = make_tokens(self.S[r, s].float(), self.G[r, g].float(), self.S_prio[r, s], self.G_prio[r, g],
                          self.sensor[r].float())
        return tok, span_mask(span, self.n), cand, glob


def round_inputs(data, device):
    """單一回合的 numpy 回合資料（env._round_data）→ forward 的參數（64 組交易）。"""
    store = RoundStore({k: np.asarray(data[k])[None] for k in ROUND_KEYS}, device)
    P = len(data["s_id"])
    t = lambda x, dt: torch.as_tensor(np.asarray(x), dtype=dt, device=device)  # noqa: E731
    return store.inputs(torch.zeros(P, dtype=torch.long, device=device), t(data["s_id"], torch.long),
                        t(data["g_id"], torch.long), t(data["span"], torch.long),
                        t(data["cand"], torch.float32), t(data["glob"], torch.float32))


def norm_stats(data, max_pairs=200000, seed=0):
    """token（全網每顆 sensor）、候選特徵、全域的平均／標準差，從最多 max_pairs 組交易估。"""
    R, P = data["s_id"].shape
    rng = np.random.default_rng(seed)
    flat = rng.choice(R * P, size=min(max_pairs, R * P), replace=False)
    r, p = flat // P, flat % P
    s, g = data["s_id"][r, p].astype(int), data["g_id"][r, p].astype(int)
    t = lambda x: torch.as_tensor(np.asarray(x))  # noqa: E731
    tok = make_tokens(t(data["S"][r, s].astype(np.float32)), t(data["G"][r, g].astype(np.float32)),
                      t(data["S_prio"][r, s]), t(data["G_prio"][r, g]),
                      t(data["sensor"][r].astype(np.float32))).numpy().reshape(-1, TOKEN_DIM)
    cand = data["cand"][r, p].reshape(-1, CAND_DIM)
    glob = data["glob"][r, p]
    return {"tok_mean": tok.mean(0), "tok_std": tok.std(0), "cand_mean": cand.mean(0), "cand_std": cand.std(0),
            "glob_mean": glob.mean(0), "glob_std": glob.std(0)}


def load_net(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = NETS[ckpt["kind"]](**ckpt["net_config"]).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net


def greedy_policy(net, device="cpu"):
    """選分數（或 Q）最大的候選。"""
    def policy(data):
        with torch.no_grad():
            return net(*round_inputs(data, device)).argmax(dim=1).cpu().numpy()
    return policy


def sample_policy(net, rng, device="cpu"):
    """依 softmax(分數) 的機率抽一個候選（監督式用；rng 是 numpy Generator）。"""
    def policy(data):
        with torch.no_grad():
            prob = torch.softmax(net(*round_inputs(data, device)), dim=1).cpu().numpy()
        u = rng.random((len(prob), 1))
        return np.minimum((prob.cumsum(axis=1) < u).sum(axis=1), prob.shape[1] - 1)
    return policy


def param_count(net):
    return sum(p.numel() for p in net.parameters())


__all__ = ["SupNet", "DuelingNet", "NETS", "RoundStore", "round_inputs", "make_tokens", "span_mask", "norm_stats",
           "load_net", "greedy_policy", "sample_policy", "param_count"]
