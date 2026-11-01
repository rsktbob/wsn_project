"""E13 網路 Net13：E12 RelFlagNet 的結構，只換輸入（env13 的定義）。

每組交易 4 個候選共用一組權重，各出 1 個分數 → 4 個 logits：
    φ：每顆 [token 24（sensor 4 ‖ S 10 ‖ G 10，z-score）‖ 在這個候選範圍內 1] → 25 → 64 → 64
    候選 k：[① 範圍內 平均‖最大 128 ‖ ② 範圍內＋覆蓋相關 128 ‖ ③ 範圍內＋路由相關 128 ‖ 候選 9 ‖ 交換 8 ‖ 全域 7]
          ＝ 408 → 128 → 64 → 1
覆蓋相關、路由相關、交換沿用 E08（features.round_extras，route／covrel 只含範圍外，這裡併上範圍內）。
"""
import numpy as np
import torch
from torch import nn

from env13 import CAND_DIM, GLOBAL_DIM, TOKEN_DIM
from features import SWAP_DIM
from relflagnet import masked_pool


def make_tokens(S, G, sensor):
    """S、G (B, N, 10)、sensor (B, N, 4) → token (B, N, 24)。"""
    return torch.cat([sensor, S, G], dim=-1)


class Net13(nn.Module):
    kind = "e13"

    def __init__(self, stats=None, d=64, hidden=128):
        super().__init__()
        self.config = {"d": d, "hidden": hidden}
        for name, dim in (("tok", TOKEN_DIM), ("cand", CAND_DIM), ("glob", GLOBAL_DIM), ("swap", SWAP_DIM)):
            mean = np.zeros(dim, np.float32) if stats is None else np.asarray(stats[f"{name}_mean"], np.float32)
            std = np.ones(dim, np.float32) if stats is None else np.asarray(stats[f"{name}_std"], np.float32)
            self.register_buffer(f"{name}_mean", torch.as_tensor(mean))
            self.register_buffer(f"{name}_std", torch.as_tensor(np.where(std > 1e-6, std, 1.0)))
        self.phi = nn.Sequential(nn.Linear(TOKEN_DIM + 1, d), nn.ReLU(), nn.Linear(d, d))
        self.mlp = nn.Sequential(nn.Linear(6 * d + CAND_DIM + SWAP_DIM + GLOBAL_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, tok, inside, route, covrel, cand, swap, glob):
        """tok (B, N, 24)、inside／route／covrel (B, 4, N) bool（route、covrel 只含範圍外）、
        cand (B, 4, 9)、swap (B, 4, 8)、glob (B, 7) → (B, 4)。"""
        C = inside.shape[1]
        t = ((tok - self.tok_mean) / self.tok_std).unsqueeze(1).expand(-1, C, -1, -1)
        h = self.phi(torch.cat([t, inside.unsqueeze(-1).float()], dim=-1))             # (B, 4, N, d)
        x = torch.cat([masked_pool(h, inside), masked_pool(h, covrel | inside), masked_pool(h, route | inside),
                       (cand - self.cand_mean) / self.cand_std, (swap - self.swap_mean) / self.swap_std,
                       ((glob - self.glob_mean) / self.glob_std).unsqueeze(1).expand(-1, C, -1)], dim=-1)
        return self.mlp(x).squeeze(-1)


class Store13:
    """回合層級資料（float16）放在裝置上；inputs 介面同 E07 RoundStore，給 E12 的 RelPairs 用。"""

    def __init__(self, rounds, device):
        t = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float16, device=device)  # noqa: E731
        self.S, self.G, self.sensor = t(rounds["S"]), t(rounds["G"]), t(rounds["sensor"])
        self.n = self.S.shape[2]

    def inputs(self, r, s, g, span, cand, glob):
        tok = make_tokens(self.S[r, s].float(), self.G[r, g].float(), self.sensor[r].float())
        ids = torch.arange(self.n, device=span.device)
        inside = (ids >= span[..., :1].long()) & (ids < span[..., 1:].long())
        return tok, inside, cand, glob


def norm_stats(data, max_pairs=200000, seed=0):
    """token（全網每顆 sensor）、候選、全域的平均／標準差，從最多 max_pairs 組交易估（同 E07 norm_stats）。"""
    R, P = data["s_id"].shape
    rng = np.random.default_rng(seed)
    flat = rng.choice(R * P, size=min(max_pairs, R * P), replace=False)
    r, p = flat // P, flat % P
    s, g = data["s_id"][r, p].astype(int), data["g_id"][r, p].astype(int)
    tok = np.concatenate([data["sensor"][r], data["S"][r, s], data["G"][r, g]], axis=-1).astype(np.float32)
    tok = tok.reshape(-1, TOKEN_DIM)
    cand = data["cand"][r, p].reshape(-1, CAND_DIM)
    glob = data["glob"][r, p]
    return {"tok_mean": tok.mean(0), "tok_std": tok.std(0), "cand_mean": cand.mean(0), "cand_std": cand.std(0),
            "glob_mean": glob.mean(0), "glob_std": glob.std(0)}


def load_net13(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = Net13(**ckpt["net_config"]).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net
