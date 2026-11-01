"""E12 網路 RelFlagNet：E08 RelNet 每顆 token 多 1 個範圍旗標，② 覆蓋相關、③ 路由相關也包含範圍內的 sensor。

每組交易 4 個候選共用一組權重，各出 1 個分數 → 4 個 logits：
    φ：每顆 [token 25（z-score）‖ 在這個候選範圍內 1] → 26 → 64 → 64（旗標隨候選不同，所以每個候選各跑一次）
    候選 k：[① 範圍內 平均‖最大 128 ‖ ② 範圍內＋覆蓋相關 128 ‖ ③ 範圍內＋路由相關 128 ‖ 候選 10 ‖ 交換 8 ‖ 全域 10]
          ＝ 412 → 128 → 64 → 1
覆蓋相關、路由相關沿用 E08 的定義（features.round_extras，只含範圍外），這裡再併上範圍內。
"""
import numpy as np
import torch
from torch import nn

from env import CAND_DIM, GLOBAL_DIM, TOKEN_DIM
from features import SWAP_DIM


def masked_pool(h, mask):
    """h (B, 4, N, d)、mask (B, 4, N) bool → (B, 4, 2d)：平均 ‖ 最大（遮罩為空時都是 0）。"""
    m = mask.float()
    count = m.sum(-1, keepdim=True)
    mean = torch.einsum("bcn,bcnd->bcd", m, h) / count.clamp(min=1.0)
    peak = h.masked_fill(~mask.unsqueeze(-1), -1e9).amax(2)
    peak = torch.where(count > 0, peak, torch.zeros_like(peak))
    return torch.cat([mean, peak], dim=-1)


class RelFlagNet(nn.Module):
    kind = "e12_relflag"

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
        """tok (B, N, 25)、inside／route／covrel (B, 4, N) bool（route、covrel 只含範圍外）、
        cand (B, 4, 10)、swap (B, 4, 8)、glob (B, 10) → (B, 4)。"""
        C = inside.shape[1]
        t = ((tok - self.tok_mean) / self.tok_std).unsqueeze(1).expand(-1, C, -1, -1)
        h = self.phi(torch.cat([t, inside.unsqueeze(-1).float()], dim=-1))             # (B, 4, N, d)
        x = torch.cat([masked_pool(h, inside), masked_pool(h, covrel | inside), masked_pool(h, route | inside),
                       (cand - self.cand_mean) / self.cand_std, (swap - self.swap_mean) / self.swap_std,
                       ((glob - self.glob_mean) / self.glob_std).unsqueeze(1).expand(-1, C, -1)], dim=-1)
        return self.mlp(x).squeeze(-1)


def load_relflagnet(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = RelFlagNet(**ckpt["net_config"]).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net
