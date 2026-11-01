"""E10 網路 OutNet：E07 SupNet 不丟掉範圍外的 sensor，範圍外和範圍內一樣處理。

每組交易 4 個候選共用一組權重，各出 1 個分數 → 4 個 logits：
    φ：token 25 → 64 → 64（100 顆共用，範圍內外同一個 φ，跑一次）
    候選 k：[範圍內 平均‖最大 128 ‖ 範圍外 平均‖最大 128 ‖ 候選 10 ‖ 全域 10] ＝ 276 → 128 → 64 → 1
輸入和 E07 相同（tok、mask、cand、glob），範圍外 ＝ ~mask。
"""
import numpy as np
import torch
from torch import nn

from env import CAND_DIM, GLOBAL_DIM, TOKEN_DIM


def masked_pool(h, mask):
    """h (B, N, d)、mask (B, 4, N) bool → (B, 4, 2d)：平均 ‖ 最大（遮罩為空時都是 0）。"""
    m = mask.float()
    count = m.sum(-1, keepdim=True)
    mean = torch.einsum("bcn,bnd->bcd", m, h) / count.clamp(min=1.0)
    peak = h.unsqueeze(1).masked_fill(~mask.unsqueeze(-1), -1e9).amax(2)
    peak = torch.where(count > 0, peak, torch.zeros_like(peak))
    return torch.cat([mean, peak], dim=-1)


class OutNet(nn.Module):
    kind = "e10_out"

    def __init__(self, stats=None, d=64, hidden=128):
        super().__init__()
        self.config = {"d": d, "hidden": hidden}
        for name, dim in (("tok", TOKEN_DIM), ("cand", CAND_DIM), ("glob", GLOBAL_DIM)):
            mean = np.zeros(dim, np.float32) if stats is None else np.asarray(stats[f"{name}_mean"], np.float32)
            std = np.ones(dim, np.float32) if stats is None else np.asarray(stats[f"{name}_std"], np.float32)
            self.register_buffer(f"{name}_mean", torch.as_tensor(mean))
            self.register_buffer(f"{name}_std", torch.as_tensor(np.where(std > 1e-6, std, 1.0)))
        self.phi = nn.Sequential(nn.Linear(TOKEN_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.mlp = nn.Sequential(nn.Linear(4 * d + CAND_DIM + GLOBAL_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, tok, inside, cand, glob):
        """tok (B, N, 25)、inside (B, 4, N) bool、cand (B, 4, 10)、glob (B, 10) → (B, 4)。"""
        h = self.phi((tok - self.tok_mean) / self.tok_std)
        C = inside.shape[1]
        x = torch.cat([masked_pool(h, inside), masked_pool(h, ~inside),
                       (cand - self.cand_mean) / self.cand_std,
                       ((glob - self.glob_mean) / self.glob_std).unsqueeze(1).expand(-1, C, -1)], dim=-1)
        return self.mlp(x).squeeze(-1)


def load_outnet(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = OutNet(**ckpt["net_config"]).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net
