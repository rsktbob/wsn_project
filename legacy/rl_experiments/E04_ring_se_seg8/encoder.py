"""E04 兩個網路共用的部分：輸入標準化與段編碼器（design_A.svg／design_B.svg 的 ② ③）。"""
from __future__ import annotations

import torch
from torch import nn

from env import GLOBAL_DIM, LMAX, SEG_DIM, TOKEN_DIM, split_state

PHI = 64
POOLED = 3 * PHI       # 平均 ‖ 最大 ‖ 總和÷13
SEG_HIDDEN = 128
D_MODEL = 64


class Normalizer(nn.Module):
    """z-score：平均／標準差由 norm_stats.py 用隨機策略的資料算一次後固定（存在 checkpoint 的 buffer）。"""

    def __init__(self, stats=None):
        super().__init__()
        for name, dim in (("token", TOKEN_DIM), ("seg", SEG_DIM), ("glob", GLOBAL_DIM)):
            mean = torch.zeros(dim) if stats is None else torch.as_tensor(stats[f"{name}_mean"], dtype=torch.float32)
            std = torch.ones(dim) if stats is None else torch.as_tensor(stats[f"{name}_std"], dtype=torch.float32)
            self.register_buffer(f"{name}_mean", mean)
            self.register_buffer(f"{name}_std", std)

    def forward(self, states):
        tokens, mask, seg, glob = split_state(states)
        tokens = (tokens - self.token_mean) / self.token_std * mask[..., None]   # 空位維持 0
        seg = (seg - self.seg_mean) / self.seg_std
        glob = (glob - self.glob_mean) / self.glob_std
        return tokens, mask, seg, glob


class SegmentEncoder(nn.Module):
    """13 token → φ 25→64→64 → 遮罩後 平均 ‖ 最大 ‖ 總和÷13（192）‖ 段特徵 7 → 199→128→64。8 段共用權重。"""

    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(TOKEN_DIM, PHI), nn.ReLU(), nn.Linear(PHI, PHI), nn.ReLU())
        self.mix = nn.Sequential(nn.Linear(POOLED + SEG_DIM, SEG_HIDDEN), nn.ReLU(),
                                 nn.Linear(SEG_HIDDEN, D_MODEL), nn.ReLU())

    def forward(self, tokens, mask, seg):
        x = self.phi(tokens)                                  # (P, 8, 13, 64)
        m = mask[..., None]
        total = (x * m).sum(dim=2)
        count = m.sum(dim=2).clamp(min=1.0)
        mean = total / count
        maximum = x.masked_fill(m == 0, float("-inf")).max(dim=2).values
        pooled = torch.cat([mean, maximum, total / LMAX], dim=-1)   # (P, 8, 192)
        return self.mix(torch.cat([pooled, seg], dim=-1))     # (P, 8, 64)


__all__ = ["Normalizer", "SegmentEncoder", "D_MODEL"]
