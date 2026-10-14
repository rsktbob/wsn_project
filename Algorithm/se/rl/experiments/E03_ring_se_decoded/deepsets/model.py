"""E03 DeepSets 打分網路與 token 組裝（訓練時在 GPU 上從回合層級特徵直接組 token，不先展開存檔）。

每個候選：範圍內每顆 sensor token 25 維 → φ 25→64→64 → 平均 ‖ 最大 ‖ 總和（÷LMAX）= 192
    ‖ 範圍外影響 4 ‖ 位置 4 ‖ 全域 10 → 打分頭 210→128→64→1。4 個候選共用同一個網路。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from env import GLOBAL_DIM, H, LMAX, OUTSIDE_DIM, POSITION_DIM, TOKEN_DIM  # noqa: E402

EXTRA_DIM = OUTSIDE_DIM + POSITION_DIM + GLOBAL_DIM


class DeepSetsScorer(nn.Module):
    def __init__(self, width=64, head=(128, 64)):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(TOKEN_DIM, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        layers, last = [], 3 * width + EXTRA_DIM
        for h in head:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.head = nn.Sequential(*layers, nn.Linear(last, 1))

    def forward(self, tokens, mask, extra):
        """tokens (B, H, L, 25)、mask (B, H, L)、extra (B, H, 18) → 分數 (B, H)。"""
        z = self.phi(tokens)
        m = mask.unsqueeze(-1).to(z.dtype)
        total = (z * m).sum(dim=2)
        mean = total / m.sum(dim=2).clamp(min=1.0)
        peak = z.masked_fill(~mask.unsqueeze(-1), -1e9).amax(dim=2).clamp(min=0.0)
        return self.head(torch.cat([mean, peak, total / LMAX, extra], dim=-1)).squeeze(-1)


def gather_tokens(store, rows):
    """store：回合層級張量（S、G、S_genes、G_genes、sensor）＋交易層級（round、s_id、g_id、spans、outside、position、global）；
    rows：要取的交易編號（tensor）。輸出和 env.build_tokens / build_batch 相同的 tokens、mask、extra。"""
    r = store["round"][rows].long()[:, None, None]
    s = store["s_id"][rows].long()[:, None, None]
    g = store["g_id"][rows].long()[:, None, None]
    spans = store["spans"][rows].long()
    left, right = spans[:, :, 1:2], spans[:, :, 2:3]
    idx = left + torch.arange(LMAX, device=spans.device)
    mask = idx < right
    idx = idx.clamp(max=store["S"].shape[2] - 1)
    S = store["S"][r, s, idx].float()
    G = store["G"][r, g, idx].float()
    sg = store["S_genes"][r, s, idx]
    gg = store["G_genes"][r, g, idx]
    sensor = store["sensor"][r, idx].float()
    on_differ = S[..., 0] != G[..., 0]
    level_differ = (S[..., 0] > 0) & (G[..., 0] > 0) & (S[..., 1] != G[..., 1])
    prio_differ = sg[..., 1] != gg[..., 1]
    diff = torch.stack([on_differ, level_differ, prio_differ], dim=-1).float()
    tokens = torch.cat([S, G, sensor, diff], dim=-1) * mask.unsqueeze(-1)
    glob = store["global"][rows].float().unsqueeze(1).expand(-1, H, -1)
    extra = torch.cat([store["outside"][rows].float(), store["position"][rows].float(), glob], dim=-1)
    return tokens, mask, extra
