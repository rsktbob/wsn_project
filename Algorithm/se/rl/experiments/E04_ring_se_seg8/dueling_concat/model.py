"""設計 A（design_A.svg）：8 個段向量依段號固定順序串接＋全域 → 經典 Dueling（Wang et al., 2016）。

    h₀ … h₇（各 64）‖ 全域 10 → 522 → 128 → 128 → V(128→64→1)、A(128→64→8)
    Q_k = V + A_k − mean(A)；第 k 個輸出固定代表第 k 段。
"""
from __future__ import annotations

import torch
from torch import nn

from encoder import D_MODEL, Normalizer, SegmentEncoder
from env import GLOBAL_DIM, H

HIDDEN = 128
BRANCH = 64


class ConcatDueling(nn.Module):
    def __init__(self, stats=None):
        super().__init__()
        self.norm = Normalizer(stats)
        self.encoder = SegmentEncoder()
        self.shared = nn.Sequential(nn.Linear(H * D_MODEL + GLOBAL_DIM, HIDDEN), nn.ReLU(),
                                    nn.Linear(HIDDEN, HIDDEN), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(HIDDEN, BRANCH), nn.ReLU(), nn.Linear(BRANCH, 1))
        self.advantage = nn.Sequential(nn.Linear(HIDDEN, BRANCH), nn.ReLU(), nn.Linear(BRANCH, H))

    def forward(self, states):
        tokens, mask, seg, glob = self.norm(states)
        h = self.encoder(tokens, mask, seg)                   # (P, 8, 64)
        z = self.shared(torch.cat([h.flatten(1), glob], dim=1))
        v = self.value(z)
        a = self.advantage(z)
        return v + a - a.mean(dim=1, keepdim=True)


__all__ = ["ConcatDueling"]
