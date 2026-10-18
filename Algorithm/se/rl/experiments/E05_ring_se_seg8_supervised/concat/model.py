"""設計 A（design_A.svg）：8 個段向量依段號固定順序串接＋全域 → MLP → 8 個分數。

    h₀ … h₇（各 64，E04 段編碼器）‖ 全域 10 → 522 → 128 → 128 → 8
    softmax(s)_k ＝ 第 k 段是 8 段中最好那段的機率；第 k 個輸出固定代表第 k 段。
"""
from __future__ import annotations

import torch
from torch import nn

from encoder import D_MODEL, Normalizer, SegmentEncoder
from env import GLOBAL_DIM, H

HIDDEN = 128


class ConcatScorer(nn.Module):
    def __init__(self, stats=None):
        super().__init__()
        self.norm = Normalizer(stats)
        self.encoder = SegmentEncoder()
        self.mlp = nn.Sequential(nn.Linear(H * D_MODEL + GLOBAL_DIM, HIDDEN), nn.ReLU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, H))

    def forward(self, states):
        tokens, mask, seg, glob = self.norm(states)
        h = self.encoder(tokens, mask, seg)                   # (P, 8, 64)
        return self.mlp(torch.cat([h.flatten(1), glob], dim=1))


__all__ = ["ConcatScorer"]
