"""設計 B（design_B.svg）：全域 token ＋ 8 個段 token 過 2 層 Transformer，V 讀全域 token、8 段共用 A 頭。

    token：全域 10→64；段 h_k ＋ 可學段嵌入 e_k（8×64）
    Transformer：pre-LN、d=64、4 頭、FFN 128、不用 dropout
    段距離 attention 偏置：score(i, j) += r[層, 頭, d(i, j)]，d = |i−j| ∈ 0…7（段之間），
        有一方是全域 token 時 d = 8；每層每頭 9 個可學參數
    V：全域 token 輸出 64→64→1；A：每個段 token 輸出 64→64→1（共用權重）
    Q_k = V + A_k − mean(A)
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from encoder import D_MODEL, Normalizer, SegmentEncoder
from env import GLOBAL_DIM, H

HEADS = 4
FFN = 128
LAYERS = 2
BRANCH = 64
N_DIST = H + 1   # 0…7 段距離，8 = 和全域 token 之間


def distance_index():
    idx = torch.full((H + 1, H + 1), H, dtype=torch.long)
    seg = torch.arange(H)
    idx[1:, 1:] = (seg[:, None] - seg[None, :]).abs()
    return idx


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(D_MODEL)
        self.qkv = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.proj = nn.Linear(D_MODEL, D_MODEL)
        self.dist_bias = nn.Parameter(torch.zeros(HEADS, N_DIST))
        self.ln2 = nn.LayerNorm(D_MODEL)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, FFN), nn.ReLU(), nn.Linear(FFN, D_MODEL))

    def forward(self, x, dist):
        P, T, _ = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(P, T, 3, HEADS, D_MODEL // HEADS).permute(2, 0, 3, 1, 4)
        score = q @ k.transpose(-1, -2) / math.sqrt(D_MODEL // HEADS) + self.dist_bias[:, dist]   # (P, 4, 9, 9)
        out = (F.softmax(score, dim=-1) @ v).transpose(1, 2).reshape(P, T, D_MODEL)
        x = x + self.proj(out)
        return x + self.ffn(self.ln2(x))


class AttentionDueling(nn.Module):
    def __init__(self, stats=None):
        super().__init__()
        self.norm = Normalizer(stats)
        self.encoder = SegmentEncoder()
        self.global_in = nn.Linear(GLOBAL_DIM, D_MODEL)
        self.segment_embedding = nn.Parameter(torch.randn(H, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.ln_out = nn.LayerNorm(D_MODEL)
        self.value = nn.Sequential(nn.Linear(D_MODEL, BRANCH), nn.ReLU(), nn.Linear(BRANCH, 1))
        self.advantage = nn.Sequential(nn.Linear(D_MODEL, BRANCH), nn.ReLU(), nn.Linear(BRANCH, 1))
        self.register_buffer("dist", distance_index(), persistent=False)

    def forward(self, states):
        tokens, mask, seg, glob = self.norm(states)
        h = self.encoder(tokens, mask, seg) + self.segment_embedding           # (P, 8, 64)
        x = torch.cat([self.global_in(glob)[:, None], h], dim=1)              # (P, 9, 64)
        for block in self.blocks:
            x = block(x, self.dist)
        x = self.ln_out(x)
        v = self.value(x[:, 0])                                                # (P, 1)
        a = self.advantage(x[:, 1:]).squeeze(-1)                               # (P, 8)
        return v + a - a.mean(dim=1, keepdim=True)


__all__ = ["AttentionDueling"]
