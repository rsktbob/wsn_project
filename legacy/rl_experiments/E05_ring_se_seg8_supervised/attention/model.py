"""設計 B（design_B.svg）：全域 token ＋ 8 個段 token 過 2 層 Transformer，8 段共用分數頭。

    token、Transformer、段距離偏置都和 E04 設計 B 相同（重用 E04 的 Block）；
    拿掉 V／A：每個段 token 輸出 → 64→64→1 → s_k；全域 token 只當背景資訊，輸出不接頭。
    softmax(s)_k ＝ 第 k 段是 8 段中最好那段的機率。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch
from torch import nn

from encoder import D_MODEL, Normalizer, SegmentEncoder
from env import GLOBAL_DIM, H

_spec = importlib.util.spec_from_file_location(
    "e04_attention_model", Path(__file__).resolve().parents[2] / "E04_ring_se_seg8" / "attention" / "model.py")
_e04 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e04)
Block, distance_index, LAYERS, BRANCH = _e04.Block, _e04.distance_index, _e04.LAYERS, _e04.BRANCH


class AttentionScorer(nn.Module):
    def __init__(self, stats=None):
        super().__init__()
        self.norm = Normalizer(stats)
        self.encoder = SegmentEncoder()
        self.global_in = nn.Linear(GLOBAL_DIM, D_MODEL)
        self.segment_embedding = nn.Parameter(torch.randn(H, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.ln_out = nn.LayerNorm(D_MODEL)
        self.score = nn.Sequential(nn.Linear(D_MODEL, BRANCH), nn.ReLU(), nn.Linear(BRANCH, 1))
        self.register_buffer("dist", distance_index(), persistent=False)

    def forward(self, states):
        tokens, mask, seg, glob = self.norm(states)
        h = self.encoder(tokens, mask, seg) + self.segment_embedding           # (P, 8, 64)
        x = torch.cat([self.global_in(glob)[:, None], h], dim=1)              # (P, 9, 64)
        for block in self.blocks:
            x = block(x, self.dist)
        return self.score(self.ln_out(x)[:, 1:]).squeeze(-1)                  # (P, 8)


__all__ = ["AttentionScorer"]
