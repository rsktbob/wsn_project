"""E11 網路 FullNet：100 顆 sensor 完整輸入（不池化），每顆多 1 個「在不在這個候選範圍內」的旗標，100 顆＋CLS 過 Transformer。

每組交易 4 個候選共用一組權重，各出 1 個分數 → 4 個 logits：
    候選 k：每顆 [token 25（z-score）‖ 範圍旗標 1] → 線性 26 → 64，前面加 CLS → 101 × 64
          → Transformer 2 層（d 64、4 頭、FFN 128、pre-norm、無 dropout）→ LayerNorm → 取 CLS 64
          → [CLS 64 ‖ 候選 10 ‖ 全域 10] ＝ 84 → 128 → 64 → 1
不加 sensor 編號的位置嵌入（避免背地圖）；4 個候選之間只有旗標不同。輸入和 E07 相同（tok、mask、cand、glob）。
"""
import numpy as np
import torch
from torch import nn

from env import CAND_DIM, GLOBAL_DIM, TOKEN_DIM


class FullNet(nn.Module):
    kind = "e11_full"

    def __init__(self, stats=None, d=64, hidden=128, layers=2, heads=4):
        super().__init__()
        self.config = {"d": d, "hidden": hidden, "layers": layers, "heads": heads}
        for name, dim in (("tok", TOKEN_DIM), ("cand", CAND_DIM), ("glob", GLOBAL_DIM)):
            mean = np.zeros(dim, np.float32) if stats is None else np.asarray(stats[f"{name}_mean"], np.float32)
            std = np.ones(dim, np.float32) if stats is None else np.asarray(stats[f"{name}_std"], np.float32)
            self.register_buffer(f"{name}_mean", torch.as_tensor(mean))
            self.register_buffer(f"{name}_std", torch.as_tensor(np.where(std > 1e-6, std, 1.0)))
        self.embed = nn.Linear(TOKEN_DIM + 1, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True),
            layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d + CAND_DIM + GLOBAL_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, tok, inside, cand, glob):
        """tok (B, N, 25)、inside (B, 4, N) bool、cand (B, 4, 10)、glob (B, 10) → (B, 4)。"""
        B, C, N = inside.shape
        t = ((tok - self.tok_mean) / self.tok_std).unsqueeze(1).expand(-1, C, -1, -1)
        x = self.embed(torch.cat([t, inside.unsqueeze(-1).float()], dim=-1).reshape(B * C, N, -1))
        x = self.encoder(torch.cat([self.cls.expand(B * C, -1, -1), x], dim=1))
        u = self.norm(x[:, 0]).reshape(B, C, -1)
        feats = torch.cat([u, (cand - self.cand_mean) / self.cand_std,
                           ((glob - self.glob_mean) / self.glob_std).unsqueeze(1).expand(-1, C, -1)], dim=-1)
        return self.mlp(feats).squeeze(-1)


def load_fullnet(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = FullNet(**ckpt["net_config"]).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net
