"""E09 網路 ChildNet：每個候選一條序列 [CLS1] child1 的 N 顆 [CLS2] child2 的 N 顆（N＝100 → 202 個 token）。

token：Linear(13 → d) ＋ 位置嵌入（環上編號，兩個子代共用）＋ 子代嵌入（child1／child2）；CLS1、CLS2 各一個可學習向量。
Transformer 3 層（d＝64、4 頭、FFN 128、pre-norm），4 個候選共用權重。
CLS1 ‖ CLS2 ‖ 全域 2 → 128 → 64 → 1 → 每個候選一個分數，4 個做 softmax。
contrast（10/26 起預設）：CLS1‖CLS2 先減掉同一組交易 4 個候選的平均、再除以標準差才接 MLP——4 條序列只差
交換的 ~20/200 個 token，不減平均時 4 個 CLS 幾乎一樣、loss 停在 ln4 學不起來（tmp/e09_diag）。
（第一版沒有距離注意力偏置：每條序列的地圖不同，偏置矩陣要逐筆算，記憶體太大。）
"""
import numpy as np
import torch
from torch import nn

from childdata import GLOBAL_DIM, TOKEN_DIM


class ChildNet(nn.Module):
    kind = "e09_child"

    def __init__(self, stats=None, n=100, d=64, heads=4, layers=3, hidden=128, contrast=True):
        super().__init__()
        self.config = {"n": n, "d": d, "heads": heads, "layers": layers, "hidden": hidden, "contrast": contrast}
        self.contrast = contrast
        for name, dim in (("tok", TOKEN_DIM), ("glob", GLOBAL_DIM)):
            mean = np.zeros(dim, np.float32) if stats is None else np.asarray(stats[f"{name}_mean"], np.float32)
            std = np.ones(dim, np.float32) if stats is None else np.asarray(stats[f"{name}_std"], np.float32)
            self.register_buffer(f"{name}_mean", torch.as_tensor(mean))
            self.register_buffer(f"{name}_std", torch.as_tensor(np.where(std > 1e-6, std, 1.0)))
        self.embed = nn.Linear(TOKEN_DIM, d)
        self.pos = nn.Parameter(torch.randn(n, d) * 0.02)
        self.child = nn.Parameter(torch.randn(2, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(2, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(2 * d + GLOBAL_DIM, hidden), nn.ReLU(), nn.Linear(hidden, d), nn.ReLU(),
                                 nn.Linear(d, 1))

    def forward(self, tokens, glob):
        """tokens (B, 4, 2, N, 13)、glob (B, 2) → 分數 (B, 4)。"""
        B, C, _, N, _ = tokens.shape
        h = self.embed((tokens - self.tok_mean) / self.tok_std) + self.pos + self.child[:, None, :]   # (B, 4, 2, N, d)
        cls = self.cls[None, None, :, None, :].expand(B, C, -1, -1, -1)                               # (B, 4, 2, 1, d)
        x = torch.cat([cls, h], dim=3).reshape(B * C, 2 * (N + 1), -1)                               # [CLS1 c1… CLS2 c2…]
        x = self.norm(self.encoder(x))
        out = torch.cat([x[:, 0], x[:, N + 1]], dim=-1).view(B, C, -1)
        if self.contrast:
            out = out - out.mean(dim=1, keepdim=True)
            out = out / out.std(dim=(1, 2), keepdim=True).clamp(min=1e-6)
        g = ((glob - self.glob_mean) / self.glob_std).unsqueeze(1).expand(-1, C, -1)
        return self.mlp(torch.cat([out, g], dim=-1)).squeeze(-1)


def load_childnet(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = ChildNet(**ckpt["net_config"]).to(device)
    net.load_state_dict(ckpt["model"])
    net.eval()
    return net
