"""E09 資料：從原始染色體直接拼子代，組成 Transformer 的 token（不解碼、不用父代的解碼狀態）。

每組交易 4 個候選範圍 [left, right)：child1 ＝ S 範圍內換成 G 的基因，child2 ＝ G 範圍內換成 S 的基因。
每個子代每顆 sensor 一個 token（TOKEN_DIM ＝ 13）：
    等級基因 % 6 ÷ 5、優先度基因 ÷ 9（子代自己的基因）
    剩餘電量比例、到 BS 距離（÷ 最遠）、energy_score（電量 ÷ 全網最大電量）、proximity_score（地圖常數）
    等級 1–5 各能覆蓋幾個 target ÷ T（地圖覆蓋表）、座標 x、y ÷ 邊界（地圖常數）
全域 2 維：搜尋進度、活著 sensor 的平均電量（E07 glob 第 0、2 欄）。
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
E07 = HERE.parent / "E07_ring_se_lifetime"
sys.path.insert(0, str(E07))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import PROBLEM_SEED, train_maps  # noqa: E402

DATA = HERE / "results" / "data"
MAP_CONST = HERE / "results" / "map_const.npz"
TOKEN_DIM = 13
GLOBAL_DIM = 2
LEVEL_OPTIONS = 6       # radius_option_counts（所有 sensor 都是 6）
RANK_MAX = 9            # 優先度基因 0–9


def map_constants(problem):
    """proximity_score (N,)、covcap (N, 7)：等級 1–5 各能覆蓋幾個 target ÷ T ‖ 座標 x、y ÷ 邊界。"""
    cover = np.asarray(problem.coverage_table) > 0                       # (T, N, 6)
    N = cover.shape[1]
    covcap = cover[:, :, 1:].sum(axis=0) / cover.shape[0]
    xy = np.asarray(problem.sensor, dtype=float)[:N] / float(problem.BOUNDARY)
    return np.asarray(problem.proximity_score, dtype=np.float32), np.concatenate([covcap, xy], axis=1).astype(np.float32)


def _one_map(name):
    from experiments.problem_setup import build_problem
    spec = next(s for s in train_maps(200) if s["name"] == name)
    return map_constants(build_problem(SimpleNamespace(_current_map=spec, fitness_service="v3"), PROBLEM_SEED))


def load_map_constants():
    """RLTRAIN01–200 的地圖常數（第一次會建出 results/map_const.npz）。"""
    if not MAP_CONST.exists():
        from concurrent.futures import ProcessPoolExecutor
        names = [s["name"] for s in train_maps(200)]
        with ProcessPoolExecutor(8) as pool:
            out = list(pool.map(_one_map, names))
        np.savez(MAP_CONST, names=np.array(names), prox=np.stack([o[0] for o in out]),
                 covcap=np.stack([o[1] for o in out]))
    z = np.load(MAP_CONST)
    return {str(n): i for i, n in enumerate(z["names"])}, z["prox"], z["covcap"]


def map_files(folder, maps=None):
    folder = Path(folder)
    if maps is not None:
        return [folder / f"RLTRAIN{i:02d}.npz" for i in range(1, maps + 1)]
    return [f for f in sorted(folder.glob("RLTRAIN*.npz")) if not f.name.endswith(".tmp.npz")]


class ChildData:
    """一個 split 的資料放在裝置上；tokens(idx) 即時組出 (B, 4, 2, N, 11) 的子代 token。"""

    def __init__(self, files, label, device):
        index, prox, covcap = load_map_constants()
        S_code, G_code, sensor, map_id = [], [], [], []
        r_of, s_id, g_id, span, lab, glob = [], [], [], [], [], []
        offset = 0
        for f in files:
            z = np.load(f)
            R, P = z["s_id"].shape
            S_code.append(z["S_code"])
            G_code.append(z["G_code"])
            sensor.append(z["sensor"].astype(np.float32))
            map_id.append(np.full(R, index[f.stem]))
            r_of.append(np.repeat(np.arange(R) + offset, P))
            s_id.append(z["s_id"].reshape(-1))
            g_id.append(z["g_id"].reshape(-1))
            span.append(z["span"].reshape(R * P, 4, 2))
            lab.append(z[label].reshape(R * P, 4))
            glob.append(z["glob"].reshape(R * P, -1)[:, [0, 2]])
            offset += R
        t = lambda x, dt: torch.as_tensor(np.concatenate(x), dtype=dt, device=device)  # noqa: E731
        self.S_code, self.G_code = t(S_code, torch.uint8), t(G_code, torch.uint8)
        self.sensor, self.map_id = t(sensor, torch.float32), t(map_id, torch.long)
        self.prox = torch.as_tensor(prox, device=device)
        self.covcap = torch.as_tensor(covcap, device=device)
        self.r, self.s, self.g = t(r_of, torch.long), t(s_id, torch.long), t(g_id, torch.long)
        self.span, self.label, self.glob = t(span, torch.long), t(lab, torch.float32), t(glob, torch.float32)
        self.size = len(self.r)
        self.n = self.S_code.shape[2] // 2
        self.informative = self.label.max(dim=1).values > 0

    def inputs(self, idx):
        """→ tokens (B, 4, 2, N, 13)、glob (B, 2)。"""
        r = self.r[idx]
        N = self.n
        S = self.S_code[r, self.s[idx]].view(-1, N, 2).long()
        G = self.G_code[r, self.g[idx]].view(-1, N, 2).long()
        ids = torch.arange(N, device=S.device)
        span = self.span[idx]
        inside = (ids >= span[..., :1]) & (ids < span[..., 1:])                     # (B, 4, N)
        m = inside.unsqueeze(-1)
        child1 = torch.where(m, G.unsqueeze(1), S.unsqueeze(1))                      # (B, 4, N, 2)
        child2 = torch.where(m, S.unsqueeze(1), G.unsqueeze(1))
        genes = torch.stack([child1, child2], dim=2)                                 # (B, 4, 2, N, 2)
        genes = torch.stack([(genes[..., 0] % LEVEL_OPTIONS) / (LEVEL_OPTIONS - 1), genes[..., 1] / RANK_MAX], dim=-1)
        sensor = self.sensor[r]                                                      # (B, N, 2)
        e = sensor[..., 0]
        escore = e / e.amax(dim=1, keepdim=True).clamp(min=1e-9)
        mid = self.map_id[r]
        feat = torch.cat([sensor, escore.unsqueeze(-1), self.prox[mid].unsqueeze(-1), self.covcap[mid]], dim=-1)
        feat = feat[:, None, None].expand(-1, 4, 2, -1, -1)                          # (B, 4, 2, N, 11)
        return torch.cat([genes, feat], dim=-1), self.glob[idx]
