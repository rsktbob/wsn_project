"""E08 新增的輸入（只用父代已評估的解碼結果＋地圖覆蓋表，不解碼子代、不做路由）。

每組交易 4 個候選範圍 [left, right) 各算：
    route  (N,) bool   ② 範圍外、S 或 G 的解裡往 BS 的路徑會經過範圍內的 sensor
    covrel (N,) bool   ③ 範圍外、和範圍內（S 或 G 開啟的）sensor 覆蓋同一個 target 的 sensor（S 或 G 開啟）
    swap   (8,) float  ⑤ 交換後覆蓋變化：child1（S 範圍內換成 G 的等級）、child2（G 範圍內換成 S 的等級）各 4 維
                       ＝ 失去覆蓋的 target ÷ T、新增覆蓋的 target ÷ T、範圍內開啟數變化 ÷ N、範圍內等級總和變化 ÷ 5N
等級用父代「解碼後」的等級（解碼器會調整等級，基因上的等級不準）；子代實際解碼可能再調整，⑤ 是近似。
"""
from __future__ import annotations

import numpy as np

SWAP_DIM = 8


def ancestors(parent):
    """parent (B, N)（往 BS 的下一跳，BS 或沒開 = −1）→ dep (B, N, N) bool，dep[b, k, j] ＝ j 在 k 往 BS 的路徑上。"""
    B, N = parent.shape
    parent = parent.astype(np.int64)
    dep = np.zeros((B, N, N), dtype=bool)
    cur = parent.copy()
    for _ in range(N):
        b, k = np.nonzero(cur >= 0)
        if len(b) == 0:
            break
        dep[b, k, cur[b, k]] = True
        cur[b, k] = parent[b, cur[b, k]]
        cur[cur < 0] = -1
    return dep


def solution_coverage(levels, cover):
    """levels (B, N) int、cover (T, N, L) bool → (B, T, N) bool：第 b 個解裡 sensor j 覆蓋 target t。"""
    N = levels.shape[1]
    cov = cover[:, np.arange(N)[None, :], levels]          # (T, B, N)
    return np.transpose(cov, (1, 0, 2)) & (levels > 0)[:, None, :]


def _prefix(x):
    """沿最後一軸的前綴和（前面補 0）→ 用 p[..., r] − p[..., l] 算 [l, r) 的總和。"""
    p = np.zeros(x.shape[:-1] + (x.shape[-1] + 1,), dtype=np.int32)
    np.cumsum(x, axis=-1, out=p[..., 1:])
    return p


def round_extras(S_lev, G_lev, S_par, G_par, cover, s_id, g_id, span):
    """一回合（8 個 searcher、8 個 goods、P 組交易 × 4 個候選）的 ②③⑤。

    S_lev、G_lev (8, N) int；S_par、G_par (8, N) int；cover (T, N, L) bool；s_id、g_id (P,)；span (P, 4, 2)
    → route、covrel (P, 4, N) bool，swap (P, 4, 8) float32。
    """
    P, H = span.shape[:2]
    N = S_lev.shape[1]
    sid, gid = np.repeat(s_id.astype(int), H), np.repeat(g_id.astype(int), H)
    left, right = span[..., 0].reshape(-1).astype(int), span[..., 1].reshape(-1).astype(int)
    inside = (np.arange(N)[None, :] >= left[:, None]) & (np.arange(N)[None, :] < right[:, None])   # (M, N)

    depS, depG = _prefix(ancestors(S_par)), _prefix(ancestors(G_par))       # (8, N, N+1)
    route = ((depS[sid, :, right] - depS[sid, :, left]) > 0) | ((depG[gid, :, right] - depG[gid, :, left]) > 0)
    route &= ~inside

    covS, covG = solution_coverage(S_lev, cover), solution_coverage(G_lev, cover)     # (8, T, N)
    pS, pG = _prefix(covS), _prefix(covG)                                              # (8, T, N+1)
    inS = pS[sid, :, right] - pS[sid, :, left]                                         # (M, T) 範圍內覆蓋者數
    inG = pG[gid, :, right] - pG[gid, :, left]
    touched = (inS > 0) | (inG > 0)
    covrel = ((covS[sid] | covG[gid]) & touched[:, :, None]).any(axis=1) & ~inside

    cntS, cntG = covS.sum(axis=2)[sid], covG.sum(axis=2)[gid]                          # (M, T)
    cnt1, cnt2 = cntS - inS + inG, cntG - inG + inS
    T = cover.shape[0]
    actS = ((S_lev[sid] > 0) & inside).sum(axis=1)
    actG = ((G_lev[gid] > 0) & inside).sum(axis=1)
    levS = (S_lev[sid] * inside).sum(axis=1)
    levG = (G_lev[gid] * inside).sum(axis=1)
    swap = np.stack([((cntS > 0) & (cnt1 == 0)).sum(1) / T, ((cntS == 0) & (cnt1 > 0)).sum(1) / T,
                     (actG - actS) / N, (levG - levS) / (5 * N),
                     ((cntG > 0) & (cnt2 == 0)).sum(1) / T, ((cntG == 0) & (cnt2 > 0)).sum(1) / T,
                     (actS - actG) / N, (levS - levG) / (5 * N)], axis=1).astype(np.float32)
    return route.reshape(P, H, N), covrel.reshape(P, H, N), swap.reshape(P, H, SWAP_DIM)


def levels_of(feat):
    """E07 的解碼特徵 feat (..., N, 10)（第 1 欄 ＝ 等級 ÷ 5）→ 等級 int。"""
    return np.rint(np.asarray(feat[..., 1], dtype=np.float32) * 5).astype(np.int64)
