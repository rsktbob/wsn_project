"""E13 環境：E07 的 Ring_SE_E07（原版 ring_se、每組交易 4 個預抽候選），只換網路輸入（_solution、_round_data）。

每個解每顆 sensor 10 維（個數類用原始值，網路裡再 z-score；不除以「這個解的全網最大」，S、G 才可比）：
    等級基因 /9、priority 基因 /9、解碼後等級 /5（0 ＝ 關）、子樹大小（路徑經過它的 sensor 數）、跳數、
    排層 cost / 初始電量、路由 cost / 初始電量、緊迫度（全網最短壽命 ÷ 這顆壽命）、覆蓋 target 數、只有它覆蓋的 target 數
sensor 本身 4 維：剩餘電量 / 初始電量、(x − BS x) / 邊長、(y − BS y) / 邊長、離 BS 距離 / 邊長
候選 9 維：起點 / N、終點 / N、長度 / N ‖ 範圍外路徑經過範圍內的開啟 sensor 數 / N（S、G）‖
    範圍內覆蓋的 target 在範圍外沒有備援的比例（S、G）‖ 瓶頸：範圍內有效差異 sensor 的緊迫度總和（S、G）
全域 7 維：活著 sensor 平均電量、S／G 覆蓋率、(S 最短壽命 − G 最短壽命) / 較大者、S／G 開啟數 / N、兩解基因不同的比例
（E07 的候選「是否跨段」、全域的搜尋進度、停滯、fitness 差距排名拿掉。）

feat_recorder 不是 None 時每回合存一筆網路輸入（收資料用；不偷看子代，推進和原版相同，標籤沿用 E07 的資料）。
"""
from __future__ import annotations

import numpy as np

from env import H, Ring_SE_E07

SOL_DIM = 10          # 每個解每顆 sensor
SENSOR_DIM = 4        # sensor 本身
TOKEN_DIM = SENSOR_DIM + 2 * SOL_DIM   # 網路 token（不含範圍旗標）＝ 24
CAND_DIM = 9
GLOBAL_DIM = 7
LEVEL = 2             # 解碼後等級 /5 的欄位
TIGHT = 7             # 緊迫度的欄位
ROUND_KEYS = ("S", "G", "S_prio", "G_prio", "S_par", "G_par", "sensor")
PAIR_KEYS = ("s_id", "g_id", "span", "cand", "glob")


class Ring_SE_E13(Ring_SE_E07):
    def __init__(self, problem, policy="random", feat_recorder=None, **kwargs):
        super().__init__(problem, policy=policy, recorder=None, **kwargs)
        self.feat_recorder = feat_recorder

    def _solution(self, problem, candidate):
        key = np.asarray(candidate.code).tobytes()
        info = self._sol_cache.get(key)
        if info is not None:
            return info
        n = problem.SENSOR_NUMBER
        code = np.asarray(candidate.code)
        state = candidate.decode(problem)
        levels = np.asarray(state.levels[:n], dtype=int)
        active = levels > 0
        radii = np.asarray(problem.state_radius(state), dtype=float)
        sched = np.asarray(problem.calculate_scheduling_cost(state, sensing_radii=radii), dtype=float)[:n]
        route = np.asarray(problem.calculate_routing_cost(state, sensing_radii=radii), dtype=float)[:n]
        sched = np.where(np.isfinite(sched), sched, 0.0)
        route = np.where(np.isfinite(route), route, 0.0)
        cost = sched + route
        energy = np.maximum(np.asarray(problem.energy[:n], dtype=float), 0.0)
        spending = active & (cost > 1e-15)
        life = np.full(n, np.inf)
        life[spending] = energy[spending] / cost[spending]
        min_life = float(life[spending].min()) if np.any(spending) else np.inf
        tight = np.zeros(n)
        if np.any(spending):
            tight[spending] = min_life / np.maximum(life[spending], 1e-15)
        depends = np.zeros((n, n), dtype=bool)           # depends[k, j] = j 在 k 往 BS 的路徑上
        hops = np.zeros(n)
        parent = np.full(n, -1, dtype=np.int16)          # 往 BS 的下一跳（BS 或沒開 = −1）
        for k, path in (getattr(state, "paths", {}) or {}).items():
            k = int(k)
            if k >= n:
                continue
            nodes = [int(x) for x in path[1:] if int(x) < n]
            depends[k, nodes] = True
            hops[k] = len(path) - 1
            if len(path) > 1 and int(path[1]) < n:
                parent[k] = int(path[1])
        coverage = np.asarray(problem.coverage_table) > 0
        cov = coverage[:, np.arange(n), levels] & active[None, :]   # (T, N)
        counts = cov.sum(axis=1)
        genes = np.stack([code[0::2], code[1::2] % 10], axis=1).astype(np.int16)
        e0 = float(problem.initial_energy)
        feat = np.zeros((n, SOL_DIM), dtype=np.float32)
        feat[:, 0] = genes[:, 0] / 9.0
        feat[:, 1] = genes[:, 1] / 9.0
        feat[:, LEVEL] = levels / 5.0
        feat[:, 3] = depends.sum(axis=0)
        feat[:, 4] = hops
        feat[:, 5] = sched / e0
        feat[:, 6] = route / e0
        feat[:, TIGHT] = tight
        feat[:, 8] = cov.sum(axis=0)
        feat[:, 9] = (cov & (counts[:, None] == 1)).sum(axis=0)
        info = {"feat": feat, "genes": genes, "active": active, "depends": depends, "cov": cov, "parent": parent,
                "min_life": min_life, "coverage": float((counts > 0).mean()), "n_active": int(active.sum())}
        self._sol_cache[key] = info
        return info

    def _round_data(self, problem, pairs):
        n = problem.SENSOR_NUMBER
        energy = np.maximum(np.asarray(problem.energy[:n], dtype=float), 0.0)
        e_ratio = energy / max(float(problem.initial_energy), 1e-15)
        side = float(problem.BOUNDARY)
        delta = (np.asarray(problem.sensor[:n], dtype=float) - np.asarray(problem.BS, dtype=float)) / side
        dist = np.asarray(problem.distances)[:n, problem.BSID] / side
        sensor = np.stack([e_ratio, delta[:, 0], delta[:, 1], dist], axis=1).astype(np.float32)
        alive = energy >= problem.liveJ
        mean_alive = float(e_ratio[alive].mean()) if np.any(alive) else 0.0
        S = [self._solution(problem, x) for x in self.searchers]
        G = [self._solution(problem, x) for x in self.goods]

        P = len(pairs)
        span = np.zeros((P, H, 2), dtype=np.int16)
        cand = np.zeros((P, H, CAND_DIM), dtype=np.float32)
        glob = np.zeros((P, GLOBAL_DIM), dtype=np.float32)
        for i, (s_id, g_id, cand_spans, _) in enumerate(pairs):
            si, gi = S[s_id], G[g_id]
            for k, (_, left, right) in enumerate(cand_spans):
                span[i, k] = (left, right)
                inside = np.zeros(n, dtype=bool)
                inside[left:right] = True
                cand[i, k, :3] = (left / n, right / n, (right - left) / n)
                for col, sol in ((0, si), (1, gi)):
                    through = sol["depends"][:, inside].any(axis=1) & ~inside & sol["active"]
                    cand[i, k, 3 + col] = through.sum() / n
                    cov_in = sol["cov"][:, inside].any(axis=1)
                    backup = sol["cov"][:, ~inside].any(axis=1)
                    cand[i, k, 5 + col] = (cov_in & ~backup).sum() / max(1, int(cov_in.sum()))
                sl = slice(left, right)
                eff = np.any(si["genes"][sl] != gi["genes"][sl], axis=1) & (si["active"][sl] | gi["active"][sl])
                cand[i, k, 7] = si["feat"][sl, TIGHT][eff].sum()
                cand[i, k, 8] = gi["feat"][sl, TIGHT][eff].sum()
            lives = (si["min_life"], gi["min_life"])
            finite = [x for x in lives if np.isfinite(x)]
            life_gap = ((lives[0] - lives[1]) / max(finite)) if len(finite) == 2 and max(finite) > 0 else 0.0
            glob[i] = (mean_alive, si["coverage"], gi["coverage"], np.clip(life_gap, -1, 1), si["n_active"] / n,
                       gi["n_active"] / n, float(np.any(si["genes"] != gi["genes"], axis=1).mean()))
        data = {
            "S": np.stack([x["feat"] for x in S]), "G": np.stack([x["feat"] for x in G]),
            "S_prio": np.stack([x["genes"][:, 1] for x in S]).astype(np.int8),
            "G_prio": np.stack([x["genes"][:, 1] for x in G]).astype(np.int8),
            "S_par": np.stack([x["parent"] for x in S]), "G_par": np.stack([x["parent"] for x in G]),
            "sensor": sensor,
            "s_id": np.array([p[0] for p in pairs], dtype=np.int8),
            "g_id": np.array([p[1] for p in pairs], dtype=np.int8),
            "span": span, "cand": cand, "glob": glob,
        }
        if self.feat_recorder is not None:
            self.feat_recorder.add_round({k: data[k] for k in ROUND_KEYS + PAIR_KEYS})
        return data


__all__ = ["Ring_SE_E13", "SOL_DIM", "SENSOR_DIM", "TOKEN_DIM", "CAND_DIM", "GLOBAL_DIM", "LEVEL",
           "ROUND_KEYS", "PAIR_KEYS"]
