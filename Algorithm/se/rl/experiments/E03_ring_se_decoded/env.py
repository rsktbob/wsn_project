"""E03 環境：Ring_SE 每組交易照常產生 4 個候選範圍，由外部策略從中選一個；
每個候選用「範圍內每顆 sensor 的解碼後資訊（token）」＋範圍外影響＋位置＋全域描述。

每個解（searcher 或 good）先解碼一次，算出每顆 sensor 10 維（PARENT_FEATURES）：
    開關、等級/5、是否轉送節點、轉送量/全網最大、子樹大小/N、跳數/最大跳數、耗能/全網最大、
    緊迫度（全網最短壽命 ÷ 這顆壽命，1＝瓶頸）、覆蓋 target 數 min(c,5)/5、只有它覆蓋的 min(c,3)/3
token（範圍內每顆 sensor，25 維）＝ searcher 10 ‖ good 10 ‖ 剩餘電量 E/E0、離 BS 距離/最大 ‖
    差異旗標 3（開關不同、兩邊都開但等級不同、優先值基因不同）
範圍外影響 4：範圍外有幾顆的路由經過範圍內（S、G，÷N）；範圍內負責的 target 在範圍外沒有備援的比例（S、G）
位置 4：起點/N、終點/N、長度/N、是否跨段
全域 10：搜尋進度、searcher 停滯/20、活著 sensor 平均電量、fitness 差距排名（64 組中）、
    S 覆蓋率、G 覆蓋率、(S 最短壽命 − G 最短壽命)/較大者、S 開啟數/N、G 開啟數/N、兩解基因不同的比例

策略 policy：
    "random"：4 選 1 均勻（等同 ring_se）
    "oracle"：作弊版，偷看 4 個候選只交換的結果（不計預算、不更新最佳），選 Δ 最大、同分比 f(child2)
    "bottleneck"：瓶頸規則（不學習），選「有效差異 sensor 在 S、G 中緊迫度總和」最大的候選，同分隨機
    callable(batch) -> actions：batch 是 build_batch() 的輸出（給網路用）
recorder 不是 None 時（只在作弊版），每回合存下該回合的 per-sensor 特徵與每組交易的標籤，供監督式學習。
交換、突變、EN 更新都和 Ring_SE 相同。
"""
from __future__ import annotations

import numpy as np

from Algorithm.se.market_components import accept_good_children, accept_searcher_children
from Algorithm.se.ring_se.Ring_SE import Ring_SE
from Algorithm.se.ring_se.ring_segment_policy import IMPROVEMENT_EPSILON, PairSegmentMarket
from State.Encoding import swap_segment

H = 4
LMAX = 50
PARENT_DIM = 10
SENSOR_DIM = 2
DIFF_DIM = 3
TOKEN_DIM = 2 * PARENT_DIM + SENSOR_DIM + DIFF_DIM   # 25
OUTSIDE_DIM, POSITION_DIM, GLOBAL_DIM = 4, 4, 10
TIGHT = 7  # PARENT_FEATURES 裡緊迫度的欄位


def build_tokens(s_feat, g_feat, sensor, s_genes, g_genes, left, right):
    """範圍 [left, right) 內每顆 sensor 的 token（L, 25）。"""
    sl = slice(left, right)
    S, G = s_feat[sl], g_feat[sl]
    on_differ = S[:, 0] != G[:, 0]
    level_differ = (S[:, 0] > 0) & (G[:, 0] > 0) & (S[:, 1] != G[:, 1])
    prio_differ = s_genes[sl, 1] != g_genes[sl, 1]
    diff = np.stack([on_differ, level_differ, prio_differ], axis=1).astype(np.float32)
    return np.concatenate([S, G, sensor[sl], diff], axis=1)


def effective_mask(s_feat, g_feat, s_genes, g_genes, left, right):
    """有效差異：基因不同、且任一方解碼後有開。"""
    sl = slice(left, right)
    differ = np.any(s_genes[sl] != g_genes[sl], axis=1)
    return differ & ((s_feat[sl, 0] > 0) | (g_feat[sl, 0] > 0))


class Ring_SE_E03(PairSegmentMarket, Ring_SE):
    NEEDS_ENERGY_FEATURES = False

    def __init__(self, problem, policy="random", recorder=None, **kwargs):
        super().__init__(problem, **kwargs)
        self.policy = policy
        self.recorder = recorder
        self.round_stats = []

    def initialize_market(self, problem, initial_state=None):
        super().initialize_market(problem, initial_state)
        self._sol_cache = {}
        self.round_stats = []

    # ---- 一個解解碼後的每顆 sensor 資訊（快取）--------------------------------
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
        cost = np.asarray(problem.calculate_total_cost(state, sensing_radii=radii), dtype=float)[:n]
        cost = np.where(np.isfinite(cost), cost, 0.0)
        energy = np.maximum(np.asarray(problem.energy[:n], dtype=float), 0.0)
        spending = active & (cost > 1e-15)
        life = np.full(n, np.inf)
        life[spending] = energy[spending] / cost[spending]
        min_life = float(life[spending].min()) if np.any(spending) else np.inf
        tight = np.zeros(n)
        if np.any(spending):
            tight[spending] = min_life / np.maximum(life[spending], 1e-15)
        # 路由：depends[k, j] = j 在 k 往 BS 的路徑上（j ≠ k）
        depends = np.zeros((n, n), dtype=bool)
        hops = np.zeros(n)
        paths = getattr(state, "paths", {}) or {}
        for k, path in paths.items():
            k = int(k)
            if k >= n:
                continue
            nodes = [int(x) for x in path[1:] if int(x) < n]
            depends[k, nodes] = True
            hops[k] = len(path) - 1
        subtree = depends.sum(axis=0)
        tx = np.asarray(state.tx_load, dtype=float)[:n]
        coverage = np.asarray(problem.coverage_table) > 0
        cov = coverage[:, np.arange(n), levels] & active[None, :]   # (T, N)
        counts = cov.sum(axis=1)
        sole = (cov & (counts[:, None] == 1)).sum(axis=0)
        feat = np.zeros((n, PARENT_DIM), dtype=np.float32)
        feat[:, 0] = active
        feat[:, 1] = levels / 5.0
        feat[:, 2] = subtree > 0
        feat[:, 3] = tx / max(float(tx.max()), 1.0)
        feat[:, 4] = subtree / n
        feat[:, 5] = hops / max(float(hops.max()), 1.0)
        feat[:, 6] = cost / max(float(cost.max()), 1e-15)
        feat[:, 7] = tight
        feat[:, 8] = np.minimum(cov.sum(axis=0), 5) / 5.0
        feat[:, 9] = np.minimum(sole, 3) / 3.0
        info = {"feat": feat, "genes": np.stack([code[0::2], code[1::2] % 10], axis=1).astype(np.int16),
                "depends": depends, "cov": cov, "min_life": min_life,
                "coverage": float((counts > 0).mean()), "n_active": int(active.sum())}
        self._sol_cache[key] = info
        return info

    # ---- 一回合的特徵 ---------------------------------------------------------
    def _round_inputs(self, problem, pairs):
        n = problem.SENSOR_NUMBER
        energy = np.maximum(np.asarray(problem.energy[:n], dtype=float), 0.0)
        e_ratio = energy / max(float(problem.initial_energy), 1e-15)
        dist = np.asarray(problem.distances)[:n, problem.BSID]
        sensor = np.stack([e_ratio, dist / max(float(dist.max()), 1e-15)], axis=1).astype(np.float32)
        alive = energy >= problem.liveJ
        mean_alive = float(e_ratio[alive].mean()) if np.any(alive) else 0.0
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        gaps = np.array([self.goods_fitness[g] - self.searcher_fitness[s] for s, g, _, _ in pairs])
        gap_rank = np.argsort(np.argsort(gaps, kind="stable"), kind="stable") / max(1, len(pairs) - 1)
        S = [self._solution(problem, x) for x in self.searchers]
        G = [self._solution(problem, x) for x in self.goods]

        spans = np.zeros((len(pairs), H, 3), dtype=np.int16)
        outside = np.zeros((len(pairs), H, OUTSIDE_DIM), dtype=np.float32)
        position = np.zeros((len(pairs), H, POSITION_DIM), dtype=np.float32)
        glob = np.zeros((len(pairs), GLOBAL_DIM), dtype=np.float32)
        for i, (s_id, g_id, cand_spans, _) in enumerate(pairs):
            si, gi = S[s_id], G[g_id]
            for k, (mode, left, right) in enumerate(cand_spans):
                spans[i, k] = (mode == "cross_ring", left, right)
                inside = np.zeros(n, dtype=bool)
                inside[left:right] = True
                for col, sol in ((0, si), (1, gi)):
                    through = sol["depends"][:, inside].any(axis=1) & ~inside & (sol["feat"][:, 0] > 0)
                    outside[i, k, col] = through.sum() / n
                    cov_in = sol["cov"][:, inside].any(axis=1)
                    backup = sol["cov"][:, ~inside].any(axis=1)
                    outside[i, k, 2 + col] = (cov_in & ~backup).sum() / max(1, int(cov_in.sum()))
                position[i, k] = (left / n, right / n, (right - left) / n, float(mode == "cross_ring"))
            lives = (si["min_life"], gi["min_life"])
            finite = [x for x in lives if np.isfinite(x)]
            life_gap = ((lives[0] - lives[1]) / max(finite)) if len(finite) == 2 and max(finite) > 0 else 0.0
            glob[i] = (progress, min(int(self.searcher_stagnation[s_id]), 20) / 20.0, mean_alive, gap_rank[i],
                       si["coverage"], gi["coverage"], np.clip(life_gap, -1, 1), si["n_active"] / n,
                       gi["n_active"] / n, float(np.any(si["genes"] != gi["genes"], axis=1).mean()))
        return {"sensor": sensor, "S": S, "G": G, "spans": spans, "outside": outside,
                "position": position, "global": glob}

    def build_batch(self, inputs, pairs):
        """給網路：tokens (P, H, LMAX, 25)、mask (P, H, LMAX)、extra (P, H, 4+4+10)。"""
        P = len(pairs)
        tokens = np.zeros((P, H, LMAX, TOKEN_DIM), dtype=np.float32)
        mask = np.zeros((P, H, LMAX), dtype=bool)
        for i, (s_id, g_id, _, _) in enumerate(pairs):
            si, gi = inputs["S"][s_id], inputs["G"][g_id]
            for k in range(H):
                _, left, right = inputs["spans"][i, k]
                tok = build_tokens(si["feat"], gi["feat"], inputs["sensor"], si["genes"], gi["genes"], left, right)
                L = min(len(tok), LMAX)
                tokens[i, k, :L] = tok[:L]
                mask[i, k, :L] = True
        extra = np.concatenate([inputs["outside"], inputs["position"],
                                np.repeat(inputs["global"][:, None, :], H, axis=1)], axis=2)
        return {"tokens": tokens, "mask": mask, "extra": extra}

    def _bottleneck_choices(self, inputs, pairs):
        choices = []
        for i, (s_id, g_id, _, _) in enumerate(pairs):
            si, gi = inputs["S"][s_id], inputs["G"][g_id]
            scores = []
            for k in range(H):
                _, left, right = inputs["spans"][i, k]
                eff = effective_mask(si["feat"], gi["feat"], si["genes"], gi["genes"], left, right)
                scores.append(float((si["feat"][left:right, TIGHT] + gi["feat"][left:right, TIGHT])[eff].sum()))
            scores = np.asarray(scores)
            best = np.flatnonzero(scores >= scores.max() - 1e-12)
            choices.append(int(best[self.random.randrange(len(best))]))
        return np.asarray(choices, dtype=int)

    def _peek(self, pairs):
        rows, crossed = [], []
        for s_id, g_id, spans, _ in pairs:
            row, kids = [], []
            for span in spans:
                c1, c2 = swap_segment(self.searchers[s_id], self.goods[g_id], span[1] * 2, span[2] * 2)
                row += [c1, c2]
                kids.append((c1, c2))
            rows.append(row)
            crossed.append(kids)
        fit = np.array([[float(np.sum(o)) for _, _, o in res] for res in self._route_to_pool(rows)],
                       dtype=float).reshape(len(pairs), H, 2)
        return fit, crossed

    # ---- 一回合 -----------------------------------------------------------------
    def vision_search(self, problem):
        goods_before = self.goods_fitness.copy()
        searcher_before = self.searcher_fitness.copy()
        context = self._round_context(problem)
        pairs = self._build_pairs(context)
        inputs = self._round_inputs(problem, pairs)

        crossed = None
        if self.policy == "random":
            actions = np.array([self.random.randrange(H) for _ in pairs], dtype=int)
        elif self.policy == "bottleneck":
            actions = self._bottleneck_choices(inputs, pairs)
        elif self.policy == "oracle":
            fit, crossed = self._peek(pairs)
            delta = (np.maximum(0.0, fit[:, :, 0] - searcher_before[[p[0] for p in pairs]][:, None])
                     + np.maximum(0.0, fit[:, :, 1] - goods_before[[p[1] for p in pairs]][:, None]))
            delta = np.where(delta > IMPROVEMENT_EPSILON, delta, 0.0)
            actions = np.array([int(np.lexsort((fit[i, :, 1], delta[i]))[-1]) for i in range(len(pairs))])
            if self.recorder is not None:
                self.recorder.add_round(inputs, pairs, delta, fit[:, :, 1], actions)
        else:
            actions = np.asarray(self.policy(self.build_batch(inputs, pairs)), dtype=int)

        children = [[] for _ in range(self.n)]
        for i, ((s_id, g_id, spans, _), k) in enumerate(zip(pairs, actions)):
            span = spans[int(k)]
            if crossed is not None:
                kids = [c.copy() for c in crossed[i][int(k)]]
            else:
                kids = list(swap_segment(self.searchers[s_id], self.goods[g_id], span[1] * 2, span[2] * 2))
            for child in kids:
                if self.random.random() < self.mutation_rate:
                    self.mutate_candidate(problem, child, sensor_span=span)
            children[s_id] += kids

        child_fitness = self.evaluate_investments(problem, children)
        f1 = child_fitness[:, 0::2]
        f2 = child_fitness[:, 1::2]
        improved_pairs = sum(int(f1[s, g] > searcher_before[s] + IMPROVEMENT_EPSILON
                                 or f2[s, g] > goods_before[g] + IMPROVEMENT_EPSILON) for s, g, _, _ in pairs)
        accept_searcher_children(self, children, f1, searcher_before)
        accept_good_children(self.goods, self.goods_fitness, goods_before, children,
                             f2, range(self.n), self.GOOD_ACCEPT_TOLERANCE)
        improved = self.searcher_fitness > searcher_before + IMPROVEMENT_EPSILON
        self.searcher_stagnation = np.where(improved, 0, self.searcher_stagnation + 1)
        self.round_stats.append({"improve_rate": improved_pairs / len(pairs),
                                 "segment_counts": np.bincount(actions, minlength=H).tolist()})


class RoundRecorder:
    """作弊版每回合：per-sensor 特徵（searcher 8、good 8、sensor）＋每組交易的標籤與候選資訊。"""

    def __init__(self, map_name=""):
        self.map_name = map_name
        self.rounds = []

    def add_round(self, inputs, pairs, delta, f2, actions):
        self.rounds.append({
            "S": np.stack([x["feat"] for x in inputs["S"]]), "G": np.stack([x["feat"] for x in inputs["G"]]),
            "S_genes": np.stack([x["genes"] for x in inputs["S"]]), "G_genes": np.stack([x["genes"] for x in inputs["G"]]),
            "sensor": inputs["sensor"], "s_id": np.array([p[0] for p in pairs], dtype=np.int8),
            "g_id": np.array([p[1] for p in pairs], dtype=np.int8), "spans": inputs["spans"],
            "outside": inputs["outside"], "position": inputs["position"], "global": inputs["global"],
            "delta": delta, "f2": f2, "choice": np.asarray(actions, dtype=np.int8), "map": self.map_name})


__all__ = ["Ring_SE_E03", "RoundRecorder", "build_tokens", "effective_mask",
           "H", "LMAX", "TOKEN_DIM", "OUTSIDE_DIM", "POSITION_DIM", "GLOBAL_DIM", "TIGHT"]
