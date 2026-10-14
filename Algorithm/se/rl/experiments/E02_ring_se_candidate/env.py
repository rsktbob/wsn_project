"""E02 環境：Ring_SE 每組交易照常產生 4 個候選範圍（4 段各抽一個，80/20 規則），
由外部策略從 4 個候選中選一個；每個候選用自己的 25 維特徵描述（以範圍為單位，不是以段號）。

每個候選 25 維：
    位置 4         起點/N、終點/N、長度/N、是否跨段
    交換會改變什麼 9  有效差異數分格 one-hot 4、無效差異 min(x,5)/5、
                     會新開（searcher 關、good 開）、會關掉（searcher 開、good 關）、
                     會改等級（兩邊都開、等級不同）、只改路由優先值（兩邊都開、等級同、優先值不同），各 min(x,10)/10
    範圍內的現況 7    searcher 開啟比例、good 開啟比例、searcher 開啟 sensor 的最低／平均電量（E/E0）、
                     瓶頸 sensor 數/5、耗能占比、單一覆蓋 target 數 ÷ 開啟數
    記憶 1           本 episode 內，和這個範圍重疊的 sensor 最近交換成功的比例（每回合衰減 0.9，沒紀錄 0.5）
    全域 4           搜尋進度、searcher 停滯/20、fitness 差距排名、活著 sensor 平均電量

策略 policy：
    "random"：4 選 1 均勻（等同 ring_se）
    "oracle"：作弊版，偷看 4 個候選只交換的結果（不計預算、不更新最佳），選 Δ 最大、同分比 f(child2)
    callable(features (P, 4, 25)) -> actions (P,)
``label_sink`` 不是 None 時（只在作弊版），每組交易存下 (features, 4 個候選的 Δ, 作弊版的選擇, f(child2))。
交換、突變、EN 更新都和 Ring_SE 相同。
"""
from __future__ import annotations

import numpy as np

from Algorithm.se.market_components import accept_good_children, accept_searcher_children
from Algorithm.se.ring_se.Ring_SE import Ring_SE
from Algorithm.se.ring_se.ring_segment_policy import IMPROVEMENT_EPSILON, PairSegmentMarket
from State.Encoding import swap_segment

H = 4
CAND_DIM = 25
BOTTLENECK_COUNT = 5
MEMORY_DECAY = 0.9

FEATURE_GROUPS = {
    "position": list(range(0, 4)),
    "change": list(range(4, 13)),
    "context": list(range(13, 20)),
    "memory": [20],
    "global": list(range(21, 25)),
}


class Ring_SE_E02(PairSegmentMarket, Ring_SE):
    NEEDS_ENERGY_FEATURES = False

    def __init__(self, problem, policy="random", label_sink=None, **kwargs):
        super().__init__(problem, **kwargs)
        self.policy = policy
        self.label_sink = label_sink
        self.round_stats = []

    def initialize_market(self, problem, initial_state=None):
        super().initialize_market(problem, initial_state)
        self._info_cache = {}
        n = problem.SENSOR_NUMBER
        self._mem_success = np.zeros(n)
        self._mem_tries = np.zeros(n)
        self.round_stats = []

    # ---- 一條染色體的解碼資訊（快取） ----------------------------------------
    def _info(self, problem, candidate):
        key = np.asarray(candidate.code).tobytes()
        info = self._info_cache.get(key)
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
        bottleneck = np.zeros(n, dtype=bool)
        spending = np.flatnonzero(active & (cost > 1e-15))
        if len(spending):
            order = spending[np.argsort(energy[spending] / cost[spending], kind="stable")]
            bottleneck[order[:BOTTLENECK_COUNT]] = True
        coverage = np.asarray(problem.coverage_table) > 0
        on = np.flatnonzero(active)
        covered = coverage[:, on, levels[on]]
        counts = covered.sum(axis=1)
        single = np.zeros(n)
        single[on] = (covered & (counts[:, None] == 1)).sum(axis=0)
        info = {"active": active, "levels": levels, "priority": code[1::2] % 10, "cost": cost,
                "bottleneck": bottleneck, "single": single, "total_cost": max(float(cost.sum()), 1e-15)}
        self._info_cache[key] = info
        return info

    # ---- 特徵 -----------------------------------------------------------------
    def _features(self, problem, context, pairs):
        n = problem.SENSOR_NUMBER
        energy_ratio = np.maximum(np.asarray(problem.energy[:n], dtype=float), 0.0) / max(
            float(problem.initial_energy), 1e-15)
        alive = np.asarray(problem.energy[:n], dtype=float) >= problem.liveJ
        mean_alive = float(energy_ratio[alive].mean()) if np.any(alive) else 0.0
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        memory = np.where(self._mem_tries > 1e-9, self._mem_success / np.maximum(self._mem_tries, 1e-9), 0.5)
        gaps = np.array([self.goods_fitness[g] - self.searcher_fitness[s] for s, g, _, _ in pairs])
        gap_rank = np.argsort(np.argsort(gaps, kind="stable"), kind="stable") / max(1, len(pairs) - 1)
        s_infos = [self._info(problem, x) for x in self.searchers]
        g_infos = [self._info(problem, x) for x in self.goods]

        X = np.zeros((len(pairs), H, CAND_DIM), dtype=np.float32)
        for i, (s_id, g_id, spans, stats) in enumerate(pairs):
            si, gi = s_infos[s_id], g_infos[g_id]
            code_s = np.asarray(self.searchers[s_id].code)
            code_g = np.asarray(self.goods[g_id].code)
            prio_differ = code_s[1::2] != code_g[1::2]
            for k, ((mode, left, right), stat) in enumerate(zip(spans, stats)):
                sl = slice(left, right)
                s_on, g_on = si["active"][sl], gi["active"][sl]
                both = s_on & g_on
                same_level = si["levels"][sl] == gi["levels"][sl]
                n_on = int(s_on.sum())
                e_on = energy_ratio[sl][s_on]
                row = X[i, k]
                row[0], row[1], row[2] = left / n, right / n, (right - left) / n
                row[3] = float(mode == "cross_ring")
                eff = stat["effective"]
                row[4 + (0 if eff == 0 else 1 if eff <= 2 else 2 if eff <= 4 else 3)] = 1.0
                row[8] = min(stat["inactive"], 5) / 5.0
                row[9] = min(int((~s_on & g_on).sum()), 10) / 10.0
                row[10] = min(int((s_on & ~g_on).sum()), 10) / 10.0
                row[11] = min(int((both & ~same_level).sum()), 10) / 10.0
                row[12] = min(int((both & same_level & prio_differ[sl]).sum()), 10) / 10.0
                row[13] = s_on.mean() if right > left else 0.0
                row[14] = g_on.mean() if right > left else 0.0
                row[15] = float(e_on.min()) if n_on else 1.0
                row[16] = float(e_on.mean()) if n_on else 1.0
                row[17] = si["bottleneck"][sl].sum() / BOTTLENECK_COUNT
                row[18] = si["cost"][sl].sum() / si["total_cost"]
                row[19] = min(1.0, si["single"][sl].sum() / max(1, n_on))
                row[20] = float(memory[sl].mean()) if right > left else 0.5
                row[21] = progress
                row[22] = min(int(self.searcher_stagnation[s_id]), 20) / 20.0
                row[23] = gap_rank[i]
                row[24] = mean_alive
        return X

    # ---- 作弊版的偷看 ---------------------------------------------------------
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
        X = self._features(problem, context, pairs)

        crossed = None
        if self.policy == "random":
            actions = np.array([self.random.randrange(H) for _ in pairs], dtype=int)
        elif self.policy == "oracle":
            fit, crossed = self._peek(pairs)
            actions = np.empty(len(pairs), dtype=int)
            for i, (s_id, g_id, _, _) in enumerate(pairs):
                f1, f2 = fit[i, :, 0], fit[i, :, 1]
                delta = np.maximum(0.0, f1 - searcher_before[s_id]) + np.maximum(0.0, f2 - goods_before[g_id])
                delta = np.where(delta > IMPROVEMENT_EPSILON, delta, 0.0)
                actions[i] = int(np.lexsort((f2, delta))[-1])
                if self.label_sink is not None:
                    self.label_sink.append((X[i].copy(), delta.astype(np.float64), int(actions[i]), f2.copy()))
        else:
            actions = np.asarray(self.policy(X), dtype=int)

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
        self._mem_success *= MEMORY_DECAY
        self._mem_tries *= MEMORY_DECAY
        improved_pairs = 0
        for (s_id, g_id, spans, _), k in zip(pairs, actions):
            _, left, right = spans[int(k)]
            ok = (f1[s_id, g_id] > searcher_before[s_id] + IMPROVEMENT_EPSILON
                  or f2[s_id, g_id] > goods_before[g_id] + IMPROVEMENT_EPSILON)
            improved_pairs += int(ok)
            self._mem_tries[left:right] += 1.0
            self._mem_success[left:right] += float(ok)

        accept_searcher_children(self, children, f1, searcher_before)
        accept_good_children(self.goods, self.goods_fitness, goods_before, children,
                             f2, range(self.n), self.GOOD_ACCEPT_TOLERANCE)
        improved = self.searcher_fitness > searcher_before + IMPROVEMENT_EPSILON
        self.searcher_stagnation = np.where(improved, 0, self.searcher_stagnation + 1)
        self.round_stats.append({"improve_rate": improved_pairs / len(pairs),
                                 "segment_counts": np.bincount(actions, minlength=H).tolist()})


__all__ = ["Ring_SE_E02", "CAND_DIM", "FEATURE_GROUPS", "H"]
