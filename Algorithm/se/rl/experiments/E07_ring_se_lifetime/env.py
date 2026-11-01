"""E07 環境（監督式用）：原版 ring_se，每組交易照常在 4 段各抽一個 80/20 候選範圍，由 policy 選 1 個。

policy："random"（4 選 1 均勻 ＝ 原版 ring_se；蒐集資料用）、callable(round_data) → actions（網路；評估用）
    或 "oracle"（作弊版天花板：偷看 4 個候選只交換的子代，選 oracle_choice；偷看不計預算）、
    "oracle_diff"（作弊版，跟 diff_cur／diff_prev 標籤一致：先照原版隨機選，其他交易固定為這組選擇算差異獎勵，
    每組選最大、全為 0 留隨機的）、"oracle_plain"（跟 plain_delta 標籤一致：選一般 Δ 最大、全為 0 留隨機的）。
交換、突變（只在選到的範圍內）、EN 更新、評估計預算都和 Ring_SE 相同。

recorder 不是 None 時（蒐集資料），每組交易另外偷看 4 個候選「只交換」的結果（不計預算、不更新最佳、不用演算法亂數
→ 推進和沒標時完全相同），每回合存一筆（欄位說明也見 results/data/README.md）：
    網路輸入（_round_data）：S、G (8, N, 10)、S_prio、G_prio、S_par、G_par (8, N)、sensor (N, 2)、
        s_id、g_id (P,)、span (P, 4, 2)、cand (P, 4, 10)、glob (P, 10)
    diff_cur      (P, 4)  差異獎勵 ÷ 這一回合的尺度（監督式主要標籤）
    diff_prev     (P, 4)  差異獎勵 ÷ 上一回合的尺度（每段第 1 回合用自己的）
    oracle_choice (P,)    作弊版會選的候選：一般 Δ 最大、同分比 f(child2)
    plain_delta   (P, 4)  一般 Δ ＝ max(0, f(c1) − f(S)) + max(0, f(c2) − f(G))，不扣其他交易
    child1_fit、child2_fit (P, 4)、searcher_fit、good_fit (8,)  原始 fitness（可事後算其他「只看交換」的標籤）
    scale_cur、scale_prev (P,)  兩種尺度；action、reward (P,)  原版實際選的候選、它實際子代的差異獎勵（參考用）
差異獎勵（其他交易固定為實際的選擇）：EN 下 searcher 只收 8 個 child1 中最好的、good 一律換成 8 個 child2 中最好的，
    r1 = max(0, f(c1) − max(f(S), 同 S 其他 7 組 child1 最好))、r2 = max(0, f(c2) − 同 G 其他 7 位 child2 最好)，
    標籤 = min((r1 + r2) ÷ 尺度, 5)，≤ 1e-12 → 0。
尺度 ＝ 實際子代（含突變）一般 Δ 的正值中位數：這一回合的 → scale_cur（這回合沒有正值時用 scale_prev）；
    上一回合的 → scale_prev（每段第 1 回合用這一回合的；都沒有時用實際子代差異獎勵的正值中位數，再沒有用 1）。

10/22 由 E03、E06 的環境合併重寫，只留監督式用到的部分。
"""
from __future__ import annotations

import numpy as np

from Algorithm.se.ring_se.ring_segment_policy import IMPROVEMENT_EPSILON, PairSegmentMarket
from State.Encoding import swap_segment

H = 4                 # 每組交易的候選數
FEAT_DIM = 10         # 每個解每顆 sensor 的解碼特徵
TOKEN_DIM = 25        # 網路的 sensor token：S 10 ‖ G 10 ‖ sensor 2 ‖ 差異旗標 3（model.make_tokens 組）
CAND_DIM = 10         # 候選特徵：位置 4 ‖ 範圍外影響 4 ‖ 瓶頸 S、G
GLOBAL_DIM = 10
TIGHT = 7             # 解碼特徵裡「緊迫度」的欄位
REWARD_CAP = 5.0
ROUND_KEYS = ("S", "G", "S_prio", "G_prio", "S_par", "G_par", "sensor")
PAIR_KEYS = ("s_id", "g_id", "span", "cand", "glob")


def effective_mask(s_feat, g_feat, s_genes, g_genes, left, right):
    """範圍 [left, right) 內的有效差異：基因不同、且任一方解碼後有開。"""
    sl = slice(left, right)
    differ = np.any(s_genes[sl] != g_genes[sl], axis=1)
    return differ & ((s_feat[sl, 0] > 0) | (g_feat[sl, 0] > 0))


def others_max(matrix, axis):
    """每個元素「同一列（axis=1）或同一欄（axis=0）其他元素」的最大值。"""
    m = np.moveaxis(np.asarray(matrix, dtype=float), axis, -1)
    order = np.argsort(m, axis=-1)
    best = np.take_along_axis(m, order[..., -1:], axis=-1)
    second = np.take_along_axis(m, order[..., -2:-1], axis=-1)
    is_best = np.arange(m.shape[-1]) == order[..., -1:]
    return np.moveaxis(np.where(is_best, second, best), -1, axis)


def positive_median(x):
    x = np.asarray(x, dtype=float).ravel()
    x = x[x > IMPROVEMENT_EPSILON]
    return float(np.median(x)) if len(x) else None


def scaled(raw, scale):
    raw = np.where(raw > IMPROVEMENT_EPSILON, raw, 0.0)
    return np.minimum(raw / scale, REWARD_CAP).astype(np.float32)


def plain_delta(fit, searcher_before, goods_before, data):
    """偷看結果 fit (P, 4, 2) → 一般 Δ (P, 4) ＝ max(0, f(c1) − f(S)) + max(0, f(c2) − f(G))，≤ 1e-12 → 0。"""
    s_idx, g_idx = data["s_id"].astype(int), data["g_id"].astype(int)
    plain = (np.maximum(0.0, fit[:, :, 0] - searcher_before[s_idx][:, None])
             + np.maximum(0.0, fit[:, :, 1] - goods_before[g_idx][:, None]))
    return np.where(plain > IMPROVEMENT_EPSILON, plain, 0.0)


def oracle_choice(plain, f2):
    """一般 Δ 最大、同分比 f(child2)。plain、f2 (P, 4) → (P,)。"""
    return np.array([int(np.lexsort((f2[i], plain[i]))[-1]) for i in range(len(plain))], dtype=np.int8)


class Ring_SE_E07(PairSegmentMarket):
    NEEDS_ENERGY_FEATURES = False

    def __init__(self, problem, policy="random", recorder=None, **kwargs):
        super().__init__(problem, **kwargs)
        self.policy = policy
        self.recorder = recorder

    def initialize_market(self, problem, initial_state=None):
        super().initialize_market(problem, initial_state)
        self._sol_cache = {}
        self.prev_scale = None

    # ---- 一個解解碼後的每顆 sensor 資訊（快取）--------------------------------
    def _solution(self, problem, candidate):
        """feat (N, 10)：開關、等級/5、是否轉送節點、轉送量/全網最大、子樹大小/N、跳數/最大跳數、耗能/全網最大、
        緊迫度（全網最短壽命 ÷ 這顆壽命）、覆蓋 target 數 min(c,5)/5、只有它覆蓋的 min(c,3)/3；
        另有基因、路由依賴、覆蓋表、父節點等，給候選特徵用。"""
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
        subtree = depends.sum(axis=0)
        tx = np.asarray(state.tx_load, dtype=float)[:n]
        coverage = np.asarray(problem.coverage_table) > 0
        cov = coverage[:, np.arange(n), levels] & active[None, :]   # (T, N)
        counts = cov.sum(axis=1)
        sole = (cov & (counts[:, None] == 1)).sum(axis=0)
        feat = np.zeros((n, FEAT_DIM), dtype=np.float32)
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
                "depends": depends, "cov": cov, "parent": parent, "min_life": min_life,
                "coverage": float((counts > 0).mean()), "n_active": int(active.sum())}
        self._sol_cache[key] = info
        return info

    # ---- 一回合的網路輸入 -------------------------------------------------------
    def _round_data(self, problem, pairs):
        """回合層級（8 個 searcher、8 個 goods 共用）＋交易層級（64 組）。
        cand：位置 4（起點/N、終點/N、長度/N、是否跨段）‖ 範圍外影響 4（範圍外有幾顆的路由經過範圍內 S、G ÷N；
        範圍內負責的 target 在範圍外沒有備援的比例 S、G）‖ 瓶頸 S、G（範圍內有效差異 sensor 的緊迫度總和）。
        glob：搜尋進度、searcher 停滯/20、活著 sensor 平均電量、fitness 差距排名、S／G 覆蓋率、
        (S 最短壽命 − G 最短壽命)/較大者、S／G 開啟數/N、兩解基因不同的比例。"""
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

        P = len(pairs)
        span = np.zeros((P, H, 2), dtype=np.int16)
        cand = np.zeros((P, H, CAND_DIM), dtype=np.float32)
        glob = np.zeros((P, GLOBAL_DIM), dtype=np.float32)
        for i, (s_id, g_id, cand_spans, _) in enumerate(pairs):
            si, gi = S[s_id], G[g_id]
            for k, (mode, left, right) in enumerate(cand_spans):
                span[i, k] = (left, right)
                inside = np.zeros(n, dtype=bool)
                inside[left:right] = True
                cand[i, k, :4] = (left / n, right / n, (right - left) / n, float(mode == "cross_ring"))
                for col, sol in ((0, si), (1, gi)):
                    through = sol["depends"][:, inside].any(axis=1) & ~inside & (sol["feat"][:, 0] > 0)
                    cand[i, k, 4 + col] = through.sum() / n
                    cov_in = sol["cov"][:, inside].any(axis=1)
                    backup = sol["cov"][:, ~inside].any(axis=1)
                    cand[i, k, 6 + col] = (cov_in & ~backup).sum() / max(1, int(cov_in.sum()))
                eff = effective_mask(si["feat"], gi["feat"], si["genes"], gi["genes"], left, right)
                cand[i, k, 8] = si["feat"][left:right, TIGHT][eff].sum()
                cand[i, k, 9] = gi["feat"][left:right, TIGHT][eff].sum()
            lives = (si["min_life"], gi["min_life"])
            finite = [x for x in lives if np.isfinite(x)]
            life_gap = ((lives[0] - lives[1]) / max(finite)) if len(finite) == 2 and max(finite) > 0 else 0.0
            glob[i] = (progress, min(int(self.searcher_stagnation[s_id]), 20) / 20.0, mean_alive, gap_rank[i],
                       si["coverage"], gi["coverage"], np.clip(life_gap, -1, 1), si["n_active"] / n,
                       gi["n_active"] / n, float(np.any(si["genes"] != gi["genes"], axis=1).mean()))
        return {
            "S": np.stack([x["feat"] for x in S]), "G": np.stack([x["feat"] for x in G]),
            "S_prio": np.stack([x["genes"][:, 1] for x in S]).astype(np.int8),
            "G_prio": np.stack([x["genes"][:, 1] for x in G]).astype(np.int8),
            "S_par": np.stack([x["parent"] for x in S]), "G_par": np.stack([x["parent"] for x in G]),
            "sensor": sensor,
            "s_id": np.array([p[0] for p in pairs], dtype=np.int8),
            "g_id": np.array([p[1] for p in pairs], dtype=np.int8),
            "span": span, "cand": cand, "glob": glob,
        }

    # ---- 作弊版偷看：4 個候選只交換的子代 fitness（不計預算、不更新最佳）--------------------------
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
        fit = np.array([[float(np.sum(o)) for _, _, o in res] for res in self.evaluator.route(rows)],
                       dtype=float).reshape(len(pairs), H, 2)
        return fit, crossed

    def _labels(self, fit, actions, data, searcher_before, goods_before):
        """偷看結果 → 差異獎勵（尚未除尺度）、一般 Δ。其他交易固定為實際的選擇。"""
        s_idx, g_idx = data["s_id"].astype(int), data["g_id"].astype(int)
        chosen = fit[np.arange(len(actions)), actions]            # (P, 2) 實際選的候選
        F1, F2 = np.zeros((self.n, self.w)), np.zeros((self.n, self.w))
        F1[s_idx, g_idx] = chosen[:, 0]
        F2[s_idx, g_idx] = chosen[:, 1]
        base1 = np.maximum(searcher_before[:, None], others_max(F1, axis=1))[s_idx, g_idx]
        base2 = others_max(F2, axis=0)[s_idx, g_idx]
        diff = np.maximum(0.0, fit[:, :, 0] - base1[:, None]) + np.maximum(0.0, fit[:, :, 1] - base2[:, None])
        return scaled(diff, 1.0).astype(float), plain_delta(fit, searcher_before, goods_before, data)

    # ---- 一回合 -----------------------------------------------------------------
    def vision_search(self, problem):
        goods_before = self.goods_fitness.copy()
        searcher_before = self.searcher_fitness.copy()
        pairs = self._build_pairs(self._round_context(problem))
        data = self._round_data(problem, pairs)
        peek = self.recorder is not None or self.policy in ("oracle", "oracle_diff", "oracle_plain")
        fit, crossed = self._peek(pairs) if peek else (None, None)
        if self.policy == "random":
            actions = np.array([self.random.randrange(H) for _ in pairs], dtype=int)
        elif self.policy == "oracle":
            plain = plain_delta(fit, searcher_before, goods_before, data)
            actions = oracle_choice(plain, fit[:, :, 1]).astype(int)
        elif self.policy == "oracle_diff":
            # 和收資料時一樣算標籤：先照原版隨機選，其他交易固定為這組選擇；每組選差異獎勵最大，全為 0 就留隨機的
            drawn = np.array([self.random.randrange(H) for _ in pairs], dtype=int)
            raw, _ = self._labels(fit, drawn, data, searcher_before, goods_before)
            actions = np.where(raw.max(axis=1) > 0, raw.argmax(axis=1), drawn)
        elif self.policy == "oracle_plain":
            # 跟 plain_delta 標籤一致：只看這筆交易，選一般 Δ 最大；全為 0 就留原版隨機的
            drawn = np.array([self.random.randrange(H) for _ in pairs], dtype=int)
            plain = plain_delta(fit, searcher_before, goods_before, data)
            actions = np.where(plain.max(axis=1) > 0, plain.argmax(axis=1), drawn)
        else:
            actions = np.asarray(self.policy(data), dtype=int)

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
        f1, f2 = child_fitness[:, 0::2], child_fitness[:, 1::2]

        # 尺度：實際子代一般 Δ 的正值中位數（這一回合 current；上一回合 prev_scale）
        s_idx, g_idx = data["s_id"].astype(int), data["g_id"].astype(int)
        actual_plain = (np.maximum(0.0, f1 - searcher_before[:, None])
                        + np.maximum(0.0, f2 - goods_before[None, :]))[s_idx, g_idx]
        base1 = np.maximum(searcher_before[:, None], others_max(f1, axis=1))
        actual_diff = (np.maximum(0.0, f1 - base1) + np.maximum(0.0, f2 - others_max(f2, axis=0)))[s_idx, g_idx]
        current = positive_median(actual_plain)
        scale_prev = self.prev_scale or current or positive_median(actual_diff) or 1.0
        scale_cur = current if current is not None else scale_prev
        if current is not None:
            self.prev_scale = current

        if self.recorder is not None:
            raw, plain = self._labels(fit, actions, data, searcher_before, goods_before)
            P = len(pairs)
            self.recorder.add_round({
                **{k: data[k] for k in ROUND_KEYS + PAIR_KEYS},
                "action": actions.astype(np.int8), "reward": scaled(actual_diff, scale_prev),
                "diff_prev": np.minimum(raw / scale_prev, REWARD_CAP).astype(np.float32),
                "diff_cur": np.minimum(raw / scale_cur, REWARD_CAP).astype(np.float32),
                "plain_delta": plain.astype(np.float32),
                "oracle_choice": oracle_choice(plain, fit[:, :, 1]),
                "child1_fit": fit[:, :, 0].astype(np.float64), "child2_fit": fit[:, :, 1].astype(np.float64),
                "searcher_fit": searcher_before.astype(np.float64), "good_fit": goods_before.astype(np.float64),
                "scale_prev": np.full(P, scale_prev, dtype=np.float32),
                "scale_cur": np.full(P, scale_cur, dtype=np.float32)})

        self.accept_searcher_children(children, f1, searcher_before)
        self.accept_good_children(self.goods, self.goods_fitness, goods_before, children, f2, range(self.n))
        improved = self.searcher_fitness > searcher_before + IMPROVEMENT_EPSILON
        self.searcher_stagnation = np.where(improved, 0, self.searcher_stagnation + 1)


class RoundRecorder:
    """一段最佳化的每回合資料；stacked() → 回合層級 (R, ...)、交易層級 (R, P, ...)，done (R,) 最後一回合為 1。"""

    def __init__(self, map_name=""):
        self.map_name = map_name
        self.rounds = []

    def add_round(self, record):
        self.rounds.append(record)

    def stacked(self):
        out = {k: np.stack([r[k] for r in self.rounds]) for k in self.rounds[0]}
        out["done"] = np.zeros(len(self.rounds), dtype=np.float32)
        out["done"][-1] = 1.0
        return out


__all__ = ["Ring_SE_E07", "RoundRecorder", "H", "FEAT_DIM", "TOKEN_DIM", "CAND_DIM", "GLOBAL_DIM"]
