"""E06 環境：原版 ring_se，每組交易照常在 4 段各抽一個 80/20 候選範圍，由外部策略選 1 個。

每回合輸出一份「回合資料」（網路的輸入，numpy），分兩層：
  回合層級（8 個 searcher、8 個 goods 共用，不隨交易重複）：
    S、G        (8, N, 10)：每個解每顆 sensor 的解碼特徵（同 E03 PARENT_FEATURES）
    S_prio、G_prio (8, N)：路由優先值基因（差異旗標用）
    S_par、G_par   (8, N)：解碼後每顆 sensor 的父節點（BS 或沒開 = −1；注意力偏置用）
    sensor      (N, 2)    ：E/E0、離 BS 距離
  交易層級（P = 64 組，順序固定為 searcher 主序）：
    s_id、g_id (P,)；span (P, 4, 2) 候選範圍 [left, right)；
    cand (P, 4, 10)：位置 4 ‖ 範圍外影響 4（同 E03）‖ 瓶頸分數 S、G（範圍內有效差異 sensor 的緊迫度總和）
    glob (P, 10)：同 E03
sensor token（100 × 25）由網路在 GPU 上組：S 10 ‖ G 10 ‖ sensor 2 ‖ 差異旗標 3（同 E03）。

策略 policy："random"（4 選 1 均勻，等同 ring_se）、"bottleneck"（瓶頸分數 S＋G 最大，同 E03 規則）、
    "oracle"（作弊版：偷看 4 個候選只交換的結果，不計預算、不更新最佳，選 Δ 最大、同分比 f(child2)，再照常突變）、
    或 callable(round_data) -> actions (P,)。

獎勵（差異獎勵，每組交易）：EN 更新時每個 searcher 只收 8 個 child1 中最好的、每個 good 一律換成 8 個 child2 中最好的，
    r1 = max(0, f(c1) − max(f(searcher), 同 searcher 其他 7 組的 child1 最好值))
    r2 = max(0, f(c2) − 同 good 其他 7 位拜訪者的 child2 最好值)
    reward = min((r1 + r2) ÷ scale, 5)，r1 + r2 ≤ 1e-12 → 0。
    scale = 上一回合「實際子代一般 Δ」的正值中位數（行動前就知道；第一回合用本回合的），
    一般 Δ = max(0, f(c1) − f(searcher)) + max(0, f(c2) − f(good))。
作弊版另存每個候選的標籤：同樣的差異獎勵，但用「只交換」的結果，其他交易固定為作弊版的選擇，scale 相同。
交換、突變、EN 更新都和 Ring_SE 相同；評估照常計入預算。
"""
from __future__ import annotations

import numpy as np

from Algorithm.se.ring_se.ring_segment_policy import IMPROVEMENT_EPSILON
from State.Encoding import swap_segment

from common import HERE, load_module

_e03 = load_module("e03_env", HERE.parent / "E03_ring_se_decoded" / "env.py")
effective_mask, TIGHT = _e03.effective_mask, _e03.TIGHT

H = 4
FEAT_DIM = _e03.PARENT_DIM      # 10
TOKEN_DIM = _e03.TOKEN_DIM      # 25
CAND_DIM = 10
GLOBAL_DIM = _e03.GLOBAL_DIM    # 10
REWARD_CAP = 5.0
ROUND_KEYS = ("S", "G", "S_prio", "G_prio", "S_par", "G_par", "sensor")
PAIR_KEYS = ("s_id", "g_id", "span", "cand", "glob")


def others_max(matrix, axis):
    """每個元素「同一列（axis=1）或同一欄（axis=0）其他元素」的最大值。"""
    m = np.moveaxis(np.asarray(matrix, dtype=float), axis, -1)
    order = np.argsort(m, axis=-1)
    best = np.take_along_axis(m, order[..., -1:], axis=-1)
    second = np.take_along_axis(m, order[..., -2:-1], axis=-1)
    is_best = np.arange(m.shape[-1]) == order[..., -1:]
    return np.moveaxis(np.where(is_best, second, best), -1, axis)


def difference_rewards(f1, f2, searcher_before, goods_before):
    """差異獎勵 (n, w)：這組交易讓 searcher／good 的下一個值多了多少（其他交易不變時）。"""
    base1 = np.maximum(searcher_before[:, None], others_max(f1, axis=1))
    r1 = np.maximum(0.0, f1 - base1)
    r2 = np.maximum(0.0, f2 - others_max(f2, axis=0))
    return r1 + r2


def plain_delta(f1, f2, searcher_before, goods_before):
    return np.maximum(0.0, f1 - searcher_before[:, None]) + np.maximum(0.0, f2 - goods_before[None, :])


def positive_median(x):
    x = np.asarray(x, dtype=float).ravel()
    x = x[x > IMPROVEMENT_EPSILON]
    return float(np.median(x)) if len(x) else None


def scaled(raw, scale):
    raw = np.where(raw > IMPROVEMENT_EPSILON, raw, 0.0)
    return np.minimum(raw / scale, REWARD_CAP).astype(np.float32)


class Ring_SE_E06(_e03.Ring_SE_E03):
    """沿用 E03 的解碼快取與特徵；多存父節點，回合資料分回合層級／交易層級。"""

    def initialize_market(self, problem, initial_state=None):
        super().initialize_market(problem, initial_state)
        self.prev_scale = None

    def _solution(self, problem, candidate):
        info = super()._solution(problem, candidate)
        if "parent" not in info:
            n = problem.SENSOR_NUMBER
            parent = np.full(n, -1, dtype=np.int16)
            for k, path in (getattr(candidate.decode(problem), "paths", {}) or {}).items():
                k = int(k)
                if k < n and len(path) > 1 and int(path[1]) < n:
                    parent[k] = int(path[1])
            info["parent"] = parent
        return info

    def _round_data(self, problem, pairs):
        inputs = self._round_inputs(problem, pairs)
        S, G = inputs["S"], inputs["G"]
        spans = inputs["spans"]
        cand = np.zeros((len(pairs), H, CAND_DIM), dtype=np.float32)
        cand[:, :, :4] = inputs["position"]
        cand[:, :, 4:8] = inputs["outside"]
        for i, (s_id, g_id, _, _) in enumerate(pairs):
            si, gi = S[s_id], G[g_id]
            for k in range(H):
                left, right = int(spans[i, k, 1]), int(spans[i, k, 2])
                eff = effective_mask(si["feat"], gi["feat"], si["genes"], gi["genes"], left, right)
                cand[i, k, 8] = si["feat"][left:right, TIGHT][eff].sum()
                cand[i, k, 9] = gi["feat"][left:right, TIGHT][eff].sum()
        return {
            "S": np.stack([x["feat"] for x in S]), "G": np.stack([x["feat"] for x in G]),
            "S_prio": np.stack([x["genes"][:, 1] for x in S]).astype(np.int8),
            "G_prio": np.stack([x["genes"][:, 1] for x in G]).astype(np.int8),
            "S_par": np.stack([x["parent"] for x in S]), "G_par": np.stack([x["parent"] for x in G]),
            "sensor": inputs["sensor"],
            "s_id": np.array([p[0] for p in pairs], dtype=np.int8),
            "g_id": np.array([p[1] for p in pairs], dtype=np.int8),
            "span": spans[:, :, 1:3].astype(np.int16), "cand": cand, "glob": inputs["global"],
        }

    def _choose(self, data, pairs, searcher_before, goods_before):
        """→ actions、（作弊版）偷看的 fit (P, 4, 2) 與交換後的子代。"""
        if self.policy == "oracle":
            fit, crossed = self._peek(pairs)
            plain = (np.maximum(0.0, fit[:, :, 0] - searcher_before[data["s_id"].astype(int)][:, None])
                     + np.maximum(0.0, fit[:, :, 1] - goods_before[data["g_id"].astype(int)][:, None]))
            plain = np.where(plain > IMPROVEMENT_EPSILON, plain, 0.0)
            actions = np.array([int(np.lexsort((fit[i, :, 1], plain[i]))[-1]) for i in range(len(pairs))])
            return actions, fit, crossed
        if self.policy == "random":
            return np.array([self.random.randrange(H) for _ in pairs], dtype=int), None, None
        if self.policy == "bottleneck":
            score = data["cand"][:, :, 8] + data["cand"][:, :, 9]
            actions = []
            for row in score:
                best = np.flatnonzero(row >= row.max() - 1e-12)
                actions.append(int(best[self.random.randrange(len(best))]))
            return np.asarray(actions, dtype=int), None, None
        return np.asarray(self.policy(data), dtype=int), None, None

    def _peek_labels(self, fit, actions, data, searcher_before, goods_before, scale):
        """作弊版標籤 (P, 4)：只交換的差異獎勵，其他交易固定為實際的選擇。"""
        n, w = self.n, self.w
        s_idx, g_idx = data["s_id"].astype(int), data["g_id"].astype(int)
        chosen = fit[np.arange(len(actions)), actions]            # (P, 2)
        F1 = np.zeros((n, w))
        F2 = np.zeros((n, w))
        F1[s_idx, g_idx] = chosen[:, 0]
        F2[s_idx, g_idx] = chosen[:, 1]
        base1 = np.maximum(searcher_before[:, None], others_max(F1, axis=1))[s_idx, g_idx]
        base2 = others_max(F2, axis=0)[s_idx, g_idx]
        raw = np.maximum(0.0, fit[:, :, 0] - base1[:, None]) + np.maximum(0.0, fit[:, :, 1] - base2[:, None])
        return scaled(raw, scale)

    def vision_search(self, problem):
        goods_before = self.goods_fitness.copy()
        searcher_before = self.searcher_fitness.copy()
        context = self._round_context(problem)
        pairs = self._build_pairs(context)
        data = self._round_data(problem, pairs)
        actions, fit, crossed = self._choose(data, pairs, searcher_before, goods_before)

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
        s_idx, g_idx = data["s_id"].astype(int), data["g_id"].astype(int)
        plain = plain_delta(f1, f2, searcher_before, goods_before)[s_idx, g_idx]
        diff = difference_rewards(f1, f2, searcher_before, goods_before)[s_idx, g_idx]
        current = positive_median(plain)
        scale = self.prev_scale or current or positive_median(diff) or 1.0
        rewards = scaled(diff, scale)
        if current is not None:
            self.prev_scale = current

        if self.recorder is not None:
            extra = {"action": actions.astype(np.int8), "reward": rewards}
            if fit is not None:
                extra["label"] = self._peek_labels(fit, actions, data, searcher_before, goods_before, scale)
                peek_plain = (np.maximum(0.0, fit[:, :, 0] - searcher_before[s_idx][:, None])
                              + np.maximum(0.0, fit[:, :, 1] - goods_before[g_idx][:, None]))
                extra["peek_plain"] = np.where(peek_plain > IMPROVEMENT_EPSILON, peek_plain, 0.0).astype(np.float32)
            self.recorder.add_round(data, extra)

        self.accept_searcher_children(children, f1, searcher_before)
        self.accept_good_children(self.goods, self.goods_fitness, goods_before, children,
                             f2, range(self.n))
        improved = self.searcher_fitness > searcher_before + IMPROVEMENT_EPSILON
        self.searcher_stagnation = np.where(improved, 0, self.searcher_stagnation + 1)
        per_searcher = actions.reshape(self.n, self.w)
        self.round_stats.append({
            "improve_rate": float((plain > IMPROVEMENT_EPSILON).mean()),
            "mean_reward": float(rewards.mean()),
            "segment_counts": np.bincount(actions, minlength=H).tolist(),
            "distinct_per_searcher": float(np.mean([len(set(row)) for row in per_searcher])),
        })


class RoundRecorder:
    """一段最佳化的每回合資料：回合層級、交易層級、action、reward（作弊版另有 label、peek_plain）。"""

    def __init__(self, map_name=""):
        self.map_name = map_name
        self.rounds = []

    def add_round(self, data, extra):
        record = {k: data[k] for k in ROUND_KEYS + PAIR_KEYS}
        record.update(extra)
        self.rounds.append(record)

    def stacked(self):
        """→ dict：回合層級 (R, ...)、交易層級 (R, P, ...)；done (R,) 最後一回合為 1。"""
        out = {k: np.stack([r[k] for r in self.rounds]) for k in self.rounds[0]}
        out["done"] = np.zeros(len(self.rounds), dtype=np.float32)
        out["done"][-1] = 1.0
        return out


__all__ = ["Ring_SE_E06", "RoundRecorder", "difference_rewards", "others_max", "H", "FEAT_DIM", "TOKEN_DIM",
           "CAND_DIM", "GLOBAL_DIM", "ROUND_KEYS", "PAIR_KEYS"]
