"""Ring 家族的逐組區段選擇：每組 searcher-good 交易各自選一個區段。

原本的 Ring_SE 讓每個 searcher 每回合選一個區段，它的 w 個 goods 都在
同一區交配。這裡改成每組交易各自決定：決策前先在每個區段用原本的 80/20
規則抽好一個範圍，再依「這個範圍裡 searcher 和 good 差在哪」選一個區段。
範圍大小、交配、突變、EN 更新都和 Ring_SE 相同（2026-10-13 起；之前 good 是 EE）。

* 有效差異：兩者在這顆 sensor 的基因不同，而且任一方解碼後有開這顆 sensor。
  兩邊都沒開的 sensor，改基因幾乎不會改變解（平坦比例分析）。
* 回饋：0.5 × [child1 比 searcher 好] + 0.5 × [child2 比 good 好]。

``LinUCBSegmentPolicy``：共用一個線性模型，模型跨 lifetime 段延續。

D3QN 版（RL_Ring_SETS 系列）、v2 特徵與分級回饋、checkpoint 的 environment_contract
已在 2026-10-13 移到 ``legacy/rl_ring/``（該處保留本檔搬移前的完整版本）。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.market_components import (
    accept_good_children,
    accept_searcher_children,
)
from State.Encoding import swap_segment

IMPROVEMENT_EPSILON = 1e-12


class PairSegmentMarket:
    """Ring_SE round where every searcher-good pair picks its own segment.

    Subclasses provide ``_choose_segments(context, pairs)`` and
    ``_observe_round(context, pairs, choices, rewards, done)``.
    """

    # 只有 D3QN（已移到 legacy/rl_ring）需要瓶頸 sensor 與電量特徵；LinUCB 不必多算成本。
    NEEDS_ENERGY_FEATURES = False
    STAGNATION_SCALE = 20

    def initialize_market(self, problem, initial_state=None):
        # 解碼只取決於染色體與目前電量；電量只在段與段之間改變，所以每段清空。
        self._decode_cache = {}
        super().initialize_market(problem, initial_state)
        self.searcher_stagnation = np.zeros(self.n, dtype=int)
        self.max_region_length = max(b - a for a, b in self.region_sensor_bounds)
        self.round_rewards = []

    # ---- 每回合共用的解碼資訊 ---------------------------------------------
    def _decoded(self, problem, candidate, need_cost=False):
        """Active mask (and per-sensor cost) of a candidate, cached by chromosome.

        Most searchers and goods are unchanged between rounds, so only new
        chromosomes are decoded.  The values are identical to decoding again.
        """
        key = np.asarray(candidate.code).tobytes()
        info = self._decode_cache.get(key)
        if info is None or (need_cost and "cost" not in info):
            state = candidate.decode(problem)
            info = {"active": np.asarray(state.levels) > 0}
            if need_cost:
                radii = np.asarray(problem.state_radius(state), dtype=float)
                info["cost"] = np.asarray(
                    problem.calculate_total_cost(state, sensing_radii=radii),
                    dtype=float,
                )[: problem.SENSOR_NUMBER]
            self._decode_cache[key] = info
        return info

    def _round_context(self, problem):
        context = {
            "progress": min(1.0, self.evatime / max(1, self.evaluation_limit)),
            "searcher_active": [],
            "goods_active": [
                self._decoded(problem, good)["active"] for good in self.goods
            ],
        }
        energy = np.maximum(np.asarray(problem.energy, dtype=float), 0.0)
        energy_ratio = energy / max(float(problem.initial_energy), 1e-15)
        context["energy_ratio"] = energy_ratio
        bottlenecks = []
        for searcher in self.searchers:
            info = self._decoded(problem, searcher, need_cost=self.NEEDS_ENERGY_FEATURES)
            context["searcher_active"].append(info["active"])
            if not self.NEEDS_ENERGY_FEATURES:
                continue
            cost = info["cost"]
            spending = np.isfinite(cost) & (cost > 1e-15)
            if np.any(spending):
                ids = np.flatnonzero(spending)
                bottlenecks.append(int(ids[np.argmin(energy[ids] / cost[ids])]))
            else:
                bottlenecks.append(-1)
        context["searcher_bottleneck"] = bottlenecks
        return context

    # ---- 每組交易的候選範圍與統計 -----------------------------------------
    def _pair_segments(self, context, searcher_id, good_id):
        """Draw one span per segment and summarize what each span exchanges."""
        code_s = np.asarray(self.searchers[searcher_id].code)
        code_g = np.asarray(self.goods[good_id].code)
        differ = (code_s[0::2] != code_g[0::2]) | (code_s[1::2] != code_g[1::2])
        active_s = context["searcher_active"][searcher_id]
        either_on = active_s | context["goods_active"][good_id]
        effective = differ & either_on
        inactive = differ & ~either_on
        spans, stats = [], []
        for region in range(self.h):
            span = self.select_sensor_span(region)
            _, left, right = span
            length = right - left
            stat = {
                "effective": int(effective[left:right].sum()),
                "inactive": int(inactive[left:right].sum()),
                "length": length / self.max_region_length,
                "active_ratio": float(active_s[left:right].sum()) / max(length, 1),
            }
            if self.NEEDS_ENERGY_FEATURES:
                bottleneck = context["searcher_bottleneck"][searcher_id]
                on = active_s[left:right]
                stat["bottleneck"] = float(left <= bottleneck < right)
                stat["energy"] = (
                    float(np.mean(context["energy_ratio"][left:right][on]))
                    if np.any(on) else 0.0
                )
            spans.append(span)
            stats.append(stat)
        return spans, stats

    # ---- 一回合 -------------------------------------------------------------
    def vision_search(self, problem):
        goods_fitness_before = self.goods_fitness.copy()
        searcher_fitness_before = self.searcher_fitness.copy()
        self._best_before = float(self.fitness)
        context = self._round_context(problem)
        pairs = self._build_pairs(context)
        self._last_pairs = pairs
        choices = self._choose_segments(context, pairs)

        children = [[] for _ in range(self.n)]
        for (s_id, g_id, spans, _), region in zip(pairs, choices):
            span = spans[int(region)]
            child1, child2 = swap_segment(
                self.searchers[s_id], self.goods[g_id], span[1] * 2, span[2] * 2
            )
            for child in (child1, child2):
                if self.random.random() < self.mutation_rate:
                    self.mutate_candidate(problem, child, sensor_span=span)
            children[s_id] += [child1, child2]

        child_fitness = self.evaluate_investments(problem, children)
        child1_fitness = child_fitness[:, 0::2]
        child2_fitness = child_fitness[:, 1::2]
        rewards = self._pair_rewards(
            pairs, child1_fitness, child2_fitness,
            searcher_fitness_before, goods_fitness_before,
        )
        self.round_rewards.append(float(np.mean(rewards)))

        accept_searcher_children(self, children, child1_fitness, searcher_fitness_before)
        # 同 Ring_SE 的 EN：good 一律換成拜訪者中最好的 child2（GOOD_ACCEPT_TOLERANCE = -inf）。
        accept_good_children(
            self.goods, self.goods_fitness, goods_fitness_before, children,
            child2_fitness, range(self.n), self.GOOD_ACCEPT_TOLERANCE,
        )
        improved = self.searcher_fitness > searcher_fitness_before + IMPROVEMENT_EPSILON
        self.searcher_stagnation = np.where(improved, 0, self.searcher_stagnation + 1)

        done = (
            self.evatime >= self.evaluation_limit
            or self.iterations_completed + 1 >= self.max_iterations
        )
        self._observe_round(context, pairs, choices, rewards, done)

    def _build_pairs(self, context):
        """One (searcher, good, spans, stats) entry per searcher-good pair."""
        return [
            (s_id, g_id, *self._pair_segments(context, s_id, g_id))
            for s_id in range(self.n)
            for g_id in range(self.w)
        ]

    def _pair_rewards(self, pairs, child1_fitness, child2_fitness,
                      searcher_before, goods_before):
        """0.5 × [child1 improves its searcher] + 0.5 × [child2 improves its good]."""
        return np.array([
            0.5 * float(child1_fitness[s_id, g_id] > searcher_before[s_id] + IMPROVEMENT_EPSILON)
            + 0.5 * float(child2_fitness[s_id, g_id] > goods_before[g_id] + IMPROVEMENT_EPSILON)
            for s_id, g_id, _, _ in pairs
        ])

    def update_search_memory(self, problem, evaluation_start):
        """No region memory; only record this round's fitness history."""
        history_end = min(int(self.evatime), len(self.history))
        if history_end > evaluation_start:
            self.history[evaluation_start:history_end] = self.fitness


class LinUCBSegmentPolicy(PairSegmentMarket):
    """Shared-parameter LinUCB over the h candidate spans of each pair.

    Features (one row per segment): constant, effective differences in
    {1-2, 3-4, >=5} (0 is the reference), inactive differences / 5, span
    length, searcher active ratio, search progress and progress × {1-2}.
    The model persists across optimizer calls of one lifetime run.
    """

    ALPHA = 0.3
    RIDGE = 1.0
    FEATURE_NAMES = (
        "constant", "effective_1_2", "effective_3_4", "effective_5_plus",
        "inactive_differences", "span_length", "searcher_active_ratio",
        "progress", "progress_x_effective_1_2",
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        size = len(self.FEATURE_NAMES)
        self.linucb_A = self.RIDGE * np.eye(size)
        self.linucb_b = np.zeros(size)

    def _linucb_features(self, stat, progress):
        e = stat["effective"]
        small = 1.0 if 1 <= e <= 2 else 0.0
        return np.array([
            1.0, small, 1.0 if 3 <= e <= 4 else 0.0, 1.0 if e >= 5 else 0.0,
            min(stat["inactive"], 5) / 5.0, stat["length"], stat["active_ratio"],
            progress, progress * small,
        ])

    def _choose_segments(self, context, pairs):
        A_inv = np.linalg.inv(self.linucb_A)
        theta = A_inv @ self.linucb_b
        self._chosen_features = []
        choices = []
        for _, _, _, stats in pairs:
            X = np.array([self._linucb_features(s, context["progress"]) for s in stats])
            scores = X @ theta + self.ALPHA * np.sqrt(
                np.einsum("ij,jk,ik->i", X, A_inv, X)
            )
            best = np.flatnonzero(scores >= scores.max() - 1e-12)
            pick = int(best[self.random.randrange(len(best))])
            choices.append(pick)
            self._chosen_features.append(X[pick])
        return choices

    def _observe_round(self, context, pairs, choices, rewards, done):
        for x, reward in zip(self._chosen_features, rewards):
            self.linucb_A += np.outer(x, x)
            self.linucb_b += reward * x


__all__ = ["LinUCBSegmentPolicy", "PairSegmentMarket"]
