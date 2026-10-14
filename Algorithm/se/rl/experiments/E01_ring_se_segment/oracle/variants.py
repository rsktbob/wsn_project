"""E01 步驟 1：ring_se 作弊版（oracle 選段）的天花板測試。

每組 searcher-good 交易：

1. 在 4 段各抽一個 80/20 候選範圍（同 PairSegmentMarket）。
2. **偷看**：4 段都只做交換（不突變），評估 4 組交換後的子代（child1、child2）。
   偷看只走評估子程序池，**不計入預算、不更新全程最佳解**。
3. **選段**：先比 Δ = max(0, f(child1) − f(searcher)) + max(0, f(child2) − f(good))（≤ 1e-12 視為 0）；
   Δ 相同（大多是都沒改善）時比 f(child2)，因為 EN 更新下 good 會被造訪者中最好的 child2 取代。
4. **實際使用**：拿選中那段的交換結果，照 ring_se 的規則讓每個子代以 μ 機率在該段突變，
   再照常評估（計入預算）、照常更新（searcher 菁英、good EN）。

所以天花板只來自「依交換結果選對段」，突變的運氣和一般 ring_se 相同。
"""
from __future__ import annotations

import numpy as np

from Algorithm.se.market_components import accept_good_children, accept_searcher_children
from Algorithm.se.ring_se.Ring_SE import Ring_SE
from Algorithm.se.ring_se.ring_segment_policy import IMPROVEMENT_EPSILON, PairSegmentMarket
from State.Encoding import swap_segment


class Ring_SE_Oracle(PairSegmentMarket, Ring_SE):
    """Peek at the 4 crossover-only children (uncounted), pick the best segment, then mutate as usual."""

    def initialize_market(self, problem, initial_state=None):
        super().initialize_market(problem, initial_state)
        self.oracle_stats = {"pairs": 0, "any_improve": 0, "chosen_improve": 0, "segment_counts": [0] * self.h}

    def _peek(self, pairs):
        """Crossover-only children for every pair × segment, and their fitness (pairs, h, 2)."""
        rows, crossed = [], []
        for s_id, g_id, spans, _ in pairs:
            row, kids = [], []
            for span in spans:
                child1, child2 = swap_segment(
                    self.searchers[s_id], self.goods[g_id], span[1] * 2, span[2] * 2
                )
                row += [child1, child2]
                kids.append((child1, child2))
            rows.append(row)
            crossed.append(kids)
        # 只走子程序池解碼評估，不經 evaluate_investments：不加 evatime、不呼叫 update_best。
        fitness = np.array(
            [[float(np.sum(obj)) for _, _, obj in results] for results in self._route_to_pool(rows)],
            dtype=float,
        ).reshape(len(pairs), self.h, 2)
        return crossed, fitness

    def _oracle_choices(self, pairs, fitness, searcher_before, goods_before):
        choices = []
        for i, (s_id, g_id, _, _) in enumerate(pairs):
            f1, f2 = fitness[i, :, 0], fitness[i, :, 1]
            delta = np.maximum(0.0, f1 - searcher_before[s_id]) + np.maximum(0.0, f2 - goods_before[g_id])
            delta = np.where(delta > IMPROVEMENT_EPSILON, delta, 0.0)
            # 先比 Δ，再比 child2；np.lexsort 以最後一個 key 為主。
            choice = int(np.lexsort((f2, delta))[-1])
            choices.append(choice)
            self.oracle_stats["pairs"] += 1
            self.oracle_stats["any_improve"] += int(delta.max() > 0)
            self.oracle_stats["segment_counts"][choice] += 1
        return choices

    def vision_search(self, problem):
        goods_fitness_before = self.goods_fitness.copy()
        searcher_fitness_before = self.searcher_fitness.copy()
        context = self._round_context(problem)
        pairs = self._build_pairs(context)

        crossed, peek_fitness = self._peek(pairs)
        choices = self._oracle_choices(pairs, peek_fitness, searcher_fitness_before, goods_fitness_before)

        children = [[] for _ in range(self.n)]
        for i, ((s_id, _, spans, _), k) in enumerate(zip(pairs, choices)):
            for child in crossed[i][k]:
                child = child.copy()
                if self.random.random() < self.mutation_rate:
                    self.mutate_candidate(problem, child, sensor_span=spans[k])
                children[s_id].append(child)

        child_fitness = self.evaluate_investments(problem, children)
        child1_fitness = child_fitness[:, 0::2]
        child2_fitness = child_fitness[:, 1::2]
        for s_id, g_id, _, _ in pairs:
            improved = (child1_fitness[s_id, g_id] > searcher_fitness_before[s_id] + IMPROVEMENT_EPSILON
                        or child2_fitness[s_id, g_id] > goods_fitness_before[g_id] + IMPROVEMENT_EPSILON)
            self.oracle_stats["chosen_improve"] += int(improved)

        accept_searcher_children(self, children, child1_fitness, searcher_fitness_before)
        accept_good_children(
            self.goods, self.goods_fitness, goods_fitness_before, children,
            child2_fitness, range(self.n), self.GOOD_ACCEPT_TOLERANCE,
        )


__all__ = ["Ring_SE_Oracle"]
