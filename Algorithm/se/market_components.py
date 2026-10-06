"""SE 三個家族（SETS、SI_SETS、Ring_SE）共用的市場元件。

這些元件和「區域怎麼定義、跟哪些商品投資」無關，因此不放在任何一個
家族的繼承鏈上，由需要的家族以 mixin 引入：

* ``beta_cdf``：區域吸引力使用的 regularized Beta CDF。
* ``AdaptiveBetaMemory``：每回合依選區結果以遞減步長更新 Beta 記憶，
  並把區域期望值轉成 Beta CDF 機率。
* ``PooledEvaluation``：以 h 個無狀態子程序平行評估每個 searcher 的投資列。
* ``accept_searcher_children``／``accept_good_children``：角色分離的菁英更新
  （child1 只能取代 searcher、child2 只能取代 good）。
"""

from __future__ import annotations

import functools

import numpy as np
from scipy.special import betainc as scipy_betainc

from Algorithm.se.persistent_worker_pool import PersistentWorkerPool
from Algorithm.se.selective_evaluation import build_evaluate_handler


def beta_cdf(a, b, x):
    """Calculate the regularized Beta CDF through the shared SciPy wrapper."""
    return float(scipy_betainc(float(a), float(b), float(x)))


class AdaptiveBetaMemory:
    """SA-SETS 的區域記憶：被選到的區域加 ta，其餘區域加 tb。

    步長 ``current_adaptive_step`` 由各家族的 ``vision_search`` 呼叫
    ``update_adaptive_step()`` 依搜尋進度遞減。
    """

    ADAPTIVE_STEP = 0.001

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.adaptive_step = self.ADAPTIVE_STEP
        self.current_adaptive_step = self.adaptive_step

    def update_adaptive_step(self):
        """Shrink the Beta step linearly with search progress; return progress."""
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        self.current_adaptive_step = self.adaptive_step * (1.0 - progress)
        return progress

    def beta_probabilities(self, expected_value):
        """Map each region/searcher expected value through its Beta CDF."""
        probability = np.empty((self.h, self.n), dtype=float)
        for region in range(self.h):
            for searcher_id in range(self.n):
                probability[region, searcher_id] = beta_cdf(
                    self.ta[region],
                    self.tb[region],
                    expected_value[region, searcher_id],
                )
        return probability

    def market_probabilities(self, goods_fitness, investment_quality):
        """SA-SETS region value: best good × investment × market share."""
        average_goods = np.mean(goods_fitness, axis=1)
        best_goods = np.max(goods_fitness, axis=1)
        total = float(np.sum(average_goods))
        if abs(total) <= np.finfo(float).eps:
            region_share = np.full(self.h, 1.0 / self.h)
        else:
            region_share = average_goods / total

        expected_value = (
            best_goods[:, None]
            * investment_quality
            * region_share[:, None]
        )
        return self.beta_probabilities(expected_value)

    def update_search_memory(self, problem, evaluation_start):
        """Adapt Beta beliefs and retain the best searcher of this round."""
        step = self.current_adaptive_step
        for selected_region in self.selected_regions:
            selected_region = int(selected_region)
            self.ta[selected_region] += step
            for region in range(self.h):
                if region != selected_region:
                    self.tb[region] += step

        history_end = min(int(self.evatime), len(self.history))
        if history_end > evaluation_start:
            self.history[evaluation_start:history_end] = self.fitness


class PooledEvaluation:
    """用 h 個子程序評估投資列，結果依原本的 searcher/good 順序合併。

    與 SETS 的 ``ParallelSEMarket``（每個子程序擁有一個區域的商品）不同，
    這些子程序不保存狀態，任何一列投資都可以交給任何一個子程序。
    """

    _pool = None

    def arrange_resources(self, problem):
        """用 h 個子程序評估投資。"""
        self._pool = PersistentWorkerPool()
        build_handler = functools.partial(build_evaluate_handler, problem)
        self._pool.start(self.h, build_handler, self.next_seed)

    def close_market(self):
        """搜尋結束或發生例外時關閉評估子程序。"""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    def _route_to_pool(self, investments):
        """Round-robin each searcher's row across the h persistent workers.

        The pool has no shared task queue of its own, so this loop does the
        load-balancing: send every row to worker ``row_index % h``, then, per
        worker, drain exactly as many responses as were sent to it, and
        finally replay the assignment order to hand results back in the
        original row order.
        """
        worker_count = len(self._pool.connections)
        assignments = [
            row_index % worker_count for row_index in range(len(investments))
        ]
        for row, worker_id in zip(investments, assignments):
            self._pool.send(worker_id, row)
        pending = [0] * worker_count
        for worker_id in assignments:
            pending[worker_id] += 1
        buffered = [[] for _ in range(worker_count)]
        for worker_id, count in enumerate(pending):
            for _ in range(count):
                buffered[worker_id].append(self._pool.recv(worker_id))
        cursors = [0] * worker_count
        for worker_id in assignments:
            yield buffered[worker_id][cursors[worker_id]]
            cursors[worker_id] += 1

    def evaluate_investments(self, problem, investments):
        """評估每一批投資，再按原 searcher/good 順序合併，保留同分規則。"""
        results_per_row = self._route_to_pool(investments)

        scores = []
        for row, results in enumerate(results_per_row):
            row_scores = []
            for col, (candidate, state, objectives) in enumerate(results):
                investments[row][col] = candidate
                fitness = float(np.sum(objectives))
                self.evatime += 1
                self.update_best(problem, state, objectives, fitness,
                                 candidate=candidate)
                row_scores.append(fitness)
            scores.append(row_scores)
        return np.asarray(scores, dtype=float)


def accept_searcher_children(algorithm, children, child1_fitness,
                             searcher_fitness_before):
    """Each searcher may take only its own best child1, and only if better.

    ``children[s]`` interleaves ``[child1, child2]`` per good.
    """
    for searcher_id in range(algorithm.n):
        good_id = int(np.argmax(child1_fitness[searcher_id]))
        score = float(child1_fitness[searcher_id, good_id])
        if score > searcher_fitness_before[searcher_id]:
            algorithm.searchers[searcher_id] = children[searcher_id][
                good_id * 2
            ].copy()
            algorithm.searcher_fitness[searcher_id] = score


def accept_good_children(goods, goods_fitness, goods_fitness_before, children,
                         child2_fitness, visitors, tolerance):
    """Each good takes its visitors' best child2 only when it strictly improves.

    ``goods``/``goods_fitness`` are one pool and are updated in place; ties
    go to the first visitor.
    """
    if not len(visitors):
        return
    for good_id in range(len(goods)):
        winner = int(
            max(
                visitors,
                key=lambda index: child2_fitness[int(index), good_id],
            )
        )
        winner_score = float(child2_fitness[winner, good_id])
        if winner_score <= goods_fitness_before[good_id] + tolerance:
            continue
        goods[good_id] = children[winner][good_id * 2 + 1].copy()
        goods_fitness[good_id] = winner_score


__all__ = [
    "AdaptiveBetaMemory",
    "PooledEvaluation",
    "accept_good_children",
    "accept_searcher_children",
    "beta_cdf",
]
