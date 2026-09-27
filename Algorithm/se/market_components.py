"""SE 三個家族（SETS、SI_SETS、Ring_SE）共用的市場元件。

這些元件和「區域怎麼定義、跟哪些商品投資」無關，因此不放在任何一個
家族的繼承鏈上，由需要的家族以 mixin 引入：

* ``beta_cdf``：區域吸引力使用的 regularized Beta CDF。
* ``AdaptiveBetaMemory``：每回合依選區結果以遞減步長更新 Beta 記憶。
* ``PooledEvaluation``：以 h 個無狀態子程序平行評估每個 searcher 的投資列。
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

    步長 ``current_adaptive_step`` 由各家族的 ``vision_search`` 依搜尋進度
    遞減後設定。
    """

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
    使用者需在 ``__init__`` 設定 ``self._pool = None``。
    """

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
            for col, (candidate, state, objectives, target_red) in enumerate(results):
                investments[row][col] = candidate
                fitness = float(np.sum(objectives))
                self.evatime += 1
                self.update_best(problem, state, objectives, fitness,
                                 candidate=candidate)
                problem.target_red = target_red
                row_scores.append(fitness)
            scores.append(row_scores)
        return np.asarray(scores, dtype=float)


__all__ = ["AdaptiveBetaMemory", "PooledEvaluation", "beta_cdf"]
