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
    """SA/SI 共用的區域記憶：被選到的區域加 ta，其餘區域加 tb。

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

    商品及更新決策由主程序保存；子程序只評估候選解。
    任何一列投資都可以交給任何一個子程序。
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

    def evaluate_many(self, problem, candidates, *, batch_size=None):
        """Evaluate a flat mutable list; preserve input and first-winner order.

        Initialization runs before the pool exists and uses Algorithm's
        sequential implementation. batch_size retains transaction row boundaries.
        """
        if getattr(self, "_pool", None) is None:
            return super().evaluate_many(problem, candidates)
        if len(candidates) == 0:
            return np.empty(0, dtype=float)
        size = len(candidates) if batch_size is None else int(batch_size)
        if size <= 0:
            raise ValueError("batch_size must be positive")
        rows = [candidates[i:i + size] for i in range(0, len(candidates), size)]
        scores = []
        index = 0
        for results in self._route_to_pool(rows):
            for candidate, state, objectives in results:
                candidates[index] = candidate
                fitness = float(np.sum(objectives))
                self.evatime += 1
                self.update_best(problem, state, objectives, fitness, candidate=candidate)
                scores.append(fitness)
                index += 1
        return np.asarray(scores, dtype=float)

    def evaluate_investments(self, problem, investments):
        """Compatibility adapter for existing external callers; flow uses evaluate_many."""
        if not investments:
            return np.empty((0, 0), dtype=float)
        width = len(investments[0])
        flat = [candidate for row in investments for candidate in row]
        scores = self.evaluate_many(problem, flat, batch_size=width)
        for i in range(len(investments)):
            investments[i][:] = flat[i * width:(i + 1) * width]
        return scores.reshape(len(investments), width)


__all__ = ["AdaptiveBetaMemory", "PooledEvaluation", "beta_cdf"]


def create_selected_investments(algorithm, problem, goods, active_regions):
    """Generate selected-scope rows; selection/fitness/replacement stay outside."""
    return [
        [child for good in algorithm.goods_for_region(region, goods)
         for child in algorithm.make_offspring(problem, algorithm.searchers[i], good, int(region))]
        for i, region in enumerate(active_regions)
    ]
