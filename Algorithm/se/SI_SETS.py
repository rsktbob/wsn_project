"""Selective-investment SETS with parallel candidate evaluation.

Uses the same :class:`~Algorithm.se.persistent_worker_pool.PersistentWorkerPool`
as ``ParallelSEMarket`` (SA_SETS's market), just with a stateless worker
handler and searcher-row task routing instead of SA_SETS's region-owned
workers -- SI-SETS' batches are per-searcher and unevenly sized round to
round, which doesn't fit ParallelSEMarket's "broadcast to h fixed region
workers" protocol (see ``evaluate_investments`` below for the routing).
"""

from __future__ import annotations

import functools

import numpy as np

from Algorithm.se.persistent_worker_pool import PersistentWorkerPool
from Algorithm.se.selective_evaluation import build_evaluate_handler

from Algorithm.se.SA_SETS import SA_SETS, beta_cdf


class SI_SETS(SA_SETS):
    """Choose a goods pool first, then invest only in that pool.

    Each region owns a persistent goods pool and the same identity-sensor
    partition used by SA-SETS.  The most recent observed investment quality
    is kept for searcher/region pairs that are not evaluated in the current
    round.
    """

    def __init__(self, problem, n=5, h=4, w=8, mu=0.2, seed=None):
        # Reuse SA-SETS initialization, partition encoding, crossover and
        # mutation verbatim.  SI-SETS only changes which pools are invested in.
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        self.goods = []
        self._pool = None
        self.goods_fitness = np.empty((self.h, self.w), dtype=float)
        self.investment_quality = np.empty((self.h, self.n), dtype=float)

    def initialize_market(self, problem, initial_state=None):
        """Initialize searchers, goods pools, and stale-value estimates."""
        # This inherited call performs exactly the same searcher initialization
        # and region assignment as SA-SETS.
        super().initialize_market(problem, initial_state)
        self.goods = [
            [self.create_candidate(problem, region=region) for _ in range(self.w)]
            for region in range(self.h)
        ]
        flat_goods = [good for region_goods in self.goods for good in region_goods]
        self.goods_fitness = self.evaluate_many(problem, flat_goods).reshape(
            self.h,
            self.w,
        )

        # Before a searcher has visited a pool, its estimated investment value
        # starts from that pool's existing average goods quality.
        initial_quality = np.mean(self.goods_fitness, axis=1)
        self.investment_quality = np.repeat(
            initial_quality[:, None],
            self.n,
            axis=1,
        )

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

        Unlike SA_SETS's region workers, these evaluators are stateless and
        own no region, so there is no affinity requirement -- any worker can
        take any row. The pool has no shared task queue of its own (that is
        what ProcessPoolExecutor gave up when we moved off it), so this
        loop does the load-balancing: send every row to worker
        ``row_index % h``, then, per worker, drain exactly as many
        responses as were sent to it, and finally replay the assignment
        order to hand results back in the original row order.
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

    def vision_search(self, problem):
        """Invest only in each searcher's already-selected goods pool."""
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        self.current_adaptive_step = self.adaptive_step * (1.0 - progress)

        active_regions = self.selected_regions.copy()
        goods_fitness_before = self.goods_fitness.copy()
        investments = [[None for _ in range(self.w)] for _ in range(self.n)]
        investment_fitness = np.empty((self.n, self.w), dtype=float)

        for searcher_id, region in enumerate(active_regions):
            region = int(region)
            candidates = [
                self.align_region(
                    problem,
                    self.invest(
                        problem,
                        self.searchers[searcher_id],
                        self.goods[region][good_id],
                    ),
                    region,
                )
                for good_id in range(self.w)
            ]
            investments[searcher_id] = candidates
        investment_fitness = self.evaluate_investments(problem, investments)
        for searcher_id, region in enumerate(active_regions):
            scores = investment_fitness[searcher_id]
            self.investment_quality[region, searcher_id] = float(
                np.mean(scores)
            )

        # Choose the next region with the same tournament and Beta rule as
        # SA-SETS.  Unvisited-region qualities are necessarily cached because
        # SI-SETS deliberately did not evaluate those investments this round.
        probabilities = self.region_probabilities(
            goods_fitness_before,
            self.investment_quality,
        )
        selected = self.select_regions(probabilities)

        # Match SA-SETS' update order: after region selection, compare the
        # searcher with the best pre-update good in that selected region.
        for searcher_id, region in enumerate(selected):
            region = int(region)
            good_id = int(np.argmax(goods_fitness_before[region]))
            good_score = float(goods_fitness_before[region, good_id])
            if good_score > self.searcher_fitness[searcher_id]:
                self.searchers[searcher_id] = self.goods[region][good_id].copy()
                self.searcher_fitness[searcher_id] = good_score

        # Preserve the current goods rule: each visited good is replaced by
        # the best investment made for that good during this round.
        for region in range(self.h):
            visitors = np.flatnonzero(active_regions == region)
            if not len(visitors):
                continue
            for good_id in range(self.w):
                best_searcher = int(
                    max(
                        visitors,
                        key=lambda searcher_id: investment_fitness[
                            int(searcher_id), good_id
                        ],
                    )
                )
                self.goods[region][good_id] = investments[best_searcher][
                    good_id
                ].copy()
                self.goods_fitness[region, good_id] = investment_fitness[
                    best_searcher, good_id
                ]

        self.selected_regions = selected

    def region_probabilities(self, goods_fitness, investment_quality):
        """Use current observations and cached values to score every pool."""
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
        probability = np.empty((self.h, self.n), dtype=float)
        for region in range(self.h):
            for searcher_id in range(self.n):
                probability[region, searcher_id] = beta_cdf(
                    self.ta[region],
                    self.tb[region],
                    expected_value[region, searcher_id],
                )
        return probability

    def update_search_memory(self, problem, evaluation_start):
        """Retain SA-SETS' adaptive Beta memory and historical-best rule."""
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


__all__ = ["SI_SETS"]
