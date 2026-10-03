"""Process-backed regional market used by the original SE implementations.

Built on :class:`~Algorithm.se.persistent_worker_pool.PersistentWorkerPool`,
which owns only process lifecycle and Pipe exchange. Everything here is the
SA_SETS-specific *policy*: each worker permanently owns one region's goods
pool and runs the full invest+evaluate step locally, so only compact
summaries (searchers in, fitness results out) cross the process boundary.
"""

from __future__ import annotations

import functools

import numpy as np

from Algorithm.se.persistent_worker_pool import PersistentWorkerPool


def _build_region_handler(algorithm, problem, worker_id, seed):
    """Own one region's goods pool for the worker process's whole life."""
    region = worker_id
    algorithm._seed_random_streams(seed)
    algorithm.evatime = 0

    goods = [
        algorithm.create_candidate(problem, region=region)
        for _ in range(algorithm.w)
    ]
    goods_fitness = algorithm.evaluate_many(problem, goods)

    def handle(request):
        nonlocal goods, goods_fitness
        searchers, searcher_fitness, market_context = request

        load_context = getattr(algorithm, "load_market_context", None)
        if callable(load_context):
            load_context(region, market_context)

        begin_round = getattr(algorithm, "begin_investment_round", None)
        if callable(begin_round):
            begin_round(region, searchers, goods, goods_fitness)

        investments = [
            [
                algorithm.align_region(
                    problem,
                    algorithm.invest(
                        problem,
                        searcher,
                        goods[good_id],
                    ),
                    region,
                )
                for good_id in range(algorithm.w)
            ]
            for searcher in searchers
        ]
        investment_fitness = np.asarray(
            [
                algorithm.evaluate_many(problem, candidates)
                for candidates in investments
            ],
            dtype=float,
        )

        record_outcomes = getattr(
            algorithm,
            "record_investment_outcomes",
            None,
        )
        if callable(record_outcomes):
            record_outcomes(
                goods_fitness,
                investment_fitness,
                searcher_fitness,
            )
        export_feedback = getattr(algorithm, "export_market_feedback", None)
        market_feedback = (
            export_feedback() if callable(export_feedback) else None
        )

        # Snapshot goods/goods_fitness now: the response must reflect this
        # round's pre-update state even though it isn't pickled by the pool
        # until after this function returns, and the loop below mutates
        # both of these in place for the next round.
        result = (
            list(goods),
            goods_fitness.copy(),
            investment_fitness,
            int(algorithm.evatime),
            algorithm.best_state,
            algorithm.best_objectives,
            algorithm.fitness,
            algorithm.coverage,
            algorithm.best_candidate,
        )
        response = (
            result if market_feedback is None else result + (market_feedback,)
        )
        algorithm.evatime = 0

        # Preserve the original SE rule: each good is replaced by the best
        # investment generated for that good in the current round.
        for good_id in range(algorithm.w):
            searcher_id = int(np.argmax(investment_fitness[:, good_id]))
            accept_replacement = getattr(
                algorithm,
                "accept_good_replacement",
                None,
            )
            if callable(accept_replacement) and not accept_replacement(
                goods_fitness[good_id],
                investment_fitness[searcher_id, good_id],
            ):
                continue
            goods[good_id] = investments[searcher_id][good_id].copy()
            goods_fitness[good_id] = investment_fitness[
                searcher_id, good_id
            ]

        return response

    return handle


class ParallelSEMarket:
    """Manage process lifecycle and exchange one SE market round."""

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.pool = PersistentWorkerPool()
        self.problem = None

    def __getstate__(self):
        """Exclude live process handles when the algorithm is pickled."""
        return {
            "algorithm": None,
            "pool": PersistentWorkerPool(),
            "problem": None,
        }

    def start(self, problem):
        self.problem = problem
        algorithm = self.algorithm
        # A plain local closure can't be pickled to hand off to a spawned
        # worker process on Windows; functools.partial over the module-level
        # _build_region_handler is picklable as long as algorithm/problem
        # are (they already have to be, to reach this point at all).
        build_handler = functools.partial(
            _build_region_handler, algorithm, problem
        )
        self.pool.start(algorithm.h, build_handler, algorithm.next_seed)

    def search(self, searchers, contexts=None):
        """Run one market round, optionally passing one context per region."""
        region_count = len(self.pool.connections)
        if contexts is not None and len(contexts) != region_count:
            raise ValueError("market contexts must contain one item per region")
        needs_feedback = callable(
            getattr(self.algorithm, "record_investment_outcomes", None)
        )
        requests = []
        for region in range(region_count):
            searcher_fitness = (
                self.algorithm.searcher_fitness if needs_feedback else None
            )
            market_context = contexts[region] if contexts is not None else None
            requests.append((searchers, searcher_fitness, market_context))
        self.pool.send_all(requests)
        responses = self.pool.recv_all()

        goods = []
        goods_fitness = []
        investment_fitness = []
        evaluations = 0
        for region, result in enumerate(responses):
            (
                region_goods,
                region_good_fitness,
                region_investments,
                count,
                best_state,
                best_objectives,
                worker_fitness,
                best_coverage,
                best_candidate,
            ) = result[:9]
            market_feedback = result[9] if len(result) > 9 else None
            goods.append(region_goods)
            goods_fitness.append(region_good_fitness)
            investment_fitness.append(region_investments)
            evaluations += int(count)
            if best_state is not None:
                # The worker already consumed the evaluation.  Merge its
                # process-local result without evaluating or counting again.
                self.algorithm.update_best(
                    self.problem,
                    best_state,
                    best_objectives,
                    worker_fitness,
                    best_coverage,
                    candidate=best_candidate,
                )
            merge_feedback = getattr(
                self.algorithm,
                "merge_market_feedback",
                None,
            )
            if callable(merge_feedback) and market_feedback is not None:
                merge_feedback(region, market_feedback)

        return (
            np.asarray(goods, dtype=object),
            np.asarray(goods_fitness, dtype=float),
            np.asarray(investment_fitness, dtype=float),
            evaluations,
        )

    def close(self, searchers):
        self.pool.close()


__all__ = ["ParallelSEMarket"]
