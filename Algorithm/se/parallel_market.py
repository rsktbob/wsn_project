"""Full-region RNG resources and batch evaluation; no search/generation flow.

Goods, operators, replacement decisions and per-region RNGs live in the
parent. Each local region context retains the former worker's random stream
and best archive, preserving generated candidates and first-winner ties.
"""
from __future__ import annotations
import copy
import functools
import numpy as np
from Algorithm.core.Algorithm import Algorithm
from Algorithm.se.persistent_worker_pool import PersistentWorkerPool
from Algorithm.se.selective_evaluation import build_evaluate_handler


class ParallelSEMarket:
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.pool = PersistentWorkerPool()
        self.problem = None
        self.regions = []
        self.goods = []
        self.goods_fitness = []

    def __getstate__(self):
        return {"algorithm": None, "pool": PersistentWorkerPool(),
                "problem": None, "regions": [], "goods": [], "goods_fitness": []}

    def start(self, problem):
        self.problem = problem
        self.regions, self.goods, self.goods_fitness = [], [], []
        self.algorithm._region_operators = None
        for _ in range(self.algorithm.h):
            region = copy.deepcopy(self.algorithm)
            region._seed_random_streams(self.algorithm.next_seed())
            region.evatime = 0
            self.regions.append(region)
            region_id = len(self.regions) - 1
            goods = [region.create_candidate(problem, region=region_id)
                     for _ in range(region.w)]
            # Former workers evaluated these before their first request.
            # Keep their FE and best results local until the first round merge.
            scores = Algorithm.evaluate_many(region, problem, goods)
            self.goods.append(goods)
            self.goods_fitness.append(scores)
        self.goods_fitness = np.asarray(self.goods_fitness)
        # Region RNG seeds have already been consumed above. Evaluation is
        # deterministic and does not consume another algorithm RNG stream.
        self.pool.start(self.algorithm.h,
                        functools.partial(build_evaluate_handler, problem),
                        lambda: 0)

    def evaluate_many(self, problem, candidates, *, batch_size):
        """Evaluate supplied regional batches and preserve archive merge order."""
        size = int(batch_size)
        if size <= 0 or len(candidates) != len(self.regions) * size:
            raise ValueError("expected one equal candidate batch per region")
        if size % self.algorithm.w:
            raise ValueError("regional batch must contain complete goods rows")
        searcher_count = size // self.algorithm.w
        batches = [candidates[i:i + size] for i in range(0, len(candidates), size)]
        custom_evaluator = type(self.algorithm).evaluate is not Algorithm.evaluate
        if custom_evaluator:
            responses = [None] * len(self.regions)
        else:
            self.pool.send_all(batches)
            responses = self.pool.recv_all()
        all_scores, evaluations = [], 0
        for region_id, (region, results) in enumerate(zip(self.regions, responses)):
            scores = []
            if custom_evaluator:
                scores = Algorithm.evaluate_many(region, problem, batches[region_id]).tolist()
            for index, (candidate, state, objectives) in enumerate(results or []):
                candidates[region_id * size + index] = candidate
                score = float(np.sum(objectives))
                region.evatime += 1
                region.update_best(problem, state, objectives, score, candidate=candidate)
                scores.append(score)
            region_scores = np.asarray(scores).reshape(searcher_count, region.w)
            all_scores.append(region_scores)
            record = getattr(region, "record_investment_outcomes", None)
            if callable(record):
                record(self.goods_fitness[region_id], region_scores,
                       self.algorithm.searcher_fitness)
            export = getattr(region, "export_market_feedback", None)
            merge = getattr(self.algorithm, "merge_market_feedback", None)
            if callable(merge):
                merge(region_id, export() if callable(export) else None)
            evaluations += int(region.evatime)
            region.evatime = 0
            if region.best_state is not None:
                self.algorithm.update_best(problem, region.best_state,
                    region.best_objectives, region.fitness, region.coverage,
                    candidate=region.best_candidate)
        self.algorithm.evatime += evaluations
        return np.asarray(all_scores).reshape(-1)

    def close(self, searchers=None):
        self.pool.close()


__all__ = ["ParallelSEMarket"]
