"""Shared high-level flow for Search Economics algorithms."""

from __future__ import annotations

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from Algorithm.se import ParallelSEMarket


class BaseSE(Algorithm):
    """Own the stable SE phases, not encoding or genetic-operator details.

    The 2023 thesis describes initialization, resource arrangement, vision
    search, and feedback from the completed round.  The last part is exposed
    as ``update_search_memory``.  Crossover and mutation rules never appear
    in this base class.
    """

    def __init__(
        self,
        problem,
        n=5,
        h=4,
        w=2,
        mu=0.2,
        code_length=None,
        player=2,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.n = int(n)
        self.h = int(h)
        self.w = int(w)
        self.player = int(player)
        self.mutation_rate = float(mu)
        self.code_length = (
            None if code_length is None else int(code_length)
        )
        # ``length`` remains the conventional chromosome-size attribute used
        # by other optimizer families and experiment diagnostics.
        self.length = self.code_length

        if self.n <= 0 or self.h <= 0 or self.w <= 0:
            raise ValueError("n, h, and w must be positive")
        if not 1 <= self.player <= self.h:
            raise ValueError("player must be between 1 and h")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mu must be between 0 and 1")

        self.name = (
            f"{self.__class__.__name__}_{self.n}_{self.h}_{self.w}_"
            f"{self.mutation_rate}"
        )
        self.market = ParallelSEMarket(self)
        self.searchers = []
        self.searcher_fitness = np.array([], dtype=float)
        self.selected_regions = np.array([], dtype=int)
        self.ta = np.ones(self.h, dtype=float)
        self.tb = np.ones(self.h, dtype=float)
        self.iterations_completed = 0
        self.evaluation_limit = 0

    def create_candidate(self, problem, region=None):
        """Create one candidate, optionally constrained to a region."""
        raise NotImplementedError

    def align_region(self, problem, candidate, region):
        """Align a candidate with an algorithm-specific region definition."""
        raise NotImplementedError

    def invest(self, problem, searcher, good):
        """Use a searcher and one regional good to make an investment."""
        raise NotImplementedError

    def create_investments(self, problem, goods):
        """Create the common ``region x searcher x good`` investment grid."""
        return [
            [
                [
                    self.align_region(
                        problem,
                        self.invest(
                            problem,
                            self.searchers[searcher_id],
                            goods[region][good_id],
                        ),
                        region,
                    )
                    for good_id in range(self.w)
                ]
                for searcher_id in range(self.n)
            ]
            for region in range(self.h)
        ]

    def select_regions(self, probabilities):
        """Select one region per searcher by the shared SE tournament rule."""
        selected = np.empty(self.n, dtype=int)
        for searcher_id in range(self.n):
            players = self.random.sample(range(self.h), self.player)
            selected[searcher_id] = max(
                players,
                key=lambda region: probabilities[region, searcher_id],
            )
        return selected

    def initialize_market(self, problem, initial_state=None):
        """Create searchers and the initial market memory."""
        self.ta = np.ones(self.h, dtype=float)
        self.tb = np.ones(self.h, dtype=float)
        self.selected_regions = np.arange(self.n, dtype=int) % self.h
        self.searchers = [
            self.create_candidate(problem, region=int(region))
            for region in self.selected_regions
        ]
        if initial_state is not None and self.searchers:
            self.searchers[0] = initial_state.copy()

        self.searcher_fitness = self.evaluate_many(problem, self.searchers)

    def arrange_resources(self, problem):
        """Create the region workers that own each region's goods."""
        self.market.start(problem)

    def vision_search(self, problem):
        """Evaluate all region investments and choose each searcher's region."""
        build_contexts = getattr(self, "build_market_contexts", None)
        market_contexts = (
            build_contexts(problem) if callable(build_contexts) else None
        )
        (
            goods,
            goods_fitness,
            investment_fitness,
            evaluations,
        ) = self.market.search(
            self.searchers,
            contexts=market_contexts,
        )
        self.evatime += evaluations

        probabilities = self.region_probabilities(
            goods_fitness,
            investment_fitness,
        )
        selected = self.select_regions(probabilities)
        for searcher_id, region in enumerate(selected):
            good_id = int(np.argmax(goods_fitness[region]))
            if (
                goods_fitness[region, good_id]
                > self.searcher_fitness[searcher_id]
            ):
                self.searchers[searcher_id] = goods[region, good_id].copy()
                self.searcher_fitness[searcher_id] = goods_fitness[
                    region, good_id
                ]
        self.selected_regions = selected

        finish_round = getattr(self, "on_market_round_finish", None)
        if callable(finish_round):
            finish_round(
                problem,
                goods,
                goods_fitness,
                investment_fitness,
                selected,
            )

    def region_probabilities(self, goods_fitness, investment_fitness):
        """Compute the original SE investment potential for every region."""
        investment_quality = np.mean(investment_fitness, axis=2)
        market_quality = np.mean(goods_fitness, axis=1)
        total_quality = float(np.sum(market_quality))
        if abs(total_quality) <= np.finfo(float).eps:
            market_share = np.full(self.h, 1.0 / self.h)
        else:
            market_share = market_quality / total_quality
        history_weight = self.tb / self.ta
        return (
            history_weight[:, None]
            * investment_quality
            * market_share[:, None]
        )

    def update_search_memory(self, problem, evaluation_start):
        """Update region beliefs and retain the best current searcher."""
        for region in self.selected_regions:
            self.ta[int(region)] += 1.0
        self.tb += 1.0
        for region in self.selected_regions:
            self.tb[int(region)] = 1.0

        history_end = min(int(self.evatime), len(self.history))
        if history_end > evaluation_start:
            self.history[evaluation_start:history_end] = self.fitness

    def close_market(self):
        """Always stop region workers, including after an exception."""
        self.market.close(self.searchers)

    def search(self, problem, budget, state=None):
        """Run the SE market and return its selected decoded WSN state."""
        self.evaluation_limit = max(1, int(budget))
        self.history = np.zeros(self.evaluation_limit, dtype=float)
        self.evatime = 0
        self.iterations_completed = 0
        self.reset_best()

        self.initialize_market(problem)
        initial_end = min(int(self.evatime), len(self.history))
        self.history[:initial_end] = self.fitness
        self.arrange_resources(problem)
        try:
            while (
                self.evatime < self.evaluation_limit
                and self.iterations_completed < self.max_iterations
                and not bool(getattr(self, "early_stopped", False))
            ):
                evaluation_start = int(self.evatime)
                self.vision_search(problem)
                self.update_search_memory(problem, evaluation_start)

                self.on_iteration_finish(
                    problem=problem,
                    state=self.best_state,
                    iteration=self.iterations_completed,
                    run=0,
                    best_value=self.best_objectives,
                    fitness=self.fitness,
                )
                self.iterations_completed += 1
                if self.evatime <= evaluation_start:
                    break
        finally:
            self.close_market()

        if self.evatime < len(self.history):
            self.history[self.evatime :] = self.fitness
        if self.best_state is None:
            self.final_coding = None
            return None

        candidate_code = getattr(self.best_candidate, "code", None)
        self.final_coding = (
            self.best_candidate.copy()
            if candidate_code is not None
            else None
        )
        return self.best_state.copy()


__all__ = ["BaseSE"]
