"""Search Economics 的流程骨架。

所有 SE 演算法共用的只有這個主迴圈：

    initialize_market → arrange_resources →
    重複 { vision_search → update_search_memory } 直到預算用完 → close_market

每回合怎麼投資、怎麼選區、怎麼更新，都寫在各演算法自己的類別裡。
"""

from __future__ import annotations

import numpy as np

from Algorithm.core.Algorithm import Algorithm


class BaseSE(Algorithm):
    """SE main loop plus the state every SE algorithm needs."""

    # 更新規則的菁英設定（各演算法自己的更新都讀這兩個；改這裡全部演算法一起改）。
    # True＝E：嚴格變好才換；False＝N：一律換成最好的候選。
    SEARCHER_ELITIST = True
    GOOD_ELITIST = False

    def __init__(
        self,
        problem,
        n=5,
        h=4,
        w=2,
        mu=0.2,
        code_length=None,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.n = int(n)
        self.h = int(h)
        self.w = int(w)
        self.mutation_rate = float(mu)
        self.code_length = (
            None if code_length is None else int(code_length)
        )
        # ``length`` remains the conventional chromosome-size attribute used
        # by other optimizer families and experiment diagnostics.
        self.length = self.code_length

        if self.n <= 0 or self.h <= 0 or self.w <= 0:
            raise ValueError("n, h, and w must be positive")
        if not 0.0 <= self.mutation_rate <= 1.0:
            raise ValueError("mu must be between 0 and 1")

        self.name = (
            f"{self.__class__.__name__}_{self.n}_{self.h}_{self.w}_"
            f"{self.mutation_rate}"
        )
        self.searchers = []
        self.searcher_fitness = np.array([], dtype=float)
        self.selected_regions = np.array([], dtype=int)
        self.iterations_completed = 0
        self.evaluation_limit = 0

    def create_candidate(self, problem, region=None):
        """Create one candidate, optionally for a region."""
        raise NotImplementedError

    def initialize_market(self, problem, initial_state=None):
        """Create searchers; searcher i starts in region i mod h."""
        self.selected_regions = np.arange(self.n, dtype=int) % self.h
        self.searchers = [
            self.create_candidate(problem, region=int(region))
            for region in self.selected_regions
        ]
        if initial_state is not None and self.searchers:
            self.searchers[0] = initial_state.copy()

        self.searcher_fitness = self.evaluate_many(problem, self.searchers)

    def arrange_resources(self, problem):
        """Start worker processes if the algorithm uses any."""

    def vision_search(self, problem):
        """One market round: invest, choose regions, update searchers and goods."""
        raise NotImplementedError

    def update_search_memory(self, problem, evaluation_start):
        """Record this round's best fitness into the evaluation history."""
        history_end = min(int(self.evatime), len(self.history))
        if history_end > evaluation_start:
            self.history[evaluation_start:history_end] = self.fitness

    def close_market(self):
        """Stop worker processes; runs even after an exception."""

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
