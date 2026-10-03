"""SI-SETS ablation with RL-SETSv4's role-separated market update."""

from __future__ import annotations

import numpy as np

from Algorithm.se.SI_SETS import SI_SETS
from State.Encoding import swap_segment
from Algorithm.se.population_updates import searchers_from_children, goods_from_children


class SI_SETSv2(SI_SETS):
    """Use fixed crossover/mutation with separate searcher and good children.

    This class contains no learning model. For every searcher-good pair,
    child1 is based on the searcher and child2 is based on the good. Child1
    may improve only its searcher; the best visiting child2 replaces its good
    only when better. This tests an elitist version of the role-separated
    market update.
    """

    IMPROVEMENT_TOLERANCE = 1e-12

    def __init__(self, problem, n=8, h=4, w=2, mu=0.4, seed=None):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        self.name = f"SI_SETSv2_{self.n}_{self.h}_{self.w}_{mu}"

    def make_children(self, problem, searcher, good, region=None):
        """Crossover both directions, then mutate each child with rate mu."""
        difference = 2
        midpoint = self.code_length // 2
        if midpoint <= difference or self.code_length - difference <= midpoint:
            child1, child2 = searcher.copy(), good.copy()
        else:
            left = self.random.randint(difference, midpoint - difference)
            right = self.random.randint(
                midpoint - difference,
                self.code_length - difference,
            )
            child1, child2 = swap_segment(searcher, good, left, right)

        for child in (child1, child2):
            if self.random.random() < self.mutation_rate:
                self.mutate_candidate(problem, child)
        return child1, child2

    def make_offspring(self, problem, searcher, good, region):
        return tuple(self.align_region(problem, child, region)
                     for child in self.make_children(problem, searcher, good, region))

    def summarize_round(self, active_regions, scores):
        return super().summarize_round(active_regions, scores[:, 0::2])

    def update_searchers(self, children, scores, selected, goods_before):
        searchers_from_children(self, [row[0::2] for row in children], scores[:, 0::2])

    def update_goods(self, children, scores, active_regions):
        child2 = [row[1::2] for row in children]
        for region in range(self.h):
            goods_from_children(self.goods[region], self.goods_fitness[region],
                child2, scores[:, 1::2], np.flatnonzero(active_regions == region),
                tolerance=self.IMPROVEMENT_TOLERANCE)


__all__ = ["SI_SETSv2"]
