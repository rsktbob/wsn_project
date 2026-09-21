"""SI-SETS ablation with RL-SETSv4's role-separated market update."""

from __future__ import annotations

import numpy as np

from Algorithm.se.SI_SETS import SI_SETS
from State.Encoding import swap_segment


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

    def vision_search(self, problem):
        """Evaluate two roles and apply the RL-SETSv4 update without RL."""
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        self.current_adaptive_step = self.adaptive_step * (1.0 - progress)

        active_regions = self.selected_regions.copy()
        goods_fitness_before = self.goods_fitness.copy()
        searcher_fitness_before = self.searcher_fitness.copy()
        children = []

        for searcher_id, region in enumerate(active_regions):
            region = int(region)
            row = []
            for good_id in range(self.w):
                child1, child2 = self.make_children(
                    problem,
                    self.searchers[searcher_id],
                    self.goods[region][good_id],
                    region=region,
                )
                row.append(self.align_region(problem, child1, region))
                row.append(self.align_region(problem, child2, region))
            children.append(row)

        child_fitness = self.evaluate_investments(problem, children)
        child1_fitness = child_fitness[:, 0::2]
        child2_fitness = child_fitness[:, 1::2]

        # Match RL-SETSv4: region attractiveness measures how well child1
        # improves the visiting searcher side of the interaction.
        for searcher_id, region in enumerate(active_regions):
            self.investment_quality[int(region), searcher_id] = float(
                np.mean(child1_fitness[searcher_id])
            )

        probabilities = self.region_probabilities(
            goods_fitness_before,
            self.investment_quality,
        )
        selected = self.select_regions(probabilities)

        # A searcher considers only its own child1 proposals and remains
        # unchanged unless the best proposal improves its previous fitness.
        for searcher_id in range(self.n):
            good_id = int(np.argmax(child1_fitness[searcher_id]))
            score = float(child1_fitness[searcher_id, good_id])
            if score > searcher_fitness_before[searcher_id]:
                self.searchers[searcher_id] = children[searcher_id][
                    good_id * 2
                ].copy()
                self.searcher_fitness[searcher_id] = score

        # A visited good accepts the best child2 only when it improves the
        # pre-investment good; otherwise the old good remains in the market.
        for region in range(self.h):
            visitors = np.flatnonzero(active_regions == region)
            if not len(visitors):
                continue
            for good_id in range(self.w):
                winner = int(
                    max(
                        visitors,
                        key=lambda index: child2_fitness[int(index), good_id],
                    )
                )
                winner_score = float(child2_fitness[winner, good_id])
                if (
                    winner_score
                    <= goods_fitness_before[region, good_id]
                    + self.IMPROVEMENT_TOLERANCE
                ):
                    continue
                self.goods[region][good_id] = children[winner][
                    good_id * 2 + 1
                ].copy()
                self.goods_fitness[region, good_id] = winner_score

        self.selected_regions = selected


__all__ = ["SI_SETSv2"]
