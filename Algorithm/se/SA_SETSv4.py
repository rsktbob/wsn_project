"""Ring-segment operator selection SA-SETS with one shared goods pool.

SA_SETSv3 tied two different decisions to the same region index: which pool
of goods a searcher trades with, and which sensor-index span the crossover
and mutation operators may touch. SA_SETSv4 splits those apart.

Goods are no longer partitioned per region. Every searcher trades against the
same shared pool of ``w`` goods every round, regardless of which ring segment
it visits. ``region`` keeps its attribute names (``self.h``,
``self.selected_regions``) for compatibility with the shared SE flow, but it
now does exactly one job: pick which ring-local sensor span this round's
crossover/mutation is allowed to modify. A segment's attractiveness is scored
purely by how much recent visits to it have improved the visiting searcher
(``child1``) -- there is no goods-pool quality or market-share term left to
blend in, because there is no longer a separate goods pool per segment.
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.SA_SETS import beta_cdf
from Algorithm.se.SA_SETSv3 import SA_SETSv3


class SA_SETSv4(SA_SETSv3):
    """Decouple the goods market from ring-segment operator selection.

    Every searcher still visits one of ``h`` ring segments each round, and
    crossover/mutation still stay inside that segment's sensor span (80%
    local / 20% cross-ring, unchanged from :class:`SA_SETSv3`,
    ``make_children``/``mutate_candidate``/``select_sensor_span`` are not
    overridden here). What changes is what a "segment" owns: it no longer
    owns a private pool of goods. All searchers trade against the same ``w``
    shared goods this round no matter which segment they picked.
    """

    def __init__(self, problem, n=8, h=4, w=2, mu=0.4, seed=None):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        self.segment_quality = np.empty((self.h, self.n), dtype=float)
        self.name = f"SA_SETSv4_{self.n}_{self.h}_{self.w}_{self.mutation_rate}"

    def initialize_market(self, problem, initial_state=None):
        """Create searchers and one shared goods pool (no per-segment pools)."""
        self.identity_sensors = []
        BaseSE.initialize_market(self, problem, initial_state)
        self.goods = [self.create_candidate(problem) for _ in range(self.w)]
        self.goods_fitness = self.evaluate_many(problem, self.goods)

        # No segment has a track record yet, so every segment starts from the
        # same neutral guess: the shared pool's current average quality.
        initial_quality = float(np.mean(self.goods_fitness))
        self.segment_quality = np.full(
            (self.h, self.n), initial_quality, dtype=float
        )

    def vision_search(self, problem):
        """Trade every searcher against the shared pool; score segments only
        by how well operating on them has paid off for the visiting searcher.
        """
        progress = min(1.0, self.evatime / max(1, self.evaluation_limit))
        self.current_adaptive_step = self.adaptive_step * (1.0 - progress)

        active_segments = self.selected_regions.copy()
        goods_fitness_before = self.goods_fitness.copy()
        searcher_fitness_before = self.searcher_fitness.copy()
        children = []

        for searcher_id, segment in enumerate(active_segments):
            segment = int(segment)
            row = []
            for good_id in range(self.w):
                child1, child2 = self.make_children(
                    problem,
                    self.searchers[searcher_id],
                    self.goods[good_id],
                    region=segment,
                )
                row.append(child1)
                row.append(child2)
            children.append(row)

        child_fitness = self.evaluate_investments(problem, children)
        child1_fitness = child_fitness[:, 0::2]
        child2_fitness = child_fitness[:, 1::2]

        # A segment's attractiveness is how well it improved the searcher who
        # just visited it -- there is no goods-pool quality or market-share
        # term to blend in, because goods are no longer partitioned by
        # segment.
        for searcher_id, segment in enumerate(active_segments):
            self.segment_quality[segment, searcher_id] = float(
                np.mean(child1_fitness[searcher_id])
            )

        probabilities = self.segment_probabilities(self.segment_quality)
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

        # The pool is shared: every searcher this round is a candidate
        # visitor for every good slot, regardless of which segment it
        # operated on. A good accepts the best child2 only when it improves
        # the pre-investment good by more than the elitist tolerance.
        for good_id in range(self.w):
            best_searcher = int(np.argmax(child2_fitness[:, good_id]))
            winner_score = float(child2_fitness[best_searcher, good_id])
            if (
                winner_score
                <= goods_fitness_before[good_id] + self.IMPROVEMENT_TOLERANCE
            ):
                continue
            self.goods[good_id] = children[best_searcher][good_id * 2 + 1].copy()
            self.goods_fitness[good_id] = winner_score

        self.selected_regions = selected

    def segment_probabilities(self, segment_quality):
        """Score each ring segment purely by its recent investment payoff.

        Unlike :meth:`SI_SETS.region_probabilities`, there is no
        ``best_goods`` or ``region_share`` term here: those measured a
        per-region goods pool that no longer exists. The Beta-CDF fairness
        correction (``ta``/``tb``) is unchanged, so segments that have not
        been visited recently still get a rising exploration bonus.
        """
        probability = np.empty((self.h, self.n), dtype=float)
        for segment in range(self.h):
            for searcher_id in range(self.n):
                probability[segment, searcher_id] = beta_cdf(
                    self.ta[segment],
                    self.tb[segment],
                    segment_quality[segment, searcher_id],
                )
        return probability


__all__ = ["SA_SETSv4"]
