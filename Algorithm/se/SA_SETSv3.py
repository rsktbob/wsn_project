"""Ring-local, role-separated SA-SETS market.

Each searcher chooses one ring region to visit. A visit exchanges a
ring-local sensor-pair segment with one good and keeps both directional
children: the searcher-side child may update only its searcher, while the
good-side child may update only its good.
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.SI_SETSv2 import SI_SETSv2
from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding


class SA_SETSv3(SI_SETSv2):
    """SA-SETS with soft ring-local operators and elitist role updates.

    For ``h=4`` and 100 ring-ordered sensors, regions are ``[0:25)``,
    ``[25:50)``, ``[50:75)``, and ``[75:100)``. On every visit, both
    directions of crossover are retained:

    * ``child1`` is searcher ``a`` with the selected segment from good ``b``;
      only ``a`` may accept it.
    * ``child2`` is good ``b`` with the selected segment from searcher ``a``;
      only ``b`` may accept it.

    Both updates are elitist through :class:`SI_SETSv2`. The selected region
    is a soft operator mask: 80% of visits operate within the selected ring
    region, while 20% include that entire region and a non-empty adjacent
    region fragment. Mutation follows the same chosen mask.
    """

    LOCAL_OPERATOR_PROBABILITY = 0.80

    def __init__(self, problem, n=8, h=4, w=2, mu=0.4, seed=None):
        super().__init__(problem, n=n, h=h, w=w, mu=mu, seed=seed)
        if problem.SENSOR_NUMBER < self.h:
            raise ValueError("SA_SETSv3 requires at least one sensor per region")
        self.region_sensor_bounds = self._build_region_sensor_bounds(
            problem.SENSOR_NUMBER
        )
        self.name = f"SA_SETSv3_{self.n}_{self.h}_{self.w}_{self.mutation_rate}"

    def _build_region_sensor_bounds(self, sensor_count):
        """Split ring-major sensor ids into contiguous, near-equal regions."""
        base_size, remainder = divmod(int(sensor_count), self.h)
        bounds = []
        start = 0
        for region in range(self.h):
            end = start + base_size + (1 if region < remainder else 0)
            bounds.append((start, end))
            start = end
        return tuple(bounds)

    def initialize_market(self, problem, initial_state=None):
        """Create unrestricted V2 searchers and persistent regional goods.

        ``SI_SETS`` normally inherits SA-SETS' identity-sensor partition.
        Ring regions replace that hard partition here, so initialization uses
        ``BaseSE`` directly and leaves every sensing gene free.
        """
        self.identity_sensors = []
        BaseSE.initialize_market(self, problem, initial_state)
        self.goods = [
            [self.create_candidate(problem, region=region) for _ in range(self.w)]
            for region in range(self.h)
        ]
        flat_goods = [good for region_goods in self.goods for good in region_goods]
        self.goods_fitness = self.evaluate_many(problem, flat_goods).reshape(
            self.h,
            self.w,
        )
        initial_quality = np.mean(self.goods_fitness, axis=1)
        self.investment_quality = np.repeat(initial_quality[:, None], self.n, axis=1)

    def create_candidate(self, problem, region=None):
        """Create the V2 coverage/routing chromosome without hard alignment."""
        candidate = SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
            rng=self.rng,
        )
        for target_candidates in problem.cover_candidates:
            if target_candidates:
                sensor_id, level = self.random.choice(target_candidates)
                candidate.code[int(sensor_id) * 2] = int(level)
        return candidate

    def align_region(self, problem, candidate, region):
        """Ring membership guides operators only; never force sensing levels."""
        return candidate

    def _resolve_region(self, region):
        region = int(region)
        if not 0 <= region < self.h:
            raise ValueError("region index is outside the configured market")
        return region

    def _local_sensor_span(self, region):
        start, end = self.region_sensor_bounds[region]
        left = self.random.randrange(start, end)
        right = self.random.randrange(left + 1, end + 1)
        return left, right

    def _cross_ring_sensor_span(self, region):
        """Include all of ``region`` plus a non-empty adjacent fragment."""
        start, end = self.region_sensor_bounds[region]
        neighbours = []
        if region > 0:
            neighbours.append(region - 1)
        if region + 1 < self.h:
            neighbours.append(region + 1)
        neighbour = self.random.choice(neighbours)
        neighbour_start, neighbour_end = self.region_sensor_bounds[neighbour]
        if neighbour < region:
            return self.random.randrange(neighbour_start, start), end
        return start, self.random.randrange(end + 1, neighbour_end + 1)

    def select_sensor_span(self, region):
        """Choose the 80/20 crossover-and-mutation scope for one visit."""
        region = self._resolve_region(region)
        if self.random.random() < self.LOCAL_OPERATOR_PROBABILITY:
            left, right = self._local_sensor_span(region)
            return "local", left, right
        left, right = self._cross_ring_sensor_span(region)
        return "cross_ring", left, right

    def make_children(self, problem, searcher, good, region=None):
        """Return role-preserving children for one selected regional visit."""
        region = self._resolve_region(region)
        sensor_span = self.select_sensor_span(region)
        _, left_sensor, right_sensor = sensor_span
        child1, child2 = swap_segment(
            searcher,
            good,
            left_sensor * 2,
            right_sensor * 2,
        )

        # Each child independently mutates with probability mu, but both use
        # the exact ring scope selected for this investor-good transaction.
        for child in (child1, child2):
            if self.random.random() < self.mutation_rate:
                self.mutate_candidate(problem, child, sensor_span=sensor_span)
        return child1, child2

    def mutate_candidate(self, problem, candidate, sensor_span):
        """Mutate one to three sensor pairs inside the chosen ring scope."""
        _, left_sensor, right_sensor = sensor_span
        mutation_count = self.random.choice([1] * 90 + [2] * 5 + [3] * 5)
        for _ in range(mutation_count):
            sensor_id = self.random.randrange(left_sensor, right_sensor)
            sensing_id = sensor_id * 2
            priority_id = sensing_id + 1
            operation = self.random.randint(0, 2)
            if operation in (0, 1):
                sensing_bound = problem.sensing_option_count(sensor_id)
                candidate.code[sensing_id] = self.random.randrange(sensing_bound)
            if operation in (0, 2):
                candidate.code[priority_id] = self.random.randrange(
                    SensorEncoding.RANK_PRECISION
                )
        return candidate


__all__ = ["SA_SETSv3"]
