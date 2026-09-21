"""Paper-oriented SETS implementation built on the common SE phases."""

from __future__ import annotations

import math

import numpy as np

from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.SA_SETS import beta_cdf
from Algorithm.core.budget import iterations_to_reach_budget
from State.SensorEncoding import SensorEncoding


class SETSv2(BaseSE):
    """2022 SETS market flow with paper-style regional archives.

    This implementation shares only the four SE phases with ``BaseSE``.  It
    deliberately replaces the legacy process-backed vision search because the
    paper evaluates searchers, goods, and investments together each round.
    """

    def __init__(
        self,
        problem,
        n=8,
        h=4,
        w=2,
        player=2,
        crossover_rate=1.0,
        mutation_rate=0.4,
        adaptive_constant=0.001,
        seed=None,
    ):
        self.crossover_rate = float(crossover_rate)
        self.adaptive_constant = float(adaptive_constant)
        code_length = problem.SENSOR_NUMBER * 2
        super().__init__(
            problem,
            n=n,
            h=h,
            w=w,
            mu=mutation_rate,
            code_length=code_length,
            player=player,
            seed=seed,
        )

        self.identity_bit_count = int(math.floor(math.log2(self.h)))
        if self.h < 2 or 2**self.identity_bit_count != self.h:
            raise ValueError("SETSv2 requires h to be a power of two")
        if not 0.0 <= self.crossover_rate <= 1.0:
            raise ValueError("crossover_rate must be between 0 and 1")

        self.identity_sensors = []
        self.searcher_regions = np.zeros(self.n, dtype=int)
        self.goods = []
        self.goods_fitness = np.empty((self.h, self.w), dtype=float)
        self.region_best_fitness = np.full(self.h, -np.inf, dtype=float)
        self.region_best_states = [None for _ in range(self.h)]
        self.global_best_state = None
        self.global_best_fitness = -np.inf
        self.total_iterations = 0
        self.name = (
            f"{self.__class__.__name__}_{self.n}_{self.h}_{self.w}_"
            f"{self.crossover_rate}_{self.mutation_rate}"
        )

    @property
    def evaluations_per_iteration(self):
        """Number of evaluations in one complete S + M + V paper round."""
        return self.n + self.h * self.w + self.h * self.n * self.w

    @property
    def identity_gene_indices(self):
        """Return sensing-gene positions reserved for region identities."""
        return {sensor_id * 2 for sensor_id in self.identity_sensors}

    def select_identity_sensors(self, problem):
        """Select the nearest sensors used by the 2022 SETS classifier."""
        distances = np.asarray(
            problem.distances[: problem.SENSOR_NUMBER, problem.BSID],
            dtype=float,
        )
        order = np.argsort(distances, kind="stable")
        return order[: self.identity_bit_count].astype(int).tolist()

    def create_candidate(self, problem, region=None):
        """Create a domain-valid sensor chromosome for one region."""
        candidate = SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
            rng=self.rng,
        )
        if region is not None:
            self.align_region(problem, candidate, region)
        return candidate

    def _active_schedule_value(self, problem, sensor_id):
        option_count = problem.sensing_option_count(sensor_id)
        if option_count <= 1:
            return 0
        return self.random.randint(1, option_count - 1)

    def align_region(self, problem, candidate, region):
        """Repair identity genes to match the paper's reversed region pattern."""
        identity_pattern = self.h - 1 - int(region)
        for bit, sensor_id in enumerate(self.identity_sensors):
            shift = self.identity_bit_count - 1 - bit
            must_open = (identity_pattern >> shift) & 1
            gene_id = sensor_id * 2
            if must_open:
                option_count = problem.sensing_option_count(sensor_id)
                if int(candidate.code[gene_id]) % option_count == 0:
                    candidate.code[gene_id] = self._active_schedule_value(
                        problem, sensor_id
                    )
            else:
                candidate.code[gene_id] = 0
        return candidate

    def choose_crossover_points(self):
        """Choose two non-identity positions for paper-style crossover."""
        candidates = [
            index
            for index in range(1, self.code_length)
            if index not in self.identity_gene_indices
        ]
        if len(candidates) < 2:
            return None
        return tuple(sorted(self.random.sample(candidates, 2)))

    def _crossover(self, searcher, good):
        child = searcher.copy()
        if self.random.random() > self.crossover_rate:
            return child
        points = self.choose_crossover_points()
        if points is None:
            return child

        first, second = points
        child_a = searcher.code.copy()
        child_b = good.code.copy()
        child_a[first:second], child_b[first:second] = (
            child_b[first:second].copy(),
            child_a[first:second].copy(),
        )
        child.code = child_a if self.random.random() < 0.5 else child_b
        return child

    def _different_random_value(self, old_value, upper_bound):
        old_value = int(old_value)
        if upper_bound <= 1:
            return old_value
        value = self.random.randrange(upper_bound - 1)
        return value + 1 if value >= old_value else value

    def _mutate(self, problem, candidate):
        if self.random.random() > self.mutation_rate:
            return candidate
        mutation_count = min(
            self.random.randint(1, 3),
            self.code_length,
        )
        for gene_id in self.random.sample(
            range(self.code_length), mutation_count
        ):
            upper_bound = (
                problem.sensing_option_count(gene_id // 2)
                if gene_id % 2 == 0
                else SensorEncoding.RANK_PRECISION
            )
            candidate.code[gene_id] = self._different_random_value(
                candidate.code[gene_id], upper_bound
            )
        return candidate

    def invest(self, problem, searcher, good):
        """Apply SETSv2 crossover and mutation before region alignment."""
        investment = self._crossover(searcher, good)
        return self._mutate(problem, investment)

    def initialize_market(self, problem, initial_state=None):
        """Initialize classification, searchers, beliefs, and archives."""
        self.ta = np.ones(self.h, dtype=float)
        self.tb = np.ones(self.h, dtype=float)
        self.region_best_fitness = np.full(self.h, -np.inf, dtype=float)
        self.region_best_states = [None for _ in range(self.h)]
        self.global_best_state = None
        self.global_best_fitness = -np.inf

        self.identity_sensors = self.select_identity_sensors(problem)
        if len(self.identity_sensors) != self.identity_bit_count:
            raise RuntimeError("unable to select all SETS identity sensors")
        self.searcher_regions = np.asarray(
            [self.random.randrange(self.h) for _ in range(self.n)],
            dtype=int,
        )
        self.selected_regions = self.searcher_regions.copy()
        self.searchers = [
            self.create_candidate(problem, region=int(region))
            for region in self.searcher_regions
        ]
        if initial_state is not None and self.searchers:
            self.searchers[0] = self.align_region(
                problem,
                initial_state.copy(),
                int(self.searcher_regions[0]),
            )

        self.total_iterations = iterations_to_reach_budget(
            self.evaluation_limit,
            initialization_evaluations=0,
            evaluations_per_iteration=self.evaluations_per_iteration,
        )

    def arrange_resources(self, problem):
        """Create the paper's in-memory goods for every region."""
        self.goods = [
            [
                self.create_candidate(problem, region=region)
                for _ in range(self.w)
            ]
            for region in range(self.h)
        ]

    def _archive_regions(
        self,
        current_regions,
        investments,
        investment_fitness,
    ):
        """Update every region's historical best candidate ``B_j``."""
        for region in range(self.h):
            candidates = []
            for searcher_id in np.where(current_regions == region)[0]:
                candidates.append(
                    (
                        float(self.searcher_fitness[searcher_id]),
                        self.searchers[searcher_id],
                    )
                )
            for good_id in range(self.w):
                candidates.append(
                    (
                        float(self.goods_fitness[region, good_id]),
                        self.goods[region][good_id],
                    )
                )
            for searcher_id in range(self.n):
                for good_id in range(self.w):
                    candidates.append(
                        (
                            float(
                                investment_fitness[
                                    region, searcher_id, good_id
                                ]
                            ),
                            investments[region][searcher_id][good_id],
                        )
                    )

            fitness, candidate = max(candidates, key=lambda item: item[0])
            if fitness > self.region_best_fitness[region]:
                self.region_best_fitness[region] = fitness
                self.region_best_states[region] = candidate.copy()

    def region_probabilities(self, goods_fitness, investment_fitness):
        """Calculate formulas (45)-(46) for paper-style region choice."""
        goods_total = float(np.sum(goods_fitness))
        if abs(goods_total) <= np.finfo(float).eps:
            region_share = np.full(self.h, 1.0 / self.h)
        else:
            region_share = np.sum(goods_fitness, axis=1) / goods_total

        probability = np.zeros((self.h, self.n), dtype=float)
        for region in range(self.h):
            historical_best = self.region_best_fitness[region]
            if not np.isfinite(historical_best):
                historical_best = 0.0
            for searcher_id in range(self.n):
                average_investment = float(
                    np.mean(investment_fitness[region, searcher_id])
                )
                benefit = (
                    historical_best
                    * average_investment
                    * region_share[region]
                )
                probability[region, searcher_id] = beta_cdf(
                    self.tb[region],
                    self.ta[region],
                    np.clip(benefit, 0.0, 1.0),
                )
        return probability

    def vision_search(self, problem):
        """Generate and evaluate S + M + V, then choose new regions."""
        current_regions = self.searcher_regions.copy()
        investments = self.create_investments(problem, self.goods)

        self.searcher_fitness = self.evaluate_many(problem, self.searchers)
        flat_goods = [good for region_goods in self.goods for good in region_goods]
        self.goods_fitness = self.evaluate_many(problem, flat_goods).reshape(
            self.h, self.w
        )
        flat_investments = [
            investments[region][searcher_id][good_id]
            for region in range(self.h)
            for searcher_id in range(self.n)
            for good_id in range(self.w)
        ]
        investment_fitness = self.evaluate_many(
            problem, flat_investments
        ).reshape(self.h, self.n, self.w)

        self._archive_regions(
            current_regions,
            investments,
            investment_fitness,
        )

        for searcher_id in range(self.n):
            own_investments = investment_fitness[:, searcher_id, :]
            flat_id = int(np.argmax(own_investments))
            region, good_id = np.unravel_index(flat_id, (self.h, self.w))
            candidate_fitness = float(own_investments[region, good_id])
            if candidate_fitness > self.searcher_fitness[searcher_id]:
                self.searchers[searcher_id] = investments[region][
                    searcher_id
                ][good_id].copy()
                self.searcher_fitness[searcher_id] = candidate_fitness

        for region in range(self.h):
            for good_id in range(self.w):
                searcher_id = int(
                    np.argmax(investment_fitness[region, :, good_id])
                )
                candidate_fitness = float(
                    investment_fitness[region, searcher_id, good_id]
                )
                if candidate_fitness > self.goods_fitness[region, good_id]:
                    self.goods[region][good_id] = investments[region][
                        searcher_id
                    ][good_id].copy()
                    self.goods_fitness[region, good_id] = candidate_fitness

        probability = self.region_probabilities(
            self.goods_fitness,
            investment_fitness,
        )
        self.selected_regions = self.select_regions(probability)

    def _archive_global(self, candidate, fitness):
        if fitness >= self.global_best_fitness:
            self.global_best_state = candidate.copy()
            self.global_best_fitness = float(fitness)

    def update_search_memory(self, problem, evaluation_start):
        """Archive the round, update beliefs, and write fitness history."""
        current_regions = self.searcher_regions.copy()
        for searcher_id, region in enumerate(current_regions):
            self._archive_global(
                self.searchers[searcher_id],
                self.searcher_fitness[searcher_id],
            )
            for good_id in range(self.w):
                self._archive_global(
                    self.goods[int(region)][good_id],
                    self.goods_fitness[int(region), good_id],
                )

        step = self.adaptive_constant * (
            1.0
            - self.iterations_completed / max(1, self.total_iterations)
        )
        for selected_region in self.selected_regions:
            selected_region = int(selected_region)
            self.ta[selected_region] += step
            for region in range(self.h):
                if region != selected_region:
                    self.tb[region] += step

        for searcher_id, region in enumerate(self.selected_regions):
            self.align_region(
                problem,
                self.searchers[searcher_id],
                int(region),
            )
        self.searcher_regions = self.selected_regions.copy()

        history_end = min(int(self.evatime), len(self.history))
        if history_end > evaluation_start:
            self.history[evaluation_start:history_end] = self.fitness

    def close_market(self):
        """SETSv2 uses an in-memory market, so no worker cleanup is needed."""


__all__ = ["SETSv2"]
