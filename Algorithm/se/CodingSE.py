"""Sensor-encoded baseline Search Economics algorithm."""

from __future__ import annotations

import math

from Algorithm.se.BaseSE import BaseSE
from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding


class CodingSE(BaseSE):
    """Standard SE market flow with sensor schedule/routing chromosomes."""

    def __init__(self, problem, n=5, h=4, w=2, mu=0.2, seed=None):
        code_length = problem.SENSOR_NUMBER * 2
        super().__init__(
            problem,
            n=n,
            h=h,
            w=w,
            mu=mu,
            code_length=code_length,
            seed=seed,
        )
        self.identity_bit_count = int(math.log2(self.h))
        if 2**self.identity_bit_count != self.h:
            raise ValueError("CodingSE requires h to be a power of two")
        self.identity_sensors = []
        problem.prepare_coding_cache()

    def _select_identity_sensors(self, problem):
        return [
            sensor_id
            for sensor_id in range(problem.SENSOR_NUMBER)
            if problem.energy[sensor_id] >= problem.liveJ
        ][: self.identity_bit_count]

    def initialize_market(self, problem, initial_state=None):
        self.identity_sensors = self._select_identity_sensors(problem)
        if len(self.identity_sensors) != self.identity_bit_count:
            raise RuntimeError("not enough live sensors for CodingSE regions")
        super().initialize_market(problem, initial_state)

    def create_candidate(self, problem, region=None):
        candidate = SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
            rng=self.rng,
        )
        if region is not None:
            self.align_region(problem, candidate, region)
        return candidate

    def align_region(self, problem, candidate, region):
        bits = int(region)
        for bit_id, sensor_id in enumerate(self.identity_sensors):
            gene_id = sensor_id * 2
            must_open = (bits >> bit_id) & 1
            if must_open:
                option_count = problem.sensing_option_count(sensor_id)
                if int(candidate.code[gene_id]) % option_count == 0:
                    candidate.code[gene_id] = self.random.randrange(
                        1, option_count
                    )
            else:
                candidate.code[gene_id] = 0
        return candidate

    def invest(self, problem, searcher, good):
        difference = 2
        midpoint = self.code_length // 2
        left = self.random.randint(difference, midpoint - difference)
        right = self.random.randint(
            midpoint - difference,
            self.code_length - difference,
        )
        first, second = swap_segment(searcher, good, left, right)
        investment = first if self.random.random() < 0.5 else second

        if self.random.random() < self.mutation_rate:
            mutation_count = self.random.choice(
                [1] * 90 + [2] * 5 + [3] * 5
            )
            upper_bound = max(
                problem.LEVEL,
                SensorEncoding.RANK_PRECISION,
            )
            for _ in range(mutation_count):
                gene_id = self.random.randrange(self.code_length)
                investment.code[gene_id] = self.random.randrange(upper_bound)
        return investment


__all__ = ["CodingSE"]
