"""Scheduling-only Search Economics algorithm."""

from __future__ import annotations

import numpy as np

from Algorithm.se.BaseSE import BaseSE
from Algorithm.Scheduling.SGA import ScheduleState


class SES(BaseSE):
    """Apply the common SE market to sensing schedules only."""

    def __init__(self, problem, n=5, h=4, w=2, seed=None):
        super().__init__(
            problem,
            n=n,
            h=h,
            w=w,
            code_length=problem.SENSOR_NUMBER,
            seed=seed,
        )
        self.Fname = ["remaining_energy", "minimum_energy", "coverage"]

    def create_candidate(self, problem, region=None):
        candidate = ScheduleState()
        candidate.CreateSchState(problem, self.rng)
        if region is not None:
            self.align_region(problem, candidate, region)
        return candidate

    def align_region(self, problem, candidate, region):
        current_region = int(np.sum(candidate.levels)) % self.h
        difference = int(region) - current_region
        if difference > 0:
            maximum_levels = problem.radius_option_counts - 1
            choices = np.where(candidate.levels < maximum_levels)[0]
            selected = self.rng.choice(
                choices,
                min(difference, len(choices)),
                replace=False,
            )
            candidate.levels[selected] += 1
        elif difference < 0:
            choices = np.where(candidate.levels > 0)[0]
            selected = self.rng.choice(
                choices,
                min(-difference, len(choices)),
                replace=False,
            )
            candidate.levels[selected] -= 1
        return candidate

    def invest(self, problem, searcher, good):
        difference = 2
        midpoint = self.code_length // 2
        left = self.random.randint(difference, midpoint - difference)
        right = self.random.randint(
            midpoint - difference,
            self.code_length - difference,
        )
        child_a = searcher.levels.copy()
        child_b = good.levels.copy()
        child_a[left:right], child_b[left:right] = (
            child_b[left:right].copy(),
            child_a[left:right].copy(),
        )
        investment = searcher.copy()
        investment.levels = (
            child_a if self.random.random() < 0.5 else child_b
        )

        if self.random.random() < self.mutation_rate:
            mutation_count = self.random.randint(1, 4)
            for _ in range(mutation_count):
                sensor_id = self.random.randrange(problem.SENSOR_NUMBER)
                if self.random.random() < 0.5:
                    if (
                        investment.levels[sensor_id]
                        < problem.sensing_option_count(sensor_id) - 1
                    ):
                        investment.levels[sensor_id] += 1
                elif investment.levels[sensor_id] > 0:
                    investment.levels[sensor_id] -= 1
        return investment

    def evaluate(self, problem, candidate):
        self.evatime += 1
        cost = problem.calculate_scheduling_cost(candidate)
        remaining_energy = problem.energy - cost
        total_energy = np.sum(remaining_energy) / np.sum(
            problem.initial_energy * problem.SENSOR_NUMBER
        )
        uncovered = problem.find_uncovered_targets(candidate.levels)
        used = cost != 0
        minimum_energy = (
            np.min(remaining_energy[used]) / problem.initial_energy
            if np.any(used)
            else 0.0
        )
        coverage = (
            problem.TARGET_NUMBER - len(uncovered)
        ) / problem.TARGET_NUMBER
        return np.asarray(
            [total_energy, minimum_energy, coverage],
            dtype=float,
        )


__all__ = ["SES"]
