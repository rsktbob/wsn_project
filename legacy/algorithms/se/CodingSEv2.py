"""Target-encoded Search Economics algorithm adapted from the 2021 project."""

from __future__ import annotations

from Algorithm.se.BaseSE import BaseSE
from State.Encoding import swap_segment
from State.TargetEncoding import TargetEncoding


class CodingSEv2(BaseSE):
    """Use one target gene for candidate selection and processing priority."""

    def __init__(self, problem, n=5, h=4, w=2, mu=1.0, seed=None):
        problem.prepare_coding_cache()
        super().__init__(
            problem,
            n=n,
            h=h,
            w=w,
            mu=mu,
            code_length=problem.TARGET_NUMBER,
            seed=seed,
        )
        self.name = f"CodingSEv2_{n}{h}{w}{mu}"

    def create_candidate(self, problem, region=None):
        candidate = TargetEncoding.random(
            problem.TARGET_NUMBER,
            rng=self.rng,
        )
        if region is not None:
            self.align_region(problem, candidate, region)
        return candidate

    def align_region(self, problem, candidate, region):
        if self.code_length:
            target_id = int(region) % self.code_length
            gene = max(0, int(candidate.code[target_id]))
            candidate.code[target_id] = (
                gene // TargetEncoding.RANK_PRECISION
            ) * TargetEncoding.RANK_PRECISION
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

        if self.random.random() < self.mutation_rate and self.code_length:
            mutation_count = self.random.randint(
                1, min(3, self.code_length)
            )
            mutation_domain = (problem.SENSOR_NUMBER + 100) * 4
            for _ in range(mutation_count):
                target_id = self.random.randrange(self.code_length)
                investment.code[target_id] = self.random.randrange(
                    mutation_domain
                )
        return investment


__all__ = ["CodingSEv2"]
