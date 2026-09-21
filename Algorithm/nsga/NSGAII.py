import random

from Algorithm.nsga.BaseNSGAII import BaseNSGAII
from Algorithm.core.initialization import (
    sample_sensor_gene,
)
from State.SensorEncoding import SensorEncoding


class NSGAII(BaseNSGAII):
    """Combined WSN NSGA-II using sensor-oriented chromosomes."""

    def __init__(
        self,
        P,
        n=50,
        generation=100,
        cu=0.9,
        mu=None,
        tournament_size=2,
    ):
        super().__init__(
            n=n,
            generation=generation,
            cu=cu,
            mu=mu,
            tournament_size=tournament_size,
            coding_cls=SensorEncoding,
        )
        self.length = P.SENSOR_NUMBER * 2
        P.prepare_coding_cache()
        self.name = "NSGAII_" + str(n)

    def _create_coding(self, P):
        return SensorEncoding.random(
            P.SENSOR_NUMBER,
            P.radius_option_counts,
        )

    def _crossover(self, parent_a, parent_b):
        child_a = parent_a.copy()
        child_b = parent_b.copy()

        if random.random() >= self.cu:
            return child_a, child_b

        diff = 2
        half = max(1, self.length // 2)
        left_high = max(diff, half - diff)
        right_low = min(self.length - 1, half + diff)
        right_high = max(right_low, self.length - diff)
        start = random.randint(diff, left_high)
        end = random.randint(right_low, right_high)

        left_code = child_a.coding.code.copy()
        right_code = child_b.coding.code.copy()
        child_a.coding.code = self._crossover_segment(
            left_code,
            right_code,
            start,
            end,
        )
        child_b.coding.code = self._crossover_segment(
            right_code,
            left_code,
            start,
            end,
        )
        return child_a, child_b

    def _random_gene_value(self, P, coding, gene_id):
        """Generate one sensor-chromosome gene."""
        return sample_sensor_gene(P)

    def _crossover_segment(self, father_code, mother_code, start, end):
        child_code = father_code.copy()
        child_code[start:end] = mother_code[start:end]
        return child_code
