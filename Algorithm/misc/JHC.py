"""以 SensorEncoding 執行的聯合爬山搜尋法。"""

import random
import time

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from Algorithm.core.initialization import (
    sample_sensor_gene,
)
from State.SensorEncoding import SensorEncoding


class JHC(Algorithm):
    """反覆產生鄰近解並保留較佳狀態。"""

    def __init__(self, problem):
        super().__init__()
        problem.prepare_coding_cache()
        self.evatime = 0
        self.history = np.array([], dtype=float)

    def _create_state(self, problem):
        return SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
        )

    def _evaluate_state(self, problem, coding):
        state = coding.decode(problem)
        self.evatime += 1
        return problem.evaluate_state(state)

    def _mutate_state(self, problem, coding):
        gene_id = random.randint(0, coding.length - 1)
        coding.code[gene_id] = sample_sensor_gene(problem)
        return coding

    def search(self, problem, budget, state=None):
        started_at = time.time()
        self.history = np.zeros(budget)
        self.evatime = 0
        coding = self._create_state(problem)
        fitness = self._evaluate_state(problem, coding)

        for iteration in range(budget):
            candidate = self._mutate_state(problem, coding.copy())
            candidate_fitness = self._evaluate_state(problem, candidate)
            if np.sum(candidate_fitness) > np.sum(fitness):
                fitness = candidate_fitness
                coding = candidate.copy()
            if iteration % 100 == 0:
                print(
                    iteration,
                    "最佳值：",
                    fitness,
                    np.sum(fitness),
                    time.time() - started_at,
                )
            self.on_iteration_finish(
                problem=problem,
                state=coding,
                iteration=iteration,
                run=0,
                Name="Test",
                time_cost=None,
                stop=False,
                scale=3,
                best_value=fitness,
                fitness=np.sum(fitness),
            )
            self.history[iteration] = np.sum(fitness)
        return coding.decode(problem)
