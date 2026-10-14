import math
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.ga.GA import GA
from Problem.Problem import Problem


GA_PARAMS = {"n": 6, "cu": 0.8}


def assert_finite_fitness(fitness):
    assert len(fitness) == 3
    for value in fitness:
        assert math.isfinite(float(value))


def main():
    np.random.seed(7)
    random.seed(7)

    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm = GA(problem, **GA_PARAMS)

    state = algorithm._create_state(problem)
    fitness = algorithm.evaluate(problem, state)
    assert_finite_fitness(fitness)

    # 2023-thesis mutation, mu=1: every chromosome redraws exactly one gene,
    # so at most one gene differs, and every gene stays in range.
    domains = GA._gene_domains(problem)
    mutated = state.copy()
    GA(problem, mu=1.0, seed=3)._mutate(problem, [mutated])
    assert int(np.sum(mutated.code != state.code)) <= 1
    assert np.all((0 <= mutated.code) & (mutated.code < domains))
    # mu=0: nothing changes.
    unchanged = state.copy()
    GA(problem, mu=0.0, seed=3)._mutate(problem, [unchanged])
    assert np.array_equal(unchanged.code, state.code)
    assert_finite_fitness(algorithm.evaluate(problem, mutated))

    best = algorithm.run(problem, budget=12).best_state
    best_fitness = problem.evaluate_state(best)
    assert_finite_fitness(best_fitness)
    assert len(algorithm.history) == 12

    print("smoke_ga_ok")


if __name__ == "__main__":
    main()
