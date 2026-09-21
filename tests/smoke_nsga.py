import math
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.nsga.NSGAII import NSGAII
from Problem.Problem import Problem


def assert_finite_state(state):
    assert state is not None
    assert state.objectives is not None
    assert math.isfinite(float(state.value))
    assert math.isfinite(float(sum(state.objectives)))


def main():
    np.random.seed(7)
    random.seed(7)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)

    algorithm = NSGAII(problem, n=4, generation=2, mu=0.2)
    individual = algorithm._create_individual(problem)
    assert_finite_state(individual.state)

    original_code = individual.coding.code.copy()
    algorithm.mu = 1.0
    mutated = algorithm._mutate(problem, individual.copy())
    assert np.count_nonzero(mutated.coding.code != original_code) == (
        individual.coding.length
    )
    algorithm.mu = 0.2

    best = algorithm.run(problem, budget=8).best_state
    assert_finite_state(best)
    assert len(algorithm.pareto_front) > 0
    assert algorithm.evatime == 8

    print("smoke_nsga_ok")


if __name__ == "__main__":
    main()
