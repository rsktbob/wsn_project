import math
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.core.Algorithm import Algorithm
from Algorithm.misc.CS import CS
from Algorithm.Scheduling.SRIME import SRIME
from Problem.Problem import Problem


def assert_finite_fitness(fitness):
    assert len(fitness) == 3
    assert all(math.isfinite(float(value)) for value in fitness)


def main():
    assert CS.__bases__ == (Algorithm,)
    assert SRIME.__bases__ == (Algorithm,)

    np.random.seed(7)
    random.seed(7)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    srime = SRIME(n=4, generation=2, route_selector=None)
    best_srime = srime.run(problem, budget=8).best_state
    assert best_srime is not None
    assert_finite_fitness(best_srime.objectives)
    assert srime.evatime == 10

    print("smoke_lightweight_bases_ok")


if __name__ == "__main__":
    main()
