"""Characterize the shared evaluation-budget contract."""

import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.misc.CS import CS
from Algorithm.misc.EDA import EDA
from Algorithm.ga.GA import GA
from Algorithm.misc.GWO import GWO
from Algorithm.nsga.NSGAII import NSGAII
from Algorithm.misc.PSO import PSO
from Algorithm.se.SETSv2 import SETSv2
from Algorithm.Scheduling.SRIME import SRIME
from Problem.Problem import Problem


def create_problem():
    return Problem(B=50, S=30, T=9, F=100, FILE=None)


def run_case(factory, budget, expected_evaluations):
    np.random.seed(17)
    random.seed(17)
    problem = create_problem()
    algorithm = factory(problem)
    result = algorithm.run(problem, budget=budget)

    assert result.best_state is not None
    assert result.evaluations == expected_evaluations
    assert algorithm.evatime == expected_evaluations
    assert expected_evaluations >= budget


def main():
    # Initialization counts toward the same budget. A started batch/generation
    # is allowed to finish, which explains the small overshoots below.
    run_case(lambda p: GA(p, n=6, cu=0.8, mu=0.1), 12, 13)
    run_case(lambda p: NSGAII(p, n=4, mu=0.2), 8, 8)
    run_case(lambda p: PSO(p, n=4), 8, 10)
    run_case(lambda p: EDA(p, n=4, alpha=0.8), 8, 10)
    run_case(lambda p: CS(p, n=4), 8, 10)
    run_case(lambda p: GWO(p, n=4), 8, 11)
    run_case(
        lambda p: SRIME(n=4, generation=99, route_selector=None),
        8,
        10,
    )

    # Paper-style SETS evaluates one complete market iteration atomically.
    np.random.seed(17)
    random.seed(17)
    problem = create_problem()
    sets = SETSv2(problem, n=2, h=4, w=1, player=2)
    result = sets.run(problem, budget=sets.evaluations_per_iteration + 1)
    assert result.evaluations == sets.evaluations_per_iteration * 2

    print("regression_evaluation_budget_ok")


if __name__ == "__main__":
    main()
