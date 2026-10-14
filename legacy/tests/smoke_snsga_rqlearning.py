import math
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.Routing.RQLearning import RQLearning
from Algorithm.Scheduling.SNSGAII import SNSGAII
from Problem.Problem import Problem


def assert_finite_state(state):
    assert state is not None
    assert state.objectives is not None
    assert math.isfinite(float(state.value))
    assert math.isfinite(float(sum(state.objectives)))


def main():
    np.random.seed(11)
    random.seed(11)

    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    problem.prepare_coding_cache()

    algorithm = SNSGAII(
        n=4,
        generation=2,
        mu=0.2,
        rqlearning_kwargs={"epsilon": 0.0, "seed": 11},
    )
    assert isinstance(algorithm.route_selector, RQLearning)

    individual = algorithm._create_individual(problem)
    assert_finite_state(individual.state)
    assert individual.coding.route_selector is algorithm.route_selector

    best = algorithm.run(problem, budget=8).best_state
    assert_finite_state(best)
    assert len(algorithm.pareto_front) > 0
    assert isinstance(algorithm.route_selector.q_table, dict)

    print("smoke_snsga_rqlearning_ok")


if __name__ == "__main__":
    main()
