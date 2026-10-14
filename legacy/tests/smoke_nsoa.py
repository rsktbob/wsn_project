import math
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.misc.NSOA import NSOA
from Problem.Problem import Problem


def main():
    np.random.seed(7)
    random.seed(7)

    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm = NSOA(problem, n=4, seed=7)

    early_mean, early_std = algorithm.ReproductionDistribution(0, 20)
    late_mean, late_std = algorithm.ReproductionDistribution(19, 20)
    assert 0.0 < early_std < 1.0
    assert 0.0 < late_std < early_std
    assert math.isclose(early_mean, 0.85)
    assert math.isclose(late_mean, 0.85)

    # WSN constraint handling is lexicographic: every feasible state outranks
    # a higher raw-fitness state that leaves a target uncovered.
    feasible_rank = (1, 0, 0, 0, 0, 0.8)
    infeasible_rank = (0, -1, -1, 0, 0, 10.0)
    assert algorithm._best_index([infeasible_rank, feasible_rank]) == 1
    assert algorithm._worst_indexes([infeasible_rank, feasible_rank], 1).tolist() == [0]

    seeded_positions = algorithm._create_initial_population(problem, 4)
    assert seeded_positions.shape == (4, problem.SENSOR_NUMBER * 2)
    assert np.all((0.0 <= seeded_positions) & (seeded_positions <= 1.0))

    result = algorithm.run(problem, budget=20)
    best = result.best_state
    assert best is not None
    assert not hasattr(result, "best_coding")
    assert algorithm.evatime == 20
    assert len(best.levels) == problem.SENSOR_NUMBER
    assert len(algorithm.history) == 20
    assert np.all(np.isfinite(algorithm.history))
    assert algorithm.elimination_history
    assert all(0.0 <= item["probability"] <= 1.0 for item in algorithm.elimination_history)
    assert all(item["count"] >= 0 for item in algorithm.elimination_history)
    assert any(item["count"] > 0 for item in algorithm.elimination_history)

    fitness = result.best_fitness
    assert len(fitness) == 3
    assert all(math.isfinite(float(value)) for value in fitness)

    position = seeded_positions[0]
    moved = algorithm.NaturalSelectionStep(
        np.array([position, position]), position
    )
    assert moved.shape == (2, len(position))
    assert np.all((0.0 <= moved) & (moved <= 1.0))

    print("smoke_nsoa_ok")


if __name__ == "__main__":
    main()
