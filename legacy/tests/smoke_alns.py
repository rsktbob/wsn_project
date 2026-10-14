import math
import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.misc.ALNS import ALNS
from Problem.Problem import Problem


def main():
    np.random.seed(7)
    random.seed(7)

    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm = ALNS(
        problem,
        initial_samples=4,
        segment_length=4,
        seed=7,
    )

    result = algorithm.run(problem, budget=24)
    best = result.best_state
    assert best is not None
    assert algorithm.evatime == 24
    assert not hasattr(result, "best_coding")
    assert len(best.levels) == problem.SENSOR_NUMBER
    assert len(algorithm.history) == 24
    assert np.all(np.isfinite(algorithm.history))
    assert algorithm.operator_history
    assert len(algorithm.destroy_weights) == len(ALNS.DESTROY_OPERATORS)
    assert len(algorithm.repair_weights) == len(ALNS.REPAIR_OPERATORS)
    assert np.all(algorithm.destroy_weights > 0)
    assert np.all(algorithm.repair_weights > 0)

    values = result.best_fitness
    assert len(values) == 3
    assert all(math.isfinite(float(value)) for value in values)

    used_destroy = {item["destroy"] for item in algorithm.operator_history}
    used_repair = {item["repair"] for item in algorithm.operator_history}
    assert used_destroy.issubset(set(ALNS.DESTROY_OPERATORS))
    assert used_repair.issubset(set(ALNS.REPAIR_OPERATORS))
    print("smoke_alns_ok")


if __name__ == "__main__":
    main()
