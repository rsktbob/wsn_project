import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.map_elites.Differential_MAP_Elites import (
    Differential_MAP_Elites,
)
from Problem.Problem import Problem
from State.TargetEncoding import TargetEncoding
from experiment_algorithms import build_algorithm


def main():
    np.random.seed(37)
    random.seed(37)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm = Differential_MAP_Elites(
        problem,
        centroid_count=64,
        initial_sample_ratio=0.125,
        minimum_initial_samples=8,
        scaling_factor=0.5,
        crossover_rate=0.9,
        cvt_sample_count=512,
        cvt_iterations=2,
        seed=37,
    )
    result = algorithm.run(problem, budget=64)
    best = result.best_state
    assert best is not None
    assert not hasattr(result, "best_coding")
    assert algorithm.evatime == 64
    assert algorithm.iteration >= 55
    assert algorithm.archive.centroids.shape == (64, 2)
    assert len(algorithm.archive.elites) >= 4
    assert algorithm.archive.additions >= 4
    assert len(best.levels) == problem.SENSOR_NUMBER
    fitness = problem.evaluate_state(best)
    assert all(math.isfinite(float(value)) for value in fitness)

    # 論文要求 target 與三個 donor 來自四個相異非空 cell。
    selected = algorithm.archive.select_four(algorithm.rng)
    assert selected is not None
    assert len({id(elite) for elite in selected}) == 4

    args = SimpleNamespace(evaluate=100)
    registered = build_algorithm(
        "differential_map_elites", problem, args, seed=37
    )
    assert isinstance(registered, Differential_MAP_Elites)
    assert registered.centroid_count == 25000
    assert registered.scaling_factor == 0.5
    assert registered.crossover_rate == 0.9
    print("smoke_differential_map_elites_ok")


if __name__ == "__main__":
    main()
