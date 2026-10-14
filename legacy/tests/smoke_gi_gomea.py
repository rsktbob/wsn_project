import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.gomea.GI_GOMEA import GI_GOMEA
from Algorithm.gomea.GI_GOMEA_Target import GI_GOMEA_Target
from Problem.Problem import Problem
from experiment_algorithms import build_algorithm


def main():
    np.random.seed(7)
    random.seed(7)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm = GI_GOMEA(
        problem,
        population_size=8,
        max_linkage_size=8,
        max_linkage_sets=24,
        seed=7,
    )

    result = algorithm.run(problem, budget=80)
    best = result.best_state
    assert best is not None
    assert not hasattr(result, "best_coding")
    assert 78 <= algorithm.evatime <= 80
    assert len(algorithm.population) == 8
    assert algorithm.last_linkage_model
    assert algorithm.gene_invariance_error() == 0
    assert algorithm.accepted_swaps + algorithm.rejected_swaps > 0

    fitness = problem.evaluate_state(best)
    assert all(math.isfinite(float(value)) for value in fitness)
    assert len(best.levels) == problem.SENSOR_NUMBER
    assert len(best.next_hops) == problem.SENSOR_NUMBER

    # 第二次執行應沿用族群，且重新評估後仍維持每個位置的等位值數量。
    second = algorithm.run(problem, budget=48).best_state
    assert second is not None
    assert algorithm.gene_invariance_error() == 0

    target_algorithm = GI_GOMEA_Target(
        problem,
        population_size=8,
        max_linkage_size=8,
        max_linkage_sets=24,
        seed=7,
    )
    target_result = target_algorithm.run(problem, budget=80)
    target_best = target_result.best_state
    assert target_best is not None
    assert target_algorithm.length == problem.TARGET_NUMBER
    assert target_algorithm.gene_invariance_error() == 0
    assert len(target_best.levels) == problem.SENSOR_NUMBER

    args = SimpleNamespace(evaluate=40)
    new_gi = build_algorithm("gi_gomea", problem, args, seed=7)
    new_target = build_algorithm(
        "gi_gomea_target",
        problem,
        args,
        seed=7,
    )
    assert type(new_gi) is GI_GOMEA
    assert type(new_target) is GI_GOMEA_Target

    result = new_gi.run(problem, budget=40)
    assert result.best_state is not None
    print("smoke_gi_gomea_ok")


if __name__ == "__main__":
    main()
