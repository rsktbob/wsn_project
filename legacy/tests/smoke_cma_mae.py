import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.map_elites.CMA_MAE import (
    CMA_MAE,
    _ArchiveElite,
    _SoftGridArchive,
)
from Problem.Problem import Problem
from State.TargetEncoding import TargetEncoding
from experiment_algorithms import build_algorithm


def _entry(problem, objective):
    state = TargetEncoding(np.zeros(problem.TARGET_NUMBER, dtype=int))
    elite = _ArchiveElite(
        theta=np.zeros(problem.TARGET_NUMBER),
        coding=state,
        objective=float(objective),
        measures=np.asarray([0.25, 0.25]),
        fitness=np.asarray([objective, 0.0, 0.0]),
        feasible=True,
        violations=0,
    )
    return {
        "elite": elite,
        "objective": float(objective),
        "measures": elite.measures,
    }


def main():
    np.random.seed(31)
    random.seed(31)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)

    # 驗證論文附錄 H：兩個同 cell 候選的門檻不受輸入順序影響。
    first_archive = _SoftGridArchive((10, 10), 0.01, 0.0)
    second_archive = _SoftGridArchive((10, 10), 0.01, 0.0)
    first_entries = [_entry(problem, 1.0), _entry(problem, 3.0)]
    second_entries = list(reversed(first_entries))
    first_archive.add_batch(first_entries)
    second_archive.add_batch(second_entries)
    index = first_archive.index_of([0.25, 0.25])
    expected = 2.0 * (1.0 - 0.99**2)
    assert math.isclose(first_archive.threshold_of(index), expected)
    assert math.isclose(
        first_archive.threshold_of(index),
        second_archive.threshold_of(index),
    )

    algorithm = CMA_MAE(
        problem,
        num_emitters=2,
        batch_size=8,
        sigma=0.2,
        archive_dims=(10, 10),
        learning_rate=0.01,
        threshold_min=0.0,
        seed=31,
    )
    result = algorithm.run(problem, budget=32)
    best = result.best_state
    assert best is not None
    assert not hasattr(result, "best_coding")
    assert algorithm.evatime == 32
    assert algorithm.generation == 2
    assert len(algorithm.emitters) == 2
    assert len(algorithm.archive.elites) > 0
    assert len(algorithm.result_archive.elites) > 0
    assert len(best.levels) == problem.SENSOR_NUMBER
    assert all(
        math.isfinite(float(value))
        for value in problem.evaluate_state(best)
    )

    args = SimpleNamespace(evaluate=100)
    registered = build_algorithm("cma_mae", problem, args, seed=31)
    assert isinstance(registered, CMA_MAE)
    assert registered.num_emitters == 15
    assert registered.batch_size == 36
    assert registered.archive_dims == (100, 100)
    print("smoke_cma_mae_ok")


if __name__ == "__main__":
    main()
