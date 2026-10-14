import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.misc.EOPT import EOPT
from Problem.Problem import Problem
from experiment_algorithms import build_algorithm, parse_args


def main():
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm = EOPT(problem, tau=1.5, initial_samples=4, seed=7)
    result = algorithm.run(problem, budget=24)

    assert result.best_state is not None
    assert result.evaluations == 24
    assert len(result.history) == 24
    assert np.all(np.isfinite(result.history))
    assert len(result.best_fitness) == 3
    assert all(math.isfinite(float(value)) for value in result.best_fitness)
    assert result.metadata["algorithm"] == "EOPT"

    # 同一實例在 lifetime 中會重複搜尋；每段都應重新取得完整 budget。
    second_result = algorithm.run(problem, budget=24)
    assert second_result.evaluations == 24
    assert len(second_result.history) == 24

    args = parse_args(["--algorithm", "eopt"])
    built = build_algorithm("eopt", problem, args, seed=7)
    assert isinstance(built, EOPT)
    print("smoke_eopt_ok")


if __name__ == "__main__":
    main()
