import math
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.CodingSEv2 import CodingSEv2
from Problem.Problem import Problem
from State.TargetEncoding import TargetEncoding


def main():
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm = CodingSEv2(problem, seed=7)

    assert isinstance(algorithm, BaseSE)
    assert (
        algorithm.n,
        algorithm.h,
        algorithm.w,
        algorithm.mutation_rate,
    ) == (5, 4, 2, 1.0)
    state = algorithm.create_candidate(problem)
    assert isinstance(state, TargetEncoding)
    assert len(state.code) == problem.TARGET_NUMBER
    assert np.all((state.code >= 0) & (state.code < 10000))

    region_state = state.copy()
    algorithm.align_region(problem, region_state, 2)
    assert region_state.code[2] % TargetEncoding.RANK_PRECISION == 0

    mutated = algorithm.invest(problem, state, state.copy())
    assert np.all(mutated.code >= 0)

    values = problem.evaluate_state(state.decode(problem))
    assert len(values) == 3
    assert all(math.isfinite(float(value)) for value in values)

    run_algorithm = CodingSEv2(problem, seed=7)
    result = run_algorithm.run(problem, budget=80)
    best = result.best_state
    assert best is not None
    assert not hasattr(result, "best_coding")
    assert run_algorithm.evatime == 93
    assert len(run_algorithm.history) == 80
    assert np.all(np.isfinite(run_algorithm.history))

    repeat = CodingSEv2(problem, seed=7).run(problem, budget=80)
    np.testing.assert_array_equal(best.levels, repeat.best_state.levels)
    np.testing.assert_array_equal(
        best.next_hops, repeat.best_state.next_hops
    )
    np.testing.assert_array_equal(best.tx_load, repeat.best_state.tx_load)
    print("smoke_codingsev2_ok")


if __name__ == "__main__":
    main()
