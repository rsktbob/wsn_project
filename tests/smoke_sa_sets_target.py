import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.se.SA_SETS_Target import SA_SETS_Target
from Problem.Problem import Problem
from State.TargetEncoding import TargetEncoding


PARAMS = {"n": 4, "h": 4, "w": 2, "mu": 0.4}


def assert_finite_fitness(fitness):
    assert len(fitness) == 3
    assert all(math.isfinite(float(value)) for value in fitness)


def main():
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm = SA_SETS_Target(problem, seed=23, **PARAMS)
    algorithm.identity_sensors = algorithm.select_identity_sensors(problem)
    state = algorithm.create_candidate(problem)

    assert isinstance(state, TargetEncoding)
    assert algorithm.code_length == problem.TARGET_NUMBER
    assert len(algorithm.critical_targets) == algorithm.identity_bit_count
    assert np.all(state.code >= 0)
    assert np.all(state.code < algorithm.TARGET_GENE_DOMAIN)

    low_region = algorithm.align_region(problem, state.copy(), 0)
    high_region = algorithm.align_region(
        problem, state.copy(), (1 << algorithm.identity_bit_count) - 1
    )
    for target_id in algorithm.critical_targets:
        _, _, reachable = algorithm._candidate_layout(problem, target_id)
        if reachable <= 1:
            continue
        split = max(1, reachable // 2)
        assert (
            algorithm._decode_candidate_index(
                problem, target_id, low_region.code[target_id]
            )
            < split
        )
        assert (
            algorithm._decode_candidate_index(
                problem, target_id, high_region.code[target_id]
            )
            >= split
        )

    assert_finite_fitness(algorithm.evaluate(problem, state))
    transitioned = algorithm._mutate(problem, state.copy())
    assert np.all(transitioned.code >= 0)
    assert np.all(transitioned.code < algorithm.TARGET_GENE_DOMAIN)
    assert_finite_fitness(algorithm.evaluate(problem, transitioned))

    result = algorithm.run(problem, budget=128)
    assert not hasattr(result, "best_coding")
    assert_finite_fitness(problem.evaluate_state(result.best_state))

    print("smoke_sa_sets_target_ok")


if __name__ == "__main__":
    main()
