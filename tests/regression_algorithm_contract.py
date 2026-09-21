"""統一 Algorithm.run 契約與固定結果的回歸測試。"""

import hashlib
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.core.Algorithm import Algorithm
from Algorithm.ga.GA import GA
from Algorithm.gomea.GI_GOMEA import GI_GOMEA
from Algorithm.nsga.NSGAII import NSGAII
from Algorithm.core import AlgorithmResult, BatchEvaluator
from Problem.Problem import Problem


def reset_seed(seed=17):
    np.random.seed(seed)
    random.seed(seed)


def create_problem():
    return Problem(B=50, S=30, T=9, F=100, FILE=None)


def digest(state):
    arrays = [
        np.asarray(getattr(state, name), dtype=np.int64).reshape(-1)
        for name in ("sch", "rou", "use")
    ]
    return hashlib.sha256(np.concatenate(arrays).tobytes()).hexdigest()


def run_contract(factory, budget):
    reset_seed()
    problem = create_problem()
    algorithm = factory(problem)
    result = algorithm.run(problem, budget=budget)
    assert isinstance(result, AlgorithmResult)
    return result, algorithm, problem


def assert_result(factory, budget, expected_digest=None):
    result, current, problem = run_contract(factory, budget)

    if expected_digest is not None:
        assert digest(result.best_state) == expected_digest
    repeated, _, _ = run_contract(factory, budget)
    assert digest(result.best_state) == digest(repeated.best_state)
    np.testing.assert_array_equal(result.history, current.history)
    assert result.evaluations == current.evatime
    assert result.iterations > 0
    np.testing.assert_allclose(
        result.best_fitness,
        problem.evaluate_state(result.best_state),
        rtol=0.0,
        atol=1.0e-15,
    )


def main():
    assert_result(
        lambda problem: GA(problem, n=6, cu=0.8, mu=0.1, seed=17),
        12,
    )
    assert_result(
        lambda problem: NSGAII(problem, n=4, generation=2, mu=0.2),
        8,
    )
    assert_result(
        lambda problem: GI_GOMEA(
            problem,
            population_size=6,
            max_linkage_size=8,
            max_linkage_sets=16,
            seed=17,
        ),
        40,
    )

    # 共用 base 不包含任何家族 operator 或多程序實作。
    assert set(Algorithm.__dict__).isdisjoint(
        {
            "crossover_population",
            "update_velocity",
            "update_probability_vector",
            "_invest",
            "_learn_linkage_model",
        }
    )
    assert BatchEvaluator.__module__ == "Algorithm.core.evaluation"

    print("regression_algorithm_contract_ok")


if __name__ == "__main__":
    main()
