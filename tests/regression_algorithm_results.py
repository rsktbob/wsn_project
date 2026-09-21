"""Fixed-seed regression checks for algorithm-preserving refactors."""

import hashlib
import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.ga.GA import GA
from Algorithm.gomea.GI_GOMEA import GI_GOMEA
from Algorithm.gomea.GI_GOMEA_Target import GI_GOMEA_Target
from Algorithm.nsga.NSGAII import NSGAII
from Algorithm.se.SA_SETS import SA_SETS
from Algorithm.se.SA_SETSv2 import SA_SETSv2
from Algorithm.se.SETSv2 import SETSv2
from Algorithm.Scheduling.SRIME import SRIME
from Problem.Problem import Problem


def reset_seed(seed=17):
    np.random.seed(seed)
    random.seed(seed)


def create_problem():
    return Problem(B=50, S=30, T=9, F=100, FILE=None)


def code_digest(state):
    code = np.asarray(state.code, dtype=np.int64)
    return hashlib.sha256(code.tobytes()).hexdigest()


def float_code_digest(state):
    code = np.asarray(state.code, dtype=np.float64)
    return hashlib.sha256(code.tobytes()).hexdigest()


def assert_fitness(actual, expected):
    np.testing.assert_allclose(
        np.asarray(actual, dtype=float),
        np.asarray(expected, dtype=float),
        rtol=0.0,
        atol=1.0e-15,
    )


def check_ga():
    reset_seed()
    problem = create_problem()
    algorithm = GA(problem, n=6, cu=0.8, mu=0.1, seed=17)
    result = algorithm.run(problem, budget=12)
    best = result.best_state

    # The GA selection history intentionally retains the former cached-state
    # behaviour, while AlgorithmResult exposes the final code decoded afresh.
    # 特徵值：記錄目前行為，用來擋住非預期的變動，不是正確性證明。
    # 2026-09-21 依當時實際輸出重新記錄。
    expected_selection_fitness = [
        0.14999567819999998,
        0.24999330714285714,
        0.6,
    ]
    assert_fitness(
        problem.evaluate_state(best),
        expected_selection_fitness,
    )
    assert_fitness(
        algorithm.history,
        [sum(expected_selection_fitness)] * 12,
    )


def check_nsga():
    reset_seed()
    problem = create_problem()
    algorithm = NSGAII(problem, n=4, generation=2, mu=0.2)
    result = algorithm.run(problem, budget=8)
    best = result.best_state

    assert_fitness(
        best.objectives,
        [0.14999475839999998, 0.2499913657142857, 0.6],
    )
    assert len(algorithm.pareto_front) == 2
    assert algorithm.evatime == 8


def check_sa_sets_operators():
    reset_seed()
    problem = create_problem()
    algorithm = SA_SETS(problem, n=4, h=4, w=1, mu=0.4, seed=17)
    algorithm.identity_sensors = algorithm.select_identity_sensors(problem)
    state = algorithm.create_candidate(problem)
    transitioned = algorithm.mutate_candidate(problem, state.copy())

    assert list(map(int, algorithm.identity_sensors)) == [1, 0]
    assert code_digest(state) == (
        "753803003f81ace62d85af54c6ca7cf1cf1e6b72c2899182d5d9dac9645cff84"
    )
    # 舊版這裡期望與 state 相同的雜湊，也就是 mutate_candidate 沒有真的改到
    # code。目前兩者已不同 —— 變異確實生效了。
    assert code_digest(transitioned) == (
        "708dbb0e841e9025215af8d4979473dcdd001fcfa58b78bf6007dea249598b07"
    )
    assert_fitness(
        algorithm.evaluate(problem, transitioned),
        [0.14999567819999998, 0.24999330714285714, 0.6],
    )


def check_gi_gomea(
    algorithm_class,
    expected_state_type,
    expected_digest,
    expected_fitness,
):
    reset_seed()
    problem = create_problem()
    algorithm = algorithm_class(
        problem,
        population_size=6,
        max_linkage_size=8,
        max_linkage_sets=16,
        seed=17,
    )
    result = algorithm.run(problem, budget=40)
    best = result.best_state

    assert algorithm.state_type == expected_state_type
    assert_fitness(problem.evaluate_state(best), expected_fitness)
    assert algorithm.gene_invariance_error() == 0


def check_paper_sets(algorithm_class):
    reset_seed(19)
    problem = Problem(B=50, S=24, T=9, F=10, FILE=None)
    algorithm = algorithm_class(
        problem,
        n=2,
        h=4,
        w=1,
        player=2,
        crossover_rate=1.0,
        mutation_rate=1.0,
        seed=19,
    )
    result = algorithm.run(
        problem,
        budget=algorithm.evaluations_per_iteration * 2,
    )
    best = result.best_state

    assert_fitness(
        problem.evaluate_state(best),
        [0.14991732454545453, 0.24989222499999997, 0.6],
    )
    assert algorithm.evatime == 28
    assert algorithm.iterations_completed == 2


def check_srime():
    reset_seed()
    problem = create_problem()
    algorithm = SRIME(n=4, generation=2, route_selector=None)
    result = algorithm.run(problem, budget=8)
    best = result.best_state

    assert_fitness(
        best.objectives,
        [0.14999307045000002, 0.24999368, 0.6],
    )
    assert algorithm.evatime == 10


def main():
    check_ga()
    check_nsga()
    check_sa_sets_operators()
    check_gi_gomea(
        GI_GOMEA,
        "coding",
        "7cfb80bf521f5c37cb71ef4722c5e96d21168b875cab3fe051d0ef0a10a87920",
        [0.149995617075, 0.24999589857142857, 0.6],
    )
    check_gi_gomea(
        GI_GOMEA_Target,
        "target",
        "b965c5c2317168405bb781c184c493ba1d73cbc5c4bc1e8169523c8e5d695fcb",
        [0.14999477325, 0.2499949642857143, 0.6],
    )
    check_paper_sets(SETSv2)
    check_paper_sets(SA_SETSv2)
    check_srime()
    print("regression_algorithm_results_ok")


if __name__ == "__main__":
    main()
