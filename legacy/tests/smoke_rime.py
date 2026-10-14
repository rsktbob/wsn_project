import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.misc.RIME import RIME
from Problem.Problem import Problem
from State.SensorEncoding import SensorEncoding
from experiment_algorithms import build_algorithm, parse_args


def create_problem(mode):
    return Problem(
        B=50,
        S=30,
        T=9,
        F=100,
        FILE=None,
        sensing_mode=mode,
    )


def assert_population_domains(algorithm, problem):
    assert len(algorithm.initial_population) == algorithm.n
    for encoding in algorithm.initial_population:
        assert isinstance(encoding, SensorEncoding)
        levels = np.asarray(encoding.code[0::2], dtype=int)
        route_weights = np.asarray(encoding.code[1::2], dtype=int)
        assert np.all(levels >= 0)
        assert np.all(levels < problem.radius_option_counts)
        assert np.all(route_weights >= 0)
        assert np.all(route_weights < SensorEncoding.RANK_PRECISION)


def main():
    domain_tables = {}
    for mode in ("discrete", "bucketed", "exact"):
        problem = create_problem(mode)
        algorithm = RIME(problem, n=4, w=5, seed=31)
        result = algorithm.run(problem, budget=12)
        assert result.best_state is not None
        assert result.evaluations == 12
        assert len(result.history) == 12
        assert algorithm.iterations_completed == 2
        assert_population_domains(algorithm, problem)
        assert np.all(
            algorithm.population_scores >= algorithm.initial_population_scores
        )
        domain_tables[mode] = problem.radius_option_counts.copy()

    # Non-discrete modes derive sensor-specific choices from target distances.
    assert np.all(domain_tables["discrete"] == domain_tables["discrete"][0])
    assert not np.array_equal(
        domain_tables["discrete"], domain_tables["exact"]
    )

    problem = create_problem("discrete")
    first = RIME(problem, n=4, seed=77).run(problem, budget=12)
    second = RIME(problem, n=4, seed=77).run(problem, budget=12)
    np.testing.assert_array_equal(first.best_state.levels, second.best_state.levels)
    np.testing.assert_array_equal(
        first.best_state.next_hops,
        second.best_state.next_hops,
    )

    args = parse_args(["--algorithm", "rime", "--evaluate", "12"])
    registered = build_algorithm("rime", problem, args, seed=9)
    assert isinstance(registered, RIME)
    assert registered.n == 50
    assert registered.w == 5

    print("smoke_rime_ok")


if __name__ == "__main__":
    main()
