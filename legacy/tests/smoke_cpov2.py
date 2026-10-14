import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.misc.CPOv2 import CPOv2
from Problem.Problem import Problem
from State.SensorEncoding import SensorEncoding
from experiment_algorithms import build_algorithm, parse_args


def create_problem(mode="discrete"):
    return Problem(B=50, S=30, T=9, F=100, FILE=None, sensing_mode=mode)


def main():
    for mode in ("discrete", "bucketed", "exact"):
        problem = create_problem(mode)
        algorithm = CPOv2(
            problem,
            n=6,
            min_population=3,
            adapt_interval=3,
            stagnation_limit=1000,
            seed=31,
        )
        result = algorithm.run(problem, budget=24)
        assert result.best_state is not None
        assert result.evaluations == 24
        assert len(result.history) == 24
        assert np.all(np.isfinite(result.history))
        assert algorithm.proposal_count == 18
        assert algorithm.restart_count == 0
        assert algorithm.encoding_change_count == 18
        assert sum(algorithm.operator_uses) == 18
        assert sum(algorithm.defence_counts.values()) == 18
        assert np.isclose(np.sum(algorithm.operator_probabilities), 1.0)
        assert np.all(algorithm.operator_probabilities >= 0.05)
        assert np.all(
            algorithm.population_scores >= algorithm.initial_population_scores
        )
        for encoding in algorithm.population:
            assert isinstance(encoding, SensorEncoding)
            assert np.all(encoding.code[0::2] < problem.radius_option_counts)
            assert np.all(encoding.code[1::2] < SensorEncoding.RANK_PRECISION)

        candidate = algorithm.population[0].copy()
        before = candidate.code.copy()
        algorithm._force_effective_change(problem, candidate)
        assert not np.array_equal(before, candidate.code)

    problem = create_problem()
    first = CPOv2(problem, n=6, min_population=3, seed=77).run(
        problem, budget=24
    )
    second = CPOv2(problem, n=6, min_population=3, seed=77).run(
        problem, budget=24
    )
    np.testing.assert_array_equal(first.best_state.levels, second.best_state.levels)
    np.testing.assert_array_equal(first.best_state.next_hops, second.best_state.next_hops)

    args = parse_args(["--algorithm", "cpo_v2", "--evaluate", "24"])
    registered = build_algorithm("cpo_v2", problem, args, seed=9)
    assert isinstance(registered, CPOv2)
    assert registered.n == 30

    print("smoke_cpov2_ok")


if __name__ == "__main__":
    main()
