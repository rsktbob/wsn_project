import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.misc.CPO import CPO
from Problem.Problem import Problem
from State.SensorEncoding import SensorEncoding
from experiment_algorithms import build_algorithm, parse_args


def create_problem(mode="discrete"):
    return Problem(B=50, S=30, T=9, F=100, FILE=None, sensing_mode=mode)


def main():
    for mode in ("discrete", "bucketed", "exact"):
        problem = create_problem(mode)
        algorithm = CPO(
            problem,
            n=6,
            min_population=3,
            cycles=2,
            seed=31,
        )
        result = algorithm.run(problem, budget=24)
        assert result.best_state is not None
        assert result.evaluations == 24
        assert len(result.history) == 24
        assert np.all(np.isfinite(result.history))
        assert algorithm.proposal_count == 18
        assert sum(algorithm.defence_counts.values()) == 18
        assert algorithm.active_population_history
        assert all(1 <= size <= 6 for size in algorithm.active_population_history)
        assert np.all(
            algorithm.population_scores >= algorithm.initial_population_scores
        )

        for encoding in algorithm.initial_population:
            assert isinstance(encoding, SensorEncoding)
            assert np.all(encoding.code[0::2] >= 0)
            assert np.all(
                encoding.code[0::2] < problem.radius_option_counts
            )
            assert np.all(encoding.code[1::2] >= 0)
            assert np.all(
                encoding.code[1::2] < SensorEncoding.RANK_PRECISION
            )

        # Every target's sampled SA-SETS seed is represented by at least one
        # selected sensor/level assignment in a single-target problem.
        one_target = Problem(B=50, S=30, T=1, F=100, FILE=None, sensing_mode=mode)
        seeded = CPO(one_target, n=2, min_population=1, seed=7)
        candidate = seeded._coverage_seeded_encoding(one_target)
        choices = {
            (int(sensor), int(level))
            for sensor, level in one_target.cover_candidates[0]
        }
        seeded_pairs = {
            (sensor, int(candidate.code[sensor * 2]))
            for sensor in range(one_target.SENSOR_NUMBER)
        }
        assert choices & seeded_pairs

    problem = create_problem()
    first = CPO(problem, n=6, min_population=3, seed=77).run(
        problem, budget=24
    )
    second = CPO(problem, n=6, min_population=3, seed=77).run(
        problem, budget=24
    )
    np.testing.assert_array_equal(first.best_state.levels, second.best_state.levels)
    np.testing.assert_array_equal(first.best_state.next_hops, second.best_state.next_hops)

    args = parse_args(["--algorithm", "cpo", "--evaluate", "24"])
    registered = build_algorithm("cpo", problem, args, seed=9)
    assert isinstance(registered, CPO)
    assert registered.n == 30
    assert registered.min_population == 10

    print("smoke_cpo_ok")


if __name__ == "__main__":
    main()
