import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.se.SA_SETS import SA_SETS
from Algorithm.se.SI_SETS import SI_SETS
from Problem.Problem import Problem


class SequentialSI(SI_SETS):
    """Reference evaluation order before process parallelization."""

    def arrange_resources(self, problem):
        pass

    def evaluate_investments(self, problem, investments):
        return np.asarray([
            self.evaluate_many(problem, candidates)
            for candidates in investments
        ])


def main():
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    operator_algorithm = SI_SETS(
        problem, n=4, h=4, w=1, mu=0.4, seed=31
    )

    assert SI_SETS.__bases__ == (SA_SETS,)
    for method_name in (
        "select_identity_sensors",
        "create_candidate",
        "align_region",
        "invest",
        "mutate_candidate",
    ):
        assert getattr(SI_SETS, method_name) is getattr(SA_SETS, method_name)
    operator_algorithm.identity_sensors = (
        operator_algorithm.select_identity_sensors(problem)
    )
    assert len(operator_algorithm.identity_sensors) == 2
    for region in range(4):
        candidate = operator_algorithm.create_candidate(problem, region)
        assert operator_algorithm.align_region(problem, candidate, region) is candidate
        for bit_id, sensor_id in enumerate(operator_algorithm.identity_sensors):
            level = int(candidate.code[sensor_id * 2]) % problem.sensing_option_count(sensor_id)
            assert (level > 0) == bool(region & (1 << bit_id))

    algorithm = SI_SETS(problem, n=4, h=4, w=1, mu=0.4, seed=31)
    result = algorithm.run(problem, budget=20)
    assert result.best_state is not None
    assert algorithm.evatime == 20
    assert algorithm.investment_quality.shape == (4, 4)
    assert algorithm.goods_fitness.shape == (4, 1)
    assert algorithm.market.pool.processes == []

    repeated = SI_SETS(
        problem,
        n=4,
        h=4,
        w=1,
        mu=0.4,
        seed=31,
    ).run(problem, budget=20)
    np.testing.assert_array_equal(
        result.best_state.levels, repeated.best_state.levels
    )
    np.testing.assert_array_equal(
        result.best_state.next_hops, repeated.best_state.next_hops
    )
    np.testing.assert_array_equal(
        result.best_state.tx_load, repeated.best_state.tx_load
    )

    assert algorithm._pool is None
    for seed in (7, 31):
        parallel = SI_SETS(problem, n=4, h=4, w=2, mu=0.4, seed=seed)
        serial = SequentialSI(problem, n=4, h=4, w=2, mu=0.4, seed=seed)
        parallel.run(problem, budget=100)
        serial.run(problem, budget=100)
        assert parallel.evatime == serial.evatime
        assert parallel.fitness == serial.fitness
        for name in ('levels', 'next_hops', 'tx_load', 'remaining_capacity'):
            np.testing.assert_array_equal(
                getattr(parallel.best_state, name), getattr(serial.best_state, name)
            )
        for name in ('history', 'goods_fitness', 'investment_quality',
                     'selected_regions', 'ta', 'tb'):
            np.testing.assert_array_equal(getattr(parallel, name), getattr(serial, name))
        for region in range(parallel.h):
            for good_id in range(parallel.w):
                np.testing.assert_array_equal(
                    parallel.goods[region][good_id].code,
                    serial.goods[region][good_id].code,
                )

    print("smoke_si_sets_ok")


if __name__ == "__main__":
    main()
