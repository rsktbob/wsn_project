import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.se.BaseSE import BaseSE
from Algorithm.se.SA_SETS import SA_SETS
from Problem.Problem import Problem
from State.SensorEncoding import SensorEncoding


SA_SETS_PARAMS = {"n": 8, "h": 4, "w": 2, "mu": 0.4}


def assert_finite_fitness(fitness):
    assert len(fitness) == 3
    for value in fitness:
        assert math.isfinite(float(value))


def main():
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm = SA_SETS(problem, seed=7, **SA_SETS_PARAMS)
    assert not hasattr(algorithm, "market_mode")
    algorithm.identity_sensors = algorithm.select_identity_sensors(problem)

    state = algorithm.create_candidate(problem)
    for sensor_id in algorithm.identity_sensors:
        state.code[sensor_id * 2] = 1
    state = algorithm.align_region(problem, state, 0)
    assert all(
        state.code[sensor_id * 2] == 0
        for sensor_id in algorithm.identity_sensors
    )

    last_region = (1 << algorithm.identity_bit_count) - 1
    for sensor_id in algorithm.identity_sensors:
        state.code[sensor_id * 2] = 0
    state = algorithm.align_region(problem, state, last_region)
    assert all(
        0 < state.code[sensor_id * 2] < problem.sensing_option_count(sensor_id)
        for sensor_id in algorithm.identity_sensors
    )
    fitness = algorithm.evaluate(problem, state)
    assert_finite_fitness(fitness)

    investment = algorithm.invest(problem, state, state.copy())
    transition_fitness = algorithm.evaluate(problem, investment)
    assert_finite_fitness(transition_fitness)

    # mutatev2 降低 level 或關閉 sensor 後，必須使用 CCS 補回所有
    # target；已開啟的替代 sensor 則保留目前值與需求值中的較大者。
    repair_code = np.zeros(problem.SENSOR_NUMBER * 2, dtype=int)
    for target_candidates in problem.CCS:
        assert target_candidates
        sensor_id, required_level = target_candidates[0]
        level_id = int(sensor_id) * 2
        repair_code[level_id] = max(
            int(repair_code[level_id]),
            int(required_level),
        )
        repair_code[level_id + 1] = SensorEncoding.RANK_PRECISION - 1
    repair_candidate = SensorEncoding(repair_code)
    assert not len(
        problem.coverage_service.find_uncovered_targets(
            repair_candidate.code[0::2]
        )
    )
    for _ in range(100):
        algorithm.mutatev2(problem, repair_candidate)
        assert not len(
            problem.coverage_service.find_uncovered_targets(
                repair_candidate.code[0::2]
            )
        )
    # 舊的純隨機 mutation 仍保留供實驗切換。
    assert algorithm.mutatev1(problem, repair_candidate.copy()) is not None

    algorithm.evatime = 0
    algorithm.initialize_market(problem)
    assert len(algorithm.searchers) == algorithm.n
    assert len(algorithm.searcher_fitness) == algorithm.n
    for value in algorithm.searcher_fitness:
        assert math.isfinite(float(value))

    assert hasattr(algorithm, "select_identity_sensors")
    assert hasattr(algorithm, "align_region")
    assert hasattr(algorithm, "region_probabilities")

    # The evaluation budget includes initial searchers and initial region goods.
    # A complete market-search batch may finish slightly beyond the limit.
    budget_problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    budget_algorithm = SA_SETS(
        budget_problem, n=4, h=4, w=1, mu=0.4, seed=7
    )
    result = budget_algorithm.run(
        budget_problem,
        budget=40,
    )
    best = result.best_state
    assert best is not None
    assert not hasattr(result, "best_coding")
    assert not hasattr(BaseSE, "_retain_best_searcher")
    assert budget_algorithm.best_state is not None
    assert len(budget_algorithm.best_objectives) == 3
    assert np.isclose(
        budget_algorithm.fitness,
        np.sum(budget_algorithm.best_objectives),
    )
    assert budget_algorithm.coverage is None
    # Common best tracking uses only scalar fitness. Coverage is metadata and
    # must not silently introduce a second ordering rule.
    tracked_state = budget_algorithm.best_state.copy()
    budget_algorithm.reset_best()
    assert budget_algorithm.update_best(
        budget_problem,
        tracked_state,
        np.asarray([0.1]),
        0.1,
        1.0,
    )
    assert budget_algorithm.update_best(
        budget_problem,
        tracked_state,
        np.asarray([0.2]),
        0.2,
        0.5,
    )
    assert budget_algorithm.fitness == 0.2
    assert budget_algorithm.coverage == 0.5
    # Initial searchers (4), initial regional goods (4), and two complete
    # investment rounds (16 each). Best tracking reuses those evaluations.
    assert budget_algorithm.evatime == 40
    assert len(result.history) == 40
    repeat_algorithm = SA_SETS(
        budget_problem, n=4, h=4, w=1, mu=0.4, seed=7
    )
    repeated = repeat_algorithm.run(budget_problem, budget=40)
    np.testing.assert_array_equal(best.levels, repeated.best_state.levels)
    np.testing.assert_array_equal(
        best.next_hops, repeated.best_state.next_hops
    )
    np.testing.assert_array_equal(best.tx_load, repeated.best_state.tx_load)

    bucketed_problem = Problem(
        B=50,
        S=30,
        T=9,
        F=100,
        FILE=None,
        sensing_mode="bucketed",
    )
    bucketed_algorithm = SA_SETS(
        bucketed_problem,
        n=4,
        h=4,
        w=1,
        mu=0.4,
        seed=17,
    )
    bucketed_algorithm.identity_sensors = (
        bucketed_algorithm.select_identity_sensors(bucketed_problem)
    )
    bucketed_state = bucketed_algorithm.create_candidate(bucketed_problem)
    for region in range(bucketed_algorithm.h):
        candidate = bucketed_state.copy()
        for sensor_id in bucketed_algorithm.identity_sensors:
            # 放入非零但解碼後為 0 的 raw gene，驗證分區會修正它。
            candidate.code[sensor_id * 2] = (
                bucketed_problem.sensing_option_count(sensor_id) * 2
            )
        bucketed_algorithm.align_region(
            bucketed_problem,
            candidate,
            region,
        )
        for bit, sensor_id in enumerate(bucketed_algorithm.identity_sensors):
            option = int(candidate.code[sensor_id * 2])
            option_count = bucketed_problem.sensing_option_count(sensor_id)
            if (region >> bit) & 1:
                assert 0 < option < option_count
            else:
                assert option == 0

    bucketed_result = bucketed_algorithm.run(
        bucketed_problem,
        budget=40,
    )
    bucketed_best = bucketed_result.best_state
    assert bucketed_best is not None
    assert np.all(
        bucketed_best.levels < bucketed_problem.radius_option_counts
    )
    assert np.allclose(
        bucketed_best.sensing_radii,
        bucketed_problem.resolve_radius(bucketed_best.levels),
    )

    print("smoke_sa_sets_ok")


if __name__ == "__main__":
    main()
