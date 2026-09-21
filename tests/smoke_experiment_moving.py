import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from experiment_algorithms import (
    configure_fitness_service,
    parse_bool,
    rebuild_problem_after_movement,
)
from Problem.services import FitnessServiceV2, RoutingService
from Problem.Problem import Problem


def main():
    assert parse_bool("true") is True
    assert parse_bool("false") is False

    problem = Problem(
        B=50,
        S=30,
        T=9,
        F=100,
        FILE=None,
        sensing_mode="bucketed",
        routing_service="v1",
    )
    configure_fitness_service(problem, "v2")
    positions = np.asarray(problem.sensor, dtype=float).copy()
    powers = np.arange(problem.SENSOR_NUMBER, dtype=float) + 1.0
    moved_sensor_id = problem.SENSOR_NUMBER - 1
    mobile_data = [
        [
            moved_sensor_id,
            positions[moved_sensor_id][0],
            positions[moved_sensor_id][1],
            positions[moved_sensor_id][0],
            positions[moved_sensor_id][1],
        ]
    ]

    distances = np.linalg.norm(positions - np.asarray(problem.BS), axis=1)
    order = np.argsort(distances, kind="stable")
    moved_problem = rebuild_problem_after_movement(
        problem, positions, powers, mobile_data
    )

    # 重排分兩段：rebuild_problem_after_movement 先依基地台距離排序，
    # Problem 再依 sensor_sort_order（sensor_ring_order 預設為 True，
    # 也就是「距離環 → 角度 → 舊 id」）重排一次。剩餘電量必須跟著
    # 這兩段一起走，才會與感測器維持相同索引。
    assert np.allclose(
        moved_problem.energy,
        powers[order][moved_problem.sensor_sort_order],
    )
    assert moved_problem.sensing_mode == "bucketed"
    assert moved_problem.fitness_service_name == "v2"
    assert type(moved_problem.fitness_service) is FitnessServiceV2
    assert moved_problem.routing_service_name == "v1"
    assert type(moved_problem.routing_service) is RoutingService
    assert np.all(
        moved_problem.radius_option_counts
        <= len(moved_problem.DISCRETE_LEVEL_RANGE)
    )
    expected_new_id = int(np.where(order == moved_sensor_id)[0][0])
    assert moved_problem.mobile_id[0][0] == expected_new_id

    print("smoke_experiment_moving_ok")


if __name__ == "__main__":
    main()
