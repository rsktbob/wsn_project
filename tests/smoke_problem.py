import random
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.ga.BaseGA import BaseGA
from Algorithm.ga.GA import GA
from Algorithm.se.SA_SETS import SA_SETS
from Problem.CodingProblem import CodingProblem
from Problem.Problem import Problem
from State.CodingState import CodingState


def build_and_evaluate(problem_cls):
    np.random.seed(3)
    random.seed(3)

    problem = problem_cls(B=50, S=30, T=9, F=100, FILE=None)
    assert int(np.min(np.sum(problem.target_sensor_mask, axis=1))) >= 3
    state = CodingState(problem)
    state.create_random_code(problem)
    state.decode(problem)
    fitness = state.evaluate(problem)
    updated_position, updated_power, mobile_id = problem.mobility_service.relocate_sensors(
        state=state,
        low_energy_targets=[],
        target_remaining_energy=np.zeros(problem.TARGET_NUMBER),
        failed_targets=[],
    )

    assert hasattr(problem, "coverage")
    assert hasattr(problem, "routing")
    assert hasattr(problem, "energy")
    assert hasattr(problem, "fitness")
    assert hasattr(problem, "mobility")
    assert len(problem.cover_candidates) == problem.TARGET_NUMBER
    assert len(problem.sensor_ids_by_bs_distance) == problem.SENSOR_NUMBER
    assert np.array_equal(
        np.sort(problem.sensor_ids_by_bs_distance),
        np.arange(problem.SENSOR_NUMBER),
    )
    # Chromosome ids are ring-major; angles within one ring run clockwise
    # from north (0 radians).
    assert np.all(np.diff(problem.sensor_ring_by_id) >= 0)
    for ring_id in np.unique(problem.sensor_ring_by_id):
        angles = problem.sensor_angle_by_id[
            problem.sensor_ring_by_id == ring_id
        ]
        assert np.all(np.diff(angles) >= 0)
    assert len(state.code) == problem.SENSOR_NUMBER * 2
    assert len(fitness) == 3
    assert np.all(np.isfinite([float(value) for value in fitness]))
    assert len(updated_position) == problem.SENSOR_NUMBER
    assert len(updated_power) == problem.SENSOR_NUMBER
    assert mobile_id == []
    return fitness


def main():
    build_and_evaluate(Problem)
    build_and_evaluate(CodingProblem)

    assert issubclass(GA, BaseGA)
    assert hasattr(SA_SETS, "create_candidate")
    assert hasattr(SA_SETS, "align_region")
    assert hasattr(CodingState, "create_random_code")
    assert hasattr(CodingState, "decode_routes")

    print("smoke_ok")


if __name__ == "__main__":
    main()
