import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Problem.Problem import Problem
from Algorithm.gomea.GI_GOMEA import GI_GOMEA
from Algorithm.gomea.GI_GOMEA_Target import GI_GOMEA_Target
from experiments.builder import build_algorithm
from State.SensorEncoding import SensorEncoding
from State.TargetEncoding import TargetEncoding
from experiment_algorithms import configured_algorithm_params


def build_problem(mode):
    np.random.seed(19)
    random.seed(19)
    return Problem(
        B=50,
        S=30,
        T=9,
        F=100,
        FILE=None,
        sensing_mode=mode,
    )


def assert_state_radius_contract(problem, state):
    expected = problem.resolve_radius(state.levels)
    assert state.sensing_radii.shape == (problem.SENSOR_NUMBER,)
    assert np.allclose(
        state.sensing_radii, expected, rtol=0.0, atol=1e-12
    )
    sensing_cost = problem.calculate_scheduling_cost(state)
    assert np.allclose(
        sensing_cost,
        expected**2 * problem.sensing_factor,
        rtol=0.0,
        atol=1e-15,
    )


def main():
    discrete = build_problem("discrete")
    assert discrete.sensing_mode == "discrete"
    assert discrete.radius_table.shape == (
        discrete.SENSOR_NUMBER,
        discrete.LEVEL,
    )
    assert np.allclose(
        discrete.radius_table,
        discrete.DISCRETE_LEVEL_RANGE[None, :],
    )

    coding = SensorEncoding.random(
        discrete.SENSOR_NUMBER,
        discrete.radius_option_counts,
    )
    state = coding.decode(discrete)
    assert_state_radius_contract(discrete, state)

    bucketed = build_problem("bucketed")
    assert bucketed.sensing_mode == "bucketed"
    assert bucketed.LEVEL <= len(bucketed.DISCRETE_LEVEL_RANGE)
    assert np.all(
        bucketed.radius_option_counts
        <= len(bucketed.DISCRETE_LEVEL_RANGE)
    )
    for sensor_id, options in enumerate(bucketed.radius_options):
        expected = [0.0]
        sensor_distances = bucketed.target_sensor_distance[:, sensor_id]
        for lower, upper in zip(
            bucketed.DISCRETE_LEVEL_RANGE[:-1],
            bucketed.DISCRETE_LEVEL_RANGE[1:],
        ):
            interval = sensor_distances[
                (sensor_distances > lower + 1e-12)
                & (sensor_distances <= upper + 1e-12)
            ]
            if len(interval) > 0:
                expected.append(float(np.max(interval)))
        assert np.allclose(options, expected, rtol=0.0, atol=1e-12)
        assert np.all(np.diff(options) > 0)

    bucketed_coding = SensorEncoding.random(
        bucketed.SENSOR_NUMBER,
        bucketed.radius_option_counts,
    )
    bucketed_state = bucketed_coding.decode(bucketed)
    assert_state_radius_contract(bucketed, bucketed_state)
    assert bucketed_state.use_continuous_radius is True

    bucketed_target_coding = TargetEncoding.random(bucketed.TARGET_NUMBER)
    bucketed_target_state = bucketed_target_coding.decode(bucketed)
    assert_state_radius_contract(bucketed, bucketed_target_state)

    exact = build_problem("exact")
    assert exact.sensing_mode == "exact"
    assert exact.radius_table.shape == (
        exact.SENSOR_NUMBER,
        exact.LEVEL,
    )
    assert np.all(exact.radius_option_counts >= 1)

    distances = exact.target_sensor_distance
    for sensor_id, options in enumerate(exact.radius_options):
        assert options[0] == 0.0
        assert np.all(np.diff(options) > 0)
        assert np.all(options <= exact.MAX_SENSING_RADIUS + 1e-12)
        for radius in options[1:]:
            assert np.any(
                np.isclose(
                    distances[:, sensor_id],
                    radius,
                    rtol=0.0,
                    atol=1e-12,
                )
            )

    target_coding = TargetEncoding.random(exact.TARGET_NUMBER)
    target_state = target_coding.decode(exact)
    assert_state_radius_contract(exact, target_state)
    assert target_state.use_continuous_radius is True

    for algorithm_class in (GI_GOMEA, GI_GOMEA_Target):
        optimizer = algorithm_class(
            exact,
            population_size=6,
            max_linkage_size=8,
            max_linkage_sets=16,
            seed=19,
        )
        best = optimizer.run(exact, budget=40).best_state
        assert best is not None
        assert_state_radius_contract(exact, best)
        assert len(exact.find_uncovered_targets(best)) == 0

    target_id = next(
        target_id
        for target_id, candidates in enumerate(exact.cover_candidates)
        if candidates
    )
    sensor_id, option_id = exact.cover_candidates[target_id][0]
    schedule = np.zeros(exact.SENSOR_NUMBER, dtype=int)
    schedule[sensor_id] = option_id
    state = SimpleNamespace(levels=schedule, sensing_radii=None)
    radius = exact.resolve_state_radius(state)
    assert np.isclose(
        radius[sensor_id],
        exact.target_sensor_distance[target_id, sensor_id],
        rtol=0.0,
        atol=1e-12,
    )
    covered = exact.count_covered_targets(state)
    direct = np.sum(
        exact.target_sensor_distance
        <= radius[None, :] + 1e-12,
        axis=1,
    )
    # Closed sensors must not cover colocated targets.
    direct -= np.sum(
        (exact.target_sensor_distance <= 1e-12)
        & (radius[None, :] <= 0),
        axis=1,
    )
    assert np.array_equal(covered, direct)

    continuous_alias = build_problem("continuous")
    assert continuous_alias.sensing_mode == "exact"
    calibrated_alias = build_problem("calibrated")
    assert calibrated_alias.sensing_mode == "bucketed"

    args = SimpleNamespace(evaluate=40)
    for sensing_problem in (bucketed, exact):
        for name in ("alns", "codingse", "gi_gomea"):
            algorithm = build_algorithm(name, sensing_problem, args, seed=19)
            result = algorithm.run(sensing_problem, budget=40)
            assert result.best_state is not None
            assert_state_radius_contract(
                sensing_problem,
                result.best_state,
            )

    print("smoke_sensing_modes_ok")


if __name__ == "__main__":
    main()
