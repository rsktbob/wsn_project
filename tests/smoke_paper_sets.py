import math
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.se.SA_SETSv2 import SA_SETSv2
from Algorithm.se.SA_SETS import beta_cdf
from Algorithm.se.SETSv2 import SETSv2
from Problem.Problem import Problem
from State.SensorEncoding import SensorEncoding


def assert_region_bits(algorithm, problem, state, region):
    identity_pattern = algorithm.h - 1 - region
    for bit, sensor_id in enumerate(algorithm.identity_sensors):
        is_active = int(state.code[sensor_id * 2]) % problem.LEVEL != 0
        shift = algorithm.identity_bit_count - 1 - bit
        assert is_active == bool((identity_pattern >> shift) & 1)


def main():
    problem = Problem(B=50, S=24, T=9, F=10, FILE=None)

    sets = SETSv2(
        problem,
        n=2,
        h=4,
        w=1,
        player=2,
        crossover_rate=1.0,
        mutation_rate=1.0,
        seed=19,
    )
    sets.identity_sensors = sets.select_identity_sensors(problem)
    distances = problem.distances[: problem.SENSOR_NUMBER, problem.BSID]
    expected_nearest = np.argsort(distances, kind="stable")[:2].tolist()
    assert sets.identity_sensors == expected_nearest

    state = SensorEncoding.random(
        problem.SENSOR_NUMBER,
        problem.radius_option_counts,
    )
    for region in range(sets.h):
        sets.align_region(problem, state, region)
        assert_region_bits(sets, problem, state, region)

    for _ in range(20):
        points = sets.choose_crossover_points()
        assert points[0] not in sets.identity_gene_indices
        assert points[1] not in sets.identity_gene_indices
        assert points[0] < points[1]

    original = state.code.copy()
    sets._mutate(problem, state)
    changed = np.flatnonzero(original != state.code)
    assert 1 <= len(changed) <= 3

    per_iteration = sets.evaluations_per_iteration
    best = sets.run(
        problem,
        budget=per_iteration * 2,
    ).best_state
    assert best is not None
    assert sets.evatime == per_iteration * 2
    assert sets.iterations_completed == 2
    assert len(sets.history) == per_iteration * 2
    assert np.all(np.isfinite(sets.history))
    assert math.isfinite(float(np.sum(problem.evaluate_state(best))))

    adaptive = SA_SETSv2(
        problem,
        n=2,
        h=4,
        w=1,
        player=2,
        energy_weight=0.5,
        distance_weight=0.5,
    )
    adaptive_ids = adaptive.select_identity_sensors(problem)
    assert len(adaptive_ids) == adaptive.identity_bit_count
    assert len(set(adaptive_ids)) == adaptive.identity_bit_count

    default_sets = SETSv2(problem)
    default_adaptive = SA_SETSv2(problem)
    assert default_sets.crossover_rate == 1.0
    assert default_sets.mutation_rate == 0.4
    assert default_adaptive.crossover_rate == 1.0
    assert default_adaptive.mutation_rate == 0.4
    assert default_adaptive.energy_weight == 0.5
    assert default_adaptive.distance_weight == 0.5

    # 參數順序必須是論文公式 I_e(tb, ta)，而不是舊版的 I_e(ta, tb)。
    x = 0.6
    assert beta_cdf(5, 1, x) < beta_cdf(1, 5, x)

    print("smoke_paper_sets_ok")


if __name__ == "__main__":
    main()
