import random
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Problem.Problem import Problem
from Problem.services.fitness_service import FitnessService
from Problem.services.fitness_servicev2 import FitnessServiceV2
from Problem.services.fitness_servicev3 import FitnessServiceV3
from State.SensorEncoding import SensorEncoding
from State.State import State


def expected_original_fitness(problem, state):
    cost = problem.calculate_total_cost(state)
    remaining = problem.energy - cost
    target_remaining = problem.calculate_target_remaining_energy(remaining)

    total_initial = np.sum(
        problem.initial_energy * problem.target_sensor_mask
    )
    energy_score = 0 if total_initial == 0 else np.sum(target_remaining) / total_initial

    min_initial = problem.initial_energy * np.min(
        np.sum(problem.target_sensor_mask, axis=1)
    )
    min_score = 0
    if len(target_remaining) and np.min(target_remaining) >= 0 and min_initial != 0:
        min_score = np.min(target_remaining) / min_initial

    if np.any(remaining < 0):
        return np.asarray([-1.0, -5.0, -1.0])

    coverage = (
        problem.TARGET_NUMBER
        - len(problem.find_uncovered_targets(state.levels))
    ) / problem.TARGET_NUMBER
    return np.asarray(
        [
            problem.fitness_service.w1 * energy_score,
            problem.fitness_service.w2 * min_score,
            problem.fitness_service.w3 * coverage,
        ]
    )


def expected_v5_fitness(problem, service, state, target_summed=False):
    energy = np.asarray(problem.energy, dtype=float)
    cost = np.asarray(problem.calculate_total_cost(state), dtype=float)
    remaining = energy - cost
    positive_cost = np.maximum(cost, 0.0)
    total_energy = np.sum(np.maximum(energy, 0.0))
    target_energy = problem.target_sensor_mask @ energy
    target_cost = problem.target_sensor_mask @ positive_cost
    if target_summed:
        aggregate_depletion = np.clip(
            np.sum(target_cost) / np.sum(target_energy),
            0.0,
            1.0,
        )
    else:
        aggregate_depletion = np.clip(
            np.sum(positive_cost) / total_energy,
            0.0,
            1.0,
        )
    target_depletion = np.ones_like(target_energy, dtype=float)
    np.divide(
        target_cost,
        target_energy,
        out=target_depletion,
        where=target_energy > service.EPSILON,
    )
    worst_target_depletion = np.clip(np.max(target_depletion), 0.0, 1.0)
    energy_component = service.w1 * (1.0 - aggregate_depletion)
    target_component = service.w2 * (1.0 - worst_target_depletion)
    quality = energy_component + target_component
    loss = 1.0 - quality

    uncovered = len(problem.find_uncovered_targets(state))
    coverage_deficit = uncovered / problem.TARGET_NUMBER
    disconnected = len(problem.find_disconnected(state))
    failed = np.any((~np.isfinite(remaining)) | (remaining < 0.0))
    if failed or disconnected:
        active = max(
            1,
            np.count_nonzero(problem.resolve_state_radius(state) > 0),
        )
        finite_deficit = np.where(
            np.isfinite(remaining),
            np.maximum(-remaining, 0.0),
            np.maximum(energy, 0.0),
        )
        desired = (
            service.INVALID_BASE_SCORE
            - coverage_deficit
            - disconnected / active
            - np.sum(finite_deficit) / total_energy
        )
    elif uncovered:
        desired = (
            -coverage_deficit
            - 0.5 / problem.TARGET_NUMBER * loss
        )
    else:
        desired = quality
    return np.asarray(
        [
            energy_component,
            target_component,
            desired - energy_component - target_component,
        ]
    )


class _FakeState:
    pass


class _FakeProblem:
    """Minimal double exposing exactly what FitnessServiceV3 reads.

    Every sensor shares the same energy (10.0) and the same flat sensing
    cost, so with an all-ones ``target_sensor_mask`` both depletion ratios
    collapse to ``cost / 10.0`` -- i.e. ``quality = 1 - cost / 10.0``. Use
    ``cost_for_quality`` to pick a cost that lands on an exact desired
    quality value.
    """

    def __init__(self, target_count, sensor_count, disconnected_ids, cost=0.0):
        self.TARGET_NUMBER = target_count
        self.energy = np.full(sensor_count, 10.0)
        self.target_sensor_mask = np.ones((target_count, sensor_count))
        self._disconnected_ids = disconnected_ids
        self._cost = cost

    @staticmethod
    def cost_for_quality(quality):
        return 10.0 * (1.0 - quality)

    def state_radius(self, state):
        return np.full(len(self.energy), 5.0)

    def calculate_total_cost(self, state, sensing_radii=None):
        return np.full(len(self.energy), self._cost)

    def calculate_target_remaining_energy(self, remaining_energy):
        return self.target_sensor_mask @ remaining_energy

    def find_uncovered_targets(self, state):
        return list(range(state.uncovered_count))

    def find_disconnected(self, state, sensing_radii=None):
        return self._disconnected_ids


def test_v3_tier_ordering_regression():
    """An infeasible state must never outscore a healthier incomplete one.

    Before the fix, V3 scored both tiers with the raw uncovered target
    count, so a mildly-infeasible state (few uncovered, one disconnected
    sensor) could outscore a badly-incomplete-but-healthy state (more
    uncovered, nothing broken) -- the opposite of the intended
    feasible > incomplete > infeasible ordering. Regression case for that.
    """
    target_count, sensor_count = 5, 2

    mildly_infeasible = _FakeState()
    mildly_infeasible.uncovered_count = 1
    infeasible_service = FitnessServiceV3(
        _FakeProblem(target_count, sensor_count, disconnected_ids=[0])
    )
    infeasible_score = float(
        np.sum(infeasible_service.evaluate_state(mildly_infeasible))
    )

    badly_incomplete_but_healthy = _FakeState()
    badly_incomplete_but_healthy.uncovered_count = 3
    incomplete_service = FitnessServiceV3(
        _FakeProblem(target_count, sensor_count, disconnected_ids=[])
    )
    incomplete_score = float(
        np.sum(incomplete_service.evaluate_state(badly_incomplete_but_healthy))
    )

    assert incomplete_score > infeasible_score, (
        "incomplete state must outrank infeasible state regardless of "
        "relative uncovered-target counts"
    )
    print("smoke_fitness_servicev3_tier_ordering_ok")


def test_v3_incomplete_beats_mediocre_feasible():
    """A near-complete, well-managed state may outscore a mediocre one.

    This is the intended, designed-in overlap: INCOMPLETE_QUALITY_WEIGHT
    (0.5) lets an almost-fully-covered state with excellent quality beat a
    fully-covered state whose quality is below the 0.5 midpoint.
    """
    target_count, sensor_count = 100, 4

    near_miss = _FakeState()
    near_miss.uncovered_count = 1
    problem = _FakeProblem(
        target_count, sensor_count, disconnected_ids=[],
        cost=_FakeProblem.cost_for_quality(1.0),
    )
    incomplete_score = float(
        np.sum(FitnessServiceV3(problem).evaluate_state(near_miss))
    )

    fully_covered_mediocre = _FakeState()
    fully_covered_mediocre.uncovered_count = 0
    mediocre_problem = _FakeProblem(
        target_count, sensor_count, disconnected_ids=[],
        cost=_FakeProblem.cost_for_quality(0.4),
    )
    feasible_score = float(
        np.sum(FitnessServiceV3(mediocre_problem).evaluate_state(
            fully_covered_mediocre
        ))
    )

    assert incomplete_score > feasible_score, (
        "a near-complete, high-quality state should beat a fully-covered "
        "but mediocre-quality one"
    )
    print("smoke_fitness_servicev3_incomplete_beats_mediocre_ok")


def test_v3_incomplete_never_beats_good_feasible():
    """Even the best possible incomplete state loses to a good feasible one.

    INCOMPLETE_QUALITY_WEIGHT (0.5) is a hard ceiling: no matter how close
    to full coverage or how perfect the quality, an incomplete state can
    never reach a fully-covered state whose quality is at or above 0.5.
    """
    target_count, sensor_count = 100, 4

    best_possible_near_miss = _FakeState()
    best_possible_near_miss.uncovered_count = 1
    problem = _FakeProblem(
        target_count, sensor_count, disconnected_ids=[],
        cost=_FakeProblem.cost_for_quality(1.0),
    )
    incomplete_score = float(
        np.sum(FitnessServiceV3(problem).evaluate_state(best_possible_near_miss))
    )

    fully_covered_good = _FakeState()
    fully_covered_good.uncovered_count = 0
    good_problem = _FakeProblem(
        target_count, sensor_count, disconnected_ids=[],
        cost=_FakeProblem.cost_for_quality(0.5),
    )
    feasible_score = float(
        np.sum(FitnessServiceV3(good_problem).evaluate_state(fully_covered_good))
    )

    assert incomplete_score < feasible_score, (
        "even the best possible incomplete state must lose to a "
        "fully-covered state with quality >= 0.5"
    )
    print("smoke_fitness_servicev3_incomplete_capped_ok")


def main():
    np.random.seed(3)
    random.seed(3)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)

    # Problem keeps the original evaluator until an experiment selects one.
    assert type(problem.fitness_service) is FitnessService
    assert problem.fitness_service.objective_names == [
        "total_remaining_energy",
        "min_remaining_energy",
        "coverage",
    ]
    np.testing.assert_allclose(
        [
            problem.fitness_service.w1,
            problem.fitness_service.w2,
            problem.fitness_service.w3,
        ],
        [0.15, 0.25, 0.60],
    )

    all_off = State.empty(problem)
    original_all_off = np.asarray(
        problem.evaluate_state(all_off),
        dtype=float,
    )
    np.testing.assert_allclose(original_all_off, [0.15, 0.25, 0.0])

    coding = SensorEncoding.random(
        problem.SENSOR_NUMBER,
        problem.radius_option_counts,
    )
    state = coding.decode(problem)
    original = np.asarray(problem.evaluate_state(state), dtype=float)
    np.testing.assert_allclose(
        original,
        expected_original_fitness(problem, state),
    )

    # V5 directly scores depletion against current energy.  Fully feasible
    # states are in [0, 1], incomplete states are below zero, and the all-off
    # state still exposes the two maximum quality components before its
    # feasibility contribution is applied.
    v5 = FitnessServiceV2(problem)
    assert v5.objective_names == [
        "energy_efficiency",
        "target_region_balance",
        "constraint_score",
    ]
    np.testing.assert_allclose([v5.w1, v5.w2], [0.3, 0.7])
    np.testing.assert_allclose(v5.evaluate_state(all_off), [0.3, 0.7, -2.0])
    assert np.isclose(np.sum(v5.evaluate_state(all_off)), -1.0)
    v5_values = np.asarray(v5.evaluate_state(state), dtype=float)
    np.testing.assert_allclose(
        v5_values,
        expected_v5_fitness(problem, v5, state),
    )
    assert np.sum(v5_values) <= 1.0

    print("smoke_fitness_service_ok")


if __name__ == "__main__":
    main()
    test_v3_tier_ordering_regression()
    test_v3_incomplete_beats_mediocre_feasible()
    test_v3_incomplete_never_beats_good_feasible()
