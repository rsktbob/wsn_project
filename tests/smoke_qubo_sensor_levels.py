"""Smoke checks for the sensor-level QUBO: encodings agree with exact search."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Problem.Problem import Problem
from State.PriorityEncoding import PriorityEncoding
from State.State import State
from Qubo import (
    ENCODINGS,
    OBJECTIVES,
    apply_with_fixed_routes,
    brute_force,
    build_level_subproblem,
    repair_and_trim,
    simulated_annealing,
    slack_weights,
    solve_exact,
)


def _problem():
    np.random.seed(3)
    return Problem(B=50, S=32, T=9, F=100, FILE=None)


def _region_near(problem, size):
    """挑一個「區域內 sensor 共同覆蓋最多 target」的鄰近區域。"""
    coverage = np.asarray(problem.coverage_table) > 0
    best = None
    for target_id in range(problem.TARGET_NUMBER):
        distance = np.linalg.norm(
            np.asarray(problem.sensor) - np.asarray(problem.target)[target_id],
            axis=1,
        )
        region = np.argsort(distance)[:size]
        reach = coverage[:, region, -1].any(axis=1).sum()
        if best is None or reach > best[0]:
            best = (reach, region)
    return best[1]


def _subproblem(problem, size, objective):
    """全部關閉的起點：每個區域內能覆蓋的 target 都成為 required。"""
    state = State.empty(problem)
    problem.resolve_state_radius(state)
    return build_level_subproblem(
        problem, state, _region_near(problem, size), objective=objective
    )


def _exact(sub):
    """列舉區域內所有等級組合，回傳最小目標值。"""
    best = np.inf
    for levels in itertools.product(*[range(int(c)) for c in sub.option_counts]):
        if sub.is_feasible(levels):
            best = min(best, sub.objective(levels))
    return best


def _level_bits(encoding, sub, region_levels, n):
    x = np.zeros(n, dtype=np.int64)
    for (j, level), index in encoding.index.items():
        if encoding.name == "thermo":
            x[index] = int(region_levels[j] >= level)
        else:
            x[index] = int(region_levels[j] == level)
    return x


def test_slack_weights_cover_every_value():
    for max_value in range(0, 14):
        weights = slack_weights(max_value)
        assert sum(weights) == max_value
        reachable = {
            sum(w for w, bit in zip(weights, bits) if bit)
            for bits in itertools.product([0, 1], repeat=len(weights))
        }
        assert reachable == set(range(max_value + 1))


def test_feasible_assignments_cost_exactly_their_objective():
    """合法且全覆蓋的等級，在最佳 slack 下 QUBO 能量等於目標值（不含獎勵）。"""
    problem = _problem()
    for objective in OBJECTIVES:
        sub = _subproblem(problem, 2, objective)
        assert len(sub.required_targets) > 0
        _check_feasible_energies(sub)


def _check_feasible_energies(sub):
    for name, encoding_type in ENCODINGS.items():
        encoding = encoding_type()
        model = encoding.build(sub)
        n = model.num_variables
        slack_ids = [i for i, label in enumerate(model.labels) if label.startswith("slack")]
        for levels in itertools.product(*[range(int(c)) for c in sub.option_counts]):
            x = _level_bits(encoding, sub, levels, n)
            energies = []
            for bits in itertools.product([0, 1], repeat=len(slack_ids)):
                x[slack_ids] = bits
                energies.append(model.energy(x))
            if sub.is_feasible(levels):
                assert abs(min(energies) - sub.objective(levels)) < 1e-9, name
            else:
                assert min(energies) > sub.objective(levels) + 1.0, name


def test_qubo_minimum_matches_exact_optimum():
    problem = _problem()
    for objective in OBJECTIVES:
        _check_minimum(_subproblem(problem, 2, objective))


def _check_minimum(sub):
    optimum = _exact(sub)
    for name, encoding_type in ENCODINGS.items():
        encoding = encoding_type()
        model = encoding.build(sub)
        x, energy = brute_force(model)
        levels, valid = encoding.decode(sub, x)
        assert valid, name
        assert sub.is_feasible(levels), name
        assert abs(sub.objective(levels) - optimum) < 1e-9, name
        assert abs(energy - optimum) < 1e-9, name


def _decoded_region(problem, size, objective):
    """從解碼後（含路由、耗能不為 0）的 state 找一個需要區域覆蓋的區域。"""
    state = PriorityEncoding.random(
        problem.SENSOR_NUMBER, rng=np.random.default_rng(5)
    ).decode(problem)
    for target_id in range(problem.TARGET_NUMBER):
        distance = np.linalg.norm(
            np.asarray(problem.sensor) - np.asarray(problem.target)[target_id],
            axis=1,
        )
        region = np.argsort(distance)[:size]
        sub = build_level_subproblem(problem, state, region, objective=objective)
        variables = ENCODINGS["thermo"]().build(sub).num_variables
        if len(sub.required_targets) and variables <= 20:
            return sub
    raise AssertionError("no decoded region needs coverage")


def test_fitness_objectives_use_target_groups():
    """耗能不為 0 時，平方項要真的出現，且 QUBO 仍與精確解一致。"""
    problem = _problem()
    for objective in ("fitness", "routing"):
        sub = _decoded_region(problem, 2, objective)
        assert np.any(sub.group_weight > 0), objective
        _check_feasible_energies(sub)
        _check_minimum(sub)
        levels, value = solve_exact(sub)
        assert abs(value - _exact(sub)) < 1e-9, objective


def test_routing_objective_matches_routing_cost():
    """路由樹不變時，routing 目標的群組耗損等於 Problem 算出的真實耗損。"""
    problem = _problem()
    state = PriorityEncoding.random(
        problem.SENSOR_NUMBER, rng=np.random.default_rng(5)
    ).decode(problem)
    radii = problem.state_radius(state)
    cost = np.asarray(problem.calculate_total_cost(state, sensing_radii=radii))
    mask = np.asarray(problem.target_sensor_mask, dtype=float)
    depletion = (mask @ cost) / (mask @ problem.energy)
    active = np.flatnonzero(state.levels > 0)
    checked = 0
    for start in range(0, len(active) - 2, 3):
        sub = build_level_subproblem(
            problem, state, active[start:start + 3], objective="routing"
        )
        current = sub.region_levels_of(state.levels)
        rows = np.arange(sub.size)
        load = sub.group_offset + sub.group[rows, current].sum(axis=0)
        assert np.allclose(load, depletion[sub.group_targets], rtol=1e-9, atol=1e-15)
        checked += len(load)
    assert checked > 0


def test_fixed_routes_reproduce_the_model():
    """原等級套用後重現原本的負載；新等級套用後真實耗損等於模型預測。"""
    problem = _problem()
    state = PriorityEncoding.random(
        problem.SENSOR_NUMBER, rng=np.random.default_rng(5)
    ).decode(problem)
    mask = np.asarray(problem.target_sensor_mask, dtype=float)
    active = np.flatnonzero(state.levels > 0)
    applied = 0
    for start in range(0, len(active) - 2, 3):
        sub = build_level_subproblem(
            problem, state, active[start:start + 3], objective="routing"
        )
        same = apply_with_fixed_routes(
            problem, state, sub, sub.region_levels_of(state.levels)
        )
        assert same is not None
        assert np.array_equal(same.tx_load, state.tx_load)
        assert np.array_equal(same.next_hops, state.next_hops)

        levels, _ = solve_exact(sub)
        new = apply_with_fixed_routes(problem, state, sub, levels)
        if new is None:
            continue
        radii = problem.state_radius(new)
        cost = np.asarray(problem.calculate_total_cost(new, sensing_radii=radii))
        depletion = (mask @ cost) / (mask @ problem.energy)
        rows = np.arange(sub.size)
        load = sub.group_offset + sub.group[rows, levels].sum(axis=0)
        assert np.allclose(load, depletion[sub.group_targets], rtol=1e-9, atol=1e-15)
        assert len(problem.find_uncovered_targets(new)) == 0
        applied += 1
    assert applied > 0


def test_annealing_returns_feasible_levels():
    problem = _problem()
    for objective in OBJECTIVES:
        _check_annealing(_subproblem(problem, 5, objective))


def _check_annealing(sub):
    optimum = _exact(sub)
    for name, encoding_type in ENCODINGS.items():
        encoding = encoding_type()
        model = encoding.build(sub)
        x, _, _ = simulated_annealing(
            model, sweeps=200, num_reads=10, rng=np.random.default_rng(0)
        )
        levels, _ = encoding.decode(sub, x)
        levels = repair_and_trim(sub, levels)
        assert sub.is_feasible(levels), name
        assert sub.objective(levels) >= optimum - 1e-9, name


if __name__ == "__main__":
    test_slack_weights_cover_every_value()
    test_feasible_assignments_cost_exactly_their_objective()
    test_qubo_minimum_matches_exact_optimum()
    test_fitness_objectives_use_target_groups()
    test_routing_objective_matches_routing_cost()
    test_fixed_routes_reproduce_the_model()
    test_annealing_returns_feasible_levels()
    print("smoke_qubo_sensor_levels_ok")
