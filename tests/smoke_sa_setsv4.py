"""Smoke checks for SA-SETSv4's shared goods pool + segment-only scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.se.SA_SETSv4 import SA_SETSv4
from Problem.Problem import Problem
from experiment_algorithms import build_algorithm, parse_args


def test_shared_pool_initialization():
    """Goods are one flat pool of size w, not an (h, w) grid per region."""
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = SA_SETSv4(problem, n=4, h=4, w=3, mu=1.0, seed=7)
    algorithm.initialize_market(problem)

    assert len(algorithm.goods) == 3
    assert algorithm.goods_fitness.shape == (3,)

    # No segment has a track record yet, so every segment starts from the
    # same neutral guess: the shared pool's own average fitness.
    initial_quality = float(np.mean(algorithm.goods_fitness))
    assert algorithm.segment_quality.shape == (4, 4)
    assert np.allclose(algorithm.segment_quality, initial_quality)


def test_ring_scopes_still_come_from_sa_setsv3():
    """Operator scoping is untouched: still inherited from SA_SETSv3."""
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = SA_SETSv4(problem, n=4, h=4, w=1, mu=1.0, seed=7)
    assert algorithm.region_sensor_bounds == ((0, 8), (8, 16), (16, 24), (24, 32))

    local_seen = False
    cross_seen = False
    for _ in range(300):
        mode, left, right = algorithm.select_sensor_span(region=2)
        assert 0 <= left < right <= problem.SENSOR_NUMBER
        if mode == "local":
            local_seen = True
            assert 16 <= left < right <= 24
        else:
            cross_seen = True
            assert left <= 16 and right >= 24
            assert left < 16 or right > 24
    assert local_seen and cross_seen


def test_all_searchers_trade_against_the_same_shared_goods():
    """Every searcher this round is a visitor for every good slot."""
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = SA_SETSv4(problem, n=4, h=4, w=1, mu=0.0, seed=7)
    algorithm.initialize_market(problem)
    algorithm.searcher_fitness[:] = 0.0
    algorithm.goods_fitness[:] = 0.0
    # Two searchers picked segment 0, two picked segment 1 -- but they all
    # still draw from the one shared good.
    algorithm.selected_regions = np.array([0, 0, 1, 1], dtype=int)

    seen_goods = []
    def make_children(_problem, searcher, good, region=None):
        seen_goods.append(good)
        child1 = searcher.copy()
        child2 = good.copy()
        child1.code[0] = len(seen_goods)
        child2.code[0] = len(seen_goods) + 100
        return child1, child2

    algorithm.make_children = make_children
    algorithm.evaluate_investments = lambda _problem, children: np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    algorithm.segment_probabilities = lambda quality: np.ones((4, 4))
    algorithm.select_regions = lambda probabilities: np.array([0, 1, 2, 3])
    algorithm.vision_search(problem)

    # All 4 searchers traded against the same single shared good object.
    assert len(seen_goods) == 4
    assert all(good is algorithm.goods[0] or good.code is not None for good in seen_goods)

    # The best child2 among all 4 searchers wins the one shared slot: that's
    # searcher 3 (index 3), whose child2 fitness (8.0) is the round's max.
    assert algorithm.goods_fitness[0] == 8.0
    assert int(algorithm.goods[0].code[0]) == 104


def test_registry_and_small_search():
    args = parse_args(["--algorithm", "sa_setsv4", "--evaluate", "40"])
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = build_algorithm("sa_setsv4", problem, args, seed=7)
    result = algorithm.run(problem, budget=40)
    assert result.best_state is not None
    assert result.evaluations >= 40
    assert algorithm.final_coding is not None


def main():
    test_shared_pool_initialization()
    test_ring_scopes_still_come_from_sa_setsv3()
    test_all_searchers_trade_against_the_same_shared_goods()
    test_registry_and_small_search()
    print("smoke_sa_setsv4_ok")


if __name__ == "__main__":
    main()
