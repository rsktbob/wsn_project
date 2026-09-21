"""Smoke checks for SA-SETSv3's soft ring-region operators."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.se.SA_SETSv3 import SA_SETSv3
from Problem.Problem import Problem
from experiment_algorithms import build_algorithm, parse_args


def test_ring_scopes_and_free_alignment():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = SA_SETSv3(problem, n=4, h=4, w=1, mu=1.0, seed=7)
    assert algorithm.region_sensor_bounds == ((0, 8), (8, 16), (16, 24), (24, 32))

    candidate = algorithm.create_candidate(problem, region=2)
    before = candidate.code.copy()
    algorithm.align_region(problem, candidate, region=2)
    assert (candidate.code == before).all()

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


def test_role_separated_ring_children():
    """child1 keeps searcher outside the span; child2 keeps good outside."""
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = SA_SETSv3(problem, n=4, h=4, w=1, mu=0.0, seed=7)
    searcher = algorithm.create_candidate(problem)
    good = algorithm.create_candidate(problem)
    searcher.code[:] = 1
    good.code[:] = 7
    algorithm.select_sensor_span = lambda region: ("local", 16, 20)

    child1, child2 = algorithm.make_children(problem, searcher, good, region=2)
    assert (child1.code[:32] == 1).all()
    assert (child1.code[32:40] == 7).all()
    assert (child1.code[40:] == 1).all()
    assert (child2.code[:32] == 7).all()
    assert (child2.code[32:40] == 1).all()
    assert (child2.code[40:] == 7).all()


def test_elitist_role_updates():
    """SA-SETSv3 must route child1 to searchers and child2 to goods only."""
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = SA_SETSv3(problem, n=4, h=4, w=1, mu=0.0, seed=7)
    algorithm.initialize_market(problem)
    algorithm.searcher_fitness[:] = 0.0
    algorithm.goods_fitness[:] = 0.0
    algorithm.selected_regions = np.array([0, 0, 1, 1], dtype=int)

    produced = []
    def make_children(_problem, searcher, good, region=None):
        child1 = searcher.copy()
        child2 = good.copy()
        child1.code[0] = len(produced) + 1
        child2.code[0] = len(produced) + 101
        produced.append((child1, child2))
        return child1, child2

    algorithm.make_children = make_children
    algorithm.evaluate_investments = lambda _problem, children: np.array(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    )
    algorithm.region_probabilities = lambda goods, quality: np.ones((4, 4))
    algorithm.select_regions = lambda probabilities: np.array([0, 1, 2, 3])
    algorithm.vision_search(problem)

    assert [int(searcher.code[0]) for searcher in algorithm.searchers] == [1, 2, 3, 4]
    assert algorithm.goods_fitness[0, 0] == 4.0
    assert algorithm.goods_fitness[1, 0] == 8.0
    assert int(algorithm.goods[0][0].code[0]) == 102
    assert int(algorithm.goods[1][0].code[0]) == 104


def test_registry_and_small_search():
    args = parse_args(["--algorithm", "sa_setsv3", "--evaluate", "40"])
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = build_algorithm("sa_setsv3", problem, args, seed=7)
    result = algorithm.run(problem, budget=40)
    assert result.best_state is not None
    assert result.evaluations >= 40
    assert algorithm.final_coding is not None


def main():
    test_ring_scopes_and_free_alignment()
    test_role_separated_ring_children()
    test_elitist_role_updates()
    test_registry_and_small_search()
    print("smoke_sa_setsv3_ok")


if __name__ == "__main__":
    main()
