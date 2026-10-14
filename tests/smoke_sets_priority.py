"""Smoke checks for SA-SETSpri / SI-SETSpri: SETS markets over PriorityEncoding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.se.sets.SA_SETSpri import SA_SETSpri
from Algorithm.se.si_sets.SI_SETSpri import SI_SETSpri
from Problem.Problem import Problem
from State.PriorityEncoding import PriorityEncoding
from experiment_algorithms import build_algorithm, parse_args


def check(algorithm_type, name, w):
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = algorithm_type(problem, n=4, h=4, w=w, mu=1.0, seed=7)

    candidate = algorithm.create_candidate(problem, region=3)
    assert type(candidate) is PriorityEncoding
    assert candidate.code.shape == (problem.SENSOR_NUMBER * 2,)
    before = candidate.code.copy()
    # 不分區：對齊不改任何基因。
    assert np.array_equal(algorithm.align_region(problem, candidate, 0).code, before)
    for _ in range(200):
        algorithm.mutatev1(problem, candidate)
        assert np.all((0 <= candidate.code) & (candidate.code < 10))

    result = algorithm.run(problem, budget=60)
    assert result.best_state is not None
    assert np.all(np.isfinite(result.best_fitness))
    assert algorithm.identity_sensors == []
    assert all(type(s) is PriorityEncoding for s in algorithm.searchers)

    args = parse_args(["--algorithm", name, "--evaluate", "40"])
    assert type(build_algorithm(name, problem, args, seed=7)) is algorithm_type


def main():
    check(SA_SETSpri, "sa_setspri", 2)
    check(SI_SETSpri, "si_setspri", 2)
    print("smoke_sets_priority_ok")


if __name__ == "__main__":
    main()
