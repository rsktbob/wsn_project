"""Smoke checks for SA-SETSv4: SA-SETSv3's market over C4 PriorityEncoding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.se.SA_SETSv4 import PriorityEncodingC4, SA_SETSv4
from Problem.Problem import Problem
from experiment_algorithms import build_algorithm, parse_args


def test_candidates_use_c4_priority_encoding():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = SA_SETSv4(problem, n=4, h=4, w=2, mu=1.0, seed=7)
    candidate = algorithm.create_candidate(problem)

    assert type(candidate) is PriorityEncodingC4
    assert candidate.ACTIVATION_BASE == "cost"
    assert candidate.ROUTING_BASE == "energy"
    assert candidate.code.shape == (problem.SENSOR_NUMBER * 2,)
    assert np.all((0 <= candidate.code) & (candidate.code < 10))


def test_mutation_stays_inside_span_and_gene_domain():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = SA_SETSv4(problem, n=4, h=4, w=1, mu=1.0, seed=7)
    for _ in range(200):
        candidate = algorithm.create_candidate(problem)
        before = candidate.code.copy()
        algorithm.mutate_candidate(problem, candidate, ("local", 8, 16))
        changed = np.flatnonzero(candidate.code != before)
        assert np.all((16 <= changed) & (changed < 32))
        assert np.all((0 <= candidate.code) & (candidate.code < 10))


def test_registry_run():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    args = parse_args(["--algorithm", "sa_setsv4", "--evaluate", "40"])
    algorithm = build_algorithm("sa_setsv4", problem, args, seed=7)
    assert type(algorithm) is SA_SETSv4

    result = algorithm.run(problem, budget=40)
    assert result.best_state is not None
    assert np.all(np.isfinite(result.best_fitness))
    assert result.evaluations >= 40


if __name__ == "__main__":
    test_candidates_use_c4_priority_encoding()
    test_mutation_stays_inside_span_and_gene_domain()
    test_registry_run()
    print("smoke_sa_setsv4_ok")
