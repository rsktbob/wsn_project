"""Smoke checks for Ring-SETSpriority: Ring-SETS' market over EF PriorityEncoding."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.se.LinUCB_Ring_SETSpriority import LinUCB_Ring_SETSpriority
from Algorithm.se.Ring_SETSpriority import (
    PriorityEncodingC4,
    PriorityEncodingEF,
    Ring_SETSpriority,
)
from Problem.Problem import Problem
from experiment_algorithms import build_algorithm, parse_args


def test_candidates_use_ef_priority_encoding():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = Ring_SETSpriority(problem, n=4, h=4, w=2, mu=1.0, seed=7)
    candidate = algorithm.create_candidate(problem)

    assert type(candidate) is PriorityEncodingEF
    assert candidate.ACTIVATION_BASE == "energy"
    assert candidate.ROUTING_BASE == "far"
    assert candidate.code.shape == (problem.SENSOR_NUMBER * 2,)
    assert np.all((0 <= candidate.code) & (candidate.code < 10))

    energy = np.asarray(problem.energy_score, dtype=float)
    to_bs = np.asarray(problem.distances)[: problem.SENSOR_NUMBER, problem.BSID]
    assert np.allclose(candidate._activation_base(problem), energy)
    assert np.allclose(candidate._routing_base(problem), energy + to_bs / to_bs.max())


def test_linucb_variant_keeps_c4():
    assert PriorityEncodingC4.ACTIVATION_BASE == "cost"
    assert PriorityEncodingC4.ROUTING_BASE == "energy"
    assert LinUCB_Ring_SETSpriority.ENCODING_TYPE is PriorityEncodingC4


def test_mutation_stays_inside_span_and_gene_domain():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    algorithm = Ring_SETSpriority(problem, n=4, h=4, w=1, mu=1.0, seed=7)
    for _ in range(200):
        candidate = algorithm.create_candidate(problem)
        before = candidate.code.copy()
        algorithm.mutate_candidate(problem, candidate, ("local", 8, 16))
        changed = np.flatnonzero(candidate.code != before)
        assert np.all((16 <= changed) & (changed < 32))
        assert np.all((0 <= candidate.code) & (candidate.code < 10))


def test_registry_run():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    args = parse_args(["--algorithm", "ring_setspriority", "--evaluate", "40"])
    algorithm = build_algorithm("ring_setspriority", problem, args, seed=7)
    assert type(algorithm) is Ring_SETSpriority

    result = algorithm.run(problem, budget=40)
    assert result.best_state is not None
    assert np.all(np.isfinite(result.best_fitness))
    assert result.evaluations >= 40


if __name__ == "__main__":
    test_candidates_use_ef_priority_encoding()
    test_linucb_variant_keeps_c4()
    test_mutation_stays_inside_span_and_gene_domain()
    test_registry_run()
    print("smoke_ring_setspriority_ok")
