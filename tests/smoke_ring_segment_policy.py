"""Smoke checks for the per-pair LinUCB segment policies (LinUCB Ring-SE).

D3QN／v2 版的測試在 2026-10-13 隨 RL 系列移到 legacy/rl_ring/smoke_ring_segment_policy.py。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.se.ring_se.LinUCB_Ring_SE import LinUCB_Ring_SE
from Algorithm.se.ring_se.LinUCB_Ring_SEpri import LinUCB_Ring_SEpri
from Problem.Problem import Problem
from experiment_algorithms import build_algorithm, parse_args


def test_linucb_learns_and_keeps_budget():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    for algorithm_type in (LinUCB_Ring_SE, LinUCB_Ring_SEpri):
        algorithm = algorithm_type(problem, n=4, h=4, w=3, mu=0.4, seed=7)
        result = algorithm.run(problem, budget=300)
        assert result.best_state is not None
        assert result.evaluations >= 300
        assert np.trace(algorithm.linucb_A) > len(algorithm.FEATURE_NAMES)


def test_registry():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    for name, algorithm_type in (
        ("linucb_ring_se", LinUCB_Ring_SE),
        ("linucb_ring_sepri", LinUCB_Ring_SEpri),
    ):
        args = parse_args(["--algorithm", name, "--evaluate", "40"])
        assert type(build_algorithm(name, problem, args, seed=7)) is algorithm_type


def main():
    test_linucb_learns_and_keeps_budget()
    test_registry()
    print("smoke_ring_segment_policy_ok")


if __name__ == "__main__":
    main()
