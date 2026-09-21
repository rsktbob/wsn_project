"""SI-SETSv2 role-separated update and experiment registration smoke test."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Algorithm.se.SI_SETSv2 import SI_SETSv2
from Problem.Problem import Problem
from experiment_algorithms import build_algorithm, build_problem, parse_args


def main():
    elitist_problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    elitist = SI_SETSv2(
        elitist_problem, n=4, h=4, w=1, mu=0.4, seed=17
    )
    elitist.evatime = 0
    elitist.evaluation_limit = 40
    elitist.reset_best()
    elitist.initialize_market(elitist_problem)
    goods_before = elitist.goods_fitness.copy()
    elitist.arrange_resources(elitist_problem)
    try:
        elitist.vision_search(elitist_problem)
    finally:
        elitist.close_market()
    assert np.all(
        elitist.goods_fitness
        >= goods_before - elitist.IMPROVEMENT_TOLERANCE
    )

    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm = SI_SETSv2(
        problem, n=4, h=4, w=1, mu=0.4, seed=7
    )
    result = algorithm.run(problem, budget=40, max_iteration=20)
    assert result.best_state is not None
    # Initial searchers + goods cost 8; each role-separated round costs 8.
    assert result.evaluations == 40
    assert algorithm.iterations_completed == 4
    assert algorithm.goods_fitness.shape == (4, 1)
    assert np.all(np.isfinite(algorithm.goods_fitness))
    assert not hasattr(algorithm, "agent")

    args = parse_args(
        [
            "--algorithm", "si_setsv2",
            "--maps", "maps/maps_100100100_a2.json",
            "--mode", "single",
            "--runs", "1",
            "--evaluate", "40",
        ]
    )
    args._current_map = args.map_specs[0]
    experiment_problem = build_problem(args, 7)
    built = build_algorithm("si_setsv2", experiment_problem, args, 7)
    assert isinstance(built, SI_SETSv2)
    assert (built.n, built.h, built.w, built.mutation_rate) == (8, 4, 2, 0.4)
    print("smoke_si_setsv2_ok")


if __name__ == "__main__":
    main()
