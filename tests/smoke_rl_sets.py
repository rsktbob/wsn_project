"""RL-SETS (D3QN over SI-SETSv2, one operator decision per searcher) smoke test."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Algorithm.se.RL_SETS import RL_SETS
from Problem.Problem import Problem
from experiment_algorithms import build_algorithm, parse_args


def main():
    model_path = ROOT / "tests" / "_rl_sets_smoke.npz"
    if model_path.exists():
        model_path.unlink()

    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm = RL_SETS(
        problem,
        n=4,
        h=4,
        w=1,
        mu=0.4,
        training=True,
        model_path=model_path,
        replay_warmup=4,
        batch_size=4,
        target_update_interval=2,
        seed=7,
    )
    assert len(algorithm.ACTION_NAMES) == 4
    result = algorithm.run(problem, budget=40, max_iteration=20)
    assert result.best_state is not None
    assert result.evaluations == 40
    assert result.metadata["rl_mode"] == "train"
    assert model_path.is_file()

    evaluation_problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    evaluation = RL_SETS(
        evaluation_problem,
        n=4,
        h=4,
        w=1,
        training=False,
        model_path=model_path,
        seed=99,
    )
    evaluation_result = evaluation.run(evaluation_problem, budget=24, max_iteration=20)
    assert evaluation_result.best_state is not None
    assert evaluation_result.metadata["rl_mode"] == "eval"
    assert len(evaluation.agent.replay) == 0

    args = parse_args(["--algorithm", "rl_sets", "--rl-model", str(model_path)])
    built = build_algorithm("rl_sets", evaluation_problem, args, seed=7)
    assert isinstance(built, RL_SETS)
    assert built.training is False

    model_path.unlink()
    print("smoke_rl_sets_ok")


if __name__ == "__main__":
    main()
