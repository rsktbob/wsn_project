"""RL-SETS (D3QN, factorized SA-centred selection) smoke test."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Algorithm.se.RL_SETSv5 import RL_SETSv5
from Algorithm.se.d3qn import D3QNAgent
from experiment_algorithms import build_algorithm, build_problem, parse_args
from train_rl_sets import (
    build_training_problem,
    main as training_main,
    parse_args as training_args,
)


def main():
    defaults = training_args(["--algorithm", "rl_setsv5"])
    assert len(defaults.map_specs) == 200
    assert len(defaults.validation_specs) == 40
    assert defaults.passes == 1
    assert defaults.validation_every == 200
    assert defaults.evaluate == 5000
    assert defaults.replay_capacity == 200000
    assert defaults.replay_group_capacity == 1000
    assert defaults.epsilon_end == 0.10

    spec = defaults.map_specs[0]
    problem = build_training_problem(defaults, 7, spec)
    algorithm = RL_SETSv5(
        problem,
        n=4,
        h=4,
        w=1,
        training=False,
        fixed_action=RL_SETSv5.SA_BASELINE_ACTION,
        seed=7,
    )
    result = algorithm.run(problem, budget=40, max_iteration=20)
    assert result.evaluations == 40
    observations = algorithm._build_observations()
    assert observations.shape == (4, 22)
    assert np.all(np.isfinite(observations))
    assert np.all((observations >= 0) & (observations <= 1))
    assert len(algorithm.ACTION_NAMES) == 9
    assert algorithm.ACTION_NAMES[4] == "sa_crossover_sa_mutation"
    assert {row["action"] for row in algorithm.reward_records} == {4}

    region = int(algorithm.selected_regions[0])
    for action in range(9):
        children = algorithm._make_children(
            problem,
            algorithm.searchers[0],
            algorithm.goods[region][0],
            action,
        )
        assert len(children) == 2
        assert all(len(child.code) == algorithm.code_length for child in children)

    feasible = np.asarray([0.9, 1, 0.1, 0.1, 0, 0.1, 0.2, 0, 1, 10.0])
    improved = feasible.copy()
    improved[0] = 0.92
    improved[9] = 11.0
    worsened = feasible.copy()
    worsened[0] = 0.88
    worsened[9] = 9.0
    assert algorithm._transition_components(feasible, improved)[0] > 0
    assert algorithm._transition_components(feasible, worsened)[0] < 0

    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        replay_checkpoint = temp / "replay.npz"
        learner = D3QNAgent(
            3,
            2,
            replay_capacity=12,
            replay_group_capacity=2,
            replay_warmup=1,
            training=True,
            seed=3,
        )
        for group in range(3):
            learner.set_replay_group(group)
            for step in range(5):
                observation = np.asarray([group, step, 1], dtype=np.float32)
                learner.remember(observation, step % 2, 0.1, observation, False, [1, 1])
        assert len(learner.replay) == 6
        assert all(
            np.sum(learner.replay.group_ids[: len(learner.replay)] == group) == 2
            for group in range(3)
        )
        learner.save(replay_checkpoint)
        restored = D3QNAgent(
            3,
            2,
            replay_capacity=12,
            replay_group_capacity=2,
            replay_warmup=1,
            training=True,
            seed=99,
        )
        restored.load(replay_checkpoint)
        assert len(restored.replay) == len(learner.replay)
        assert np.array_equal(
            restored.replay.group_ids[: len(restored.replay)],
            learner.replay.group_ids[: len(learner.replay)],
        )

        model = temp / "rl_setsv5.npz"
        algorithm.save_model(model)
        experiment = parse_args(
            ["--algorithm", "rl_setsv5", "--rl-model", str(model)]
        )
        experiment._current_map = spec
        frozen_problem = build_problem(experiment, 7)
        frozen = build_algorithm("rl_setsv5", frozen_problem, experiment, 7)
        assert isinstance(frozen, RL_SETSv5)
        assert not frozen.random_policy and frozen.fixed_action is None

        # The fixed-action ablation is no longer reachable from the experiment
        # CLI, but RL_SETSv5 still accepts it directly.
        fixed = RL_SETSv5(frozen_problem, n=4, h=4, w=1, fixed_action=4)
        assert fixed.fixed_action == 4 and not fixed.random_policy

        training_checkpoint = temp / "trained_rl_setsv5.npz"
        manifest = temp / "maps.json"
        manifest.write_text(json.dumps({"maps": ["RLTRAIN01"]}), encoding="utf-8")
        summaries = training_main(
            [
                "--algorithm", "rl_setsv5",
                "--rl-model", str(training_checkpoint),
                "--maps", str(manifest),
                "--episodes", "1",
                "--evaluate", "16",
                "--max-rounds", "1",
                "--segment-slot-limit", "1",
                "--max-lifetime", "1",
                "--n", "4",
                "--w", "1",
                "--replay-capacity", "32",
                "--replay-group-capacity", "8",
                "--replay-warmup", "4",
                "--batch-size", "4",
                "--validation-every", "0",
            ]
        )
        assert len(summaries) == 1
        report = json.loads(
            training_checkpoint.with_suffix(".training.json").read_text(
                encoding="utf-8"
            )
        )
        assert report["protocol"]["algorithm"] == "rl_setsv5"
        assert report["protocol"]["learner"]["replay_capacity"] == 32
        assert report["protocol"]["reward_version"] == algorithm.REWARD_VERSION
    print("smoke_rl_setsv5_ok")


if __name__ == "__main__":
    main()
