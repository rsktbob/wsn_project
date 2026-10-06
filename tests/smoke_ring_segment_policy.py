"""Smoke checks for the per-pair segment policies (LinUCB / D3QN Ring-SETS)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Algorithm.se.LinUCB_Ring_SETS import LinUCB_Ring_SETS
from Algorithm.se.LinUCB_Ring_SETSpriority import LinUCB_Ring_SETSpriority
from Algorithm.se.RL_Ring_SETS import RL_Ring_SETS
from Algorithm.se.RL_Ring_SETSpriority import RL_Ring_SETSpriority
from Algorithm.se.ring_segment_policy import SharedSegmentDuelingNetwork
from Algorithm.se.LinUCB_Ring_SETSv2 import LinUCB_Ring_SETSv2
from Algorithm.se.LinUCB_Ring_SETSpriorityv2 import LinUCB_Ring_SETSpriorityv2
from Algorithm.se.RL_Ring_SETSv2 import RL_Ring_SETSv2
from Algorithm.se.RL_Ring_SETSpriorityv2 import RL_Ring_SETSpriorityv2
from Problem.Problem import Problem
from experiment_algorithms import build_algorithm, parse_args


def test_network_gradients_and_equivariance():
    rng = np.random.default_rng(0)
    net = SharedSegmentDuelingNetwork(3, 4, 2, 6, rng)
    for name in net.parameters:
        net.parameters[name] = net.parameters[name].astype(np.float64)
    observations = rng.random((4, 3 * 4 + 2))
    actions = rng.integers(0, 3, 4)
    targets = rng.normal(0.0, 0.5, 4)
    _, gradients = net.loss_and_gradients(observations, actions, targets)
    weights = net.parameters["weight1"]
    numeric = np.zeros_like(weights)
    for index in np.ndindex(weights.shape):
        old = weights[index]
        weights[index] = old + 1e-6
        plus, _ = net.loss_and_gradients(observations, actions, targets)
        weights[index] = old - 1e-6
        minus, _ = net.loss_and_gradients(observations, actions, targets)
        weights[index] = old
        numeric[index] = (plus - minus) / 2e-6
    assert np.allclose(numeric, gradients["weight1"], atol=1e-7)

    permutation = [2, 0, 1]
    segments = observations[:, :12].reshape(4, 3, 4)[:, permutation].reshape(4, 12)
    permuted = np.concatenate((segments, observations[:, 12:]), axis=1)
    assert np.allclose(net.predict(permuted), net.predict(observations)[:, permutation])


def test_linucb_learns_and_keeps_budget():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    for algorithm_type in (LinUCB_Ring_SETS, LinUCB_Ring_SETSpriority):
        algorithm = algorithm_type(problem, n=4, h=4, w=3, mu=0.4, seed=7)
        result = algorithm.run(problem, budget=300)
        assert result.best_state is not None
        assert result.evaluations >= 300
        assert np.trace(algorithm.linucb_A) > len(algorithm.FEATURE_NAMES)


def test_d3qn_training_save_and_frozen_load():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    with tempfile.TemporaryDirectory() as folder:
        for algorithm_type in (RL_Ring_SETS, RL_Ring_SETSpriority):
            path = Path(folder) / f"{algorithm_type.__name__}.npz"
            learner = algorithm_type(
                problem, n=4, h=4, w=3, mu=0.4, seed=7, training=True,
                replay_warmup=16, batch_size=8,
            )
            learner.run(problem, budget=300)
            assert len(learner.agent.replay) > 0
            assert learner.agent.training_steps > 0
            learner.save_model(path)

            frozen = algorithm_type(
                problem, n=4, h=4, w=3, mu=0.4, seed=8, model_path=path,
            )
            assert frozen.agent.epsilon == 0.0
            frozen.run(problem, budget=200)
            assert len(frozen.agent.replay) == 0

            try:
                other = RL_Ring_SETSpriority if algorithm_type is RL_Ring_SETS else RL_Ring_SETS
                other(problem, n=4, h=4, w=3, mu=0.4, seed=8, model_path=path)
            except ValueError:
                pass
            else:
                raise AssertionError("a checkpoint loaded into the other encoding")


def test_v2_policies():
    """v2: gamma 0, tiered rewards in {0, 0.5, 1, 5, 5.5, 10}, save/load, frozen run."""
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    for algorithm_type in (LinUCB_Ring_SETSv2, LinUCB_Ring_SETSpriorityv2):
        algorithm = algorithm_type(problem, n=4, h=4, w=3, mu=0.4, seed=7)
        result = algorithm.run(problem, budget=300)
        assert result.evaluations >= 300
        assert algorithm.linucb_A.shape == (18, 18)
    with tempfile.TemporaryDirectory() as folder:
        for algorithm_type in (RL_Ring_SETSv2, RL_Ring_SETSpriorityv2):
            learner = algorithm_type(problem, n=4, h=4, w=3, mu=0.4, seed=7, training=True,
                                     replay_warmup=16, batch_size=8)
            assert learner.agent.gamma == 0.0
            assert learner.agent.observation_size == 4 * 9 + 7
            learner.run(problem, budget=300)
            assert learner.agent.training_steps > 0
            shares = learner.policy_statistics()["reward_share"]
            assert abs(sum(shares.values()) - 1.0) < 1e-2  # 各比例已四捨五入到小數 4 位
            path = Path(folder) / f"{algorithm_type.__name__}.npz"
            learner.save_model(path)
            frozen = algorithm_type(problem, n=4, h=4, w=3, mu=0.4, seed=8, model_path=path)
            frozen.run(problem, budget=200)
            assert len(frozen.agent.replay) == 0
            try:
                RL_Ring_SETS(problem, n=4, h=4, w=3, mu=0.4, seed=8, model_path=path)
            except ValueError:
                pass
            else:
                raise AssertionError("a v2 checkpoint loaded into v1")


def test_registry():
    problem = Problem(B=50, S=32, T=9, F=100, FILE=None)
    args = parse_args(["--algorithm", "linucb_ring_setspriority", "--evaluate", "40"])
    assert type(build_algorithm("linucb_ring_setspriority", problem, args, seed=7)) is LinUCB_Ring_SETSpriority
    try:
        parse_args(["--algorithm", "rl_ring_sets", "--evaluate", "40"])
    except SystemExit:
        pass
    else:
        raise AssertionError("rl_ring_sets must require --rl-model")


def main():
    test_network_gradients_and_equivariance()
    test_linucb_learns_and_keeps_budget()
    test_d3qn_training_save_and_frozen_load()
    test_v2_policies()
    test_registry()
    print("smoke_ring_segment_policy_ok")


if __name__ == "__main__":
    main()
