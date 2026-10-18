"""Qdit-TS: register shape, Givens update, tabu, floor, budget, determinism, runner hookup."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Algorithm.quantum.QditTS import QditTS
from Problem.Problem import Problem
from State.SensorEncoding import SensorEncoding
from experiment_algorithms import parse_args, build_algorithm


def main():
    np.random.seed(11)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    levels = problem.radius_option_counts
    algo = QditTS(problem, seed=3)
    algo.run(problem, budget=10)
    domains = algo.domains
    assert np.array_equal(domains[0::2], levels)
    assert np.all(domains[1::2] == SensorEncoding.RANK_PRECISION)
    assert int(algo.valid.sum()) == np.sum(levels) + len(levels) * SensorEncoding.RANK_PRECISION
    np.testing.assert_allclose(algo.probabilities.sum(axis=1), 1.0)
    assert np.all(algo.probabilities[~algo.valid] == 0)

    # Observations stay inside every gene's own domain.
    for _ in range(200):
        code = algo.observe().code
        assert np.all((code >= 0) & (code < domains))

    # Hand example from qbit格式.md: uniform 6 options, best 3, worst 1, 9 degrees.
    hand = QditTS(problem, rotation=np.deg2rad(9), seed=3)
    hand.run(problem, budget=10)
    hand.amplitudes[0, :6] = 1 / np.sqrt(6)
    best = np.minimum(domains - 1, 3)
    worst = best.copy()
    worst[0] = 1
    before = hand.probabilities.copy()
    hand._update(best, worst)
    after = hand.probabilities
    # The (a_3, a_1) pair has radius^2 = 1/3 at 45 degrees and turns to 54 degrees.
    s3, s1 = np.sin(np.deg2rad(54)) ** 2 / 3, np.sin(np.deg2rad(36)) ** 2 / 3
    np.testing.assert_allclose(after[0, :6], [1 / 6, s1, 1 / 6, s3, 1 / 6, 1 / 6])
    assert round(s3, 3) == 0.218 and round(s1, 3) == 0.115
    np.testing.assert_array_equal(after[1:], before[1:])
    np.testing.assert_allclose(after.sum(axis=1), before.sum(axis=1))

    # Tabu: best == worst moves nothing.
    p = hand.probabilities.copy()
    hand._update(best, best)
    np.testing.assert_array_equal(hand.probabilities, p)

    # Floor: repeated pressure stops the worst option at the floor, not at zero.
    for _ in range(500):
        hand._update(best, worst)
    assert abs(hand.probabilities[0, 1] - hand.probability_floor) < 1e-9
    assert np.all(hand.probabilities[hand.valid] >= hand.probability_floor - 1e-12)
    np.testing.assert_allclose(hand.probabilities.sum(axis=1), 1.0)

    for budget in (1, 7, 25):
        result = QditTS(problem, n=6, seed=77).run(problem, budget=budget)
        assert result.evaluations == budget and len(result.history) == budget
        assert np.all(np.diff(result.history) >= 0)
        repeat = QditTS(problem, n=6, seed=77).run(problem, budget=budget)
        np.testing.assert_array_equal(result.history, repeat.history)
    # Budget is not cut by the default 800-iteration cap.
    assert QditTS(problem, n=2, seed=1).run(problem, budget=1700).evaluations == 1700

    args = parse_args(['--algorithm', 'qdit_ts', '--evaluate', '7'])
    built = build_algorithm('qdit_ts', problem, args, seed=9)
    assert built.n == 10 and built.run(problem, budget=7).evaluations == 7
    print('smoke_qdit_ts_ok')


if __name__ == '__main__':
    main()
