"""Qdit-TS smooth (Givens): kernel (1,) == QditTS, hand example, boundary, tabu, floor, budget, determinism, runner hookup."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Algorithm.quantum.QditTS import QditTS
from Algorithm.quantum.QditTS_Smooth import QditTS_Smooth
from Problem.Problem import Problem
from experiment_algorithms import parse_args, build_algorithm


def main():
    np.random.seed(11)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algo = QditTS_Smooth(problem, seed=3)
    algo.run(problem, budget=10)
    domains = algo.domains
    np.testing.assert_allclose(algo.probabilities.sum(axis=1), 1.0)
    assert np.all(algo.probabilities[~algo.valid] == 0)
    for _ in range(200):
        code = algo.observe().code
        assert np.all((code >= 0) & (code < domains))

    # kernel (1,) is exactly QditTS: same update on random registers, same run.
    rng = np.random.default_rng(5)
    plain = QditTS(problem, rotation=0.2 * np.pi, seed=3)
    single = QditTS_Smooth(problem, rotation=0.2 * np.pi, kernel=(1.0,), seed=3)
    plain._reset_register()
    single._reset_register()
    for _ in range(50):
        best = rng.integers(0, domains)
        worst = np.where(rng.random(len(domains)) < 0.3, best, rng.integers(0, domains))
        plain._update(best, worst)
        single._update(best, worst)
    np.testing.assert_allclose(single.amplitudes, plain.amplitudes, atol=1e-12)
    a = QditTS(problem, n=6, rotation=0.2 * np.pi, seed=77).run(problem, budget=60)
    b = QditTS_Smooth(problem, n=6, rotation=0.2 * np.pi, kernel=(1.0,), seed=77).run(problem, budget=60)
    np.testing.assert_array_equal(a.history, b.history)

    # Hand example: 6 levels uniform, b=3, w=1, kernel (1, 0.5).
    hand = QditTS_Smooth(problem, rotation=0.05 * np.pi, kernel=(1.0, 0.5), seed=3)
    hand.run(problem, budget=10)
    row = int(np.flatnonzero(domains == 6)[0])
    hand.amplitudes[row, :6] = np.sqrt(1 / 6)
    best = np.minimum(domains - 1, 3)
    worst = best.copy()
    worst[row] = 1
    before = hand.amplitudes.copy()
    hand._update(best, worst)
    P = hand.probabilities[row, :6]
    np.testing.assert_allclose(P.sum(), 1.0)
    assert P[3] > P[4] > 1 / 6 and P[1] < P[0] < 1 / 6 and P[5] == 1 / 6
    others = np.arange(len(domains)) != row
    np.testing.assert_array_equal(hand.amplitudes[others], before[others])

    # Boundary: b=5 gives gain 1/3, 2/3 on levels 4, 5.
    gain = hand._centred(np.array([5]), hand.valid[[row]])
    np.testing.assert_allclose(gain[0, :6], [0, 0, 0, 0, 1 / 3, 2 / 3])

    # Tabu.
    amplitudes = hand.amplitudes.copy()
    hand._update(best, best)
    np.testing.assert_array_equal(hand.amplitudes, amplitudes)

    # Floor: repeated updates keep every option at or above the floor, norm 1.
    worst[row] = 0
    for _ in range(2000):
        hand._update(best, worst)
    floor = np.minimum(hand.probability_floor, 1 / domains)
    assert np.all(hand.probabilities[hand.valid] >= floor[np.nonzero(hand.valid)[0]] - 1e-12)
    np.testing.assert_allclose(hand.probabilities.sum(axis=1), 1.0)
    assert np.isclose(hand.probabilities[row, 0], 0.01) and np.isclose(hand.probabilities[row, 1], 0.01)

    for budget in (1, 7, 25):
        result = QditTS_Smooth(problem, n=6, seed=77).run(problem, budget=budget)
        assert result.evaluations == budget and len(result.history) == budget
        assert np.all(np.diff(result.history) >= 0)
        repeat = QditTS_Smooth(problem, n=6, seed=77).run(problem, budget=budget)
        np.testing.assert_array_equal(result.history, repeat.history)

    args = parse_args(['--algorithm', 'qdit_ts_smooth', '--evaluate', '7'])
    built = build_algorithm('qdit_ts_smooth', problem, args, seed=9)
    assert built.n == 30 and np.isclose(built.rotation, 0.2 * np.pi)
    np.testing.assert_array_equal(built.kernel, [1.0, 0.25])
    assert built.run(problem, budget=7).evaluations == 7
    print('smoke_qdit_ts_smooth_ok')


if __name__ == '__main__':
    main()
