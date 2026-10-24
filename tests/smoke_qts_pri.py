"""QTSpri / QditTSpri: register size, PriorityEncoding candidates, uniform prior, tabu, budget, determinism, runner hookup."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Algorithm.quantum.QditTSpri import QditTSpri
from Algorithm.quantum.QTSpri import QTSpri
from Problem.Problem import Problem
from State.PriorityEncoding import PriorityEncoding
from experiment_algorithms import parse_args, build_algorithm


def main():
    np.random.seed(11)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    sensors = problem.SENSOR_NUMBER
    precision = PriorityEncoding.RANK_PRECISION

    # QTSpri: 2 genes x 9 gates per sensor, uniform over the 10 values.
    qts = QTSpri(problem, seed=3)
    assert qts.encoding.length == 2 * sensors * (precision - 1)
    assert np.all(qts.encoding.domains == precision)
    qts.run(problem, budget=10)
    rng = np.random.default_rng(0)
    draws = np.concatenate([qts.encoding.observe(qts.encoding.initial_probabilities, rng)[0].code
                            for _ in range(400)])
    np.testing.assert_allclose(np.bincount(draws, minlength=precision) / len(draws), 0.1, atol=0.01)
    candidate, _ = qts.encoding.observe(qts.probabilities, qts.rng)
    assert isinstance(candidate, PriorityEncoding) and len(candidate.code) == 2 * sensors
    best = np.full(2 * sensors, 5)
    p = qts.probabilities.copy()
    qts._update(best, best)
    np.testing.assert_array_equal(qts.probabilities, p)

    # QditTSpri: one row of 10 amplitudes per gene.
    qdit = QditTSpri(problem, seed=3)
    qdit.run(problem, budget=10)
    assert qdit.amplitudes.shape == (2 * sensors, precision) and np.all(qdit.valid)
    np.testing.assert_allclose(qdit.probabilities.sum(axis=1), 1.0)
    candidate = qdit.observe()
    assert isinstance(candidate, PriorityEncoding)
    assert np.all((candidate.code >= 0) & (candidate.code < precision))
    a = qdit.amplitudes.copy()
    qdit._update(best, best)
    np.testing.assert_array_equal(qdit.amplitudes, a)

    for cls in (QTSpri, QditTSpri):
        for budget in (1, 7, 25):
            result = cls(problem, n=6, seed=77).run(problem, budget=budget)
            assert result.evaluations == budget and len(result.history) == budget
            assert np.all(np.diff(result.history) >= 0)
            repeat = cls(problem, n=6, seed=77).run(problem, budget=budget)
            np.testing.assert_array_equal(result.history, repeat.history)

    for name, n, rotation in (('qts_pri', 30, 0.08 * np.pi), ('qdit_ts_pri', 10, 0.03 * np.pi)):
        args = parse_args(['--algorithm', name, '--evaluate', '7'])
        built = build_algorithm(name, problem, args, seed=9)
        assert built.n == n and np.isclose(built.rotation, rotation)
        assert built.run(problem, budget=7).evaluations == 7
    print('smoke_qts_pri_ok')


if __name__ == '__main__':
    main()
