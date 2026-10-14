"""QTS: gate count, tabu update direction, budget, determinism, runner hookup."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Algorithm.quantum.QTS import QTS
from Problem.Problem import Problem
from State.SensorEncoding import SensorEncoding
from experiment_algorithms import parse_args, build_algorithm


def main():
    np.random.seed(11)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algo = QTS(problem, seed=3)
    levels = problem.radius_option_counts
    assert algo.encoding.length == np.sum(levels - 1) + len(levels) * (SensorEncoding.RANK_PRECISION - 1)

    # Tabu update: only best-path gates where worst disagrees move, toward best.
    algo.run(problem, budget=10)
    domains = algo.encoding.domains
    best = np.minimum(domains - 1, 3)
    worst = best.copy()
    worst[0] = 0
    before = algo.probabilities.copy()
    algo._update(best, worst)
    delta = algo.probabilities - before
    target, path = algo.encoding.elite_path(best)
    moved = np.flatnonzero(np.abs(delta) > 1e-12)
    assert len(moved) and np.all(algo.encoding.gene_ids[moved] == 0)
    assert np.all(delta[path & target] >= -1e-14) and np.all(delta[path & ~target] <= 1e-14)
    algo2 = QTS(problem, seed=3)
    algo2.run(problem, budget=10)
    p = algo2.probabilities.copy()
    algo2._update(best, best)
    np.testing.assert_array_equal(algo2.probabilities, p)

    for init in ('uniform', 'half'):
        for budget in (1, 7, 25):
            result = QTS(problem, n=6, init=init, seed=77).run(problem, budget=budget)
            assert result.evaluations == budget and len(result.history) == budget
            assert np.all(np.diff(result.history) >= 0)
            repeat = QTS(problem, n=6, init=init, seed=77).run(problem, budget=budget)
            np.testing.assert_array_equal(result.history, repeat.history)
    # Budget is not cut by the default 800-iteration cap.
    assert QTS(problem, n=2, seed=1).run(problem, budget=1700).evaluations == 1700

    args = parse_args(['--algorithm', 'qts', '--evaluate', '7'])
    built = build_algorithm('qts', problem, args, seed=9)
    assert built.n == 10 and built.run(problem, budget=7).evaluations == 7
    print('smoke_qts_ok')


if __name__ == '__main__':
    main()
