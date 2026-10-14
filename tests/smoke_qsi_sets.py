"""Q-SI-SETS：預算、可重現、旋轉只動觀測過的段且兩個方向都會轉、CLI 整合。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Algorithm.quantum.QSI_SETS import QSI_SETS
from Problem.Problem import Problem
from State.SensorEncoding import SensorEncoding
from experiment_algorithms import build_algorithm, parse_args


def main():
    np.random.seed(11)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    for budget in (1, 40, 300):
        algo = QSI_SETS(problem, n=4, h=2, w=3, seed=77)
        result = algo.run(problem, budget=budget)
        assert type(algo.best_candidate) is SensorEncoding
        assert result.evaluations >= budget
        assert np.all(np.diff(result.history) >= 0)
        assert np.isclose(result.history[-1], result.best_fitness.sum())
        repeat = QSI_SETS(problem, n=4, h=2, w=3, seed=77).run(problem, budget=budget)
        np.testing.assert_array_equal(result.history, repeat.history)
        np.testing.assert_array_equal(result.best_state.next_hops, repeat.best_state.next_hops)

    # 旋轉：段外不動；較差轉向 b，較好轉向 child1，且兩支都真的有轉。
    algo = QSI_SETS(problem, n=4, h=2, w=3, seed=5)
    algo.run(problem, budget=40)
    enc = algo.q_encoding
    good = algo.goods[0][0].code.copy()
    other = SensorEncoding.random(problem.SENSOR_NUMBER, problem.radius_option_counts,
                                  rng=np.random.default_rng(1)).code
    mask = algo._gate_mask(0, 20)
    for better, reference in ((False, good), (True, other)):
        before = algo.q_probabilities[0, 0].copy()
        fitness = 1.0 if better else -1.0
        algo.rotate(0, 0, [mask], [other], [fitness], good, 0.0)
        delta = algo.q_probabilities[0, 0] - before
        target, path = enc.elite_path(reference)
        np.testing.assert_allclose(delta[~mask], 0, atol=1e-14)
        assert np.all(delta[mask & path & target] >= -1e-14)
        assert np.all(delta[mask & path & ~target] <= 1e-14)
        assert np.abs(delta).max() > 1e-6
    lo, hi = np.sin(algo.angle_low) ** 2, np.sin(algo.angle_high) ** 2
    assert np.all(algo.q_probabilities >= lo - 1e-12)
    assert np.all(algo.q_probabilities <= hi + 1e-12)

    # 偏置初始化：init_confidence 高時，觀測多半等於 good。
    biased = QSI_SETS(problem, n=4, h=2, w=3, init_confidence=0.95, seed=3)
    biased.run(problem, budget=1)
    sample, _ = biased.q_encoding.observe(biased.q_probabilities[1, 0], biased.rng)
    assert np.mean(sample.code == biased.goods[1][0].code) > 0.6

    args = parse_args(['--algorithm', 'qsi_sets', '--evaluate', '7'])
    built = build_algorithm('qsi_sets', problem, args, seed=9)
    assert isinstance(built, QSI_SETS) and built.w == 8
    assert built.run(problem, budget=7).evaluations >= 7
    print('smoke_qsi_sets_ok')


if __name__ == '__main__':
    main()
