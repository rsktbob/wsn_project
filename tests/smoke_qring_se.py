"""Q-Ring-SE：預算、可重現、旋轉只動觀測過的段且方向正確、CLI 整合。"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Algorithm.quantum.QRing_SE import QRing_SE
from Problem.Problem import Problem
from State.SensorEncoding import SensorEncoding
from experiment_algorithms import build_algorithm, parse_args


def main():
    np.random.seed(11)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    for budget in (1, 40, 300):
        algo = QRing_SE(problem, n=4, h=3, w=3, seed=77)
        result = algo.run(problem, budget=budget)
        assert type(algo.best_candidate) is SensorEncoding
        assert result.evaluations >= budget
        assert np.all(np.diff(result.history) >= 0)
        assert np.isclose(result.history[-1], result.best_fitness.sum())
        repeat = QRing_SE(problem, n=4, h=3, w=3, seed=77).run(problem, budget=budget)
        np.testing.assert_array_equal(result.history, repeat.history)
        np.testing.assert_array_equal(result.best_state.next_hops, repeat.best_state.next_hops)

    # 旋轉：段外不動；比 b_k 差時轉向 b_k，比 b_k 好時轉向觀測解。
    algo = QRing_SE(problem, n=4, h=3, w=3, seed=5)
    algo.run(problem, budget=40)
    enc = algo.q_encoding
    good = algo.goods[0].code.copy()
    observed, bits = enc.observe(algo.q_probabilities[0], algo.rng)
    mask = algo._gate_mask(0, 10)
    for better, reference in ((False, good), (True, observed.code)):
        before = algo.q_probabilities[0].copy()
        fitness = 1.0 if better else -1.0
        algo.rotate(0, [(bits, mask)], [observed.code], [fitness], good, 0.0)
        delta = algo.q_probabilities[0] - before
        target, path = enc.elite_path(reference)
        np.testing.assert_allclose(delta[~mask], 0, atol=1e-14)
        assert np.all(delta[mask & path & target] >= -1e-14)
        assert np.all(delta[mask & path & ~target] <= 1e-14)
    lo, hi = np.sin(algo.angle_low) ** 2, np.sin(algo.angle_high) ** 2
    assert np.all(algo.q_probabilities >= lo - 1e-12)
    assert np.all(algo.q_probabilities <= hi + 1e-12)

    # 偏置初始化：init_confidence 高時，觀測多半等於 good。
    biased = QRing_SE(problem, n=4, h=3, w=3, init_confidence=0.95, seed=3)
    biased.run(problem, budget=1)
    sample, _ = biased.q_encoding.observe(biased.q_probabilities[0], biased.rng)
    assert np.mean(sample.code == biased.goods[0].code) > 0.6

    args = parse_args(['--algorithm', 'qring_se', '--evaluate', '7'])
    built = build_algorithm('qring_se', problem, args, seed=9)
    assert isinstance(built, QRing_SE) and built.w == 8
    assert built.run(problem, budget=7).evaluations >= 7
    print('smoke_qring_se_ok')


if __name__ == '__main__':
    main()
