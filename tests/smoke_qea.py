"""QEA invariants, meaningful update controls, and shared-runner integration."""
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from Algorithm.quantum.QEA import QEA
from Problem.Problem import Problem
from State.SensorEncoding import SensorEncoding
from State.QuantumSensorEncoding import QuantumSensorEncoding
from experiment_algorithms import parse_args, build_algorithm


def main():
    sampler = QuantumSensorEncoding([1, 3, 6])
    rng = np.random.default_rng(123)
    counts = np.zeros(6, dtype=int)
    for _ in range(12000):
        candidate, _ = sampler.observe(sampler.initial_probabilities, rng)
        assert type(candidate) is SensorEncoding
        assert np.all(candidate.code >= 0)
        assert np.all(candidate.code < sampler.domains)
        counts[candidate.code[4]] += 1
    assert np.max(np.abs(counts / counts.sum() - 1 / 6)) < 0.02
    lo, _ = sampler.observe(np.zeros(sampler.length), rng)
    hi, _ = sampler.observe(np.ones(sampler.length), rng)
    np.testing.assert_array_equal(lo.code, 0)
    np.testing.assert_array_equal(hi.code, sampler.domains - 1)

    for mode in ('discrete', 'bucketed', 'exact'):
        np.random.seed(11)
        problem = Problem(B=50, S=30, T=9, F=100, FILE=None, sensing_mode=mode)
        for budget in (1, 7, 25):
            first = QEA(problem, n=6, seed=77)
            result = first.run(problem, budget=budget)
            assert type(first.best_candidate) is SensorEncoding
            decoded = first.best_candidate.copy().decode(problem)
            np.testing.assert_array_equal(decoded.levels, result.best_state.levels)
            np.testing.assert_array_equal(decoded.next_hops, result.best_state.next_hops)
            np.testing.assert_array_equal(decoded.tx_load, result.best_state.tx_load)
            assert result.evaluations == budget
            assert len(result.history) == budget
            assert np.all(np.isfinite(result.history))
            assert np.all(np.diff(result.history) >= 0)
            assert np.isclose(result.history[-1], result.best_fitness.sum())
            repeat = QEA(problem, n=6, seed=77).run(problem, budget=budget)
            np.testing.assert_array_equal(result.history, repeat.history)
            np.testing.assert_array_equal(result.best_state.levels, repeat.best_state.levels)
            np.testing.assert_array_equal(result.best_state.next_hops, repeat.best_state.next_hops)
        initial = QEA(problem, n=6, seed=4)
        initial.run(problem, budget=6, max_iteration=1)
        target, path = initial.encoding.elite_path(initial.best_candidate.code)
        before = initial.probabilities.copy()
        initial._update(np.tile(~target, (6, 1)))
        delta = initial.probabilities - before
        assert np.all(delta[:, path & target] >= -1e-14)
        assert np.all(delta[:, path & ~target] <= 1e-14)
        np.testing.assert_allclose(delta[:, ~path], 0, atol=1e-14)
        np.testing.assert_allclose(np.cos(initial.angles)**2 + np.sin(initial.angles)**2, 1)
        first_batches = []
        for rule in ('rotation', 'none', 'classical'):
            control = QEA(problem, n=6, update_rule=rule, seed=5)
            result = control.run(problem, budget=25)
            first_batches.append(result.history[:6])
            if rule == 'none':
                np.testing.assert_array_equal(control.probabilities,
                    np.tile(control.encoding.initial_probabilities, (6, 1)))
        for history in first_batches[1:]:
            np.testing.assert_array_equal(history, first_batches[0])
        limited = QEA(problem, n=6, seed=7).run(problem, budget=25, max_iteration=2)
        assert limited.evaluations == 12 and limited.metadata['early_stopped']
        reused = QEA(problem, n=6, seed=9)
        reused.run(problem, budget=7)
        assert reused.run(problem, budget=3).evaluations == 3

    args = parse_args(['--algorithm', 'qea', 'qea_random', 'qea_classical', '--evaluate', '7'])
    for name, rule in [('qea', 'rotation'), ('qea_random', 'none'), ('qea_classical', 'classical')]:
        algorithm = build_algorithm(name, problem, args, seed=9)
        assert algorithm.update_rule == rule
        assert algorithm.run(problem, budget=7).evaluations == 7
    for kwargs in ({'n':0}, {'rotation':float('nan')}, {'probability_floor':0}, {'update_rule':'bad'}):
        try:
            QEA(problem, **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(kwargs)
    print('smoke_qea_ok')


if __name__ == '__main__':
    main()
