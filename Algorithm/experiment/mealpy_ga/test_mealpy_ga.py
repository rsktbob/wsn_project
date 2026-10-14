"""Real WSN integration checks, including MEALPY call counts and lifetime."""
import contextlib
import io
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from Algorithm.experiment.mealpy_ga.MealpyGA import MealpyGA
from experiments.cli import parse_args, selected_algorithms
from experiments.builder import build_algorithm
from experiments.problem_setup import build_problem
from experiments.runner import run_lifetime_case


def main():
    assert 'mealpy_ga' not in selected_algorithms(['all'])
    for mode in ('discrete', 'bucketed', 'exact'):
        args = parse_args(['--algorithm', 'mealpy_ga', '--sensing-mode', mode, '--seed', '7'])
        args._current_map = args.map_specs[0]
        problem = build_problem(args, 7)
        for budget in (1, 9, 30, 31, 60, 67):
            alg = build_algorithm('mealpy_ga', problem, args, 7)
            with patch.object(problem, 'evaluate_state', wraps=problem.evaluate_state) as objective:
                result = alg.run(problem, budget=budget)
            assert objective.call_count == budget == result.evaluations == len(result.history)
            assert np.all(np.isfinite(result.history)) and np.all(np.diff(result.history) >= 0)
            assert np.isclose(result.history[-1], result.best_fitness.sum())
            decoded = alg.best_candidate.copy().decode(problem)
            for field in ('levels', 'next_hops', 'tx_load'):
                np.testing.assert_array_equal(getattr(decoded, field), getattr(result.best_state, field))
            repeat = MealpyGA(problem, seed=7).run(problem, budget=budget)
            np.testing.assert_array_equal(result.history, repeat.history)
        limited = MealpyGA(problem, seed=7).run(problem, budget=100, max_iteration=1)
        assert limited.evaluations == 60 and limited.metadata['early_stopped']
        reuse = MealpyGA(problem, seed=7)
        reuse.run(problem, budget=31)
        previous_seed = reuse.last_mealpy_seed
        assert reuse.run(problem, budget=9).evaluations == 9
        assert previous_seed != reuse.last_mealpy_seed
    args = parse_args(['--algorithm', 'mealpy_ga_baseline', '--evaluate', '1000', '--max-lifetime', '3'])
    args._current_map = args.map_specs[0]
    problem = build_problem(args, 7)
    baseline = build_algorithm('mealpy_ga_baseline', problem, args, 7)
    result = baseline.run(problem, budget=67)
    assert baseline.pc == 0.8 and baseline.n == 30
    assert baseline.effective_pm == baseline.model.pm == 1/(2*problem.SENSOR_NUMBER)
    assert result.evaluations == 67
    assert 'mealpy_ga_baseline' not in selected_algorithms(['all'])
    official = build_algorithm('mealpy_ga_official', problem, args, 7)
    assert (official.n, official.pc, official.pm) == (100, 0.95, 0.025)
    assert official.run(problem, budget=101).evaluations == 101
    assert 'mealpy_ga_official' not in selected_algorithms(['all'])
    args._current_map = args.map_specs[0]
    result, trace = run_lifetime_case('mealpy_ga_baseline', 0, args)
    assert result['lifetime'] == 3 and result['stop_reason'] == 'lifetime_cap'
    assert result['actual_evatime'] == 1000
    assert sum(r['segment_length'] for r in trace) == 3
    print('MEALPY WSN checks passed: true budget, integer decoding, repeatability, iteration limits and shared lifetime.')


if __name__ == '__main__':
    with contextlib.redirect_stdout(io.StringIO()) as output:
        main()
    print(output.getvalue().splitlines()[-1])
