"""Paired global-QEA pilot with raw per-evaluation histories and controls.

Run from any directory: python /path/to/experiments/benchmark_qea.py
Budget 3006 = GA initialization 30 + 96 complete generations of 31 calls.
This avoids changing GA or giving it an evaluation-budget overshoot.
"""
import argparse
import contextlib
import csv
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.cli import parse_args
from experiments.builder import build_algorithm, algorithm_params
from experiments.problem_setup import build_problem
from experiments.evaluation import state_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', default='maps/maps_100100100.json')
    parser.add_argument('--map-count', type=int, default=3)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--budget', type=int, default=3006)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', type=Path, default=ROOT / 'experiment_results' / 'qea_pilot')
    options = parser.parse_args()
    if min(options.map_count, options.runs) < 1 or options.budget < 30 or (options.budget - 30) % 31:
        parser.error('positive maps/runs and budget=30+31*k required for exact GA comparison')
    names = ['qea', 'qea_classical', 'qea_random', 'ga']
    args = parse_args(['--algorithm', *names, '--maps', options.maps, '--mode', 'single',
                       '--evaluate', str(options.budget), '--runs', str(options.runs),
                       '--seed', str(options.seed)])
    output = options.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records, histories = [], {}
    # Warm up the shared decoder/Numba kernels outside the measured runs.
    args._current_map = args.map_specs[0]
    with contextlib.redirect_stdout(io.StringIO()):
        warm_problem = build_problem(args, options.seed)
        build_algorithm('qea', warm_problem, args, options.seed).run(warm_problem, budget=1)
    for spec in args.map_specs[:options.map_count]:
        args._current_map = spec
        for run_id in range(options.runs):
            seed = options.seed + run_id
            # Rotate run order to reduce systematic timing-order bias.
            order = names[run_id % len(names):] + names[:run_id % len(names)]
            for name in order:
                with contextlib.redirect_stdout(io.StringIO()):
                    problem = build_problem(args, seed)
                    algorithm = build_algorithm(name, problem, args, seed)
                    history = []
                    original_evaluate = algorithm.evaluate
                    def measured_evaluate(*a, **kw):
                        value = original_evaluate(*a, **kw)
                        history.append(algorithm.fitness)
                        return value
                    algorithm.evaluate = measured_evaluate
                    start = time.perf_counter()
                    result = algorithm.run(problem, budget=options.budget,
                                           max_iteration=options.budget)
                    elapsed = time.perf_counter() - start
                assert result.evaluations == options.budget == len(history)
                metrics = state_metrics(problem, result.best_state, decode=False,
                                        objectives=result.best_fitness)
                row = dict(map=spec['name'], seed=seed, algorithm=name,
                           evaluations=result.evaluations, seconds=elapsed, **metrics)
                row['feasible'] = not any(row[key] for key in
                    ('uncovered_count', 'disconnected_count', 'energy_failed_count'))
                records.append(row)
                histories[f"{spec['name']}_{seed}_{name}"] = np.asarray(history)
                print(f"{spec['name']} seed={seed} {name}: fitness={row['fitness']:.6f} feasible={row['feasible']} {elapsed:.2f}s", flush=True)
    with (output / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    np.savez_compressed(output / 'histories.npz', **histories)
    summary = {}
    for name in names:
        rows = [row for row in records if row['algorithm'] == name]
        summary[name] = dict(mean_fitness=float(np.mean([r['fitness'] for r in rows])),
                             std_fitness=float(np.std([r['fitness'] for r in rows], ddof=1)) if len(rows)>1 else 0.,
                             mean_seconds=float(np.mean([r['seconds'] for r in rows])),
                             feasible_runs=sum(r['feasible'] for r in rows), runs=len(rows))
    pairs = {}
    for name in names[1:]:
        delta = []
        for q in (r for r in records if r['algorithm'] == 'qea'):
            other = next(r for r in records if r['algorithm']==name and r['map']==q['map'] and r['seed']==q['seed'])
            delta.append(q['fitness'] - other['fitness'])
        pairs[name] = dict(wins=int(np.sum(np.asarray(delta)>1e-12)),
                           ties=int(np.sum(np.abs(delta)<=1e-12)),
                           losses=int(np.sum(np.asarray(delta)<-1e-12)),
                           mean_delta=float(np.mean(delta)))
    report = dict(budget=options.budget, maps=args.map_specs[:options.map_count],
                  seeds=list(range(options.seed, options.seed+options.runs)),
                  fitness_service=args.fitness_service, sensing_mode=args.sensing_mode,
                  parameters={name: algorithm_params(name,args) for name in names},
                  summary=summary, qea_paired_comparisons=pairs,
                  limitations=['Pilot only; no hyperparameter tuning or significance claim.',
                               'GA uses its existing coverage-seeded initialization; QEA controls share a uniform prior.',
                               'Single optimization; does not measure network lifetime or quantum hardware advantage.'])
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9,5))
    for name in names:
        curves = np.asarray([v for k,v in histories.items() if k.endswith('_'+name)])
        mean = curves.mean(axis=0)
        ax.plot(np.arange(1,len(mean)+1), mean, label=name)
    ax.set(xlabel='Actual fitness evaluations', ylabel='Mean best-so-far fitness (v2)',
           title=f'Global QEA pilot: {len(report["maps"])} maps x {options.runs} seeds')
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/'convergence.png',dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    print(json.dumps(pairs, indent=2))


if __name__ == '__main__':
    main()
