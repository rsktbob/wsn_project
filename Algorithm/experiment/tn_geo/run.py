"""Run matched fixed-state or lifetime TN-GEO experiments."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.dont_write_bytecode = True
os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE / '.numba_cache'))
sys.path.insert(0, str(ROOT))

from Algorithm.experiment.tn_geo.optimizer import TNGEO
from experiments.maps import load_map_specs
from experiments.problem_setup import build_problem
from experiments.runner import _advance_lifetime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', default='MAP01')
    parser.add_argument('--maps', default='maps/maps_100100100.json')
    parser.add_argument('--model', choices=['mps', 'independent'], default='mps')
    parser.add_argument('--order', choices=['identity', 'random', 'coverage'], default='coverage')
    parser.add_argument('--mode', choices=['fixed', 'lifetime'], default='fixed')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--budget', type=int, default=1000, help='fitness calls per optimization')
    parser.add_argument('--rank', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--initial-size', type=int, default=64)
    parser.add_argument('--train-size', type=int, default=64)
    parser.add_argument('--train-steps', type=int, default=3)
    parser.add_argument('--learning-rate', type=float, default=.02)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--exploration', type=float, default=.1)
    parser.add_argument('--mutation', type=float, default=.2)
    parser.add_argument('--sensing-mode', choices=['discrete', 'bucketed', 'exact'], default='discrete')
    parser.add_argument('--fitness-service', choices=['v1', 'v2', 'v3'], default='v2')
    parser.add_argument('--max-lifetime', type=int, default=2000)
    parser.add_argument('--max-segments', type=int, default=2000)
    parser.add_argument('--output', default='results')
    args = parser.parse_args()
    if min(args.budget, args.max_lifetime, args.max_segments) < 1:
        parser.error('budget and limits must be positive')
    specs = [s for s in load_map_specs(args.maps) if s['name'] == args.map]
    if len(specs) != 1:
        parser.error('unknown map')
    args._current_map = specs[0]
    kwargs = {key: getattr(args, key) for key in ('seed', 'model', 'order', 'rank', 'batch_size',
              'initial_size', 'train_size', 'train_steps', 'learning_rate', 'temperature', 'exploration', 'mutation')}
    try:
        algorithm = TNGEO(**kwargs)
    except ValueError as exc:
        parser.error(str(exc))
    out = (HERE / args.output).resolve()
    if HERE not in out.parents:
        parser.error('output must be a subdirectory of tn_geo')
    out.mkdir(parents=True, exist_ok=True)
    filename = out / f'{args.map}_{args.model}_{args.order}_{args.mode}_seed{args.seed}.json'
    if filename.exists():
        parser.error(f'output already exists: {filename}; choose another --output')
    problem = build_problem(args, args.seed)
    records, life, total_fe = [], 0, 0
    started = time.perf_counter()
    reason = 'max_segments'
    for segment in range(1 if args.mode == 'fixed' else args.max_segments):
        result = algorithm.run(problem, budget=args.budget, max_iteration=args.budget)
        state = result.best_state
        cost = problem.calculate_total_cost(state)
        feasible = bool(problem.LifeCheck(state, cost))
        total_fe += result.evaluations
        row = dict(segment=segment, evaluations=result.evaluations, best_score=result.best_score,
                   objectives=result.best_fitness.tolist(), feasible=feasible, lifetime_start=life,
                   levels=state.levels.tolist(), next_hops=state.next_hops.tolist(),
                   history=result.history.tolist(), metadata=result.metadata)
        records.append(row)
        if args.mode == 'fixed':
            reason = 'fixed_state_complete'
            break
        if not feasible:
            reason = 'infeasible_solution'
            break
        life, slots, status = _advance_lifetime(problem, state, cost, life, 5000, args.max_lifetime)
        problem.prepare_coding_cache(cost*slots)
        row['slots'] = slots
        print(json.dumps(dict(segment=segment, lifetime=life, evaluations=total_fe)), flush=True)
        if status == 'lifetime_cap':
            reason = status
            break
    report = dict(config=vars(args), total_evaluations=total_fe, elapsed_seconds=time.perf_counter()-started,
                  lifetime=life if args.mode == 'lifetime' else None, stop_reason=reason,
                  truncated=reason in ('max_segments', 'lifetime_cap'), records=records)
    filename.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(output=str(filename), best_score=records[-1]['best_score'],
                         total_evaluations=total_fe, stop_reason=reason)), flush=True)


if __name__ == '__main__':
    main()
