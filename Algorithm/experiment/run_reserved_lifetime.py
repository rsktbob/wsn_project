"""Controlled SA-SETS scheduling + distributed routing lifetime experiment."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
# Keep bytecode/Numba cache and every experiment output within this directory.
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(HERE / '.numba_cache')
sys.path.insert(0, str(ROOT))
from experiments.maps import load_map_specs
from experiments.problem_setup import build_problem
from experiments.builder import build_algorithm
from experiments.runner import _advance_lifetime
from Algorithm.experiment.reserved_routing import ReservedRouting


def run_case(task):
    spec, mode, seed, budget, out, max_rounds, cap = task
    started = time.time()
    out = Path(out)
    stem = f"{spec['name']}_{mode}_{seed}"
    with (out / f'{stem}.log').open('w') as log, contextlib.redirect_stdout(log):
        args = SimpleNamespace(_current_map=spec, sensing_mode='discrete', fitness_service='v2')
        p = build_problem(args, seed)
        algorithm = build_algorithm('sa_sets', p, args, seed)
        life, evaluations, paid_control, messages = 0, 0, 0., 0
        trace = []
        reason = 'max_rounds'
        for segment in range(max_rounds):
            result = algorithm.run(p, budget=budget)
            evaluations += result.evaluations
            state = result.best_state
            if state is None:
                reason = 'no_state'
                break
            control = np.zeros(p.SENSOR_NUMBER)
            sent = 0
            # Same centralized scheduling method in all arms; local router
            # reads ONLY selected levels, never the optimizer's next hops.
            if mode != 'sa_sets':
                built = ReservedRouting(128 if mode == 'distributed_control' else 0).build(p, state.levels)
                state, control, sent = built.state, built.control, built.messages
            data_cost = p.calculate_total_cost(state)
            record = dict(segment=segment, lifetime_start=life,
                          evaluations=int(result.evaluations), active=int(np.count_nonzero(state.levels)),
                          routing_cost=float(p.calculate_routing_cost(state).sum()),
                          scheduling_cost=float(p.calculate_scheduling_cost(state).sum()),
                          control_cost=float(control.sum()), control_messages=sent,
                          levels=state.levels.tolist(), next_hops=state.next_hops.tolist())
            if not p.LifeCheck(state, data_cost + control):
                reason = 'infeasible_solution'
                record.update(slots=0, lifetime_end=life, status=reason, control_paid=False)
                trace.append(record)
                break
            # Routing messages are paid ONCE on installing a new schedule.
            p.energy -= control
            p.prepare_coding_cache(control)
            paid_control += float(control.sum())
            messages += sent
            life, slots, status = _advance_lifetime(p, state, data_cost, life, 5000, cap)
            p.prepare_coding_cache(data_cost * slots)
            record.update(slots=slots, lifetime_end=life, status=status, control_paid=True)
            trace.append(record)
            print(segment, life, slots, flush=True)
            if status == 'lifetime_cap':
                reason = status
                break
            if slots == 0:
                reason = 'no_lifetime_progress'
                break
        summary = dict(map=spec['name'], mode=mode, seed=seed, lifetime=life,
                       stop_reason=reason, truncated=reason in ('max_rounds','lifetime_cap'),
                       evaluations=int(evaluations), segments=len(trace),
                       control_energy=paid_control, control_messages=messages,
                       seconds=time.time()-started)
        (out / f'{stem}.json').write_text(json.dumps(dict(summary=summary, trace=trace), indent=2))
    return summary


import numpy as np


def source_hashes():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in ROOT.rglob('*') if p.is_file() and HERE not in p.parents
            and p.suffix in ('.py', '.md', '.json') and '__pycache__' not in p.parts}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', default=None)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--budget', type=int, default=10000)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-rounds', type=int, default=2000)
    parser.add_argument('--lifetime-cap', type=int, default=None)
    parser.add_argument('--output', default='results/reserved_v2')
    args = parser.parse_args()
    if min(args.runs, args.budget, args.workers, args.max_rounds) < 1:
        parser.error('counts must be positive')
    out = (HERE / args.output).resolve()
    if HERE not in out.parents:
        parser.error('output must be inside Algorithm/experiment')
    out.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    specs = load_map_specs('maps/maps_100100100.json')
    if args.map:
        specs = [s for s in specs if s['name'] == args.map]
        if not specs:
            parser.error('unknown map')
    modes = ('sa_sets', 'distributed_free', 'distributed_control')
    config = dict(vars(args), maps=specs, modes=modes,
                  scope='centralized SA-SETS scheduling + distributed routing; static nodes',
                  control_model='128-bit query/reply and reservation/ack only at schedule change; ideal reliable control channel',
                  common_overhead='central sensing schedule distribution and neighbor discovery excluded in all arms')
    (out / 'config.json').write_text(json.dumps(config, indent=2))
    tasks = [(s, m, args.seed+r, args.budget, str(out), args.max_rounds, args.lifetime_cap)
             for s in specs for r in range(args.runs) for m in modes]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for row in pool.map(run_case, tasks):
            rows.append(row)
            print(json.dumps(row), flush=True)
            with (out / 'summary.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(rows)
    after = source_hashes()
    unchanged = before == after
    (out / 'external_source_audit.json').write_text(json.dumps(dict(unchanged=unchanged, before=before, after=after), indent=2))
    if not unchanged:
        raise RuntimeError('outside source files changed during experiment; inspect audit')


if __name__ == '__main__':
    main()
