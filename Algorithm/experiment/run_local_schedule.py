"""Small no-training lifetime validation; output stays in this directory."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(HERE / '.numba_cache')
sys.path.insert(0, str(ROOT))
from experiments.maps import load_map_specs
from experiments.problem_setup import build_problem
from experiments.runner import _advance_lifetime
from Algorithm.experiment.local_scheduling import LocalScheduleRouting
from Algorithm.experiment.run_reserved_lifetime import source_hashes


def simulate(p, paid, cap=2000, max_rounds=2000):
    algorithm = LocalScheduleRouting(paid)
    life, trace = 0, []
    reason = 'max_rounds'
    for segment in range(max_rounds):
        result = algorithm.build(p)
        state = result.state
        cost = p.calculate_total_cost(state)
        missing = list(map(int, p.find_uncovered_targets(state)))
        disconnected = list(map(int, p.find_disconnected(state)))
        failed = np.flatnonzero(cost + result.control > p.energy).tolist()
        row = dict(segment=segment, lifetime_start=life, levels=state.levels.tolist(),
                   next_hops=state.next_hops.tolist(), energy_before=p.energy.tolist(),
                   control_energy=result.control.tolist(), control_messages=result.messages,
                   scheduling_cost=float(p.calculate_scheduling_cost(state).sum()),
                   routing_cost=float(p.calculate_routing_cost(state).sum()),
                   uncovered=missing, disconnected=disconnected, energy_failed=failed,
                   claims=result.schedule.claims)
        if not p.LifeCheck(state, cost + result.control):
            reason = 'infeasible_solution'
            row.update(slots=0, status=reason, control_paid=False)
            trace.append(row)
            break
        p.energy -= result.control
        p.prepare_coding_cache(result.control)
        life, slots, status = _advance_lifetime(p, state, cost, life, 5000, cap)
        p.prepare_coding_cache(cost * slots)
        row.update(slots=slots, status=status, control_paid=True)
        trace.append(row)
        if status == 'lifetime_cap':
            reason = status
            break
    return dict(lifetime=life, stop_reason=reason, truncated=reason in ('max_rounds','lifetime_cap'),
                segments=len(trace), policy='local_schedule_and_routing',
                control_charged=paid, training=False), trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', default='MAP01')
    parser.add_argument('--max-lifetime', type=int, default=2000)
    parser.add_argument('--output', default='results/local_schedule_v1')
    args = parser.parse_args()
    if args.max_lifetime <= 0:
        parser.error('max-lifetime must be positive')
    out = (HERE / args.output).resolve()
    if HERE not in out.parents:
        parser.error('output must stay inside Algorithm/experiment')
    out.mkdir(parents=True, exist_ok=True)
    before = source_hashes()
    specs = [s for s in load_map_specs('maps/maps_100100100.json') if s['name'] == args.map]
    if len(specs) != 1:
        parser.error('unknown map')
    spec = specs[0]
    (out/'config.json').write_text(json.dumps(dict(map=spec, max_lifetime=args.max_lifetime,
        seed=7, control_header_bits=64, control_target_id_bits=16, routing_message_bits=128,
        assumption='known overlap peers, ideal serialized timers, reliable direct control links; no discovery cost'), indent=2))
    rows = []
    for paid in (False, True):
        p = build_problem(SimpleNamespace(_current_map=spec, sensing_mode='discrete', fitness_service='v2'), 7)
        summary, trace = simulate(p, paid, args.max_lifetime)
        summary['map'] = spec['name']
        rows.append(summary)
        (out/f'{spec["name"]}_{"paid" if paid else "free"}.json').write_text(json.dumps(dict(summary=summary,trace=trace),indent=2))
        print(json.dumps(summary), flush=True)
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    after = source_hashes()
    (out/'external_source_audit.json').write_text(json.dumps(dict(unchanged=before==after,before=before,after=after),indent=2))
    assert before == after, 'outside source changed'


if __name__ == '__main__':
    main()
