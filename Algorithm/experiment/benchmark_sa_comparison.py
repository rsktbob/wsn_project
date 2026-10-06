"""Matched initial-state timing, plus a sequential full-lifetime timing."""
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(HERE / '.numba_cache')
sys.path.insert(0, str(HERE.parents[1]))
from Algorithm.experiment.run_local_schedule import build_problem, simulate
from Algorithm.experiment.local_scheduling import LocalScheduleRouting
from Algorithm.experiment.run_reserved_lifetime import source_hashes
from experiments.builder import build_algorithm
from experiments.runner import _advance_lifetime


def main():
    before = source_hashes()
    out = HERE / 'results/local_schedule_v3'
    config = json.loads((out/'config.json').read_text())
    args = SimpleNamespace(_current_map=config['map'], sensing_mode='discrete', fitness_service='v2')
    initial = {'local': [], 'sa_sets': []}
    # First pair warms kernels/process machinery and is excluded.
    for repeat in range(6):
        for name in ('local', 'sa_sets'):
            p = build_problem(args, 7)
            alg = LocalScheduleRouting(True) if name == 'local' else build_algorithm('sa_sets', p, args, 7)
            start = time.perf_counter()
            if name == 'local':
                result = alg.build(p)
            else:
                result = alg.run(p, budget=10000)
            elapsed = time.perf_counter() - start
            if name == 'sa_sets':
                assert result.evaluations == 10000
            if repeat:
                initial[name].append(elapsed)
    print('matched_initial_means', {k:statistics.mean(v) for k,v in initial.items()}, flush=True)
    p = build_problem(args, 7)
    alg = build_algorithm('sa_sets', p, args, 7)
    life = 0
    calls = []
    start_full = time.perf_counter()
    for segment in range(2000):
        start = time.perf_counter()
        result = alg.run(p, budget=10000)
        calls.append(time.perf_counter()-start)
        state = result.best_state
        cost = p.calculate_total_cost(state)
        if not p.LifeCheck(state, cost):
            break
        life, slots, reason = _advance_lifetime(p, state, cost, life, 5000, None)
        p.prepare_coding_cache(cost * slots)
    sa_full = time.perf_counter() - start_full
    assert life == 1062
    p = build_problem(args, 7)
    start = time.perf_counter()
    summary, trace = simulate(p, True)
    local_full = time.perf_counter()-start
    assert summary['lifetime'] == 776
    report = dict(map='MAP01', seed=7, sa_budget_per_call=10000,
                  timing_scope='perf_counter; warm kernels; sequential arms; includes algorithm internal worker startup, excludes map/algorithm construction and file I/O',
                  matched_initial_samples=initial,
                  matched_initial_mean_seconds={k:statistics.mean(v) for k,v in initial.items()},
                  initial_speed_ratio=statistics.mean(initial['sa_sets'])/statistics.mean(initial['local']),
                  full_lifetime_single_run=dict(sa_seconds=sa_full, sa_lifetime=life, sa_calls=len(calls),
                      sa_mean_optimizer_call_seconds=statistics.mean(calls), sa_call_seconds=calls,
                      local_seconds=local_full, local_lifetime=summary['lifetime'],local_calls=len(trace)),
                  external_sources_unchanged=before==source_hashes())
    assert report['external_sources_unchanged']
    (out/'timing_sa_comparison.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
