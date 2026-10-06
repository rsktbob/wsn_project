"""Wall-clock timing of the existing local scheduling/routing implementation."""
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
import numpy as np


def main():
    out = HERE / 'results/local_schedule_v3'
    config = json.loads((out/'config.json').read_text())
    saved = json.loads((out/'MAP01_paid.json').read_text())
    args = SimpleNamespace(_current_map=config['map'], sensing_mode='discrete', fitness_service='v2')
    p = build_problem(args, 7)
    initial_caps = p.link_capacity.copy()
    algorithm = LocalScheduleRouting(True)
    records = []
    for epoch in saved['trace']:
        p.energy = np.asarray(epoch['energy_before'], dtype=float)
        p.link_capacity = initial_caps.copy()
        p.prepare_coding_cache(p.initial_energy - p.energy)
        algorithm.build(p)  # warm-up outside timing
        for repeat in range(5):
            start = time.perf_counter()
            result = algorithm.build(p)
            elapsed = time.perf_counter() - start
            records.append(dict(segment=epoch['segment'], repeat=repeat,
                                seconds=elapsed, claims=len(result.schedule.claims)))
    complete = []
    for repeat in range(3):
        p = build_problem(args, 7)  # setup excluded
        start = time.perf_counter()
        summary, trace = simulate(p, True)
        elapsed = time.perf_counter() - start
        assert summary['lifetime'] == saved['summary']['lifetime']
        complete.append(dict(seconds=elapsed, lifetime=summary['lifetime'], segments=len(trace)))
    times = [r['seconds'] for r in records]
    report = dict(map='MAP01', sensors=100, targets=100, control_charged=True,
                  measurement='perf_counter wall time; warm process; excludes imports/map loading/disk writes; includes Python message simulation',
                  build_samples=len(times), build_mean_seconds=statistics.mean(times),
                  build_median_seconds=statistics.median(times), build_min_seconds=min(times),
                  build_max_seconds=max(times),
                  successful_build_mean_seconds=statistics.mean(r['seconds'] for r in records if saved['trace'][r['segment']]['slots'] > 0),
                  full_lifetime_mean_seconds=statistics.mean(r['seconds'] for r in complete),
                  full_lifetime_runs=complete, build_timings=records)
    (out/'timing.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'build_timings'}, indent=2))


if __name__ == '__main__':
    main()
