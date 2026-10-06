"""Paired six-map experiment for quantum-inspired local scheduling v2."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['NUMBA_CACHE_DIR']=str(HERE/'.numba_cache')
sys.path.insert(0,str(ROOT))
import numpy as np
from Algorithm.experiment.quantum_local_v2 import QuantumLocalV2
from Algorithm.experiment.local_scheduling import LocalScheduleRouting
from Algorithm.experiment.run_local_schedule import build_problem
from Algorithm.experiment.run_reserved_lifetime import source_hashes
from Algorithm.experiment.run_local_six_maps import verify
from experiments.runner import _advance_lifetime
from experiments.maps import load_map_specs

def simulate_variant(p, mode, seed, cap=None, max_rounds=2000):
    paid = True
    algorithm = LocalScheduleRouting(paid) if mode == "greedy" else QuantumLocalV2(mode,seed,paid)
    life, trace = 0, []
    reason = 'max_rounds'
    for segment in range(max_rounds):
        result = algorithm.build(p)
        state = result.state
        cost = p.calculate_total_cost(state)
        if mode != "greedy":
            algorithm.observe_local_outcomes(state.levels,state.next_hops,p.energy,cost,result.control)
        missing = list(map(int, p.find_uncovered_targets(state)))
        disconnected = list(map(int, p.find_disconnected(state)))
        failed = np.flatnonzero(cost + result.control > p.energy).tolist()
        row = dict(segment=segment, lifetime_start=life, levels=state.levels.tolist(),
                   next_hops=state.next_hops.tolist(), energy_before=p.energy.tolist(),
                   control_energy=result.control.tolist(), control_messages=result.messages,
                   scheduling_cost=float(p.calculate_scheduling_cost(state).sum()),
                   routing_cost=float(p.calculate_routing_cost(state).sum()),
                   uncovered=missing, disconnected=disconnected, energy_failed=failed,
                   claims=result.schedule.claims,
                   local_feedback=[] if mode == "greedy" else algorithm.last_feedback)
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
                segments=len(trace), policy=mode, seed=seed,
                control_charged=paid, training=False), trace


def run_case(task):
    spec,mode,seed,out=task
    out=Path(out)
    stem=f"{spec['name']}_{mode}_{seed}"
    with (out/f'{stem}.log').open('w') as log, contextlib.redirect_stdout(log):
        p=build_problem(SimpleNamespace(_current_map=spec,sensing_mode='discrete',fitness_service='v2'),7)
        start=time.perf_counter()
        summary,trace=simulate_variant(p,mode,seed)
        summary.update(map=spec['name'],seconds=time.perf_counter()-start,
                       final_uncovered=len(trace[-1]['uncovered']),
                       final_disconnected=len(trace[-1]['disconnected']),
                       final_energy_failed=len(trace[-1]['energy_failed']),
                       paid_control_energy=sum(sum(t['control_energy']) for t in trace if t['control_paid']))
        assert not summary['truncated']
        verify(p,trace,summary)
        for t in trace:
            for f in t['local_feedback']:
                np.testing.assert_allclose(np.sin(f['angles'])**2,f['probabilities'],atol=1e-12)
        (out/f'{stem}.json').write_text(json.dumps(dict(summary=summary,trace=trace),indent=2))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=int,default=5)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--map',default=None)
    parser.add_argument('--output',default='results/quantum_local_v2_six_maps')
    args=parser.parse_args()
    if min(args.runs,args.workers)<1:
        parser.error('counts must be positive')
    out=(HERE/args.output).resolve()
    if HERE not in out.parents:
        parser.error('output must remain in Algorithm/experiment')
    out.mkdir(parents=True,exist_ok=True)
    specs=load_map_specs('maps/maps_100100100.json')
    if args.map:
        specs=[s for s in specs if s['name']==args.map]
        if not specs: parser.error('unknown map')
    modes=['random','classical','quantum']
    sources=['local_scheduling.py','reserved_routing.py','probabilistic_scheduling.py','quantum_local_v2.py','run_quantum_local_v2.py']
    hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in sources}
    for name in sources:
        (out/('snapshot_'+name)).write_bytes((HERE/name).read_bytes())
    before=source_hashes()
    config=dict(maps=specs,seeds=list(range(7,7+args.runs)),exploration=.15,rotation_step_radians=.01*np.pi,
                classical_rate=.05,probability_floor=.01,reward_horizon_scale=100,baseline_ema=.2,
                initial_baseline=.5,control_charged=True,source_hashes=hashes,
                representation='K-1 prefix gates for K positive levels; off remains protocol inactivity',
                modes=['greedy']+modes,lifetime_cap=None,max_rounds=2000,workers=args.workers)
    (out/'config.json').write_text(json.dumps(config,indent=2))
    tasks=[(s,'greedy',7,str(out)) for s in specs]
    tasks += [(s,m,seed,str(out)) for s in specs for seed in config['seeds'] for m in modes]
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_case,t) for t in tasks]
        for future in as_completed(futures):
            row=future.result();rows.append(row)
            print(json.dumps(row),flush=True)
            with (out/'summary.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    rows.sort(key=lambda r:(r['map'],r['policy'],r['seed']))
    with (out/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    aggregates={m:statistics.mean(r['lifetime'] for r in rows if r['policy']==m) for m in ['greedy']+modes}
    lines=['# 分散式排程第二版：量子啟發式局部探索','',
           '演算法參數固定，控制成本全部計入；貪婪基準每圖一次，其餘方法同一組 seeds 配對。未使用量子硬體，也不是原始 QEA 或 QEA3。', '',
           '| 地圖 | 貪婪 | 隨機探索、不學習 | 傳統機率更新 | 量子角度更新 |','| --- | ---: | ---: | ---: | ---: |']
    for s in specs:
        vals=[statistics.mean(r['lifetime'] for r in rows if r['map']==s['name'] and r['policy']==m) for m in ['greedy']+modes]
        lines.append('| '+s['name']+' | '+' | '.join(f'{v:.2f}' for v in vals)+' |')
    lines += ['| 平均 | '+' | '.join(f'{aggregates[m]:.2f}' for m in ['greedy']+modes)+' |','']
    for m in ['greedy','random','classical']:
        wins=ties=losses=0
        for q in [r for r in rows if r['policy']=='quantum']:
            ref=next(r for r in rows if r['map']==q['map'] and r['policy']==m and (m=='greedy' or r['seed']==q['seed']))
            delta=q['lifetime']-ref['lifetime'];wins+=delta>0;ties+=delta==0;losses+=delta<0
        lines.append(f'量子對 {m}：{wins} 勝 / {ties} 平 / {losses} 負。')
    lines += ['',f'共 {len(rows)} 次完整實驗，全部未截斷；保存逐段覆蓋、路由、能量與局部學習軌跡。已核對能量守恆、無環路徑與覆蓋。', '',
              '固定探索比例 15%，量子和傳統更新率未做等效步長校準；本比較只能評估目前具體設定，不能证明量子表示具有普遍優勢。局部協調沿用第一版理想通訊假設。']
    after=source_hashes()
    assert hashes=={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in sources}
    preserved=json.loads((HERE/'quantum_v2_preserved_sources.json').read_text())
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==h for path,h in preserved.items())
    (out/'external_source_audit.json').write_text(json.dumps(dict(unchanged=before==after,baseline_preserved=True,before=before,after=after),indent=2))
    assert before==after
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
