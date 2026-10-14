"""Paired fixed-state comparison with SA/SI actual batch-rounded FE budgets."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from statistics import mean

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE / '.numba_cache'))
sys.path.insert(0, str(HERE.parents[2]))
from Algorithm.experiment.tn_geo.optimizer import TNGEO
from experiments.maps import load_map_specs
from experiments.problem_setup import build_problem
from experiments.builder import build_algorithm, algorithm_params


def run_case(name, spec, seed, budget):
    args = SimpleNamespace(_current_map=spec, sensing_mode='discrete', fitness_service='v2')
    problem = build_problem(args, seed)
    if name == 'mps':
        algorithm = TNGEO(seed=seed, order='coverage')
        params = algorithm.config.copy()
    else:
        algorithm = build_algorithm(name, problem, args, seed)
        params = algorithm_params(name, args)
    # Warm decode/fitness kernels outside timing, without drawing algorithm RNG.
    from State.SensorEncoding import SensorEncoding
    import numpy as np
    warm = SensorEncoding.random(problem.SENSOR_NUMBER, problem.radius_option_counts,
                                 rng=np.random.default_rng(0)).decode(problem)
    problem.evaluate_state(warm)
    started = time.perf_counter()
    result = algorithm.run(problem, budget=budget)
    elapsed = time.perf_counter()-started
    state = result.best_state
    cost = problem.calculate_total_cost(state)
    return dict(algorithm=name, map=spec, seed=seed, requested_budget=budget,
                evaluations=result.evaluations, seconds=elapsed, best_score=result.best_score,
                objectives=result.best_fitness.tolist(), feasible=bool(problem.LifeCheck(state, cost)),
                weighted_depletion=1-result.best_score, parameters=params,
                levels=state.levels.tolist(), next_hops=state.next_hops.tolist(),
                history=result.history.tolist(), metadata=result.metadata,
                sensing_mode='discrete', fitness_service='v2')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', default='MAP01')
    parser.add_argument('--budget', type=int, default=1000)
    parser.add_argument('--seeds', nargs='+', type=int, default=[7,17,27])
    parser.add_argument('--output', default='sets_comparison_20261009')
    args=parser.parse_args()
    if args.budget < 64:
        parser.error('budget must be at least 64 for this comparison')
    out=(HERE/args.output).resolve()
    if HERE not in out.parents:
        parser.error('output must stay within tn_geo')
    out.mkdir(parents=True, exist_ok=True)
    specs=[s for s in load_map_specs('maps/maps_100100100.json') if s['name']==args.map]
    if len(specs)!=1:
        parser.error('unknown map')
    spec=specs[0]
    rows=[]
    for seed in args.seeds:
        for baseline in ('sa_sets','si_sets'):
            file=out/f'{args.map}_{baseline}_seed{seed}.json'
            if file.exists():
                base=json.loads(file.read_text())
                if base['requested_budget'] != args.budget or base['map'] != spec:
                    raise ValueError('cached baseline config mismatch')
            else:
                print(f'START {baseline} seed={seed}', flush=True)
                base=run_case(baseline,spec,seed,args.budget)
                file.write_text(json.dumps(base,indent=2,allow_nan=False))
            print(json.dumps({k:base[k] for k in ('algorithm','seed','evaluations','best_score','seconds')}),flush=True)
            budget=base['evaluations']
            file=out/f'{args.map}_mps_fe{budget}_seed{seed}.json'
            if file.exists():
                mps=json.loads(file.read_text())
                if mps['map'] != spec or mps['requested_budget'] != budget:
                    raise ValueError('cached MPS config mismatch')
            else:
                print(f'START MPS matched to {baseline}: seed={seed} FE={budget}',flush=True)
                mps=run_case('mps',spec,seed,budget)
                file.write_text(json.dumps(mps,indent=2,allow_nan=False))
            assert mps['evaluations']==budget
            print(json.dumps({k:mps[k] for k in ('algorithm','seed','evaluations','best_score','seconds')}),flush=True)
            for result in (base,mps):
                rows.append(dict(comparison=baseline, algorithm=result['algorithm'], seed=seed,
                    evaluations=result['evaluations'], best_score=result['best_score'],
                    weighted_depletion=result['weighted_depletion'], feasible=result['feasible'],
                    seconds=result['seconds']))
    with (out/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregate=[]
    for comparison in ('sa_sets','si_sets'):
        for name in (comparison,'mps'):
            group=[r for r in rows if r['comparison']==comparison and r['algorithm']==name]
            aggregate.append(dict(comparison=comparison,algorithm=name,runs=len(group),
                evaluations=[r['evaluations'] for r in group],
                mean_best_score=mean(r['best_score'] for r in group),
                mean_weighted_depletion=mean(r['weighted_depletion'] for r in group),
                mean_seconds=mean(r['seconds'] for r in group),feasible_runs=sum(r['feasible'] for r in group)))
    (out/'aggregate.json').write_text(json.dumps(aggregate,indent=2))
    lines=['# TN-GEO 與 SA-SETS、SI-SETS：固定電量配對比較','',
           f"MAP01；種子 {args.seeds}；SETS 名義預算 {args.budget}，MPS 配對其實際 FE。",
           '相同地圖、初始電量、discrete 感測、SensorEncoding 預設解碼及 v2 fitness。',
           '保留各方法原有初始化與參數，並非相同初始種群。SETS 使用現有多程序，MPS 為 NumPy 單程序；耗時是現有實作牆鐘時間，不是等 CPU 資源比较。',
           '计时含 run 的初始化、訓練和程序啟停，排除 Problem/演算法建構、主程序 kernel 暖機及報告寫檔。','',
           '| 配對 | 方法 | 實際 FE | 平均最佳 fitness ↑ | 平均 1−fitness ↓ | 平均秒數 | 可行 |',
           '|---|---|---|---:|---:|---:|---|']
    for a in aggregate:
        lines.append(f"| {a['comparison']} | {a['algorithm']} | {a['evaluations']} | {a['mean_best_score']:.9f} | {a['mean_weighted_depletion']:.9f} | {a['mean_seconds']:.3f} | {a['feasible_runs']}/{a['runs']} |")
    lines += ['', '1−fitness 是可行解的加權能量耗損代理，不是 lifetime。單圖三種子只作初步診斷，不宣稱統計顯著或跨地圖泛化。',
              'SA 預設 n=8,h=4,w=2,mu=0.4；SI 預設 n=8,h=4,w=8,mu=0.4。MPS rank=4，初始64、batch32、訓練64樣本/3步、覆蓋排序。',
              '完整參數、解碼結果和 history 已存各次 JSON。比較重跑使用全新 output，以避免混用不同版本的快取結果。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(aggregate,indent=2),flush=True)


if __name__=='__main__':
    main()
