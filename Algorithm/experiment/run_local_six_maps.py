"""Freeze and validate the local-policy prototype on all six maps."""
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['NUMBA_CACHE_DIR'] = str(HERE / '.numba_cache')
sys.path.insert(0, str(ROOT))
from Algorithm.experiment.run_local_schedule import build_problem, simulate
from Algorithm.experiment.run_reserved_lifetime import source_hashes
from experiments.maps import load_map_specs
from State.State import State
import numpy as np


def verify(p, trace, summary):
    assert sum(t['slots'] for t in trace) == summary['lifetime']
    for index, t in enumerate(trace):
        if t['slots'] == 0:
            assert index == len(trace)-1
            continue
        assert not(t['uncovered'] or t['disconnected'] or t['energy_failed'])
        ids = [v for claim in t['claims'] for v in claim['targets']]
        assert len(ids) == len(set(ids))
        state = State.empty(p)
        state.levels[:] = t['levels']
        state.next_hops[:] = t['next_hops']
        for source in np.flatnonzero(state.levels):
            node, seen = int(source), set()
            while node != p.BSID:
                assert 0 <= node < p.SENSOR_NUMBER and node not in seen
                assert state.levels[node] > 0
                seen.add(node)
                parent = int(state.next_hops[node])
                assert 0 <= parent < p.DEVICE_NUMBER
                assert p.distances[node, parent] < p.blocked_distance / 100
                state.tx_load[node] += p.generated_load[source]
                node = parent
        p.resolve_state_radius(state)
        assert not len(p.find_uncovered_targets(state))
        cost = p.calculate_total_cost(state)
        np.testing.assert_allclose(cost.sum(), t['routing_cost']+t['scheduling_cost'], atol=1e-10)
        remaining = np.asarray(t['energy_before']) - np.asarray(t['control_energy']) - cost*t['slots']
        assert np.all(remaining >= -1e-9)
        if index+1 < len(trace):
            np.testing.assert_allclose(remaining, trace[index+1]['energy_before'], atol=1e-9)


def main():
    out = HERE / 'results/local_schedule_six_maps_20261005'
    out.mkdir(parents=True, exist_ok=True)
    sources = ('local_scheduling.py', 'reserved_routing.py', 'run_local_schedule.py', 'run_local_six_maps.py')
    hashes = {name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in sources}
    for name in sources:
        (out/('snapshot_'+name)).write_bytes((HERE/name).read_bytes())
    outside_before = source_hashes()
    specs = load_map_specs('maps/maps_100100100.json')
    assert len(specs) == 6
    config = dict(maps=specs, seed=7, lifetime_cap=None, max_rounds=2000,
                  control_modes=[False,True], training=False, source_hashes=hashes,
                  design='frozen local_schedule_v3; ideal coordination assumptions unchanged',
                  timing='wall time of simulate(), excludes map loading, validation, file I/O')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    rows=[]
    for spec in specs:
        for paid in (False,True):
            p=build_problem(SimpleNamespace(_current_map=spec,sensing_mode='discrete',fitness_service='v2'),7)
            start=time.perf_counter()
            summary,trace=simulate(p,paid,cap=None,max_rounds=2000)
            elapsed=time.perf_counter()-start
            assert not summary['truncated'], 'incomplete run; do not report as finished'
            verify(p,trace,summary)
            last=trace[-1]
            summary.update(map=spec['name'],seconds=elapsed,
                           final_uncovered=len(last['uncovered']),final_disconnected=len(last['disconnected']),
                           final_energy_failed=len(last['energy_failed']),
                           paid_control_energy=sum(sum(t['control_energy']) for t in trace if t['control_paid']))
            rows.append(summary)
            (out/f"{spec['name']}_{'paid' if paid else 'free'}.json").write_text(json.dumps(dict(summary=summary,trace=trace),indent=2))
            print(json.dumps(summary),flush=True)
            with (out/'summary.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    baseline_file=ROOT.parents[1]/'artifacts/sets_vs_qea_full_lifetime_budget10000_20261004/all_runs_with_qea.csv'
    baseline={}
    with baseline_file.open(newline='') as f:
        for r in csv.DictReader(f):
            if r['algorithm']=='sa_sets' and int(r['seed'])==7:
                assert r['fitness_service']=='v2' and r['sensing_mode']=='discrete'
                assert int(r['evaluate_budget'])==10000 and r['moving']=='False'
                assert r['stop_reason']=='infeasible_solution'
                baseline[r['map']]=int(r['lifetime'])
    assert len(baseline)==6
    (out/'baseline_reference.json').write_text(json.dumps(dict(source=str(baseline_file),sha256=hashlib.sha256(baseline_file.read_bytes()).hexdigest(),seed=7,lifetimes=baseline),indent=2))
    lines=['# 固定版本：六圖完整 lifetime 驗證','',
           '所有 12 組均執行至 infeasible_solution，無 lifetime 上限、未碰到 max_rounds。政策為確定性規則，每圖每條件跑一次，seed 7。MAP01 曾用於開發，MAP02–06 本輪未用於調整政策。','',
           '| 地圖 | 不扣控制成本 | 計入控制成本 | SA-SETS seed 7（既有結果） | 計入控制 / SA-SETS |',
           '| --- | ---: | ---: | ---: | ---: |']
    for s in specs:
        name=s['name']; free=next(r['lifetime'] for r in rows if r['map']==name and not r['control_charged']);paid=next(r['lifetime'] for r in rows if r['map']==name and r['control_charged'])
        lines.append(f'| {name} | {free} | {paid} | {baseline[name]} | {paid/baseline[name]:.2%} |')
    mf=statistics.mean(r['lifetime'] for r in rows if not r['control_charged']);mp=statistics.mean(r['lifetime'] for r in rows if r['control_charged']);mb=statistics.mean(baseline.values())
    lines.append(f'| 平均 | {mf:.2f} | {mp:.2f} | {mb:.2f} | {mp/mb:.2%} |')
    holdout=[s['name'] for s in specs if s['name']!='MAP01']
    hp=statistics.mean(r['lifetime'] for r in rows if r['control_charged'] and r['map'] in holdout)
    hb=statistics.mean(baseline[n] for n in holdout)
    lines+=['',f'MAP02–06 計入控制平均 {hp:.2f}；對應 SA-SETS 平均 {hb:.2f}，平均值比例 {hp/hb:.2%}。', '',
            'SA-SETS 使用既有相同 seed、每次預算 10000 的結果，本輪未重跑，也不是六圖五 seed 平均。新方法仍採已知鄰居、醒睡名單及可靠理想控制通訊等假設，不能當作真實部署的公平總成本比較。', '',
            '已獨立重建成功段的路徑/流量，核對覆蓋、合法鏈路、無環可達、逐段能量守恆與 lifetime 加總；固定政策原始碼快照與 SHA256 保存於此目錄。外部來源前後比對見 external_source_audit.json。', '',
            '停止原因表示本策略未找到下一個可服務配置，不代表剩餘能量完全耗盡或不存在其他可行解。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    assert hashes=={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in sources}
    after=source_hashes()
    (out/'external_source_audit.json').write_text(json.dumps(dict(unchanged=outside_before==after,before=outside_before,after=after),indent=2))
    assert outside_before==after
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    main()
