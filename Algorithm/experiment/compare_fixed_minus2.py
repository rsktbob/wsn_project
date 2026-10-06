"""Report matched lifetime changes for the temporary v2 penalty experiment."""
import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'results/sets_v2_fixed_minus2_20261005'
DATA=HERE/'results/sets_v2_fixed_minus2_20261005_02'
OLD=ROOT.parents[1]/'artifacts/sets_vs_qea_full_lifetime_budget10000_20261004/all_runs_with_qea.csv'
NAMES=['sa_sets','si_sets','ring_sets','ring_setspriority']


def main():
    baseline={(r['algorithm'],r['map'],r['seed']):r for r in csv.DictReader(OLD.open()) if r['algorithm'] in NAMES}
    rows=[];report={};maps={};paired=[]
    compare_keys=['mode','seed','map','boundary','sensors','targets','full_energy','file','moving','sensing_mode','fitness_service','routing_service','evaluate_budget','max_rounds','segment_slot_limit','lifetime_cap']
    for name in NAMES:
        group=[];trace=[]
        for path in sorted((DATA/name).glob('*/summary.csv')):
            group.extend(csv.DictReader(path.open()))
        for path in sorted((DATA/name).glob('*/trace.csv')):
            trace.extend(csv.DictReader(path.open()))
        assert len(group)==30,(name,len(group))
        assert len({(r['map'],r['seed']) for r in group})==30
        assert all(r['stop_reason']=='infeasible_solution' for r in group), 'truncated or unexpected stop'
        old=[];new=[];deltas=[]
        for r in group:
            b=baseline[(name,r['map'],r['seed'])]
            for k in compare_keys:
                assert r[k]==b[k],(name,k,r[k],b[k])
            old.append(int(b['lifetime']));new.append(int(r['lifetime']));delta=int(r['lifetime'])-int(b['lifetime']);deltas.append(delta)
            paired.append(dict(algorithm=name,map=r['map'],seed=r['seed'],old_lifetime=b['lifetime'],new_lifetime=r['lifetime'],delta=delta))
        bycase={}
        for t in trace:
            key=(t['map'],t['seed']);bycase[key]=bycase.get(key,0)+int(t['segment_length'])
        for r in group:
            assert bycase[(r['map'],r['seed'])]==int(r['lifetime'])
        report[name]=dict(runs=len(group),old_mean=statistics.mean(old),new_mean=statistics.mean(new),
                         delta_mean=statistics.mean(deltas),percent_change=(statistics.mean(new)/statistics.mean(old)-1)*100,
                         wins=sum(d>0 for d in deltas),ties=sum(d==0 for d in deltas),losses=sum(d<0 for d in deltas),
                         new_std=statistics.stdev(new),actual_evaluations_per_segment=sorted({int(t['actual_evatime']) for t in trace}),
                         stop_reasons=dict(Counter(r['stop_reason'] for r in group)))
        maps[name]={m:dict(old_mean=statistics.mean(int(baseline[(name,m,r['seed'])]['lifetime']) for r in group if r['map']==m),new_mean=statistics.mean(int(r['lifetime']) for r in group if r['map']==m)) for m in sorted({r['map'] for r in group})}
        rows.extend(group)
    for path,data in [(OUT/'all_runs.csv',rows),(OUT/'paired_changes.csv',paired)]:
        with path.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    before=json.loads((OUT/'source_manifest_before.json').read_text())
    changed=[name for name,h in before.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    (OUT/'source_audit.json').write_text(json.dumps(dict(unchanged_during_experiment=True,files_checked=len(before)),indent=2))
    (OUT/'comparison.json').write_text(json.dumps(dict(summary=report,per_map=maps,baseline_source=str(OLD),result_source=str(DATA),source_unchanged=True),indent=2))
    text=['# Fitness v2 暫改固定 -2：SETS lifetime 比較','',
          '只將斷線或能量不足分支固定為 -2；僅覆蓋不足的分支不變。六圖、每圖 seeds 7–11、每次預算 10000，固定節點、discrete、SensorEncoding v3（priority 方法用其既有編碼）。比較基準為之前已完成的相同設定結果。','',
          '| 方法 | 修改前平均 | 固定 -2 平均 | 差值 | 變化 | 配對勝/平/負 |','| --- | ---: | ---: | ---: | ---: | --- |']
    for n,r in report.items():
        text.append(f"| {n} | {r['old_mean']:.2f} | {r['new_mean']:.2f} | {r['delta_mean']:+.2f} | {r['percent_change']:+.2f}% | {r['wins']}/{r['ties']}/{r['losses']} |")
    text+=['','全部 120 次到 infeasible_solution，未觸及 lifetime/max_rounds 上限。逐段 lifetime 加總及地圖/seed/能量/預算/服務設定已核對。實驗期間外部原始碼未變動。','',
           '實際評估次數（沿用各方法整批評估行為）：']
    for n,r in report.items():text.append(f"- {n}: {r['actual_evaluations_per_segment']}")
    text+=['',f'原始逐場與逐段 CSV 在：{DATA}', '', '本次未自動還原 fitness，等待使用者要求還原。原始與暫改版本快照均在本目錄；原備份路徑另記於 penalty_change.json。', '',
           '修改前後使用相同 seeds，但 fitness 改變會使搜尋軌跡不同；勝負是配對實驗結果，不代表單一 seed 的決策逐步一致。']
    (OUT/'README.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text))


if __name__=='__main__':main()
