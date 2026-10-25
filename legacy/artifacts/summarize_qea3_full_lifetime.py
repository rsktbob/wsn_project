"""Compare completed SETS lifetime experiments to saved QEA runs."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

base=Path(__file__).resolve().parent
root=base/'qea3_full_lifetime_budget10000_20261005'
sroot=base/'sets_vs_qea_full_lifetime_budget10000_20261004'
qroot=base/'qea_full_lifetime_budget10000_20261004'
names=['qea3','qea','sa_sets','si_sets','ring_sets','ring_setspriority']
rows=[]
for name in names:
    source=root if name=='qea3' else (qroot if name=='qea' else sroot)
    group=[]
    for path in sorted((source/name).glob('*/summary.csv')):
        with path.open() as f:
            group.extend(csv.DictReader(f))
    assert len(group)==30, (name,len(group))
    assert len({(r['map'],r['seed']) for r in group})==30
    assert all(r['stop_reason'] not in ('max_rounds','lifetime_cap') for r in group), 'Truncated lifetime run'
    rows.extend(group)
keys=['mode','seed','map','boundary','sensors','targets','full_energy','file',
      'moving','sensing_mode','fitness_service','routing_service','evaluate_budget',
      'max_rounds','segment_slot_limit','lifetime_cap']
qrows={(r['map'],r['seed']):r for r in rows if r['algorithm']=='qea3'}
for r in rows:
    q=qrows[(r['map'],r['seed'])]
    for key in keys:
        assert r[key]==q[key], (r['algorithm'],key,r[key],q[key])
summary={}
pairs={}
per_map={}
for name in names:
    group=[r for r in rows if r['algorithm']==name]
    values=np.array([int(r['lifetime']) for r in group])
    actual=[]
    source=root if name=='qea3' else (qroot if name=='qea' else sroot)
    for path in (source/name).glob('*/trace.csv'):
        with path.open() as f:
            for trace in csv.DictReader(f):
                # Trace field names are validated against the saved file.
                actual.append(int(trace['actual_evatime']))
    summary[name]={'mean_lifetime':float(values.mean()),'std_lifetime':float(values.std(ddof=1)),
       'min_lifetime':int(values.min()),'max_lifetime':int(values.max()),'runs':len(group),
       'mean_seconds':float(np.mean([float(r['elapsed_seconds']) for r in group])),
       'actual_evaluations_per_segment':sorted(set(actual)),
       'total_evaluations':sum(int(r['actual_evatime']) for r in group),
       'stop_reasons':dict(Counter(r['stop_reason'] for r in group)),
       'early_stop_count':sum(int(r['early_stop_count']) for r in group)}
    per_map[name]={m:float(np.mean([int(r['lifetime']) for r in group if r['map']==m])) for m in sorted({r['map'] for r in group})}
    if name!='qea3':
        delta=np.array([int(qrows[(r['map'],r['seed'])]['lifetime'])-int(r['lifetime']) for r in group])
        pairs[name]={'qea3_wins':int(sum(delta>0)),'ties':int(sum(delta==0)),'qea3_losses':int(sum(delta<0)),
                     'mean_delta_qea3_minus_baseline':float(delta.mean()),
                     'qea3_gain_percent':float(100*(summary['qea3']['mean_lifetime']/values.mean()-1))}
report={'requested_evaluations_per_optimization':10000,'summary':summary,'qea3_paired_comparisons':pairs,'per_map_mean_lifetime':per_map}
(root/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (root/'all_runs.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
text='''# QEA3、QEA 與 SETS 家族：完整網路壽命比較

MAP01–MAP06，各五個種子 7–11；每輪要求 10,000 次評估，固定感測器，fitness v2。
補跑 QEA3 30 次完整壽命；QEA 與四種 SETS 沿用先前完成的同條件 150 次結果。
已逐筆核對地圖、種子、問題規模、服務版本、預算與 lifetime 設定一致。

| 方法 | 平均壽命（時槽）↑ | 標準差 | 最小–最大 | 平均計算秒數 | 每段實際評估次數 |
|---|---:|---:|---:|---:|---|
'''
for n,v in summary.items():
    text+=f"| {n} | {v['mean_lifetime']:.2f} | {v['std_lifetime']:.2f} | {v['min_lifetime']}–{v['max_lifetime']} | {v['mean_seconds']:.2f} | {v['actual_evaluations_per_segment']} |\n"
text+='\n| 對照方法 | QEA3 勝／平／負 | QEA3 平均壽命相對變化 |\n|---|---:|---:|\n'
for n,v in pairs.items():
    text+=f"| {n} | {v['qea3_wins']}／{v['ties']}／{v['qea3_losses']} | {v['qea3_gain_percent']:+.2f}% |\n"
text+='''
沒有案例因 lifetime_cap 或 max_rounds 截斷。infeasible_solution 表示該演算法
在目前能源下找不到可行排程，不是全網能源耗盡的證明。
預算是每輪最佳化，不是整場總量；不同存活時間與重排程次數會有不同總評估量。
SETS 的完整批次可能使實際評估次數超過 10,000，以上表記錄為準。
Ring-SETSpriority 使用自己的 priority 編碼；這是既有方法的整體比較，並非只替換單一運算子。
SETS 案例內還有市場工作程序，與 QEA/QEA3 的計算資源使用不同；耗時不能直接解讀為單核心速度。
所有設定未額外調參，不以此宣稱量子硬體優勢。

原始逐段與每場結果在各方法/地圖的 trace.csv、summary.csv；
all_runs.csv 合併 180 筆結果；comparison.json 包含每張地圖的平均壽命與成對勝負。
'''
(root/'README.md').write_text(text)
print(json.dumps(report,indent=2))
