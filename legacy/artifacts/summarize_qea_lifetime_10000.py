"""Summarize full-lifetime runs; reject truncated or incomplete experiments."""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent / 'qea_full_lifetime_budget10000_20261004'
rows=[]
traces=[]
for path in sorted(root.glob('*/*/summary.csv')):
    with path.open() as f:
        rows.extend(csv.DictReader(f))
for path in sorted(root.glob('*/*/trace.csv')):
    with path.open() as f:
        traces.extend(csv.DictReader(f))
assert len(rows)==120, f'Expected 120 completed cases, got {len(rows)}'
assert not any(r['stop_reason'] in ('max_rounds','lifetime_cap') for r in rows), 'Truncated case: inspect before reporting full lifetime'
names=['qea','ga','qea_classical','qea_random']
summary={}
for name in names:
    group=[r for r in rows if r['algorithm']==name]
    assert len(group)==30
    values=np.array([int(r['lifetime']) for r in group])
    summary[name]={
        'runs':len(group),
        'mean_lifetime':float(values.mean()),
        'std_lifetime':float(values.std(ddof=1)),
        'min_lifetime':int(values.min()),
        'max_lifetime':int(values.max()),
        'mean_seconds':float(np.mean([float(r['elapsed_seconds']) for r in group])),
        'mean_reoptimizations':float(np.mean([int(r['reoptimizations']) for r in group])),
        'total_evaluations':sum(int(r['actual_evatime']) for r in group),
        'stop_reasons':dict(Counter(r['stop_reason'] for r in group)),
        'early_stop_count':sum(int(r['early_stop_count']) for r in group),
    }
pairs={}
for name in names[1:]:
    diffs=[]
    for row in [r for r in rows if r['algorithm']=='qea']:
        other=next(r for r in rows if r['algorithm']==name and r['map']==row['map'] and r['seed']==row['seed'])
        diffs.append(int(row['lifetime'])-int(other['lifetime']))
    d=np.asarray(diffs)
    pairs[name]={'wins':int(sum(d>0)),'ties':int(sum(d==0)), 'losses':int(sum(d<0)), 'mean_lifetime_delta':float(d.mean())}
report={'requested_budget_per_optimization':10000,'mode':'lifetime','moving':False,'summary':summary,'qea_paired_comparisons':pairs}
(root/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
for filename, records in [('all_runs.csv',rows),('all_segments.csv',traces)]:
    with (root/filename).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
text='''# 完整網路壽命比較：每輪最佳化預算 10,000

2026-10-04；maps/maps_100100100.json 的 MAP01–MAP06；每張種子 7–11。
四種方法共 120 次 lifetime 模擬，fitness v2、discrete sensing、固定感測器。
每段排程持續扣除能源至 LifeCheck 不通過，再重新最佳化；本批無 lifetime_cap，
且已檢查沒有案例因 max_rounds 提前截斷。壽命單位是模擬 time slots，不是秒。

| 方法 | 平均壽命 ↑ | 標準差 | 最小–最大 | 平均計算秒數 | 平均成功重排程次數 |
|---|---:|---:|---:|---:|---:|
'''
for name,v in summary.items():
    text+=f"| {name} | {v['mean_lifetime']:.2f} | {v['std_lifetime']:.2f} | {v['min_lifetime']}–{v['max_lifetime']} | {v['mean_seconds']:.2f} | {v['mean_reoptimizations']:.2f} |\n"
text+='\n| QEA 對照對象 | 勝 | 平 | 負 | 平均壽命差 |\n|---|---:|---:|---:|---:|\n'
for name,v in pairs.items():
    text+=f"| {name} | {v['wins']} | {v['ties']} | {v['losses']} | {v['mean_lifetime_delta']:+.2f} |\n"
text+='''
預算 10,000 指每次最佳化，不是整場模擬。既有 GA 按完整世代結束，實際每輪
10,012；QEA 三組每輪 10,000。網路存活越久，可能重新最佳化越多次，故整場
總計算預算不相等。四個案例平行執行，計時受到共享 CPU 影響。

infeasible_solution 表示該演算法在當時能源下未找到通過 LifeCheck 的排程；
不是證明全網能源全部耗盡或不存在其他可行解。壽命是既有模擬流程下的操作性
結果。三種 QEA 採均勻先驗，GA 保留覆蓋導向初始化，皆未額外調參。

`all_runs.csv` 保留 120 個案例；`all_segments.csv` 保留所有排程段；
`comparison.json` 提供彙整與停止原因；`algorithm_parameters.json` 記錄設定。
'''
(root/'README.md').write_text(text)
print(json.dumps(report,indent=2))
