"""Summarize saved single-run experiment CSVs without re-evaluating solutions."""
import csv
import json
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent / 'qea_maps_100100100_budget10000_20261004'
rows=[]
for path in sorted(root.glob('*/*/summary.csv')):
    rows.extend(csv.DictReader(path.open()))
assert len(rows)==120, f'Expected 120 completed runs, got {len(rows)}'
names=['qea','ga','qea_classical','qea_random']
summary={}
for name in names:
    group=[r for r in rows if r['algorithm']==name]
    assert len(group)==30
    summary[name]={
        'runs':len(group),
        'mean_fitness':float(np.mean([float(r['fitness']) for r in group])),
        'std_fitness':float(np.std([float(r['fitness']) for r in group],ddof=1)),
        'mean_seconds':float(np.mean([float(r['elapsed_seconds']) for r in group])),
        'actual_evaluations':sorted({int(r['actual_evatime']) for r in group}),
        'feasible_runs':sum(all(int(r[k])==0 for k in ('uncovered_count','disconnected_count','energy_failed_count')) for r in group),
    }
pairs={}
for name in names[1:]:
    diffs=[]
    for row in [r for r in rows if r['algorithm']=='qea']:
        other=next(r for r in rows if r['algorithm']==name and r['map']==row['map'] and r['seed']==row['seed'])
        diffs.append(float(row['fitness'])-float(other['fitness']))
    d=np.asarray(diffs)
    pairs[name]={'wins':int(sum(d>1e-12)),'ties':int(sum(abs(d)<=1e-12)), 'losses':int(sum(d<-1e-12)), 'mean_delta':float(d.mean())}
report={'requested_budget':10000,'mode':'single','summary':summary,'qea_paired_comparisons':pairs}
(root/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
with (root/'all_runs.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
text='''# QEA 完整地圖清單比較：預算 10,000

2026-10-04；maps/maps_100100100.json 的 MAP01–MAP06；每張種子 7–11。
四種方法共 120 次 single 最佳化，fitness v2、discrete sensing，並非 lifetime 實驗。

| 方法 | 平均最佳 fitness ↑ | 標準差 | 平均秒數 | 實際評估次數 | 可行解 |
|---|---:|---:|---:|---:|---:|
'''
for name,v in summary.items():
    text+=f"| {name} | {v['mean_fitness']:.9f} | {v['std_fitness']:.9f} | {v['mean_seconds']:.3f} | {v['actual_evaluations']} | {v['feasible_runs']}/30 |\n"
text+='\n| QEA 對照對象 | 勝 | 平 | 負 | 平均 fitness 差 |\n|---|---:|---:|---:|---:|\n'
for name,v in pairs.items():
    text+=f"| {name} | {v['wins']} | {v['ties']} | {v['losses']} | {v['mean_delta']:+.9f} |\n"
text+='''
所有方法要求預算均為 10,000。QEA 三種方法嚴格使用 10,000；既有 GA 按完整世代
結束，因此實際為 10,012（多 0.12%），這是本次比較限制，未修改其演算法。
GA 使用原有覆蓋導向初始化，三種 QEA 使用相同均勻先驗。使用尚未調參的設定，
不以此宣稱統計顯著性、網路壽命改善或量子硬體優勢。單次執行時間包含搜尋，
不包含問題建立；未額外暖機或輪替演算法順序，時間應視為概略。

`all_runs.csv` 保留 120 筆原始結果；`comparison.json` 提供彙整；
`algorithm_parameters.json` 記錄實驗參數；各方法/地圖目錄保留原始 summary.csv。
'''
(root/'README.md').write_text(text)
print(json.dumps(report,indent=2))
