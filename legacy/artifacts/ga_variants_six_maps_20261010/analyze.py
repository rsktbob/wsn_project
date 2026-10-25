from pathlib import Path
import csv
import json
import hashlib
import statistics as st
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
NAMES = ['ga', 'ga_elitist', 'ga_adaptive']
LABELS = ['GA', 'Elitist GA', 'Adaptive GA']
rows = []
for name in NAMES:
    for path in sorted((ROOT / 'results' / name).glob('MAP*/summary.csv')):
        rows.extend(csv.DictReader(path.open()))
assert len(rows) == 54, len(rows)
keys = {(r['algorithm'], r['map'], r['seed']) for r in rows}
assert len(keys) == 54
assert {r['seed'] for r in rows} == {'7', '8', '9'}
assert all(int(r['evaluate_budget']) == 1000 for r in rows)
assert all(int(r['actual_evatime']) == int(r['requested_evaluate_budget_total']) for r in rows)
assert not any(r['stop_reason'] in ('lifetime_cap', 'max_rounds') for r in rows)
traces = []
for name in NAMES:
    for path in (ROOT / 'results' / name).glob('MAP*/trace.csv'):
        traces.extend(csv.DictReader(path.open()))
assert traces
assert all(int(r['actual_evatime']) == 1000 for r in traces)
with (ROOT / 'all_runs.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
maps=sorted({r['map'] for r in rows})
summary=[]
for m in maps:
    baseline=st.mean(float(r['lifetime']) for r in rows if r['map']==m and r['algorithm']=='ga')
    for name in NAMES:
        group=[r for r in rows if r['map']==m and r['algorithm']==name]
        life=[float(r['lifetime']) for r in group]
        summary.append(dict(map=m,algorithm=name,lifetime_mean=st.mean(life),lifetime_sd=st.stdev(life),
                            change_percent=100*(st.mean(life)/baseline-1),
                            elapsed_mean=st.mean(float(r['elapsed_seconds']) for r in group),
                            evaluations_mean=st.mean(float(r['actual_evatime']) for r in group)))
with (ROOT/'aggregate.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
overall={}
for name in NAMES:
    group=[r for r in rows if r['algorithm']==name]
    delta=[]
    for r in group:
        base=next(b for b in rows if b['algorithm']=='ga' and b['map']==r['map'] and b['seed']==r['seed'])
        delta.append(float(r['lifetime'])-float(base['lifetime']))
    overall[name]=dict(lifetime_mean=st.mean(float(r['lifetime']) for r in group),
                       elapsed_mean=st.mean(float(r['elapsed_seconds']) for r in group),
                       evaluations_mean=st.mean(float(r['actual_evatime']) for r in group),
                       wins=sum(d>0 for d in delta),ties=sum(d==0 for d in delta),losses=sum(d<0 for d in delta),
                       stop_reasons=dict(Counter(r['stop_reason'] for r in group)))
(ROOT/'overall.json').write_text(json.dumps(overall,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(10,5))
x=np.arange(len(maps)); width=.25
for i,(name,label) in enumerate(zip(NAMES,LABELS)):
    group=[r for r in summary if r['algorithm']==name]
    ax.bar(x+(i-1)*width,[r['lifetime_mean'] for r in group],width,
           yerr=[r['lifetime_sd'] for r in group],capsize=3,label=label)
ax.set_xticks(x,maps); ax.set_ylabel('Service lifetime (slots)'); ax.set_title('Six maps: mean ± sample SD, seeds 7–9\n1,000 evaluations per optimization; full lifetime simulation')
ax.legend(); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig(ROOT/'lifetime_comparison.png',dpi=180); plt.close(fig)
lines=['# 六圖 GA 變體比較','',
'設定：MAP01–MAP06；每圖 100 sensors / 100 targets；seed 7、8、9；每次最佳化 1,000 FE；完整 lifetime、無移動；fitness v2、routing v1、discrete sensing、ring ordering。族群 30、交配率 0.8。六個程序平行執行，時間僅供同批參考。','',
'表中為壽命 slots 的平均 ± 樣本標準差（每格 n=3）。','',
'| 地圖 | GA | 菁英 GA | 自適應 GA | 菁英對 GA | 自適應對 GA |',
'|---|---:|---:|---:|---:|---:|']
for m in maps:
    g=[next(r for r in summary if r['map']==m and r['algorithm']==n) for n in NAMES]
    cells=[f"{r['lifetime_mean']:.1f} ± {r['lifetime_sd']:.1f}" for r in g]
    lines.append('| '+m+' | '+' | '.join(cells)+f" | {g[1]['change_percent']:+.1f}% | {g[2]['change_percent']:+.1f}% |")
lines += ['', '## 整體（每種 18 次）','', '| 方法 | 平均壽命 | 相對 GA | 平均秒數 | 平均總 FE | 配對勝／和／負 |','|---|---:|---:|---:|---:|---|']
for name,label in zip(NAMES,LABELS):
    r=overall[name]
    lines.append(f"| {label} | {r['lifetime_mean']:.1f} | {100*(r['lifetime_mean']/overall['ga']['lifetime_mean']-1):+.1f}% | {r['elapsed_mean']:.2f} | {r['evaluations_mean']:.0f} | {r['wins']}/{r['ties']}/{r['losses']} |")
lines += ['', '配對是同地圖、同 seed；整體百分比是平均壽命的比值，非各次百分比的平均。','',
'限制：這是 3 seeds、1,000 FE 的初步比較，不能推論其他預算下的排名或統計顯著性。每次最佳化預算一致，但各方法重新最佳化次數不同，因此整段 lifetime 總 FE 不固定。停止於搜尋未找到可繼續服務的解（infeasible_solution）不等同於證明剩餘網路不存在可行解。所有執行均未碰到 lifetime cap 或 max_rounds。','',
'檔案：`all_runs.csv` 為 54 筆原始 summary 合併；`aggregate.csv` 為各圖統計；`overall.json` 為整體統計；`results/` 含原 runner summary、trace、參數；`run.log` 為執行日誌。']
(ROOT/'README.md').write_text('\n'.join(lines)+'\n')
print('\n'.join(lines))
