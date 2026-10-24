"""E06 lifetime 對照：MAP01–06 × seed 7–36，預算 1000；成對 Wilcoxon 對 ring_se，另報拿到作弊版天花板的比例。"""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

R = Path("/home/g114056067/wsn_project/codes/2027WSN/experiment_results")
ROWS = [("ring_se（隨機）", "s736_ring_se_eval1000", "ring_se"),
        ("E03 DeepSets", "s736_e03_scorer_w64_eval1000", "e03_scorer"),
        ("E03 瓶頸規則", "s736_e03_bottleneck_eval1000", "e03_bottleneck"),
        ("監督式", "s736_e06_sup_eval1000", "e06_net"),
        ("監督式 0 層消融", "s736_e06_sup_l0_eval1000", "e06_net"),
        ("RL γ=0 從零", "s736_e06_g0_scratch_eval1000", "e06_net"),
        ("RL γ=0 暖啟動", "s736_e06_g0_warm_eval1000", "e06_net"),
        ("RL γ=0.5 暖啟動", "s736_e06_g05_warm_eval1000", "e06_net"),
        ("作弊版", "s736_oracle_eval1000", "ring_se_oracle")]
PAIRS = (("監督式", "E03 瓶頸規則"), ("監督式", "E03 DeepSets"), ("監督式", "監督式 0 層消融"),
         ("RL γ=0 暖啟動", "監督式"), ("RL γ=0 暖啟動", "RL γ=0 從零"), ("RL γ=0.5 暖啟動", "RL γ=0 暖啟動"),
         ("RL γ=0 暖啟動", "E03 瓶頸規則"))


def load(exp, algo):
    fs = sorted((R / exp / algo).glob("MAP*/summary.csv"))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f).assign(map=f.parent.name) for f in fs]).set_index(["map", "seed"])["lifetime"].astype(float)


def compare(x, y):
    c = x.index.intersection(y.index)
    d = x[c] - y[c]
    return f"{100 * d.mean() / y[c].mean():+.2f}%（{(d > 0).sum()}/{(d < 0).sum()}，p={wilcoxon(x[c], y[c]).pvalue:.4f}）"


data = {name: load(exp, algo) for name, exp, algo in ROWS}
base, ceiling = data["ring_se（隨機）"], data["作弊版"]
print("| 方法 | n | lifetime | 對 ring_se | 拿到天花板 | 最差 |")
print("| --- | --- | --- | --- | --- | --- |")
for name, v in data.items():
    if v is None:
        print(f"| {name} | — | 尚無結果 | | | |")
        continue
    vs = "—" if v is base else compare(v, base)
    share = f"{100 * (v.mean() - base.mean()) / (ceiling.mean() - base.mean()):.0f}%" if v is not base else "—"
    print(f"| {name} | {len(v)} | {v.mean():.1f} | {vs} | {share} | {v.min():.0f} |")
print()
for a, b in PAIRS:
    if data[a] is not None and data[b] is not None:
        print(f"{a} 對 {b}：{compare(data[a], data[b])}")
