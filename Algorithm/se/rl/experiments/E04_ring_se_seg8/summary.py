"""E04 lifetime 對照：MAP01–06 × seed 7–36，預算 1000；成對 Wilcoxon。"""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

R = Path("/home/g114056067/wsn_project/codes/2027WSN/experiment_results")
ROWS = [("8 段隨機", "s736_e04_random_eval1000", "e04_random"),
        ("8 段瓶頸規則", "s736_e04_bottleneck_eval1000", "e04_bottleneck"),
        ("設計 A 串接 Dueling", "s736_e04_concat_eval1000", "e04_concat"),
        ("設計 B Attention", "s736_e04_attention_eval1000", "e04_attention"),
        ("監督式 A", "s736_e04_sup_concat_eval1000", "e04_concat"),
        ("監督式 B", "s736_e04_sup_attention_eval1000", "e04_attention"),
        ("ring_se（4 段隨機範圍）", "s736_ring_se_eval1000", "ring_se"),
        ("8 段作弊版（seed 7–11）", "wholeseg8_6maps_eval1000", "ring_se_wholeseg8_oracle")]


def load(exp, algo):
    fs = sorted((R / exp / algo).glob("MAP*/summary.csv"))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f).assign(map=f.parent.name) for f in fs]).set_index(["map", "seed"])["lifetime"].astype(float)


data = {name: load(exp, algo) for name, exp, algo in ROWS}
base = data["8 段隨機"]
for name, v in data.items():
    if v is None:
        print(f"{name}：尚無結果")
        continue
    line = f"{name:22s} n={len(v):3d} 平均 {v.mean():7.1f} 最差 {v.min():.0f}"
    if v is not base:
        c = base.index.intersection(v.index)
        d = v[c] - base[c]
        line += f"  對 8 段隨機 {100 * d.mean() / base[c].mean():+.2f}%（{(d > 0).sum()}/{(d < 0).sum()}，p={wilcoxon(v[c], base[c]).pvalue:.4f}）"
    print(line)
    print("   各圖", v.groupby(level="map").mean().round(0).astype(int).to_dict())
for a, b in (("設計 B Attention", "設計 A 串接 Dueling"), ("設計 A 串接 Dueling", "8 段瓶頸規則"), ("設計 B Attention", "8 段瓶頸規則"),
             ("監督式 B", "設計 B Attention"), ("監督式 A", "設計 A 串接 Dueling"), ("監督式 B", "監督式 A")):
    x, y = data[a], data[b]
    c = x.index.intersection(y.index)
    d = x[c] - y[c]
    print(f"{a} 對 {b}：{100 * d.mean() / y[c].mean():+.2f}%（{(d > 0).sum()}/{(d < 0).sum()}，p={wilcoxon(x[c], y[c]).pvalue:.4f}）")
