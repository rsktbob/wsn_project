"""E05 lifetime 對照（含 E04）：MAP01–06 × seed 7–36，預算 1000；成對 Wilcoxon。"""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

R = Path("/home/g114056067/wsn_project/codes/2027WSN/experiment_results")
ROWS = [("8 段隨機", "s736_e04_random_eval1000", "e04_random"),
        ("E04 RL B", "s736_e04_attention_eval1000", "e04_attention"),
        ("E04 監督式回歸 B", "s736_e04_sup_attention_eval1000", "e04_attention"),
        ("E05 A 抽樣", "s736_e05_concat_sample_eval1000", "e05_concat_sample"),
        ("E05 A 取最大", "s736_e05_concat_argmax_eval1000", "e05_concat_argmax"),
        ("E05 B 抽樣", "s736_e05_attention_sample_eval1000", "e05_attention_sample"),
        ("E05 B 取最大", "s736_e05_attention_argmax_eval1000", "e05_attention_argmax"),
        ("8 段作弊版（seed 7–11）", "wholeseg8_6maps_eval1000", "ring_se_wholeseg8_oracle")]
PAIRS = [("E05 B 抽樣", "E04 RL B"), ("E05 A 抽樣", "E04 RL B"), ("E05 B 抽樣", "E05 A 抽樣"),
         ("E05 B 抽樣", "E05 B 取最大"), ("E05 A 抽樣", "E05 A 取最大"), ("E05 B 抽樣", "E04 監督式回歸 B")]


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
base = data["8 段隨機"]
print("| 方法 | MAP01 | MAP02 | MAP03 | MAP04 | MAP05 | MAP06 | 全部 | 對 8 段隨機 |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for name, v in data.items():
    m = v.groupby(level="map").mean().round(0).astype(int)
    rel = "—" if v is base else compare(v, base)
    print(f"| {name} | " + " | ".join(str(m[k]) for k in sorted(m.index)) + f" | {v.mean():.1f} | {rel} |")
print()
for a, b in PAIRS:
    print(f"{a} 對 {b}：{compare(data[a], data[b])}")
