"""E01 天花板：作弊版（偷看 4 段交換結果選段，再照常突變）對 ring_se（每組隨機選段，正式版）與舊版 ring_se，MAP01–06 × seed 7–11。"""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

R = Path("/home/g114056067/wsn_project/codes/2027WSN/experiment_results")


def load(exp, algo):
    frames = [pd.read_csv(f).assign(map=f.parent.name) for f in sorted((R / exp / algo).glob("MAP*/summary.csv"))]
    return pd.concat(frames).set_index(["map", "seed"])["lifetime"].astype(float) if frames else None


def compare(a, b, name_a, name_b):
    common = a.index.intersection(b.index)
    if len(common) < 6:
        return
    d = b[common] - a[common]
    p = wilcoxon(b[common], a[common]).pvalue if (d != 0).any() else float("nan")
    print(f"  {name_b} 對 {name_a}：{100 * d.mean() / a[common].mean():+.1f}%（{(d > 0).sum()}/{(d < 0).sum()}，p={p:.3f}，n={len(common)}）")


for budget in (1000, 3000):
    rows = {
        "ring_se（每組隨機，正式版）": load(f"ringpair_prod_6maps_eval{budget}", "ring_se"),
        "ring_se 舊版（每 searcher 一段）": load(f"se3_6maps_eval{budget}", "ring_sets"),
        "作弊版": load(f"e01_oracle_6maps_eval{budget}", "ring_se_oracle"),
    }
    print(f"== 預算 {budget}")
    for name, v in rows.items():
        if v is None:
            print(f"  {name}：尚無結果")
            continue
        per_map = v.groupby(level="map").mean().round(0).astype(int).to_dict()
        print(f"  {name:24s} n={len(v):2d} 平均 {v.mean():7.1f}  CV {100 * v.std() / v.mean():4.1f}%  最差 {v.min():4.0f}  各圖 {per_map}")
    base = rows["ring_se（每組隨機，正式版）"]
    if base is None:
        base = rows["ring_se 舊版（每 searcher 一段）"]
    if rows["作弊版"] is not None and base is not None:
        compare(base, rows["作弊版"], "ring_se", "作弊版")
