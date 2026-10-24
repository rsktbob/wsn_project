"""E07 lifetime 對照：MAP01–06 × seed 7–11（30 場），預算 1000／3000／10000；
成對 Wilcoxon 對同預算的 ring_se（s736_ring_se_eval{B} 裡同樣的地圖、seed）。E07 的模型都是預算 10000 的標籤訓練的；
E06 監督式是 E06 的 best.pt（預算 1000 的作弊版標籤），1000 取 s736 裡的 seed 7–11。所有結果都只取 seed 7–11。"""
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

R = Path("/home/g114056067/wsn_project/codes/2027WSN/experiment_results")
BUDGETS = (1000, 3000, 10000)
METHODS = (("監督式（取最大）", "s711_e07_sup_eval{B}", "e07_net"),
           ("監督式（抽樣）", "s711_e07_sup_sample_eval{B}", "e07_net_sample"),
           ("監督式（差異獎勵回歸，取最大）", "s711_e07_supdiff_eval{B}", "e07_net"),
           ("監督式（差異獎勵回歸，用 E06 資料）", "s711_e07_supdiff_e06data_eval{B}", "e07_net"),
           ("監督式（差異獎勵回歸，預算 1000 標籤）", "s711_e07_supdiff_b1000_eval{B}", "e07_net"),
           ("監督式（差異獎勵回歸，預算 1000 標籤、60 張圖）", "s711_e07_supdiff_b1000_m60_eval{B}", "e07_net"),
           ("D3QN（預算 1000 訓練）", "s711_e07_d3qn_b1000_eval{B}", "e07_net"),
           ("E06 監督式", "s711_e06_sup_eval{B}", "e06_net"),
           ("E06 監督式", "s736_e06_sup_eval{B}", "e06_net"))
SEEDS = range(7, 12)


def load(exp, algo):
    fs = sorted((R / exp / algo).glob("MAP*/summary.csv"))
    if not fs:
        return None
    v = pd.concat([pd.read_csv(f).assign(map=f.parent.name) for f in fs]).set_index(["map", "seed"])["lifetime"].astype(float)
    return v[v.index.get_level_values("seed").isin(SEEDS)]


def compare(x, y):
    c = x.index.intersection(y.index)
    d = x[c] - y[c]
    return f"{100 * d.mean() / y[c].mean():+.2f}%（{(d > 0).sum()}/{(d < 0).sum()}，p={wilcoxon(x[c], y[c]).pvalue:.4f}）"


print("| 預算 | 方法 | n | lifetime | 同場 ring_se | 對 ring_se（勝/負，p） | 最差 |")
print("| --- | --- | --- | --- | --- | --- | --- |")
for B in BUDGETS:
    base = load(f"s736_ring_se_eval{B}", "ring_se")
    for name, exp, algo in METHODS:
        v = load(exp.format(B=B), algo)
        if v is None:
            continue
        c = v.index.intersection(base.index)
        print(f"| {B} | {name} | {len(v)} | {v.mean():.1f} | {base[c].mean():.1f} | {compare(v, base)} | {v.min():.0f} |")


# ---- 3×3：60 張圖、驗證早停的差異獎勵模型（標籤預算 T）× 評估預算 E ----
print()
print("3×3（60 張圖、差異獎勵、驗證早停；格內：lifetime（對同場 ring_se，勝/負，p））")
print("| 訓練標籤預算 ＼ 評估預算 | " + " | ".join(str(E) for E in BUDGETS) + " |")
print("| --- |" + " --- |" * len(BUDGETS))
bases = {E: load(f"s736_ring_se_eval{E}", "ring_se") for E in BUDGETS}
print("| ring_se（隨機） | " + " | ".join(f"{bases[E].mean():.1f}" for E in BUDGETS) + " |")
for T in BUDGETS:
    cells = []
    for E in BUDGETS:
        v = load(f"s711_e07_m60v_t{T}_eval{E}", "e07_net")
        cells.append("—" if v is None else f"{v.mean():.1f}（{compare(v, bases[E])}）")
    print(f"| {T} | " + " | ".join(cells) + " |")
cells = []
for E in BUDGETS:
    v = load(f"s711_e07_m60v_t1000cur_eval{E}", "e07_net")
    cells.append("—" if v is None else f"{v.mean():.1f}（{compare(v, bases[E])}）")
print("| 1000（標籤 ÷ 這一回合尺度） | " + " | ".join(cells) + " |")
cells = []
for E in BUDGETS:
    v = load(f"s736_oracle_eval{E}", "ring_se_oracle")
    cells.append("—" if v is None else f"{v.mean():.1f}（{compare(v, bases[E])}）")
print("| 作弊版 | " + " | ".join(cells) + " |")
