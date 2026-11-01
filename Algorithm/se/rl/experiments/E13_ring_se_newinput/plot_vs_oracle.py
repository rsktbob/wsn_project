"""三預算 lifetime 長條圖（MAP01–06 × seed 7–11）：原版 ring_se、E13 監督式、ring_se 作弊版（plain Δ 最大）。
輸出 figures/e13_vs_oracle.png。

用法（在 2027WSN 根目錄）：python Algorithm/se/rl/experiments/E13_ring_se_newinput/plot_vs_oracle.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Noto Sans CJK TC"

GRAY, BLUE, ORANGE = "#8a8984", "#2a78d6", "#eb6834"
SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BUDGETS = (1000, 3000, 10000)
SERIES = [("原版 ring_se", "s711_e07cmp_ring_se_eval{}", GRAY),
          ("E13 監督式", "s711_e13_b1000_m140_plain_delta_soft1_eval{}", BLUE),
          ("作弊版（plain Δ）", "s711_e07cmp_e07_oracle_plain_eval{}", ORANGE)]


def load(exp):
    fs = sorted((ROOT / "experiment_results" / exp).glob("*/MAP*/summary.csv"))
    return pd.concat([pd.read_csv(f).assign(map=f.parent.name) for f in fs]).set_index(["map", "seed"])["lifetime"].astype(float)


fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
width, gap = 0.26, 0.02
top = 0
for b, budget in enumerate(BUDGETS):
    runs = [load(exp.format(budget)) for _, exp, _ in SERIES]
    base = runs[0]
    for k, ((_, _, color), v) in enumerate(zip(SERIES, runs)):
        x = b + (k - 1) * (width + gap)
        ax.bar(x, v.mean(), width=width, color=color, zorder=2)
        label = f"{v.mean():.0f}"
        if k:
            c = v.index.intersection(base.index)
            p = wilcoxon(v[c], base[c]).pvalue
            label += f"\n{100 * (v[c].mean() / base[c].mean() - 1):+.1f}%" + ("＊" if p < 0.05 else "")
        ax.text(x, v.mean() + 10, label, ha="center", va="bottom", fontsize=8.5, color=TEXT, zorder=3)
        top = max(top, v.mean())
ax.set_xticks(range(len(BUDGETS)))
ax.set_xticklabels([f"預算 {b}" for b in BUDGETS], fontsize=10, color=TEXT)
ax.set_ylabel("lifetime（平均，30 場）", color=MUTED)
ax.set_ylim(0, top * 1.16)
ax.yaxis.grid(True, color=GRID, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="y", colors=MUTED, length=0)
ax.tick_params(axis="x", length=0)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in SERIES], labels=[n for n, _, _ in SERIES],
          loc="upper left", ncol=3, frameon=False, fontsize=9, labelcolor=TEXT)
ax.set_title("ring_se、E13 監督式、作弊版的 lifetime（MAP01–06 × seed 7–11）", fontsize=12, color=TEXT, loc="left")
fig.text(0.01, 0.01, "E13：預算 1000 標籤、soft τ=1 訓練；作弊版偷看 4 個候選只交換的子代，選 plain Δ 最大（全 0 留隨機）。"
         "百分比對同場原版 ring_se，＊＝成對 Wilcoxon p<0.05", fontsize=7, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = HERE / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "e13_vs_oracle.png", facecolor=fig.get_facecolor())
print(out / "e13_vs_oracle.png")
