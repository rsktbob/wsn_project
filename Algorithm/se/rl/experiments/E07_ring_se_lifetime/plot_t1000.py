"""預算 1000 標籤訓練的監督式（60 張圖、差異獎勵、驗證早停）在預算 1000／3000／10000 的 lifetime，
和原版 ring_se、作弊版並排（MAP01–06 × seed 7–11）。輸出 figures/t1000_vs_budgets.png。

用法（在 2027WSN 根目錄）：python Algorithm/se/rl/experiments/E07_ring_se_lifetime/plot_t1000.py
"""
import contextlib
import io
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
with contextlib.redirect_stdout(io.StringIO()):
    from summary import load  # noqa: E402

font_manager.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = "Noto Sans CJK TC"

SURFACE, TEXT, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
BUDGETS = (1000, 3000, 10000)
SERIES = [("原版 ring_se", "s736_ring_se_eval{B}", "ring_se", "#8a8984"),
          ("監督式", "s711_e07_m60v_t1000_eval{B}", "e07_net", "#2a78d6"),
          ("作弊版", "s736_oracle_eval{B}", "ring_se_oracle", "#eb6834")]

fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
width, gap = 0.26, 0.02
top = 0
for j, (name, exp, algo, color) in enumerate(SERIES):
    for i, B in enumerate(BUDGETS):
        v = load(exp.format(B=B), algo)
        base = load(f"s736_ring_se_eval{B}", "ring_se")
        x = i + (j - 1) * (width + gap)
        ax.bar(x, v.mean(), width=width, color=color, zorder=2, label=name if i == 0 else None)
        label = f"{v.mean():.1f}"
        if j:
            c = v.index.intersection(base.index)
            pct = 100 * (v[c] - base[c]).mean() / base[c].mean()
            p = wilcoxon(v[c], base[c]).pvalue
            label += f"\n{pct:+.2f}%" + ("" if p < 0.05 else "（ns）")
        ax.text(x, v.mean() + 10, label, ha="center", va="bottom", fontsize=8.5, color=TEXT,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1), zorder=3)
        top = max(top, v.mean())
ax.set_xticks(range(len(BUDGETS)))
ax.set_xticklabels([f"預算 {B}" for B in BUDGETS], fontsize=11, color=TEXT)
ax.set_ylabel("lifetime（平均，30 場）", color=MUTED)
ax.set_ylim(0, top * 1.16)
ax.yaxis.grid(True, color=GRID, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="y", colors=MUTED, length=0)
ax.tick_params(axis="x", length=0)
ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=9.5, bbox_to_anchor=(0, 1.0))
ax.set_title("原版、監督式、作弊版在不同預算下的 lifetime（MAP01–06 × seed 7–11）", fontsize=12, color=TEXT,
             loc="left", pad=22)
fig.text(0.01, 0.01, "監督式：60 張訓練圖、差異獎勵回歸、驗證早停；百分比為對同場原版 ring_se，"
         "成對 Wilcoxon p ≥ 0.05 標 ns", fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = HERE / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "t1000_vs_budgets.png", facecolor=fig.get_facecolor())
print(out / "t1000_vs_budgets.png")
