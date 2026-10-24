"""預算 1000 的 lifetime 長條圖（MAP01–06 × seed 7–11）：原版 ring_se、三個監督式（60 張圖、差異獎勵、驗證早停，
標籤預算 1000／3000／10000）、作弊版。輸出 figures/budget1000_lifetime.png。

用法（在 2027WSN 根目錄）：python Algorithm/se/rl/experiments/E07_ring_se_lifetime/plot_budget1000.py
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

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_manager.fontManager.addfont(FONT)
plt.rcParams["font.family"] = "Noto Sans CJK TC"

GRAY, BLUE, ORANGE = "#8a8984", "#2a78d6", "#eb6834"
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BARS = [("原版 ring_se", "s736_ring_se_eval1000", "ring_se", GRAY),
        ("監督式\n（1000 標籤）", "s711_e07_m60v_t1000_eval1000", "e07_net", BLUE),
        ("監督式\n（3000 標籤）", "s711_e07_m60v_t3000_eval1000", "e07_net", BLUE),
        ("監督式\n（10000 標籤）", "s711_e07_m60v_t10000_eval1000", "e07_net", BLUE),
        ("作弊版", "s736_oracle_eval1000", "ring_se_oracle", ORANGE)]

data = [(name, load(exp, algo), color) for name, exp, algo, color in BARS]
base = data[0][1]
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
fig.patch.set_facecolor("#fcfcfb")
ax.set_facecolor("#fcfcfb")
x = range(len(data))
means = [v.mean() for _, v, _ in data]
ax.bar(x, means, width=0.6, color=[c for _, _, c in data], zorder=2)
for i, (name, v, _) in enumerate(data):
    label = f"{v.mean():.1f}"
    if i:
        c = v.index.intersection(base.index)
        d = v[c] - base[c]
        p = wilcoxon(v[c], base[c]).pvalue
        label += f"\n{100 * d.mean() / base[c].mean():+.2f}%（p={p:.4f}）" if p >= 1e-4 else \
            f"\n{100 * d.mean() / base[c].mean():+.2f}%（p<0.0001）"
    ax.text(i, v.mean() + 12, label, ha="center", va="bottom", fontsize=9, color=TEXT,
            bbox=dict(facecolor="#fcfcfb", edgecolor="none", pad=1.5), zorder=3)
ax.set_xticks(list(x))
ax.set_xticklabels([n for n, _, _ in data], fontsize=10, color=TEXT)
ax.set_ylabel("lifetime（平均，30 場）", color=MUTED)
ax.set_ylim(0, max(means) * 1.18)
ax.yaxis.grid(True, color=GRID, zorder=0)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.tick_params(axis="y", colors=MUTED, length=0)
ax.tick_params(axis="x", length=0)
ax.set_title("預算 1000 的 lifetime（MAP01–06 × seed 7–11）", fontsize=12, color=TEXT, loc="left")
fig.text(0.01, 0.01, "監督式：60 張訓練圖、差異獎勵回歸、驗證早停；百分比與 p 值為對同場原版 ring_se 的成對 Wilcoxon",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.03, 1, 1))
out = HERE / "figures"
out.mkdir(exist_ok=True)
fig.savefig(out / "budget1000_lifetime.png", facecolor=fig.get_facecolor())
print(out / "budget1000_lifetime.png")
