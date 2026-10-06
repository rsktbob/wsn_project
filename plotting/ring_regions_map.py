"""畫出 Ring_SETS 的區段劃分：sensor 依所屬區段上色、標上編號（＝染色體位置）。

sensor 編號與實驗相同（以 BS 為中心的環狀順序，sensor_ring_id=true），區段是
Ring_SETS.region_sensor_bounds：h 段連續的編號範圍。原點在左上角（BS 顯示在左上）。
用法：python plotting/ring_regions_map.py [--maps maps/maps_100100100_a2.json] [--h 4] [--output 檔名.png]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = ["Noto Sans CJK TC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Algorithm.se.Ring_SETS import Ring_SETS  # noqa: E402
from experiment_algorithms import parse_args as experiment_args  # noqa: E402
from experiments.maps import load_map_specs  # noqa: E402
from experiments.problem_setup import build_problem  # noqa: E402

# 分類色（dataviz 參考調色盤 slot 1/2/3/7）＋不同形狀，顏色不是唯一的辨識方式。
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
MARKERS = ["o", "s", "^", "D"]
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def build_map_problem(manifest, map_index=0, seed=7):
    """The experiment's problem for one map of a manifest, plus its spec."""
    args = experiment_args(["--algorithm", "ring_sets", "--mode", "single", "--maps", manifest])
    spec = load_map_specs(str(ROOT / manifest))[map_index]
    args._current_map = spec
    return build_problem(args, seed), spec


def region_bounds(problem, h):
    return Ring_SETS(problem, n=8, h=h, w=8, mu=0.4, seed=7).region_sensor_bounds


def draw_regions(problem, bounds, title, output, region_labels=None, active=None):
    """Scatter the sensors by region; hollow markers for sensors that are off in ``active``."""
    if len(bounds) > len(COLORS):
        raise ValueError(f"at most {len(COLORS)} regions can be drawn distinguishably")
    sensors = np.asarray(problem.sensor, dtype=float)
    targets = np.asarray(problem.target, dtype=float)
    bs = np.asarray(problem.BS, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.scatter(targets[:, 0], targets[:, 1], marker="x", s=14, linewidths=0.8,
               color="#b5b4ae", label="target", zorder=1)
    for region, (start, end) in enumerate(bounds):
        ids = np.arange(start, end)
        label = (region_labels[region] if region_labels
                 else f"區段 {region}（sensor {start}–{end - 1}）")
        on = ids if active is None else ids[active[ids]]
        off = np.array([], dtype=int) if active is None else ids[~active[ids]]
        ax.scatter(sensors[on, 0], sensors[on, 1], s=70, marker=MARKERS[region], color=COLORS[region],
                   edgecolors=SURFACE, linewidths=1.5, zorder=3, label=label)
        if len(off):
            ax.scatter(sensors[off, 0], sensors[off, 1], s=60, marker=MARKERS[region], facecolors="none",
                       edgecolors=COLORS[region], linewidths=1.2, zorder=3)
        for sensor_id in ids:
            ax.annotate(str(sensor_id), sensors[sensor_id], xytext=(4, 4), textcoords="offset points",
                        fontsize=7, color=INK, zorder=4)
    ax.scatter([bs[0]], [bs[1]], marker="*", s=320, color=INK, edgecolors=SURFACE,
               linewidths=1.5, zorder=5, label="基地台 BS")
    if active is not None:
        ax.scatter([], [], s=60, marker="o", facecolors="none", edgecolors=MUTED, label="空心：未開啟")

    boundary = float(problem.BOUNDARY)
    ax.set_xlim(-3, boundary + 3)
    ax.set_ylim(-3, boundary + 3)
    ax.invert_yaxis()  # 原點 (0, 0) 在左上角，BS 顯示在左上
    ax.set_aspect("equal")
    ax.grid(color="#e4e3df", linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(title, fontsize=13, color=INK, loc="left", pad=12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=10,
              labelcolor=INK, borderaxespad=0)
    fig.savefig(output, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--maps", default="maps/maps_100100100_a2.json")
    parser.add_argument("--h", type=int, default=4)
    parser.add_argument("--output", default="plotting/ring_regions_A2.png")
    cli = parser.parse_args()
    problem, spec = build_map_problem(cli.maps)
    output = ROOT / cli.output
    draw_regions(problem, region_bounds(problem, cli.h),
                 f"{spec['name']}：Ring_SETS 的 {cli.h} 個區段（編號＝染色體中的 sensor 位置）", output)
    print("saved", output)


if __name__ == "__main__":
    main()
