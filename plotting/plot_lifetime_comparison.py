"""Draw a matched-seed lifetime comparison from experiment summary CSV files."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp") / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt


RESULTS = (
    ("CPOv2", "cpov2_lifetime_a2_5runs", "cpo_v2"),
    ("GA", "ga_lifetime_a2_5runs", "ga"),
    ("SA-SETS", "sa_sets_lifetime_a2_5runs", "sa_sets"),
    ("SI-SETSv2", "si_setsv2_lifetime_a2_5runs", "si_setsv2"),
)


def read_lifetimes(root: Path, result_dir: str, algorithm: str) -> np.ndarray:
    path = root / result_dir / algorithm / "A2" / "summary.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return np.asarray([float(row["lifetime"]) for row in rows], dtype=float)


def main() -> None:
    root = Path("experiment_results")
    output_dir = root / "cpov2_ga_sa_si_setsv2_lifetime_a2"
    output_dir.mkdir(parents=True, exist_ok=True)

    names = []
    values = []
    for name, result_dir, algorithm in RESULTS:
        names.append(name)
        values.append(read_lifetimes(root, result_dir, algorithm))
    means = np.asarray([series.mean() for series in values])
    stds = np.asarray([series.std(ddof=1) for series in values])

    figure, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    positions = np.arange(len(names))
    colors = ("tab:red", "tab:orange", "tab:blue", "tab:green")
    axis.bar(positions, means, yerr=stds, capsize=7, color=colors, alpha=0.78)
    offsets = np.linspace(-0.12, 0.12, len(values[0]))
    for position, series, color in zip(positions, values, colors):
        axis.scatter(
            np.full(series.size, position) + offsets,
            series,
            color=color,
            edgecolor="black",
            linewidth=0.45,
            zorder=3,
        )
    for position, mean in zip(positions, means):
        axis.text(position, mean + stds[position] + 12, f"{mean:.1f}", ha="center")
    axis.set_xticks(positions, names)
    axis.set_ylabel("network lifetime (slots)")
    axis.set_title("A2, 5 matched seeds, Fitness V5, 10,000 FEs per segment")
    axis.grid(axis="y", alpha=0.3)
    figure.savefig(output_dir / "lifetime_comparison.png", dpi=180)
    plt.close(figure)

    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "lifetimes", "mean", "sample_std", "min", "max"])
        for name, series in zip(names, values):
            writer.writerow([name, *series.tolist(), series.mean(), series.std(ddof=1), series.min(), series.max()])


if __name__ == "__main__":
    main()
