"""Lifetime tables (markdown) for one result folder.

One table per algorithm: per map mean / best / worst / sample std over runs.
With two or more algorithms, a comparison table follows: per map
mean ± std for every algorithm, the winner (highest mean) and the paired
t-test p value of the winner against the runner-up, paired by (map, seed).

Reads the folder written by write_grouped_results, so older results can be
redone with:
    python3 -m experiments.lifetime_tables <result folder>
"""

import csv
import json
import statistics as st
import sys
from pathlib import Path

from scipy import stats

OUTPUT_NAME = "lifetime_tables.md"


def load_lifetimes(result_root):
    """Return [(name, label)] in run order and {name: {map: {seed: lifetime}}}."""
    params = json.loads((result_root / "algorithm_parameters.json").read_text())
    order = [(a["name"], a.get("label", a["name"])) for a in params["algorithms"]]
    data = {}
    for name, _ in order:
        per_map = {}
        for path in sorted((result_root / name).glob("*/summary.csv")):
            with path.open(newline="", encoding="utf-8") as csv_file:
                for row in csv.DictReader(csv_file):
                    per_map.setdefault(row["map"], {})[row["seed"]] = int(row["lifetime"])
        data[name] = per_map
    return order, data


def _std(values):
    return st.stdev(values) if len(values) > 1 else 0.0


def _all_runs(per_map):
    """Flatten {map: {seed: lifetime}} to {(map, seed): lifetime}."""
    return {
        (map_name, seed): value
        for map_name, by_seed in per_map.items()
        for seed, value in by_seed.items()
    }


def paired_p_value(runs_a, runs_b):
    """Two-sided paired t-test on the runs both sides share; None if undefined."""
    keys = sorted(set(runs_a) & set(runs_b))
    if len(keys) < 2:
        return None
    diffs = [runs_a[key] - runs_b[key] for key in keys]
    if all(diff == diffs[0] for diff in diffs):
        # Constant differences: zero variance, the t statistic is undefined.
        return None
    return float(stats.ttest_rel([runs_a[key] for key in keys], [runs_b[key] for key in keys]).pvalue)


def _format_p(p_value):
    if p_value is None:
        return "-"
    text = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    return text + " *" if p_value < 0.05 else text


def _stats_row(label, runs):
    values = list(runs.values())
    return (
        f"| {label} | {len(values)} | {st.mean(values):.1f} | "
        f"{max(values)} | {min(values)} | {_std(values):.1f} |"
    )


def algorithm_table(label, per_map):
    lines = [
        f"### {label}",
        "",
        "| 地圖 | 場數 | 平均 | 最佳 | 最差 | 標準差 |",
        "|---|---|---|---|---|---|",
    ]
    for map_name in sorted(per_map):
        lines.append(_stats_row(map_name, per_map[map_name]))
    lines.append(_stats_row("**全部**", _all_runs(per_map)))
    return "\n".join(lines)


def comparison_table(order, data):
    """Per map mean ± std of every algorithm; highest mean is bold and wins.

    p compares the winner with the runner-up (second highest mean).
    """
    labels = [label for _, label in order]
    p_header = "p" if len(order) == 2 else "p（勝出 vs 第二）"
    lines = [
        "### 比較",
        "",
        "| 地圖 | " + " | ".join(labels) + f" | 勝出 | {p_header} |",
        "|---|" + "---|" * len(labels) + "---|---|",
    ]
    maps = sorted({map_name for per_map in data.values() for map_name in per_map})
    rows = [
        (map_name, {name: {(map_name, seed): value for seed, value in data[name].get(map_name, {}).items()} for name, _ in order})
        for map_name in maps
    ]
    rows.append(("**全部**", {name: _all_runs(data[name]) for name, _ in order}))
    for row_label, values_by_name in rows:
        means = {
            name: st.mean(runs.values())
            for name, runs in values_by_name.items()
            if runs
        }
        best = max(means.values()) if means else None
        cells = []
        for name, _ in order:
            runs = values_by_name[name]
            if not runs:
                cells.append("-")
                continue
            cell = f"{means[name]:.1f} ± {_std(list(runs.values())):.1f}"
            cells.append(f"**{cell}**" if means[name] == best else cell)
        winners = [label for name, label in order if means.get(name) == best]
        ranked = sorted(means, key=means.get, reverse=True)
        p_value = (
            paired_p_value(values_by_name[ranked[0]], values_by_name[ranked[1]])
            if len(ranked) >= 2
            else None
        )
        lines.append(
            f"| {row_label} | " + " | ".join(cells) + " | " + "、".join(winners)
            + f" | {_format_p(p_value)} |"
        )
    lines += [
        "",
        "p：勝出者對第二名的配對 t 檢定（同地圖、同 seed 配對，雙尾）；* 表示 p < 0.05。",
    ]
    return "\n".join(lines)


def build_report(result_root):
    order, data = load_lifetimes(result_root)
    sections = [f"# {result_root.name}"]
    for name, label in order:
        sections.append(algorithm_table(f"{label}（{name}）", data[name]))
    if len(order) >= 2:
        sections.append(comparison_table(order, data))
    return "\n\n".join(sections) + "\n"


def write_lifetime_tables(result_root):
    """Write lifetime_tables.md into the result folder and print it."""
    result_root = Path(result_root)
    report = build_report(result_root)
    output_path = result_root / OUTPUT_NAME
    output_path.write_text(report, encoding="utf-8")
    print(report)
    print("tables", output_path)
    return output_path


if __name__ == "__main__":
    write_lifetime_tables(sys.argv[1])
