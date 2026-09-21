"""公平比較不同 optimizer 在 GA lifetime 軌跡上的 fitness 曲線。

每個 re-optimization segment 都先複製 GA 目前的 WSN 狀態。GA 和指定的
比較演算法分別在這個相同 snapshot 上求解；只有 GA 找到的解會真正消耗能量、
推進到下一個 segment。因此比較演算法不會因為前一段自己的決策而走到不同環境。
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import experiment_algorithms as experiment  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Compare optimizers on exactly the same WSN energy snapshot; "
            "GA alone advances the lifetime."
        )
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        required=True,
        choices=experiment.ALGORITHM_NAMES,
        help="Algorithms to compare with GA. 'ga' is added automatically.",
    )
    parser.add_argument(
        "--map",
        "--maps",
        dest="map",
        default="maps/maps_100100100.json",
        metavar="JSON",
        help=(
            "Map manifest path, same meaning as experiment_algorithms.py --maps. "
            "You may pass one or more map entries in this JSON manifest."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluate", type=int, default=10000)
    parser.add_argument("--max-segments", type=int, default=30)
    parser.add_argument("--segment-slot-limit", type=int, default=5000)
    parser.add_argument("--max-lifetime", type=int, default=None)
    parser.add_argument(
        "--fitness-service", choices=tuple(experiment.FITNESS_SERVICES), default="v2"
    )
    parser.add_argument(
        "--sensing-mode", choices=("discrete", "bucketed", "exact"), default="discrete"
    )
    parser.add_argument("--routing-service", choices=experiment.ROUTING_SERVICES, default="v1")
    parser.add_argument("--rl-model", default=None)
    parser.add_argument(
        "--output",
        default="experiment_results/ga_led_fitness_comparison",
        help="Output directory (one subdirectory is created for each map).",
    )
    args = parser.parse_args(argv)
    if args.evaluate <= 0 or args.max_segments <= 0 or args.segment_slot_limit <= 0:
        parser.error("--evaluate, --max-segments, and --segment-slot-limit must be positive")
    if args.max_lifetime is not None and args.max_lifetime <= 0:
        parser.error("--max-lifetime must be positive")
    return args


def build_experiment_args(cli_args):
    """Reuse the existing runner's presets and problem/algorithm factories."""
    inner_argv = [
        "--algorithm", "ga",
        "--mode", "lifetime",
        "--runs", "1",
        "--seed", str(cli_args.seed),
        "--maps", cli_args.map,
        "--evaluate", str(cli_args.evaluate),
        "--max-rounds", str(cli_args.max_segments),
        "--segment-slot-limit", str(cli_args.segment_slot_limit),
        "--sensing-mode", cli_args.sensing_mode,
        "--fitness-service", cli_args.fitness_service,
        "--routing-service", cli_args.routing_service,
        "--draw-fitness-history", "false",
        "--draw-save", "false",
        "--show", "false",
    ]
    if cli_args.max_lifetime is not None:
        inner_argv += ["--max-lifetime", str(cli_args.max_lifetime)]
    if cli_args.rl_model is not None:
        inner_argv += ["--rl-model", cli_args.rl_model]
    return experiment.parse_args(inner_argv)


def ordered_algorithms(requested):
    """GA is first and visually treated as the reference curve."""
    return ["ga"] + [name for name in requested if name != "ga"]


def best_so_far(history):
    values = np.asarray(history, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return values
    return np.maximum.accumulate(values)


def solve_snapshot(name, snapshot, experiment_args, seed):
    """Run one algorithm on an isolated copy of an identical environment."""
    problem = copy.deepcopy(snapshot)
    experiment.set_seed(seed)
    algorithm = experiment.build_algorithm(name, problem, experiment_args, seed)
    budget = experiment.effective_evaluate_budget(algorithm, experiment_args)
    started = time.perf_counter()
    result = experiment.run_optimizer(algorithm, problem, budget)
    elapsed = time.perf_counter() - started
    state = result["state"]
    if state is not None:
        state = experiment.prepare_result_state(algorithm, problem, state)
        cost = problem.calculate_total_cost(state)
        feasible = bool(problem.LifeCheck(state, cost))
    else:
        cost = None
        feasible = False
    return {
        "result": result,
        "state": state,
        "cost": cost,
        "feasible": feasible,
        "elapsed_seconds": elapsed,
        "requested_budget": budget,
    }


def draw_line_chart(output_path, title, x_label, y_label, series, labels):
    """Write a dependency-light PNG chart using Pillow (no Matplotlib required)."""
    width, height = 1600, 900
    left, top, right, bottom = 140, 90, 360, 130
    chart_left, chart_top = left, top
    chart_right, chart_bottom = width - right, height - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    colors = [(31, 119, 180), (214, 39, 40), (44, 160, 44), (148, 103, 189), (255, 127, 14)]
    nonempty = [(name, np.asarray(x, dtype=float), np.asarray(y, dtype=float)) for name, x, y in series if len(x) and len(y)]
    if not nonempty:
        draw.text((left, top), "No finite fitness history was returned.", fill="black", font=font)
        image.save(output_path)
        return
    all_x = np.concatenate([x[np.isfinite(x)] for _, x, _ in nonempty])
    all_y = np.concatenate([y[np.isfinite(y)] for _, _, y in nonempty])
    x_min, x_max = float(all_x.min()), float(all_x.max())
    y_min, y_max = float(all_y.min()), float(all_y.max())
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        pad = max(abs(y_min) * 0.02, 0.01)
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.06
        y_min -= pad
        y_max += pad

    def point(x, y):
        px = chart_left + (x - x_min) / (x_max - x_min) * (chart_right - chart_left)
        py = chart_bottom - (y - y_min) / (y_max - y_min) * (chart_bottom - chart_top)
        return round(px), round(py)

    draw.text((left, 25), title, fill="black", font=font)
    for tick in range(6):
        fraction = tick / 5
        y = chart_bottom - fraction * (chart_bottom - chart_top)
        value = y_min + fraction * (y_max - y_min)
        draw.line((chart_left, y, chart_right, y), fill=(225, 225, 225), width=1)
        draw.text((15, y - 5), f"{value:.5g}", fill=(70, 70, 70), font=font)
    draw.line((chart_left, chart_top, chart_left, chart_bottom), fill="black", width=2)
    draw.line((chart_left, chart_bottom, chart_right, chart_bottom), fill="black", width=2)
    draw.text((chart_left, height - 55), x_label, fill="black", font=font)
    draw.text((15, chart_top - 25), y_label, fill="black", font=font)
    for index, (name, x, y) in enumerate(nonempty):
        color = (0, 0, 0) if name == "ga" else colors[index % len(colors)]
        points = [point(float(xi), float(yi)) for xi, yi in zip(x, y)]
        if len(points) == 1:
            px, py = points[0]
            draw.ellipse((px - 3, py - 3, px + 3, py + 3), fill=color)
        else:
            draw.line(points, fill=color, width=4 if name == "ga" else 2)
        legend_y = chart_top + index * 28
        draw.line((chart_right + 25, legend_y + 7, chart_right + 55, legend_y + 7), fill=color, width=4 if name == "ga" else 2)
        draw.text((chart_right + 65, legend_y), labels[name], fill="black", font=font)
    image.save(output_path)


def plot_segment(output_path, segment_id, lifetime, curves, labels):
    draw_line_chart(
        output_path,
        f"Same energy snapshot: segment {segment_id}, GA lifetime {lifetime}",
        "Fitness evaluation",
        "Best-so-far fitness",
        [(name, np.arange(1, len(values) + 1), values) for name, values in curves.items()],
        labels,
    )


def plot_final_summary(output_path, rows, labels):
    if not rows:
        return
    by_algorithm = {}
    for row in rows:
        by_algorithm.setdefault(row["algorithm"], []).append(row)
    draw_line_chart(
        output_path,
        "Final best fitness on each GA-controlled energy snapshot",
        "Re-optimization segment",
        "Best-so-far fitness at end of budget",
        [
            (
                name,
                np.asarray([row["segment"] for row in algorithm_rows]),
                np.asarray([row["best_fitness"] for row in algorithm_rows]),
            )
            for name, algorithm_rows in by_algorithm.items()
        ],
        labels,
    )


def compare_one_map(cli_args, experiment_args, map_spec, algorithms):
    experiment_args._current_map = map_spec
    labels = {name: experiment.algorithm_label(name) for name in algorithms}
    output_dir = Path(cli_args.output) / map_spec["name"] / f"seed_{cli_args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    problem = experiment.build_problem(experiment_args, cli_args.seed)
    lifetime = 0
    rows = []
    histories = {}
    stop_reason = "max_segments"

    for segment_id in range(cli_args.max_segments):
        # This snapshot is created once, before any optimizer starts.  Every curve
        # in this segment therefore sees identical sensor energy and topology.
        snapshot = copy.deepcopy(problem)
        segment_results = {}
        for algorithm_index, name in enumerate(algorithms):
            segment_seed = cli_args.seed + segment_id * 1009 + algorithm_index * 37
            segment_results[name] = solve_snapshot(
                name, snapshot, experiment_args, segment_seed
            )

        curves = {}
        for name, solved in segment_results.items():
            curve = best_so_far(solved["result"]["history"])
            curves[name] = curve
            histories[f"segment_{segment_id:04d}_{name}"] = curve
            rows.append(
                {
                    "segment": segment_id,
                    "lifetime_start": lifetime,
                    "algorithm": name,
                    "label": labels[name],
                    "evaluations": solved["result"]["evaluations"],
                    "requested_budget": solved["requested_budget"],
                    "history_length": len(curve),
                    "best_fitness": float(curve[-1]) if len(curve) else float("nan"),
                    "feasible": solved["feasible"],
                    "elapsed_seconds": round(solved["elapsed_seconds"], 6),
                }
            )
        plot_segment(
            output_dir / f"segment_{segment_id:04d}_life_{lifetime:06d}.png",
            segment_id,
            lifetime,
            curves,
            labels,
        )

        ga_result = segment_results["ga"]
        if ga_result["state"] is None:
            stop_reason = "ga_no_state"
            break
        if not ga_result["feasible"]:
            stop_reason = "ga_infeasible_solution"
            break

        segment_length = 0
        for _ in range(cli_args.segment_slot_limit):
            if not problem.LifeCheck(ga_result["state"], ga_result["cost"]):
                break
            problem.energy = problem.calculate_remaining_energy(ga_result["cost"])
            lifetime += 1
            segment_length += 1
            if cli_args.max_lifetime is not None and lifetime >= cli_args.max_lifetime:
                stop_reason = "lifetime_cap"
                break
        if segment_length:
            problem.prepare_coding_cache(ga_result["cost"] * segment_length)
        else:
            stop_reason = "ga_no_lifetime_progress"
            break

        print(
            f"[ga-led] {map_spec['name']} segment {segment_id} "
            f"life {lifetime - segment_length} -> {lifetime} "
            f"GA fitness {rows[-len(algorithms)]['best_fitness']:.6f}"
        )
        if stop_reason == "lifetime_cap":
            break

    fieldnames = list(rows[0]) if rows else ["segment", "algorithm"]
    with (output_dir / "segment_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output_dir / "fitness_histories.npz", **histories)
    plot_final_summary(output_dir / "final_fitness_by_segment.png", rows, labels)
    metadata = {
        "leader": "ga",
        "algorithms": algorithms,
        "map": map_spec["name"],
        "seed": cli_args.seed,
        "evaluate": cli_args.evaluate,
        "completed_lifetime": lifetime,
        "completed_segments": len({row["segment"] for row in rows}),
        "stop_reason": stop_reason,
        "fairness_rule": (
            "Within each segment all algorithms solve independent deep copies of "
            "the same GA-controlled problem snapshot; only GA consumes energy."
        ),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir, metadata


def main(argv=None):
    cli_args = parse_args(argv)
    algorithms = ordered_algorithms(cli_args.algorithms)
    experiment_args = build_experiment_args(cli_args)
    maps = experiment_args.map_specs
    for map_spec in maps:
        output_dir, metadata = compare_one_map(
            cli_args, experiment_args, map_spec, algorithms
        )
        print(
            f"[done] {output_dir} | lifetime={metadata['completed_lifetime']} "
            f"stop={metadata['stop_reason']}"
        )


if __name__ == "__main__":
    main()
