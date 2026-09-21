"""Plot one SA-SETS optimization run for several V5 fitness weight settings.

This is intentionally a single-optimization experiment: it does not execute
the outer WSN lifetime loop or consume sensor energy between optimizations.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Algorithm.se.SA_SETS import SA_SETS
from Problem.Problem import Problem
from Problem.services.fitness_servicev2 import FitnessServiceV2


WEIGHT_SETTINGS = (
    ("V5 current (0.3, 0.7)", 0.3, 0.7),
    ("V5 balanced (0.5, 0.5)", 0.5, 0.5),
    ("V5 global (0.7, 0.3)", 0.7, 0.3),
)


def build_problem(map_name: str, sensing_mode: str, w1: float, w2: float):
    problem = Problem(
        B=100,
        S=100,
        T=100,
        F=10.0,
        FILE=map_name,
        sensing_mode=sensing_mode,
        routing_service="v1",
    )
    problem.fitness_service = FitnessServiceV2(problem, w1=w1, w2=w2)
    problem.fitness_service_name = "v5"
    return problem


def run_setting(label: str, w1: float, w2: float, args):
    problem = build_problem(args.map, args.sensing_mode, w1, w2)
    algorithm = SA_SETS(
        problem,
        n=args.searchers,
        h=args.regions,
        w=args.goods,
        mu=args.mutation_rate,
        seed=args.seed,
    )
    algorithm.search(problem, args.evaluate)
    history = np.asarray(algorithm.history, dtype=float)
    return {
        "label": label,
        "w1": w1,
        "w2": w2,
        "history": history,
        "best_fitness": float(algorithm.fitness),
        "actual_evaluations": int(algorithm.evatime),
        "iterations": int(algorithm.iterations_completed),
    }


def write_csv(results, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "evaluation",
                "label",
                "global_weight",
                "worst_target_weight",
                "best_fitness",
            ]
        )
        for result in results:
            for evaluation, value in enumerate(result["history"], start=1):
                writer.writerow(
                    [
                        evaluation,
                        result["label"],
                        result["w1"],
                        result["w2"],
                        float(value),
                    ]
                )


def draw(results, output_path: Path):
    figure, (raw_axis, gap_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )
    for result in results:
        evaluations = np.arange(1, len(result["history"]) + 1)
        fitness = result["history"]
        raw_axis.step(evaluations, fitness, where="post", label=result["label"])
        gap_axis.step(
            evaluations,
            np.maximum(1.0 - fitness, 1e-12),
            where="post",
            label=result["label"],
        )

    raw_axis.set_title("SA-SETS: one optimization run on A2")
    raw_axis.set_ylabel("best V5 fitness")
    raw_axis.grid(alpha=0.3)
    raw_axis.legend()

    gap_axis.set_xlabel("fitness evaluations")
    gap_axis.set_ylabel("1 - best fitness (log scale)")
    gap_axis.set_yscale("log")
    gap_axis.grid(alpha=0.3, which="both")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(
        description="Compare V5 weight settings in one SA-SETS optimization."
    )
    parser.add_argument("--map", default="A2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluate", type=int, default=8000)
    parser.add_argument("--searchers", type=int, default=8)
    parser.add_argument("--regions", type=int, default=4)
    parser.add_argument("--goods", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.4)
    parser.add_argument("--sensing-mode", default="discrete")
    parser.add_argument(
        "--output-dir",
        default="experiment_results/fitness_curve_single_A2",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = [run_setting(*setting, args) for setting in WEIGHT_SETTINGS]
    write_csv(results, output_dir / "fitness_history.csv")
    draw(results, output_dir / "fitness_curve.png")

    for result in results:
        print(
            f"{result['label']}: best={result['best_fitness']:.9f}, "
            f"evaluations={result['actual_evaluations']}, "
            f"iterations={result['iterations']}"
        )
    print("csv", output_dir / "fitness_history.csv")
    print("plot", output_dir / "fitness_curve.png")


if __name__ == "__main__":
    main()
