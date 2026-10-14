"""Plot first-segment convergence across repeated A2 region-policy runs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from Algorithm.se.EXP3_SA_SETS import EXP3_SA_SETS
from Algorithm.se.LinUCB_SA_SETS import LinUCB_SA_SETS
from Algorithm.se.SA_SETS import SA_SETS
from Algorithm.se.Thompson_SA_SETS import Thompson_SA_SETS
from Problem.Problem import Problem
from Problem.services import FitnessServiceV2


POLICIES = (
    ("SA-SETS Beta-CDF", SA_SETS),
    ("EXP3-SA-SETS", EXP3_SA_SETS),
    ("LinUCB-SA-SETS", LinUCB_SA_SETS),
    ("Thompson-SA-SETS", Thompson_SA_SETS),
)


def build_problem():
    problem = Problem(
        B=100, S=100, T=100, F=10.0, FILE="A2",
        sensing_mode="discrete", routing_service="v1",
    )
    problem.fitness_service = FitnessServiceV2(problem)
    problem.fitness_service_name = "v5"
    return problem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-seed", type=int, default=7)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--evaluate", type=int, default=500)
    parser.add_argument(
        "--output-dir",
        default="experiment_results/region_policy_a2_full_5runs",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    curves = {}
    for label, algorithm_type in POLICIES:
        histories = []
        for seed in range(args.first_seed, args.first_seed + args.runs):
            problem = build_problem()
            algorithm = algorithm_type(
                problem, n=8, h=4, w=2, mu=0.4, seed=seed
            )
            algorithm.run(problem, budget=args.evaluate)
            history = np.maximum.accumulate(
                np.asarray(algorithm.history, dtype=float)
            )
            histories.append(history)
            records.extend(
                (label, seed, evaluation, float(value))
                for evaluation, value in enumerate(history, start=1)
            )
        curves[label] = np.vstack(histories)

    with (output_dir / "first_segment_fitness_history_5runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(("algorithm", "seed", "evaluation", "best_fitness"))
        writer.writerows(records)

    figure, axis = plt.subplots(figsize=(10, 6), constrained_layout=True)
    evaluations = np.arange(1, args.evaluate + 1)
    for label, values in curves.items():
        for run in values:
            axis.plot(evaluations, run, color="gray", alpha=0.12, linewidth=0.8)
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0, ddof=1)
        line = axis.plot(evaluations, mean, linewidth=2.1, label=label)[0]
        axis.fill_between(
            evaluations, mean - std, mean + std,
            color=line.get_color(), alpha=0.16,
        )
    axis.set_xlabel("Fitness evaluations in first segment")
    axis.set_ylabel("Best-so-far V5 fitness")
    axis.set_title("A2: first-segment convergence across five seeds")
    axis.grid(alpha=0.3)
    axis.legend()
    figure.savefig(output_dir / "first_segment_fitness_curve_5runs.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
