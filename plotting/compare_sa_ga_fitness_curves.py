"""Compare one SA-SETS and one GA best-so-far fitness curve fairly."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path("tmp") / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
import matplotlib.pyplot as plt

from Algorithm.ga.GA import GA
from Algorithm.se.SA_SETS import SA_SETS
from Algorithm.se.SI_SETS import SI_SETS
from Problem.Problem import Problem
from Problem.services.fitness_servicev2 import FitnessServiceV2


def build_problem(map_name: str):
    problem = Problem(
        B=100,
        S=100,
        T=100,
        F=10.0,
        FILE=map_name,
        sensing_mode="discrete",
        routing_service="v1",
    )
    problem.fitness_service = FitnessServiceV2(problem)
    problem.fitness_service_name = "v5"
    return problem


def run_algorithm(name: str, seed: int, evaluate: int):
    problem = build_problem("A2")
    if name == "SA-SETS":
        algorithm = SA_SETS(problem, n=8, h=4, w=2, mu=0.4, seed=seed)
    elif name == "SI-SETS (w=8)":
        algorithm = SI_SETS(problem, n=8, h=4, w=8, mu=0.4, seed=seed)
    else:
        algorithm = GA(problem, n=30, cu=0.8, mu=0.1, seed=seed)
    result = algorithm.run(problem, budget=evaluate)
    history = np.maximum.accumulate(np.asarray(result.history, dtype=float))
    return history, float(result.best_fitness.sum()), int(result.evaluations)


def normalized_gain(history: np.ndarray):
    gain = history[-1] - history[0]
    if gain <= 1e-15:
        return np.zeros_like(history)
    return (history - history[0]) / gain


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluate", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        default="experiment_results/sa_sets_vs_ga_fitness_curve_a2",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in ("SA-SETS", "SI-SETS (w=8)", "GA"):
        results[name] = run_algorithm(name, args.seed, args.evaluate)

    figure, (fitness_axis, progress_axis) = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
        constrained_layout=True,
    )
    colors = {
        "SA-SETS": "tab:blue",
        "SI-SETS (w=8)": "tab:green",
        "GA": "tab:orange",
    }
    for name, (history, _, _) in results.items():
        evaluations = np.arange(1, len(history) + 1)
        fitness_axis.step(
            evaluations, history, where="post", label=name, color=colors[name]
        )
        progress_axis.step(
            evaluations,
            normalized_gain(history),
            where="post",
            label=name,
            color=colors[name],
        )

    fitness_axis.set_title("A2, seed 7, Fitness V5, 10,000 evaluations")
    fitness_axis.set_ylabel("best-so-far fitness")
    fitness_axis.grid(alpha=0.3)
    fitness_axis.legend()
    progress_axis.set_xlabel("fitness evaluations")
    progress_axis.set_ylabel("fraction of each run's final gain")
    progress_axis.set_ylim(-0.02, 1.02)
    progress_axis.grid(alpha=0.3)
    figure.savefig(output_dir / "sa_sets_vs_ga_fitness_curve.png", dpi=180)
    plt.close(figure)

    with (output_dir / "fitness_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["algorithm", "evaluation", "best_so_far_fitness"])
        for name, (history, _, _) in results.items():
            for evaluation, value in enumerate(history, start=1):
                writer.writerow([name, evaluation, float(value)])

    for name, (history, final, actual_evaluations) in results.items():
        print(
            f"{name}: start={history[0]:.9f}, final={final:.9f}, "
            f"actual_evaluations={actual_evaluations}"
        )


if __name__ == "__main__":
    main()
