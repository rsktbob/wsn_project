"""以較低評估預算篩選 GI-GOMEA state 與參數組合。"""

import random
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.gomea.GI_GOMEA import GI_GOMEA
from Algorithm.gomea.GI_GOMEA_Target import GI_GOMEA_Target
from Problem.Problem import Problem


CONFIGURATIONS = {
    "coding_base": {
        "variant": "coding",
        "population_size": 24,
        "statistical_weight": 0.7,
        "graph_weight": 0.3,
        "max_linkage_size": 16,
        "max_linkage_sets": 160,
    },
    "coding_p16": {
        "variant": "coding",
        "population_size": 16,
        "statistical_weight": 0.7,
        "graph_weight": 0.3,
        "max_linkage_size": 16,
        "max_linkage_sets": 160,
    },
    "coding_p40": {
        "variant": "coding",
        "population_size": 40,
        "statistical_weight": 0.7,
        "graph_weight": 0.3,
        "max_linkage_size": 16,
        "max_linkage_sets": 160,
    },
    "coding_graph50": {
        "variant": "coding",
        "population_size": 24,
        "statistical_weight": 0.5,
        "graph_weight": 0.5,
        "max_linkage_size": 16,
        "max_linkage_sets": 160,
    },
    "target_base": {
        "variant": "target",
        "population_size": 24,
        "statistical_weight": 0.7,
        "graph_weight": 0.3,
        "max_linkage_size": 16,
        "max_linkage_sets": 120,
    },
    "target_p16": {
        "variant": "target",
        "population_size": 16,
        "statistical_weight": 0.7,
        "graph_weight": 0.3,
        "max_linkage_size": 16,
        "max_linkage_sets": 120,
    },
    "target_p40": {
        "variant": "target",
        "population_size": 40,
        "statistical_weight": 0.7,
        "graph_weight": 0.3,
        "max_linkage_size": 16,
        "max_linkage_sets": 120,
    },
    "target_graph50": {
        "variant": "target",
        "population_size": 24,
        "statistical_weight": 0.5,
        "graph_weight": 0.5,
        "max_linkage_size": 16,
        "max_linkage_sets": 120,
    },
}


def projected_duration(problem, cost):
    used = np.asarray(cost, dtype=float) > 1e-15
    if not np.any(used):
        return 0.0
    return float(
        np.min(np.asarray(problem.energy)[used] / np.asarray(cost)[used])
    )


def main():
    seeds = (7, 8, 9)
    evaluation_budget = 1000
    results = []
    for name, configuration in CONFIGURATIONS.items():
        for seed in seeds:
            parameters = configuration.copy()
            variant = parameters.pop("variant")
            algorithm_class = (
                GI_GOMEA_Target if variant == "target" else GI_GOMEA
            )
            np.random.seed(seed)
            random.seed(seed)
            problem = Problem(B=100, S=100, T=100, F=10, FILE="A2")
            algorithm = algorithm_class(
                problem,
                elite_fraction=0.5,
                seed=seed,
                **parameters,
            )
            started_at = time.time()
            state = algorithm.run(
                problem,
                budget=evaluation_budget,
            ).best_state
            elapsed = time.time() - started_at
            state.Decode(problem)
            cost = problem.calculate_total_cost(state)
            feasible = problem.LifeCheck(state, cost)
            result = {
                "name": name,
                "seed": seed,
                "fitness": float(np.sum(state.Evaluate(problem))),
                "duration": projected_duration(problem, cost) if feasible else 0.0,
                "feasible": int(feasible),
                "elapsed": elapsed,
            }
            results.append(result)
            print("PARAM_RESULT", result, flush=True)

    print("PARAM_SUMMARY")
    for name in CONFIGURATIONS:
        rows = [result for result in results if result["name"] == name]
        print(
            name,
            "feasible=",
            sum(result["feasible"] for result in rows),
            "/",
            len(rows),
            "duration_mean=",
            round(float(np.mean([result["duration"] for result in rows])), 3),
            "fitness_mean=",
            round(float(np.mean([result["fitness"] for result in rows])), 6),
            "seconds_mean=",
            round(float(np.mean([result["elapsed"] for result in rows])), 3),
        )


if __name__ == "__main__":
    main()
