"""每個 Ring 區段的平均排層（感測）cost 與路由 cost。

對 manifest 裡每張地圖、每個 seed：用指定演算法解第 0 段（滿電），取最好的解，
以 energy_service 拆成每顆 sensor 的排層 cost（calculate_scheduling_cost）與路由 cost
（calculate_routing_cost，含自己傳送與幫別人轉送），再依區段統計。
輸出（在 <output-dir>/<演算法>/）：CSV（每個 seed 一列／區段）、每張地圖一張區段圖（seed 第一個的解：實心＝開啟），
圖例附上該區段跨 seed 的平均值。
用法：python plotting/ring_region_costs.py [--maps maps/maps_100100100.json] [--algorithm ring_setspriority]
區段是 Ring 的 sensor 編號範圍；SA_SETS／SI_SETS 自己的分區不同，這裡只是共同的空間分組。
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiment_algorithms import build_algorithm  # noqa: E402
from experiments.maps import load_map_specs  # noqa: E402
from plotting.ring_regions_map import build_map_problem, draw_regions, region_bounds  # noqa: E402

FIELDS = ("map", "seed", "region", "sensors", "active", "sensing_total", "routing_total",
          "sensing_mean_active", "routing_mean_active", "cost_share", "fitness")


def region_rows(problem, state, bounds, map_name, seed, fitness):
    count = problem.SENSOR_NUMBER
    radii = problem.state_radius(state)
    sensing = np.asarray(problem.energy_service.calculate_scheduling_cost(state, sensing_radii=radii), float)[:count]
    routing = np.asarray(problem.energy_service.calculate_routing_cost(state, sensing_radii=radii), float)[:count]
    active = np.asarray(state.levels)[:count] > 0
    total = float(np.sum(sensing + routing))
    rows = []
    for region, (start, end) in enumerate(bounds):
        part = slice(start, end)
        on = int(active[part].sum())
        rows.append({
            "map": map_name, "seed": seed, "region": region, "sensors": end - start, "active": on,
            "sensing_total": float(sensing[part].sum()), "routing_total": float(routing[part].sum()),
            "sensing_mean_active": float(sensing[part][active[part]].mean()) if on else 0.0,
            "routing_mean_active": float(routing[part][active[part]].mean()) if on else 0.0,
            "cost_share": float((sensing[part] + routing[part]).sum() / total) if total > 0 else 0.0,
            "fitness": fitness,
        })
    return rows, active


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--maps", default="maps/maps_100100100.json")
    parser.add_argument("--algorithm", default="ring_setspriority")
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9, 10, 11])
    parser.add_argument("--evaluate", type=int, default=10000)
    parser.add_argument("--h", type=int, default=4)
    parser.add_argument("--output-dir", default="experiment_results/ring_region_costs")
    cli = parser.parse_args()
    out_dir = ROOT / cli.output_dir / cli.algorithm
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, spec in enumerate(load_map_specs(str(ROOT / cli.maps))):
        first_active, bounds, first_problem = None, None, None
        for seed in cli.seeds:
            problem, spec = build_map_problem(cli.maps, index, seed)
            bounds = region_bounds(problem, cli.h)
            algorithm = build_algorithm(cli.algorithm, problem, SimpleNamespace(rl_model=None, evaluate=cli.evaluate), seed)
            algorithm.run(problem, budget=cli.evaluate)
            map_rows, active = region_rows(problem, algorithm.best_state, bounds, spec["name"], seed,
                                           float(algorithm.fitness))
            rows += map_rows
            if first_active is None:
                first_active, first_problem = active, problem
        labels = []
        for region, (start, end) in enumerate(bounds):
            sub = [r for r in rows if r["map"] == spec["name"] and r["region"] == region]
            labels.append(f"區段 {region}（{start}–{end - 1}）開 {np.mean([r['active'] for r in sub]):.1f} 顆｜"
                          f"排層 {np.mean([r['sensing_mean_active'] for r in sub]):.4f}｜"
                          f"路由 {np.mean([r['routing_mean_active'] for r in sub]):.4f}")
        draw_regions(first_problem, bounds,
                     f"{spec['name']}：區段與每顆開啟 sensor 的平均 cost（{cli.algorithm}，第 0 段，{len(cli.seeds)} 個 seed）",
                     out_dir / f"{spec['name']}.png", region_labels=labels, active=first_active)
        print("done", spec["name"], flush=True)

    with (out_dir / "region_costs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'地圖':6} {'區段':4} {'開啟數':>6} {'排層cost/顆':>12} {'路由cost/顆':>12} {'路由÷排層':>9} {'占全網cost':>9}")
    for map_name in dict.fromkeys(r["map"] for r in rows):
        for region in range(cli.h):
            sub = [r for r in rows if r["map"] == map_name and r["region"] == region]
            sensing = np.mean([r["sensing_mean_active"] for r in sub])
            routing = np.mean([r["routing_mean_active"] for r in sub])
            print(f"{map_name:6} {region:>4} {np.mean([r['active'] for r in sub]):6.1f} {sensing:12.5f} {routing:12.5f}"
                  f" {routing / sensing if sensing else float('nan'):9.2f} {np.mean([r['cost_share'] for r in sub]):9.1%}")
    print("\nsaved", out_dir)


if __name__ == "__main__":
    main()
