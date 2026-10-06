"""彙整 ring_region_costs.py 的結果：各演算法、各區段的平均（6 張地圖 × seed 的平均）。"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1] / "experiment_results" / "ring_region_costs"
ALGORITHMS = sys.argv[1:] or ["sa_sets", "si_sets", "ring_sets", "ring_setspriority"]

print(f"{'演算法':18} {'區段':>4} {'開啟數':>6} {'排層cost/顆':>12} {'路由cost/顆':>12} {'路由÷排層':>9} {'占全網cost':>9}")
for algorithm in ALGORITHMS:
    path = ROOT / algorithm / "region_costs.csv"
    if not path.is_file():
        print(f"{algorithm:18} （沒有結果）")
        continue
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for region in sorted({int(r["region"]) for r in rows}):
        sub = [r for r in rows if int(r["region"]) == region]
        mean = {key: np.mean([float(r[key]) for r in sub])
                for key in ("active", "sensing_mean_active", "routing_mean_active", "cost_share")}
        ratio = mean["routing_mean_active"] / mean["sensing_mean_active"]
        print(f"{algorithm:18} {region:>4} {mean['active']:6.1f} {mean['sensing_mean_active']:12.5f}"
              f" {mean['routing_mean_active']:12.5f} {ratio:9.2f} {mean['cost_share']:9.1%}")
    fitness = np.mean([float(r["fitness"]) for r in rows if int(r["region"]) == 0])
    total_active = np.mean([sum(float(r["active"]) for r in rows if r["map"] == m and r["seed"] == s)
                            for m, s in {(r["map"], r["seed"]) for r in rows}])
    print(f"{'':18} 全網：平均開啟 {total_active:.1f} 顆，第 0 段最好 fitness 平均 {fitness:.6f}")
