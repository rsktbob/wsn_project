"""E01 訓練／驗證快照：在訓練圖上用 ring_se（正式版）跑 lifetime（預算 1000），每 STRIDE 段存一份問題狀態。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E01_ring_se_segment/snapshots.py --shard 0 --shards 4
地圖分配：maps_100100100_train.json 的 200 張，最後 10 張當驗證、其餘 190 張訓練。
輸出：data/snapshots/{地圖}.pkl.gz，內容 [{"map", "segment", "lifetime", "problem": pickle bytes}, ...]
"""
from __future__ import annotations

import argparse
import copy
import gzip
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from experiments.evaluation import prepare_result_state, run_optimizer  # noqa: E402
from experiments.maps import load_map_specs  # noqa: E402
from experiments.problem_setup import build_problem  # noqa: E402
from experiment_algorithms import build_algorithm  # noqa: E402

MAPS = "maps/maps_100100100_train.json"
N_VALIDATION = 10
PROBLEM_SEED = 7
BUDGET = 1000
STRIDE = 3
MAX_SEGMENTS = 45
SLOT_LIMIT = 5000
OUT = HERE / "data" / "snapshots"


def split_maps():
    specs = load_map_specs(MAPS)
    return specs[:-N_VALIDATION], specs[-N_VALIDATION:]


def build(spec):
    return build_problem(SimpleNamespace(_current_map=spec, fitness_service="v3"), PROBLEM_SEED)


def snapshot_map(spec, algo_seed):
    problem = build(spec)
    algorithm = build_algorithm("ring_se", problem, SimpleNamespace(rl_model=None, evaluate=BUDGET), algo_seed)
    snaps, lifetime = [], 0
    for segment in range(MAX_SEGMENTS):
        if segment % STRIDE == 0:
            snaps.append({"map": spec["name"], "segment": segment, "lifetime": lifetime,
                          "problem": pickle.dumps(copy.deepcopy(problem), protocol=pickle.HIGHEST_PROTOCOL)})
        result = run_optimizer(algorithm, problem, BUDGET)
        state = prepare_result_state(algorithm, problem, result["state"])
        if state is None:
            break
        cost = problem.calculate_total_cost(state)
        length = 0
        for _ in range(SLOT_LIMIT):
            if not problem.LifeCheck(state, cost):
                break
            problem.energy = problem.calculate_remaining_energy(cost)
            length += 1
        if length == 0:
            break
        lifetime += length
        problem.prepare_coding_cache(cost * length)
    return snaps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--shards", type=int, default=1)
    args = parser.parse_args()
    train, val = split_maps()
    OUT.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(train + val):
        if index % args.shards != args.shard:
            continue
        path = OUT / f"{spec['name']}.pkl.gz"
        if path.exists():
            continue
        snaps = snapshot_map(spec, algo_seed=1000 + index)
        tmp = path.with_suffix(".tmp")
        with gzip.open(tmp, "wb") as handle:
            pickle.dump(snaps, handle, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.rename(path)
        print(f"[snapshots] {spec['name']} {len(snaps)} snapshots, last lifetime {snaps[-1]['lifetime']}", flush=True)


if __name__ == "__main__":
    main()
