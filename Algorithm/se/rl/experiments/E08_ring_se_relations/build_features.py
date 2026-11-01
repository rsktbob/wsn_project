"""用 E07 收好的資料算 E08 新增的 ②③⑤（不重收）：E07 results/data/b{預算}/{split}/RLTRAIN??.npz
→ E08 results/data/b{預算}/{split}/RLTRAIN??.npz（route、covrel 用 packbits 存，swap float16）。

用法（2027WSN 根目錄）：python Algorithm/se/rl/experiments/E08_ring_se_relations/build_features.py --budget 1000
"""
import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
E07 = HERE.parent / "E07_ring_se_lifetime"
sys.path.insert(0, str(E07))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common import DATA as E07_DATA, PROBLEM_SEED, train_maps  # noqa: E402
from features import levels_of, round_extras  # noqa: E402

DATA = HERE / "results" / "data"
SPECS = {spec["name"]: spec for spec in train_maps(200)}


def map_cover(name):
    from experiments.problem_setup import build_problem
    problem = build_problem(SimpleNamespace(_current_map=SPECS[name], fitness_service="v3"), PROBLEM_SEED)
    return np.asarray(problem.coverage_table) > 0


def build(src, dst):
    if dst.exists():
        return dst.name, "skip"
    z = np.load(src)
    cover = map_cover(src.stem)
    route, covrel, swap = [], [], []
    for r in range(len(z["s_id"])):
        out = round_extras(levels_of(z["S"][r]), levels_of(z["G"][r]), z["S_par"][r], z["G_par"][r], cover,
                           z["s_id"][r], z["g_id"][r], z["span"][r])
        route.append(np.packbits(out[0], axis=-1))
        covrel.append(np.packbits(out[1], axis=-1))
        swap.append(out[2].astype(np.float16))
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, route=np.stack(route), covrel=np.stack(covrel), swap=np.stack(swap))
    tmp.rename(dst)
    return dst.name, "done"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    jobs = []
    for split in ("train", "val", "test"):
        for src in sorted((E07_DATA / f"b{args.budget}" / split).glob("RLTRAIN*.npz")):
            if not src.name.endswith(".tmp.npz"):
                jobs.append((src, DATA / f"b{args.budget}" / split / src.name))
    with ProcessPoolExecutor(args.jobs) as pool:
        for i, (name, status) in enumerate(pool.map(build, *zip(*jobs)), 1):
            if i % 20 == 0 or i == len(jobs):
                print(f"[E08 features] {i}/{len(jobs)} {name} {status}", flush=True)


if __name__ == "__main__":
    main()
