"""E02 擴大訓練：學習曲線（訓練資料量 × 驗證成績）。

* 驗證：固定的驗證地圖（190 張訓練圖中隨機 15%，seed 0），這些地圖上的所有交易都只拿來驗證。
* 訓練：其餘地圖的資料，依 episode 數取 50／100／200／400／全部（巢狀抽樣，seed 0）；每個 episode 512 組交易。
* 網路：完整 25 維、寬 64（同 capacity.py）；資料最多時另訓練寬 256。目標、損失、早停同 capacity.py。
輸出：results/scale.json、results/scale_{episodes}ep_w{width}.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E02 = HERE.parent
sys.path.insert(0, str(E02.parents[4]))
sys.path.insert(0, str(E02))
sys.path.insert(0, str(HERE))
from capacity import FEATURE_SETS, metrics, targets, train  # noqa: E402
from snapshot_pool import split_maps  # noqa: E402

PAIRS_PER_EPISODE = 512
SIZES = [50, 100, 200, 400, None]  # None = 全部
COLS = FEATURE_SETS["完整（25 維）"]
KEYS = ("X", "delta", "f2", "choice", "map")


def load_with_episodes():
    data = {k: [] for k in KEYS}
    episode, next_id = [], 0
    for path in sorted((E02 / "data" / "labels").glob("shard*.npz")):
        part = np.load(path)
        n = len(part["X"])
        assert n % PAIRS_PER_EPISODE == 0, path
        for k in KEYS:
            data[k].append(part[k])
        episode.append(next_id + np.arange(n) // PAIRS_PER_EPISODE)
        next_id += n // PAIRS_PER_EPISODE
    out = {k: np.concatenate(v) for k, v in data.items()}
    out["episode"] = np.concatenate(episode)
    return out


def main():
    data = load_with_episodes()
    train_specs, _ = split_maps()
    names = sorted(s["name"] for s in train_specs)
    rng = np.random.default_rng(0)
    val_maps = set(rng.choice(names, size=int(round(0.15 * len(names))), replace=False))
    is_va = np.array([m in val_maps for m in data["map"]])
    va = {k: v[is_va] for k, v in data.items()}
    pool = {k: v[~is_va] for k, v in data.items()}
    pool_episodes = np.unique(pool["episode"])
    order = np.random.default_rng(0).permutation(pool_episodes)
    ref = pool["delta"][pool["delta"] > 0]
    y_va = targets(va["delta"], ref)

    report = {"episodes_total": int(len(np.unique(data["episode"]))), "episodes_train_pool": int(len(pool_episodes)),
              "val_maps": len(val_maps), "pairs_val": int(len(va["X"])),
              "random": metrics(np.random.default_rng(1).random(va["delta"].shape), va), "runs": []}
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, ensure_ascii=False), flush=True)

    runs = [(size, 64) for size in SIZES] + [(None, 256)]
    for size, width in runs:
        chosen = order if size is None else order[:size]
        mask = np.isin(pool["episode"], chosen)
        tr = {k: v[mask] for k, v in pool.items()}
        net, result = train(COLS, width, tr, va, targets(tr["delta"], ref), y_va, epochs=60, patience=6)
        result.update(episodes=int(len(chosen)), pairs=int(mask.sum()), width=width)
        report["runs"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        torch.save({"state": net.state_dict(), "cols": COLS, "width": width},
                   HERE / "results" / f"scale_{len(chosen)}ep_w{width}.pt")
    (HERE / "results" / "scale.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
