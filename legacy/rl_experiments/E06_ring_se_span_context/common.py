"""E06 共用：路徑設定、快照（沿用 E01 的訓練／驗證快照）、跑一段最佳化。"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ROOT)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pool = load_module("e03_snapshot_pool", HERE.parent / "E03_ring_se_decoded" / "snapshot_pool.py")
SnapshotPool, split_maps = _pool.SnapshotPool, _pool.split_maps

RESULTS = HERE / "results"          # symlink → runs/2027WSN/rl/E06_ring_se_span_context
DATA = RESULTS / "data"


def run_episode(problem, policy, seed, budget, recorder=None):
    """原版 ring_se 參數（n8 h4 w8 μ0.4），每組交易 4 個預抽候選由 policy 選。"""
    from env import Ring_SE_E06
    algorithm = Ring_SE_E06(problem, policy=policy, recorder=recorder, n=8, h=4, w=8, mu=0.4, seed=seed)
    algorithm.run(problem, budget=budget)
    return algorithm


def validation_items(snapshot_index=(1, 4)):
    """10 張驗證圖 × 快照 snapshot_index → [(map, segment, problem_blob)]。"""
    _, val = split_maps()
    pool = SnapshotPool(val)
    items = []
    for name in pool.names:
        snaps = pool.get(name)
        for idx in snapshot_index:
            if idx < len(snaps):
                items.append((name, snaps[idx]["segment"], snaps[idx]["problem"]))
    return items


def summarize_rounds(stats):
    counts = np.sum([r["segment_counts"] for r in stats], axis=0)
    return {"improve_rate": float(np.mean([r["improve_rate"] for r in stats])),
            "mean_reward": float(np.mean([r["mean_reward"] for r in stats])),
            "segment_share": np.round(counts / max(1, counts.sum()), 3).tolist(),
            "distinct_per_searcher": float(np.mean([r["distinct_per_searcher"] for r in stats]))}


def validate(items, policy, budget):
    """每份驗證快照跑一段最佳化（seed 900000 + i，各方法相同）→ 最終 fitness、回合統計摘要。"""
    fits, stats = [], []
    for i, (_, _, blob) in enumerate(items):
        algorithm = run_episode(pickle.loads(blob), policy, 900000 + i, budget)
        fits.append(float(algorithm.fitness))
        stats += algorithm.round_stats
    return np.array(fits), summarize_rounds(stats)


def validation_baseline(items, budget):
    """隨機與瓶頸規則在驗證快照上的結果（算一次存檔，各訓練共用）。"""
    path = DATA / f"val_baseline_{len(items)}_{budget}.json"
    if path.exists():
        return json.loads(path.read_text())
    out = {}
    for name in ("random", "bottleneck"):
        fits, summary = validate(items, name, budget)
        out[name] = {"fitness": fits.tolist(), **summary}
    DATA.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


__all__ = ["HERE", "ROOT", "RESULTS", "DATA", "SnapshotPool", "split_maps", "load_module",
           "run_episode", "validation_items", "validate", "validation_baseline",
           "summarize_rounds"]
