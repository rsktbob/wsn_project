"""E04 共用：路徑設定、快照（沿用 E01 的訓練／驗證快照）、跑一段最佳化、載入網路類別。"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
for path in (str(ROOT), str(HERE)):
    if path not in sys.path:
        sys.path.insert(0, path)
os.chdir(ROOT)


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_pool = _load("e03_snapshot_pool", HERE.parent / "E03_ring_se_decoded" / "snapshot_pool.py")
SnapshotPool, split_maps = _pool.SnapshotPool, _pool.split_maps

from env import Ring_SE_E04  # noqa: E402

NETS = {"concat": ("dueling_concat", "ConcatDueling"), "attention": ("attention", "AttentionDueling")}
DATA = HERE / "data"


def net_class(net):
    folder, cls = NETS[net]
    return getattr(_load(f"e04_{folder}_model", HERE / folder / "model.py"), cls)


def run_episode(problem, policy, seed, budget, recorder=None):
    """ring_se 參數（n8 w8 μ0.4），h=8 整段。"""
    algorithm = Ring_SE_E04(problem, policy=policy, recorder=recorder, n=8, w=8, mu=0.4, seed=seed)
    algorithm.run(problem, budget=budget)
    return algorithm


__all__ = ["HERE", "ROOT", "DATA", "NETS", "SnapshotPool", "split_maps", "net_class", "run_episode"]
