"""E05 共用：沿用 E04 的環境、輸入、段編碼器、快照、作弊版標籤與標準化統計（E04_ring_se_seg8）。"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
E04 = HERE.parent / "E04_ring_se_seg8"
if str(E04) not in sys.path:
    sys.path.append(str(E04))   # 放最後：E05 自己的 train.py 優先於 E04 的

from common import DATA, ROOT, SnapshotPool, run_episode, split_maps  # noqa: E402,F401  （E04 的 common；會設 cwd。本檔不能叫 common.py，會撞名）

NETS = {"concat": ("concat", "ConcatScorer"), "attention": ("attention", "AttentionScorer")}


def net_class(net):
    folder, cls = NETS[net]
    spec = importlib.util.spec_from_file_location(f"e05_{folder}_model", HERE / folder / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, cls)


__all__ = ["HERE", "E04", "DATA", "ROOT", "NETS", "SnapshotPool", "run_episode", "split_maps", "net_class"]
