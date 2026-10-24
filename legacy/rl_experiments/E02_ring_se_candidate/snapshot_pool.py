"""沿用 E01 的訓練／驗證快照（E01_ring_se_segment/data/snapshots，由 E01/snapshots.py 產生）。

E01/snapshots.py 以檔案路徑載入（避免和本資料夾的 env.py 等模組撞名）；只用它的地圖切分與快照路徑。
"""
from __future__ import annotations

import gzip
import importlib.util
import pickle
from pathlib import Path

_E01 = Path(__file__).resolve().parent.parent / "E01_ring_se_segment"
_spec = importlib.util.spec_from_file_location("e01_snapshots", _E01 / "snapshots.py")
_snap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_snap)
split_maps = _snap.split_maps
SNAPSHOT_DIR = _snap.OUT


class SnapshotPool:
    """依需要讀進快照檔（每張圖一個檔），讀過的留在記憶體。"""

    def __init__(self, specs):
        self.names = [s["name"] for s in specs if (SNAPSHOT_DIR / f"{s['name']}.pkl.gz").exists()]
        self.cache = {}

    def get(self, name):
        if name not in self.cache:
            with gzip.open(SNAPSHOT_DIR / f"{name}.pkl.gz", "rb") as handle:
                self.cache[name] = pickle.load(handle)
        return self.cache[name]

    def sample(self, rng):
        name = self.names[int(rng.integers(len(self.names)))]
        snaps = self.get(name)
        snap = snaps[int(rng.integers(len(snaps)))]
        return snap, pickle.loads(snap["problem"])
