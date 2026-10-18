"""自己推進的檢查（E04 監督式的問題在這裡才看得出來）：12 份驗證快照，預算 1000，
比較單段最終 fitness、每步改善率、選段分布：8 段隨機、作弊版、E05 A／B × {抽樣, 取最大}。

用法（在 2027WSN 根目錄）：python Algorithm/se/rl/experiments/E05_ring_se_seg8_supervised/rollout_check.py
輸出：印出並存 rollout_check.txt
"""
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from e05_common import E04, run_episode, split_maps, SnapshotPool  # noqa: E402,F401
from train import load_policy  # noqa: E402

N_ITEMS = 12


def items():
    _, val = split_maps()
    pool = SnapshotPool(val)
    out = []
    for name in pool.names:
        snaps = pool.get(name)
        out += [snaps[i]["problem"] for i in (1, 4) if i < len(snaps)]
    return out[:N_ITEMS]


def main():
    policies = {"random": lambda i: "random", "oracle": lambda i: "oracle"}
    for net in ("concat", "attention"):
        ckpt = HERE / net / "results" / "seed0" / "best.pt"
        for mode in ("sample", "argmax"):
            policies[f"{net}_{mode}"] = (lambda c, m: (lambda i: load_policy(c, m, seed=900000 + i)))(ckpt, mode)
    lines = []
    blobs = items()
    for name, make in policies.items():
        counts, fits, imp = np.zeros(8), [], []
        for i, blob in enumerate(blobs):
            alg = run_episode(pickle.loads(blob), make(i), 900000 + i, 1000)
            counts += np.sum([r["segment_counts"] for r in alg.round_stats], axis=0)
            fits.append(alg.fitness)
            imp.append(np.mean([r["improve_rate"] for r in alg.round_stats]))
        line = f"{name:18s} fit {np.mean(fits):.6f} improve_rate {np.mean(imp):.3f} seg share {np.round(counts / counts.sum(), 3)}"
        print(line, flush=True)
        lines.append(line)
    (HERE / "rollout_check.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
