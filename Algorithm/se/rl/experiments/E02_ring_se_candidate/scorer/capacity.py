"""E02 容量測試：每個候選共用一個打分網路，監督式預測 4 個候選只交換後的好壞，看能不能學會作弊版的選法。

目標 y_k：Δ_k ≤ 1e-12 → 0；Δ_k > 0 → 0.5 + 0.5 × （Δ_k 在訓練集所有正 Δ 中的百分位）。損失：4 個候選的 MSE。
依地圖切 15% 當驗證（沒見過的地圖）。比較輸入組合 × 網路寬度。
指標（驗證集）：
    一致率：選到作弊版那個候選的比例（隨機 25%）
    命中率：4 個候選裡有改善時，選到的那個也有改善的比例
    Δ 比例：選到的 Δ ÷ 最大 Δ（有改善的交易平均）
輸出：results/capacity.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
E02 = HERE.parent
sys.path.insert(0, str(E02.parents[4]))
sys.path.insert(0, str(E02))
from env import FEATURE_GROUPS  # noqa: E402

torch.set_num_threads(4)
SEED = 0
FEATURE_SETS = {
    "位置＋全域（8 維）": FEATURE_GROUPS["position"] + FEATURE_GROUPS["global"],
    "完整（25 維）": sum(FEATURE_GROUPS.values(), []),
}
RUNS = [("位置＋全域（8 維）", 256), ("完整（25 維）", 64), ("完整（25 維）", 256), ("完整（25 維）", 1024)]


def load():
    parts = [np.load(p) for p in sorted((E02 / "data" / "labels").glob("shard*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("X", "delta", "f2", "choice", "map")}


def targets(delta, positive_ref):
    ref = np.sort(positive_ref)
    pct = np.searchsorted(ref, delta, side="right") / len(ref)
    return np.where(delta > 0, 0.5 + 0.5 * pct, 0.0).astype(np.float32)


def scorer(in_dim, width):
    return nn.Sequential(nn.Linear(in_dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))


def metrics(scores, data):
    pick = scores.argmax(axis=1)
    rows = np.arange(len(pick))
    delta, choice = data["delta"], data["choice"]
    any_pos = delta.max(axis=1) > 0
    hit = (delta[rows, pick] > 0)[any_pos].mean()
    ratio = (delta[rows, pick][any_pos] / delta.max(axis=1)[any_pos]).mean()
    return {"agree": float((pick == choice).mean()), "hit": float(hit), "delta_ratio": float(ratio)}


def train(cols, width, tr, va, y_tr, y_va, epochs=40, patience=5):
    torch.manual_seed(SEED)
    net = scorer(len(cols), width)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xtr = torch.as_tensor(tr["X"][:, :, cols]); Ytr = torch.as_tensor(y_tr)
    Xva = torch.as_tensor(va["X"][:, :, cols]); Yva = torch.as_tensor(y_va)
    rng = np.random.default_rng(SEED)
    best, best_state, wait = np.inf, None, 0
    for epoch in range(epochs):
        net.train()
        for idx in np.array_split(rng.permutation(len(Xtr)), max(1, len(Xtr) // 512)):
            loss = ((net(Xtr[idx]).squeeze(-1) - Ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            val = float(((net(Xva).squeeze(-1) - Yva) ** 2).mean())
        if val < best - 1e-6:
            best, best_state, wait = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    with torch.no_grad():
        s_tr = net(Xtr).squeeze(-1).numpy(); s_va = net(Xva).squeeze(-1).numpy()
        train_loss = float(((net(Xtr).squeeze(-1) - Ytr) ** 2).mean())
    return net, {"params": sum(p.numel() for p in net.parameters()), "epochs": epoch + 1,
                 "train_mse": train_loss, "val_mse": best, "train": metrics(s_tr, tr), "val": metrics(s_va, va)}


def main():
    data = load()
    maps = np.unique(data["map"])
    rng = np.random.default_rng(SEED)
    val_maps = set(rng.choice(maps, size=max(1, int(round(0.15 * len(maps)))), replace=False))
    is_va = np.array([m in val_maps for m in data["map"]])
    tr = {k: v[~is_va] for k, v in data.items()}
    va = {k: v[is_va] for k, v in data.items()}
    ref = tr["delta"][tr["delta"] > 0]
    y_tr, y_va = targets(tr["delta"], ref), targets(va["delta"], ref)

    report = {"pairs_train": int(len(tr["X"])), "pairs_val": int(len(va["X"])), "maps_train": int(len(maps) - len(val_maps)),
              "maps_val": len(val_maps), "any_improve_val": float((va["delta"].max(axis=1) > 0).mean())}
    rand = np.random.default_rng(1).random(va["delta"].shape)
    report["random"] = metrics(rand, va)
    report["max_effective_heuristic"] = metrics(va["X"][:, :, 4:8] @ np.array([0, 1, 2, 3], dtype=np.float32)
                                                + 1e-3 * rand, va)
    print(json.dumps({k: v for k, v in report.items()}, ensure_ascii=False), flush=True)
    report["runs"] = []
    for name, width in RUNS:
        net, result = train(FEATURE_SETS[name], width, tr, va, y_tr, y_va)
        result.update(features=name, width=width)
        report["runs"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        torch.save({"state": net.state_dict(), "cols": FEATURE_SETS[name], "width": width},
                   HERE / "results" / f"scorer_{len(FEATURE_SETS[name])}d_w{width}.pt")
    (HERE / "results" / "capacity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
