"""E03 容量測試：DeepSets 打分網路（解碼後 token），監督式預測 4 個候選只交換後的好壞。

資料：data/labels/shard*.npz（collect.py）。驗證圖和 E02 scale.py 相同（190 張訓練圖中 seed 0 抽 15%＝28 張）。
目標、指標同 E02（Δ>0 → 0.5+0.5×百分位；一致率、命中率、Δ 比例）。另列兩個不學習的參考：
    隨機；瓶頸規則（有效差異 sensor 在 S、G 中緊迫度總和最大）。
停損：驗證命中率要明顯超過 E02 的 65.4%。
輸出：results/capacity.json、results/deepsets_w{width}.pt
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
E03 = HERE.parent
sys.path.insert(0, str(E03.parents[4]))
sys.path.insert(0, str(E03))
sys.path.insert(0, str(HERE))
from env import TIGHT  # noqa: E402
from model import DeepSetsScorer, gather_tokens  # noqa: E402
from snapshot_pool import split_maps  # noqa: E402

SEED = 0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROUND_KEYS = ("S", "G", "S_genes", "G_genes", "sensor")
PAIR_KEYS = ("s_id", "g_id", "spans", "outside", "position", "global", "delta", "f2", "choice")
WIDTHS = (64, 256)


def load():
    out = {k: [] for k in ROUND_KEYS + PAIR_KEYS + ("round", "map")}
    offset = 0
    for path in sorted((E03 / "data" / "labels").glob("shard*.npz")):
        part = np.load(path)
        for k in ROUND_KEYS + PAIR_KEYS + ("map",):
            out[k].append(part[k])
        out["round"].append(part["round"] + offset)
        offset += len(part["S"])
    return {k: np.concatenate(v) for k, v in out.items()}


def val_maps():
    train_specs, _ = split_maps()
    names = sorted(s["name"] for s in train_specs)
    return set(np.random.default_rng(0).choice(names, size=int(round(0.15 * len(names))), replace=False))


def targets(delta, positive_ref):
    ref = np.sort(positive_ref)
    pct = np.searchsorted(ref, delta, side="right") / len(ref)
    return np.where(delta > 0, 0.5 + 0.5 * pct, 0.0).astype(np.float32)


def metrics(scores, delta, choice):
    pick = scores.argmax(axis=1)
    rows = np.arange(len(pick))
    any_pos = delta.max(axis=1) > 0
    return {"agree": float((pick == choice).mean()), "hit": float((delta[rows, pick] > 0)[any_pos].mean()),
            "delta_ratio": float((delta[rows, pick][any_pos] / delta.max(axis=1)[any_pos]).mean())}


def to_store(data):
    store = {k: torch.as_tensor(data[k]).to(DEVICE) for k in ROUND_KEYS + ("round", "s_id", "g_id", "spans",
                                                                           "outside", "position", "global")}
    return store


def scores_of(net, store, rows, batch=4096):
    out = []
    net.eval()
    with torch.no_grad():
        for idx in torch.split(rows, batch):
            out.append(net(*gather_tokens(store, idx)).cpu())
    return torch.cat(out).numpy()


def bottleneck_scores(store, rows, batch=8192):
    out = []
    for idx in torch.split(rows, batch):
        tokens, mask, _ = gather_tokens(store, idx)
        r = store["round"][idx].long()[:, None, None]
        spans = store["spans"][idx].long()
        sensors = (spans[:, :, 1:2] + torch.arange(mask.shape[-1], device=DEVICE)).clamp(max=store["S"].shape[2] - 1)
        sg = store["S_genes"][r, store["s_id"][idx].long()[:, None, None], sensors]
        gg = store["G_genes"][r, store["g_id"][idx].long()[:, None, None], sensors]
        effective = (sg != gg).any(-1) & ((tokens[..., 0] > 0) | (tokens[..., 10] > 0)) & mask
        tight = tokens[..., TIGHT] + tokens[..., 10 + TIGHT]
        out.append((tight * effective).sum(-1).cpu())
    s = torch.cat(out).numpy()
    return s + 1e-6 * np.random.default_rng(1).random(s.shape)


def train(width, store, tr_rows, va_rows, y_tr, y_va, epochs=60, patience=6, batch=512):
    torch.manual_seed(SEED)
    net = DeepSetsScorer(width=width).to(DEVICE)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Ytr, Yva = torch.as_tensor(y_tr, device=DEVICE), torch.as_tensor(y_va, device=DEVICE)
    gen = torch.Generator(device="cpu").manual_seed(SEED)
    best, best_state, wait = np.inf, None, 0
    for epoch in range(epochs):
        net.train()
        perm = torch.randperm(len(tr_rows), generator=gen).to(DEVICE)
        for idx in torch.split(perm, batch):
            loss = ((net(*gather_tokens(store, tr_rows[idx])) - Ytr[idx]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        val = float(((torch.as_tensor(scores_of(net, store, va_rows), device=DEVICE) - Yva) ** 2).mean())
        print(f"  w{width} epoch {epoch + 1} val_mse {val:.5f}", flush=True)
        if val < best - 1e-6:
            best, best_state, wait = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= patience:
                break
    net.load_state_dict(best_state)
    s_tr, s_va = scores_of(net, store, tr_rows), scores_of(net, store, va_rows)
    return net, {"params": sum(p.numel() for p in net.parameters()), "epochs": epoch + 1, "val_mse": best,
                 "train_mse": float(((s_tr - y_tr) ** 2).mean()), "s_tr": s_tr, "s_va": s_va}


def main():
    data = load()
    is_va = np.isin(data["map"], sorted(val_maps()))
    store = to_store(data)
    tr_rows = torch.as_tensor(np.flatnonzero(~is_va), device=DEVICE)
    va_rows = torch.as_tensor(np.flatnonzero(is_va), device=DEVICE)
    d_tr, d_va = data["delta"][~is_va], data["delta"][is_va]
    c_tr, c_va = data["choice"][~is_va], data["choice"][is_va]
    ref = d_tr[d_tr > 0]
    y_tr, y_va = targets(d_tr, ref), targets(d_va, ref)

    report = {"pairs_train": int(len(tr_rows)), "pairs_val": int(len(va_rows)),
              "episodes_rounds": int(len(data["S"])), "any_improve_val": float((d_va.max(axis=1) > 0).mean()),
              "random": metrics(np.random.default_rng(1).random(d_va.shape), d_va, c_va),
              "bottleneck": metrics(bottleneck_scores(store, va_rows), d_va, c_va), "runs": []}
    print(json.dumps({k: v for k, v in report.items() if k != "runs"}, ensure_ascii=False), flush=True)
    for width in WIDTHS:
        net, result = train(width, store, tr_rows, va_rows, y_tr, y_va)
        result["train"] = metrics(result.pop("s_tr"), d_tr, c_tr)
        result["val"] = metrics(result.pop("s_va"), d_va, c_va)
        result["width"] = width
        report["runs"].append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        torch.save({"state": net.state_dict(), "width": width}, HERE / "results" / f"deepsets_w{width}.pt")
    (HERE / "results" / "capacity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
