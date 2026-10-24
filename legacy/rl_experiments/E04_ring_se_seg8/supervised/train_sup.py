"""監督式診斷：同一個網路（設計 A 或 B），回歸作弊版 8 段只交換的 Δ，輸出 8 個 Q、選 Q 最大。

目標：log(1 + Δ ÷ 該回合所有交易 × 8 段中正 Δ 的中位數)；沒改善為 0。
    不用 RL reward 的上限 5：截斷會讓最好的幾段同分，完美預測也只命中 93%；log 保持大小順序又壓縮長尾。
損失：8 個輸出的 MSE。驗證：訓練圖依地圖切 10% 出來，看
    命中率（選到的段是 Δ 最大的段之一，只算至少一段有改善的交易）、Δ 比例（選到的 Δ ÷ 最大 Δ）。
best.pt 依驗證 Δ 比例選；格式和 RL 的 checkpoint 相同（../train.py load_policy、../evaluate.py 可直接讀）。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E04_ring_se_seg8/supervised/train_sup.py --net attention
輸出：results/{net}/（log.jsonl、best.pt）
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import DATA, NETS, net_class  # noqa: E402

VAL_FRAC = 0.1


def load_labels():
    parts = [np.load(p) for p in sorted((DATA / "labels").glob("shard*.npz"))]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ("states", "delta", "round", "map")}
    target = np.zeros_like(d["delta"])
    order = np.argsort(d["round"], kind="stable")
    rounds, starts = np.unique(d["round"][order], return_index=True)
    for idx in np.split(order, starts[1:]):
        x = d["delta"][idx]
        pos = x[x > 0]
        if len(pos):
            target[idx] = np.log1p(x / np.median(pos))
    d["target"] = target.astype(np.float32)
    return d


def metrics(q, delta):
    """q、delta：(N, 8) numpy。只看至少一段有改善的交易。"""
    has = delta.max(axis=1) > 0
    q, delta = q[has], delta[has]
    choice = q.argmax(axis=1)
    best = delta.max(axis=1)
    picked = delta[np.arange(len(delta)), choice]
    return {"hit": float((picked >= best - 1e-15).mean()), "delta_ratio": float(picked.sum() / best.sum()),
            "random_hit": float((delta >= best[:, None] - 1e-15).mean()),
            "random_delta_ratio": float(delta.mean(axis=1).sum() / best.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", choices=tuple(NETS), required=True)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = HERE / "results" / args.net
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    d = load_labels()
    maps = np.unique(d["map"])
    val_maps = np.random.default_rng(0).choice(maps, size=max(1, int(len(maps) * VAL_FRAC)), replace=False)
    is_val = np.isin(d["map"], val_maps)
    tr, va = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    states = torch.as_tensor(d["states"], device=device)            # float16
    target = torch.as_tensor(d["target"], device=device)
    write({"event": "data", "train": len(tr), "val": len(va), "maps": len(maps), "val_maps": len(val_maps)})

    stats = dict(np.load(DATA / "norm.npz"))
    net = net_class(args.net)(stats).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    cfg = {"net": args.net, "lr": args.lr, "batch": args.batch, "seed": args.seed, "training": "supervised"}

    def predict(idx):
        net.eval()
        with torch.no_grad():
            q = torch.cat([net(states[torch.as_tensor(idx[i:i + 4096], device=device)].float())
                           for i in range(0, len(idx), 4096)])
        net.train()
        return q

    best, wait = -np.inf, 0
    rng = np.random.default_rng(args.seed)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = rng.permutation(tr)
        losses = []
        for i in range(0, len(perm), args.batch):
            idx = torch.as_tensor(perm[i:i + args.batch], device=device)
            loss = F.mse_loss(net(states[idx].float()), target[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            losses.append(loss.item())
        q = predict(va)
        val_loss = float(F.mse_loss(q, target[torch.as_tensor(va, device=device)]).item())
        m = metrics(q.cpu().numpy(), d["delta"][va])
        mt = metrics(predict(tr[:50000]).cpu().numpy(), d["delta"][tr[:50000]])
        write({"event": "epoch", "epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss,
               **{f"val_{k}": v for k, v in m.items()}, "train_hit": mt["hit"], "train_delta_ratio": mt["delta_ratio"],
               "seconds": round(time.time() - t0, 1)})
        if m["delta_ratio"] > best:
            best, wait = m["delta_ratio"], 0
            torch.save({"online": net.state_dict(), "cfg": cfg, "updates": epoch, "extra": m}, out / "best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                break
    write({"event": "done", "best_val_delta_ratio": best})


if __name__ == "__main__":
    main()
