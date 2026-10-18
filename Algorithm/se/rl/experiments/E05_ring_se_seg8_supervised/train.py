"""E05 監督式訓練（兩個網路共用；只換網路）：組內排序（listwise）交叉熵。

標籤：E04 收好的 8 段作弊標籤（E04_ring_se_seg8/data/labels，作弊版推進、800 episode）。
    p*_k ＝ 1/m（Δ_k 是 8 段中最大的 m 段之一），否則 0；8 段都沒改善的交易不拿來訓練。
損失：−Σ_k p*_k · log softmax(s)_k。
驗證（訓練圖依地圖切 10%，同 E04 診斷）：
    交叉熵；取最大的命中率與 Δ 比例；抽樣的期望命中率與期望 Δ 比例；
    選段分布（取最大、抽樣）和作弊版選段分布的總變異距離。
best.pt 依驗證交叉熵選（抽樣直接用 softmax 的機率，交叉熵就是在量這個機率準不準）。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E05_ring_se_seg8_supervised/train.py --net attention
輸出：{concat|attention}/results/seed{seed}/（log.jsonl、best.pt）
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e05_common import DATA, HERE, NETS, net_class  # noqa: E402

VAL_FRAC = 0.1
TIE = 1e-15


def load_labels():
    parts = [np.load(p) for p in sorted((DATA / "labels").glob("shard*.npz"))]
    d = {k: np.concatenate([p[k] for p in parts]) for k in ("states", "delta", "map")}
    best = d["delta"].max(axis=1)
    keep = best > 0
    is_best = d["delta"] >= best[:, None] - TIE
    target = is_best / is_best.sum(axis=1, keepdims=True)
    return {"states": d["states"][keep], "delta": d["delta"][keep], "map": d["map"][keep],
            "target": target[keep].astype(np.float32), "dropped": int((~keep).sum())}


def metrics(logits, delta, target):
    """logits、delta、target：(N, 8) numpy。"""
    z = logits - logits.max(axis=1, keepdims=True)
    prob = np.exp(z)
    prob /= prob.sum(axis=1, keepdims=True)
    best = delta.max(axis=1)
    is_best = delta >= best[:, None] - TIE
    choice = logits.argmax(axis=1)
    rows = np.arange(len(delta))
    oracle_share = target.mean(axis=0)
    argmax_share = np.bincount(choice, minlength=8) / len(choice)
    sample_share = prob.mean(axis=0)
    return {"ce": float(-(target * np.log(prob + 1e-12)).sum(axis=1).mean()),
            "argmax_hit": float(is_best[rows, choice].mean()),
            "argmax_delta_ratio": float(delta[rows, choice].sum() / best.sum()),
            "sample_hit": float((prob * is_best).sum(axis=1).mean()),
            "sample_delta_ratio": float((prob * delta).sum() / best.sum()),
            "random_hit": float(is_best.mean()), "random_delta_ratio": float(delta.mean(axis=1).sum() / best.sum()),
            "argmax_tv": float(0.5 * np.abs(argmax_share - oracle_share).sum()),
            "sample_tv": float(0.5 * np.abs(sample_share - oracle_share).sum()),
            "argmax_share": np.round(argmax_share, 3).tolist(), "sample_share": np.round(sample_share, 3).tolist(),
            "oracle_share": np.round(oracle_share, 3).tolist()}


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
    out = HERE / NETS[args.net][0] / "results" / f"seed{args.seed}"
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
    net = net_class(args.net)(dict(np.load(DATA / "norm.npz"))).to(device)
    write({"event": "data", "train": len(tr), "val": len(va), "dropped_no_improvement": d["dropped"],
           "maps": len(maps), "val_maps": len(val_maps), "params": sum(p.numel() for p in net.parameters())})
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    cfg = {"net": args.net, "lr": args.lr, "batch": args.batch, "seed": args.seed, "training": "listwise_ce"}

    def logits_of(idx):
        net.eval()
        with torch.no_grad():
            out_ = torch.cat([net(states[torch.as_tensor(idx[i:i + 4096], device=device)].float())
                              for i in range(0, len(idx), 4096)])
        net.train()
        return out_.cpu().numpy()

    best, wait = np.inf, 0
    rng = np.random.default_rng(args.seed)
    sub = tr[:50000]
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = rng.permutation(tr)
        losses = []
        for i in range(0, len(perm), args.batch):
            idx = torch.as_tensor(perm[i:i + args.batch], device=device)
            loss = -(target[idx] * F.log_softmax(net(states[idx].float()), dim=1)).sum(dim=1).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            losses.append(loss.item())
        m = metrics(logits_of(va), d["delta"][va], d["target"][va])
        mt = metrics(logits_of(sub), d["delta"][sub], d["target"][sub])
        write({"event": "epoch", "epoch": epoch, "train_loss": float(np.mean(losses)),
               **{f"val_{k}": v for k, v in m.items()}, "train_ce": mt["ce"], "train_argmax_hit": mt["argmax_hit"],
               "seconds": round(time.time() - t0, 1)})
        if m["ce"] < best:
            best, wait = m["ce"], 0
            torch.save({"online": net.state_dict(), "cfg": cfg, "epoch": epoch, "extra": m}, out / "best.pt")
        else:
            wait += 1
            if wait >= args.patience:
                break
    write({"event": "done", "best_val_ce": best})


def load_policy(path, mode, seed=0, device="cpu"):
    """讀 checkpoint，回傳策略 callable(states) -> actions。mode：sample（依 softmax 機率抽）／argmax。"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = net_class(ckpt["cfg"]["net"])().to(device)
    net.load_state_dict(ckpt["online"])
    net.eval()
    rng = np.random.default_rng(seed)

    def policy(states):
        with torch.no_grad():
            logits = net(torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)).cpu().numpy()
        if mode == "argmax":
            return logits.argmax(axis=1)
        prob = np.exp(logits - logits.max(axis=1, keepdims=True))
        cum = np.cumsum(prob / prob.sum(axis=1, keepdims=True), axis=1)
        return np.minimum((cum < rng.random((len(cum), 1))).sum(axis=1), cum.shape[1] - 1)
    return policy


if __name__ == "__main__":
    main()
