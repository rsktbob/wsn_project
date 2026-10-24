"""E06 監督式：作弊版標籤回歸 4 個 Q（Huber），網路同 RL。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E06_ring_se_span_context/train_sup.py --seed 0
    python Algorithm/se/rl/experiments/E06_ring_se_span_context/train_sup.py --seed 0 --sensor-layers 0   # 消融
標籤：results/data/labels（collect.py --policy oracle）；標準化：results/data/norm.npz。
輸出：results/supervised{_l0}/seed{seed}/（log.jsonl、config.json、best.pt）。
驗證集：訓練圖依地圖切 val_frac；早停看驗證 Huber。訓練完用 best.pt 在 20 份驗證快照自己推進，和隨機、瓶頸規則比。
指標（只看至少一個候選標籤 > 0 的交易）：命中率＝選到標籤最大的候選、標籤比例＝選到的標籤 ÷ 最大標籤；
另以「只交換的一般 Δ」算同樣兩個量（和 E03 可比）。
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import DATA, HERE, RESULTS, validate, validation_baseline, validation_items
from model import E06Net, RoundStore, greedy_policy, param_count


def load_labels():
    parts = [dict(np.load(f)) for f in sorted((DATA / "labels").glob("shard*.npz"))]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


class PairData:
    """交易層級資料攤平成 (R × 64)，放在裝置上。"""

    def __init__(self, data, device):
        R, P = data["s_id"].shape
        t = lambda x, dt: torch.as_tensor(np.ascontiguousarray(x), dtype=dt, device=device)  # noqa: E731
        self.r = t(np.repeat(np.arange(R), P), torch.long)
        self.s = t(data["s_id"].reshape(-1), torch.long)
        self.g = t(data["g_id"].reshape(-1), torch.long)
        self.span = t(data["span"].reshape(R * P, 4, 2), torch.long)
        self.cand = t(data["cand"].reshape(R * P, 4, -1), torch.float32)
        self.glob = t(data["glob"].reshape(R * P, -1), torch.float32)
        self.label = t(data["label"].reshape(R * P, 4), torch.float32)
        self.plain = t(data["peek_plain"].reshape(R * P, 4), torch.float32)
        self.map = np.repeat(data["map"], P)

    def inputs(self, store, idx):
        return store.inputs(self.r[idx], self.s[idx], self.g[idx], self.span[idx], self.cand[idx], self.glob[idx])


def evaluate(net, store, pairs, idx, chunk=2048):
    net.eval()
    qs = []
    with torch.no_grad():
        for start in range(0, len(idx), chunk):
            qs.append(net(*pairs.inputs(store, idx[start:start + chunk])))
    q = torch.cat(qs)
    label, plain = pairs.label[idx], pairs.plain[idx]
    out = {"loss": float(F.smooth_l1_loss(q, label).item())}
    pick = q.argmax(dim=1, keepdim=True)
    for name, target in (("label", label), ("plain", plain)):
        best = target.max(dim=1).values
        has = best > 0
        got = target.gather(1, pick).squeeze(1)
        out[f"{name}_hit"] = float((got[has] >= best[has] - 1e-9).float().mean().item())
        out[f"{name}_ratio"] = float((got[has] / best[has]).mean().item())
        out[f"{name}_random_ratio"] = float((target[has].mean(dim=1) / best[has]).mean().item())
    out["choice_share"] = np.round(np.bincount(pick.squeeze(1).cpu().numpy(), minlength=4) / len(idx), 3).tolist()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sensor-layers", type=int, default=None)
    args = parser.parse_args()
    cfg = json.loads((HERE / "config.json").read_text())
    sup = cfg["supervised"]
    net_cfg = dict(cfg["net"])
    if args.sensor_layers is not None:
        net_cfg["sensor_layers"] = args.sensor_layers
    name = "supervised" if net_cfg["sensor_layers"] == cfg["net"]["sensor_layers"] else f"supervised_l{net_cfg['sensor_layers']}"
    out = RESULTS / name / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"supervised": sup, "net": net_cfg, "seed": args.seed}, indent=2))
    log = open(out / "log.jsonl", "a")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    torch.set_num_threads(int(cfg["torch_threads"]))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_labels()
    store = RoundStore(data, device)
    pairs = PairData(data, device)
    maps = np.unique(data["map"])
    val_maps = set(rng.choice(maps, size=max(1, int(round(sup["val_frac"] * len(maps)))), replace=False))
    is_val = np.array([m in val_maps for m in pairs.map])
    train_idx = torch.as_tensor(np.flatnonzero(~is_val), device=device)
    val_idx = torch.as_tensor(np.flatnonzero(is_val), device=device)

    stats = dict(np.load(DATA / "norm.npz"))
    net = E06Net(stats, n_sensor=store.n, **net_cfg).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=sup["lr"], weight_decay=sup["l2"])
    write({"event": "start", "params": param_count(net), "net": net.config, "train_pairs": len(train_idx),
           "val_pairs": len(val_idx), "val_maps": len(val_maps), "device": str(device),
           "label_positive": float((pairs.label > 0).float().mean().item())})

    best, stale = np.inf, 0
    for epoch in range(1, int(sup["max_epochs"]) + 1):
        t0 = time.time()
        net.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        losses = []
        for start in range(0, len(perm), int(sup["batch"])):
            idx = perm[start:start + int(sup["batch"])]
            loss = F.smooth_l1_loss(net(*pairs.inputs(store, idx)), pairs.label[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), sup["grad_clip"])
            opt.step()
            losses.append(loss.item())
        record = {"event": "epoch", "epoch": epoch, "train_loss": float(np.mean(losses)),
                  **{f"val_{k}": v for k, v in evaluate(net, store, pairs, val_idx).items()},
                  "seconds": round(time.time() - t0, 1)}
        write(record)
        if record["val_loss"] < best - 1e-6:
            best, stale = record["val_loss"], 0
            torch.save({"model": net.state_dict(), "net_config": net.config, "epoch": epoch}, out / "best.pt")
        else:
            stale += 1
            if stale >= int(sup["patience"]):
                break

    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    write({"event": "best", "epoch": ckpt["epoch"],
           **{f"train_{k}": v for k, v in evaluate(net, store, pairs, train_idx).items()},
           **{f"val_{k}": v for k, v in evaluate(net, store, pairs, val_idx).items()}})

    items = validation_items(tuple(cfg["val_snapshot_index"]))
    base = validation_baseline(items, cfg["budget"])
    fits, summary = validate(items, greedy_policy(net, device), cfg["budget"])
    random_fit, rule_fit = np.array(base["random"]["fitness"]), np.array(base["bottleneck"]["fitness"])
    write({"event": "rollout", "n": len(items), "mean_diff": float((fits - random_fit).mean()),
           "wins": int((fits > random_fit).sum()), "vs_bottleneck": float((fits - rule_fit).mean()), **summary,
           "random": {k: v for k, v in base["random"].items() if k != "fitness"},
           "bottleneck": {k: v for k, v in base["bottleneck"].items() if k != "fitness"}})
    (out / "TRAIN_DONE").touch()


if __name__ == "__main__":
    main()
