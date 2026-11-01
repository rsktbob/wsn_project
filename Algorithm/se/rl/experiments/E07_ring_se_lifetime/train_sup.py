"""E07 監督式（網路 SupNet，每組交易輸出 4 個分數，全部交易都用）。

資料：results/data/b{預算}/{train,val,test}/RLTRAIN*.npz（collect.py；train ＝ 第 1–140 張、val ＝ 141–170、test ＝ 171–200）。
--label 選要學的標籤欄位（欄位說明見 env.py、results/data/README.md）：
    diff_cur（預設）：差異獎勵（看其他交易）；plain_delta：一般 Δ（只看這筆交易）
    （10/25 起拿掉 diff_prev、oracle_choice 兩種標籤；資料檔裡的欄位照舊保留）
--loss huber（預設）：回歸 4 個候選的標籤值
--loss cls：學「標籤最大的那個分數最高」（softmax 交叉熵），同分時目標平分給並列的候選；輸出資料夾加 _cls。
    （10/25 以前有 --loss rank：同分取編號最前面的，已拿掉；舊模型 sup_*_rank 照舊保留。）
--loss soft --tau T：軟標籤，目標 ＝ softmax(標籤 ÷ 這筆最大值 ÷ T)，交叉熵；輸出資料夾加 _soft{T}。
--loss prop：按比例分，目標 ＝ 標籤 ÷ 4 個標籤總和（0 的候選目標就是 0），交叉熵；輸出資料夾加 _prop。
4 個標籤全為 0 的交易：cls、soft、prop 的目標是均勻 [0.25] × 4，huber 是全 0。
任一 loss 後面加 _nz（huber_nz、cls_nz、soft_nz、prop_nz）：拿掉全為 0 的交易不訓練（驗證損失也只算其餘），資料夾再加 _nz
    （huber_nz 的資料夾是 _huber_nz）。
    plain_delta ＝ 只看這筆交易：max(0, f(c1) − f(S)) + max(0, f(c2) − f(G))，不看其他交易。
--attn L：範圍內 sensor 之間加 L 層 Transformer（見 model.py），輸出資料夾加 _attn{L}。
--topk K：範圍內最緊迫的 K 顆 sensor 各自的向量直接接起來（不只平均‖最大），輸出資料夾加 _top{K}。
--full D：全網 100 顆各壓成 D 維、範圍外補 0、照編號接起來（不只平均‖最大），輸出資料夾加 _full{D}。
--maps N：訓練用 train 的前 N 張；val、test 一律用整個資料夾（各 30 張）。
--val：每個 epoch 後算驗證損失，最多 --epochs 個 epoch，連續 patience 個 epoch 沒進步就停，最好的存 best.pt（評估用它）；
       不給 --val 就固定 epoch 數、評估用 final.pt。訓練完在 test 上算一次「猜得準不準」（用 best.pt 或 final.pt）。
例（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E07_ring_se_lifetime/train_sup.py --budget 1000 --maps 70 --val --epochs 20
輸出：results/sup_b{預算}_m{N}_{label}/seed{seed}/（log.jsonl、config.json、epoch{e}.pt、best.pt、final.pt、TRAIN_DONE）。
標準化統計從這 N 張訓練圖估，存成 train/norm_m{N}.npz。
--labels 可直接指一個資料夾（例如 E06 的 shard*.npz；舊欄位名會自動對應）。
指標：交叉熵／Huber、和作弊版一致的比例（作弊版＝選訓練標籤最大的候選，同分都算一致；10/25 以前是一般 Δ 最大、同分比 f(child2)）、選到的標籤 ÷ 最大標籤（label_ratio，只算至少一個標籤 > 0 的交易）、
同樣用一般 Δ 算的 plain_ratio、4 個候選位置被選比例、best_hit（只算至少一個標籤 > 0 的交易，選到標籤最大的比例）。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import DATA, HERE, RESULTS
from model import RoundStore, SupNet, norm_stats, param_count

ROUND_KEYS = ("S", "G", "S_prio", "G_prio", "sensor")
PAIR_KEYS = ("s_id", "g_id", "span", "cand", "glob", "plain_delta")
OLD_NAMES = {"label": "diff_prev", "label_cur": "diff_cur", "oracle": "oracle_choice", "peek_plain": "plain_delta"}
LABELS = ("diff_cur", "plain_delta")


def load_labels(folder, label, maps=None):
    """讀資料夾裡的 RLTRAIN??.npz（maps 給了就只讀 RLTRAIN01–maps）或 E06 的 shard*.npz。
    舊檔的欄位名照 OLD_NAMES 對應。"""
    folder = Path(folder)
    if maps is not None:
        files = [folder / f"RLTRAIN{i:02d}.npz" for i in range(1, maps + 1)]
        missing = [f.name for f in files if not f.exists()]
        if missing:
            raise FileNotFoundError(f"{folder} 缺 {missing}（先用 collect.py 收）")
    else:
        files = sorted(f for f in folder.glob("*.npz") if not f.name.endswith(".tmp.npz") and not f.name.startswith("norm"))
    parts = []
    for f in files:
        with np.load(f) as z:
            raw = {OLD_NAMES.get(k, k): z[k] for k in z.files}
        part = {k: raw[k] for k in ROUND_KEYS + PAIR_KEYS}
        part["target"] = raw[label]
        parts.append(part)
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}, len(files)


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
        self.plain = t(data["plain_delta"].reshape(R * P, 4), torch.float32)
        self.label = t(data["target"].reshape(R * P, 4), torch.float32)
        self.size = R * P
        self.informative = self.label.max(dim=1).values > 0

    def inputs(self, store, idx):
        return store.inputs(self.r[idx], self.s[idx], self.g[idx], self.span[idx], self.cand[idx], self.glob[idx])


def ratio(values, pick):
    """選到的 ÷ 4 個裡最大的（只算最大 > 0 的交易）；另回傳隨機選的同一個量。"""
    best = values.max(dim=1).values
    has = best > 0
    got = values.gather(1, pick[:, None]).squeeze(1)
    return float((got[has] / best[has]).mean().item()), float((values[has].mean(dim=1) / best[has]).mean().item())


def loss_fn(label, loss="huber", tau=0.3):
    loss = loss.removesuffix("_nz")
    if loss == "soft":
        def soft_ce(out, pairs, idx):
            label = pairs.label[idx]
            target = F.softmax(label / label.max(dim=1, keepdim=True).values.clamp(min=1e-12) / tau, dim=1)
            return -(target * F.log_softmax(out, dim=1)).sum(dim=1).mean()
        return soft_ce
    if loss == "prop":
        def prop_ce(out, pairs, idx):
            label = pairs.label[idx]
            total = label.sum(dim=1, keepdim=True)
            target = torch.where(total > 0, label / total.clamp(min=1e-12), torch.full_like(label, 0.25))
            return -(target * F.log_softmax(out, dim=1)).sum(dim=1).mean()
        return prop_ce
    if loss == "cls":
        def tie_ce(out, pairs, idx):
            label = pairs.label[idx]
            top = (label >= label.max(dim=1, keepdim=True).values).float()
            return -(top / top.sum(dim=1, keepdim=True) * F.log_softmax(out, dim=1)).sum(dim=1).mean()
        return tie_ce
    return lambda out, pairs, idx: F.smooth_l1_loss(out, pairs.label[idx])


def metrics(net, store, pairs, idx, label, loss="huber", chunk=8192, tau=0.3):
    if net.config.get("attn"):
        chunk = 1024                            # 注意力矩陣 (chunk × 4 × 頭 × N × N) 才放得進 GPU
    net.eval()
    outs = []
    with torch.no_grad():
        for start in range(0, len(idx), chunk):
            outs.append(net(*pairs.inputs(store, idx[start:start + chunk])))
    out = torch.cat(outs)
    pick = out.argmax(dim=1)
    # 作弊版跟著訓練標籤：選這個標籤最大的候選；同分（含全為 0）時選哪個都算一致
    labels = pairs.label[idx]
    agree = labels.gather(1, pick[:, None]).squeeze(1) >= labels.max(dim=1).values
    oracle = labels.argmax(dim=1)
    plain_ratio, plain_random = ratio(pairs.plain[idx], pick)
    label_ratio, label_random = ratio(pairs.label[idx], pick)
    info = pairs.informative[idx]
    hit = pick[info] == pairs.label[idx][info].argmax(dim=1)
    keep = info if loss.endswith("_nz") else torch.ones_like(info)
    return {"loss": float(loss_fn(label, loss, tau)(out[keep], pairs, idx[keep]).item()),
            "best_hit": float(hit.float().mean().item()), "informative": float(info.float().mean().item()),
            "agree": float(agree.float().mean().item()),
            "label_ratio": label_ratio, "random_label_ratio": label_random,
            "plain_ratio": plain_ratio, "random_plain_ratio": plain_random,
            "choice_share": np.round(np.bincount(pick.cpu().numpy(), minlength=4) / len(idx), 3).tolist(),
            "oracle_share": np.round(np.bincount(oracle[info].cpu().numpy(), minlength=4) / max(int(info.sum()), 1), 3).tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", choices=LABELS, default="diff_cur")
    parser.add_argument("--budget", type=int, default=1000, help="用 results/data/b{預算}")
    parser.add_argument("--maps", type=int, default=140, help="訓練用 train 的前幾張")
    parser.add_argument("--val", action="store_true", help="用 val（第 141–170 張）依驗證損失早停")
    parser.add_argument("--epochs", type=int, default=None, help="（最多）幾個 epoch，預設 config.json")
    parser.add_argument("--loss", choices=tuple(f"{b}{n}" for b in ("huber", "cls", "soft", "prop") for n in ("", "_nz")),
                        default="huber", help="cls：學最大的那個、同分平分；soft：軟標籤；prop：按比例分；_nz：拿掉全為 0 的交易")
    parser.add_argument("--tau", type=float, default=0.3, help="--loss soft 的溫度")
    parser.add_argument("--attn", type=int, default=0, help="範圍內 sensor 之間的 Transformer 層數（0 ＝ 簡單版）")
    parser.add_argument("--topk", type=int, default=0, help="範圍內最緊迫的幾顆 sensor 直接接起來（0 ＝ 只用平均‖最大）")
    parser.add_argument("--full", type=int, default=0, help="全網每顆 sensor 壓成幾維後照編號接起來（0 ＝ 不用）")
    parser.add_argument("--labels", default=None, help="直接指訓練資料夾（不分 train/val/test，例如 E06 的資料）")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    cfg = json.loads((HERE / "config.json").read_text())
    sup = dict(cfg["supervised"])
    if args.epochs is not None:
        sup["epochs"] = args.epochs
    root = DATA / f"b{args.budget}"
    if args.labels:
        train_dir, maps, val_dir, test_dir = Path(args.labels), None, None, None
        norm_path = train_dir.parent / "norm.npz" if train_dir.name == "labels" else train_dir / "norm.npz"
    else:
        train_dir, maps = root / "train", args.maps
        val_dir = root / "val" if args.val else None
        test_dir = root / "test" if (root / "test").exists() else None
        norm_path = train_dir / f"norm_m{maps}.npz"
    tag = (f"sup_b{args.budget}_m{args.maps}_{args.label}" + ({"cls": "_cls", "soft": f"_soft{args.tau:g}", "prop": "_prop", "huber": "_huber"}[args.loss.removesuffix("_nz")]
              + "_nz" if args.loss.endswith("_nz") else {"cls": "_cls", "soft": f"_soft{args.tau:g}", "prop": "_prop"}.get(args.loss, ""))
           + (f"_attn{args.attn}" if args.attn else "") + (f"_top{args.topk}" if args.topk else "")
           + (f"_full{args.full}" if args.full else ""))
    out = Path(args.out) if args.out else RESULTS / tag / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"supervised": sup, "net": cfg["net"], "seed": args.seed, "label": args.label, "loss": args.loss, "tau": args.tau, "attn": args.attn, "topk": args.topk, "full": args.full,
                                                 "train": str(train_dir), "maps": maps, "val": str(val_dir) if val_dir else None,
                                                 "test": str(test_dir) if test_dir else None}, indent=2))
    log = open(out / "log.jsonl", "a")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    torch.set_num_threads(int(cfg["torch_threads"]))
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    data, n_maps = load_labels(train_dir, args.label, maps)
    if not norm_path.exists():
        np.savez(norm_path, **norm_stats(data, seed=args.seed))
    stats = dict(np.load(norm_path))
    store, pairs = RoundStore(data, device), PairData(data, device)
    del data

    def load_eval(folder):
        d, n = load_labels(folder, args.label)
        return RoundStore(d, device), PairData(d, device), n

    val = load_eval(val_dir) if val_dir else None
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    metric_idx = torch.randperm(pairs.size, generator=gen)[:int(sup["metric_pairs"])].to(device)
    train_idx = (pairs.informative.nonzero().squeeze(1) if args.loss.endswith("_nz")
                 else torch.arange(pairs.size, device=device))

    net = SupNet(stats, **cfg["net"], attn=args.attn, topk=args.topk, full=args.full, n=store.n).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=sup["lr"], weight_decay=sup["l2"])
    batch = int(sup["batch"])
    steps = int(sup["epochs"]) * ((len(train_idx) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    write({"event": "start", "params": param_count(net), "label": args.label, "loss": args.loss, "tau": args.tau, "maps": n_maps,
           "pairs": pairs.size, "train_pairs": len(train_idx),
           "val_maps": val[2] if val else 0, "val_pairs": val[1].size if val else 0,
           "device": str(device), "load_seconds": round(time.time() - t0, 1),
           **{f"init_{k}": v for k, v in metrics(net, store, pairs, metric_idx, args.label, args.loss, tau=args.tau).items()}})

    def save(path, epoch):
        torch.save({"kind": net.kind, "model": net.state_dict(), "net_config": net.config, "epoch": epoch}, path)

    criterion = loss_fn(args.label, args.loss, args.tau)
    best, stale = np.inf, 0
    for epoch in range(1, int(sup["epochs"]) + 1):
        t0 = time.time()
        net.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        losses = []
        for start in range(0, len(perm), batch):
            idx = perm[start:start + batch]
            loss = criterion(net(*pairs.inputs(store, idx)), pairs, idx)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), sup["grad_clip"])
            opt.step()
            sched.step()
            losses.append(loss.item())
        save(out / f"epoch{epoch}.pt", epoch)
        record = {"event": "epoch", "epoch": epoch, "train_loss": float(np.mean(losses)),
                  **metrics(net, store, pairs, metric_idx, args.label, args.loss, tau=args.tau)}
        if val:
            val_idx = torch.arange(val[1].size, device=device)
            record.update({f"val_{k}": v for k, v in metrics(net, val[0], val[1], val_idx, args.label, args.loss, tau=args.tau).items()})
        record["seconds"] = round(time.time() - t0, 1)
        write(record)
        if val:
            if record["val_loss"] < best - 1e-6:
                best, stale = record["val_loss"], 0
                save(out / "best.pt", epoch)
            else:
                stale += 1
                if stale >= int(sup["patience"]):
                    break
    save(out / "final.pt", epoch)
    chosen = out / ("best.pt" if val else "final.pt")
    ckpt = torch.load(chosen, map_location=device, weights_only=False)
    if val:
        write({"event": "best", "epoch": ckpt["epoch"], "val_loss": best, "epochs_run": epoch})
    if test_dir:
        net.load_state_dict(ckpt["model"])
        del store, pairs, val, metric_idx, train_idx, perm   # b10000 的 train＋val＋test 同時放 GPU 會 OOM
        torch.cuda.empty_cache()
        test = load_eval(test_dir)
        write({"event": "test", "checkpoint": chosen.name, "epoch": ckpt["epoch"], "test_maps": test[2],
               **{f"test_{k}": v for k, v in
                  metrics(net, test[0], test[1], torch.arange(test[1].size, device=device), args.label,
                                  args.loss, tau=args.tau).items()}})
    (out / "TRAIN_DONE").touch()


if __name__ == "__main__":
    main()
