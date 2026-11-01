"""E10 監督式訓練：OutNet（E07 的輸入，範圍外也池化）。資料、標籤、loss、驗證早停、指標都沿用 E07 train_sup。

資料：E07 results/data/b{預算}（不另外收）。
用法（2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E10_ring_se_outside/train_out.py --budget 1000 --maps 140 \
        --label plain_delta --loss soft --tau 1 --val --epochs 20
輸出：E10 results/out_b{預算}_m{N}_{label}{_soft{τ}…}/seed{seed}/（log.jsonl、config.json、best.pt、final.pt、TRAIN_DONE）。
test 另外算「同分算對」的準確度（全部交易／正值交易），和 E08 一樣。
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
E07 = HERE.parent / "E07_ring_se_lifetime"
sys.path.insert(0, str(E07))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import DATA as E07_DATA  # noqa: E402
from model import RoundStore, norm_stats  # noqa: E402
from outnet import OutNet  # noqa: E402
from train_sup import LABELS, PairData, load_labels, loss_fn, metrics  # noqa: E402

RESULTS = HERE / "results"
LOSSES = tuple(f"{b}{n}" for b in ("huber", "cls", "soft", "prop") for n in ("", "_nz"))


def tie_accuracy(net, store, pairs):
    """同分算對：全部交易、正值交易（至少一個標籤 > 0）選到最大的比例。"""
    net.eval()
    picks = []
    with torch.no_grad():
        for s in range(0, pairs.size, 8192):
            idx = torch.arange(s, min(s + 8192, pairs.size), device=pairs.label.device)
            picks.append(net(*pairs.inputs(store, idx)).argmax(1))
    pick = torch.cat(picks)
    best = pairs.label.max(1).values
    ok = pairs.label.gather(1, pick[:, None]).squeeze(1) >= best
    return float(ok.float().mean()), float(ok[best > 0].float().mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", choices=LABELS, default="plain_delta")
    parser.add_argument("--loss", choices=LOSSES, default="soft")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--maps", type=int, default=140)
    parser.add_argument("--val", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()

    sup = json.loads((E07 / "config.json").read_text())["supervised"]
    base = args.loss.removesuffix("_nz")
    suffix = {"huber": "", "cls": "_cls", "soft": f"_soft{args.tau:g}", "prop": "_prop"}[base]
    if args.loss.endswith("_nz"):
        suffix = (suffix or "_huber") + "_nz"
    if args.epochs != 20:                       # 餘弦排程跟著 epoch 數，所以是重新訓練、另開資料夾
        suffix += f"_e{args.epochs}"
    out = RESULTS / f"out_b{args.budget}_m{args.maps}_{args.label}{suffix}" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "supervised": sup}, indent=2))
    log = open(out / "log.jsonl", "a")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    t0 = time.time()
    data, n_maps = load_labels(E07_DATA / f"b{args.budget}" / "train", args.label, args.maps)
    stats = norm_stats(data, seed=args.seed)
    store, pairs = RoundStore(data, device), PairData(data, device)
    del data
    val = None
    if args.val:
        d, _ = load_labels(E07_DATA / f"b{args.budget}" / "val", args.label)
        val = (RoundStore(d, device), PairData(d, device))
        del d
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    metric_idx = torch.randperm(pairs.size, generator=gen)[:int(sup["metric_pairs"])].to(device)
    train_idx = (pairs.informative.nonzero().squeeze(1) if args.loss.endswith("_nz")
                 else torch.arange(pairs.size, device=device))

    net = OutNet(stats).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=sup["lr"], weight_decay=sup["l2"])
    batch = int(sup["batch"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * ((len(train_idx) + batch - 1) // batch))
    criterion = loss_fn(args.label, args.loss, args.tau)
    write({"event": "start", "params": sum(p.numel() for p in net.parameters()), "maps": n_maps,
           "pairs": pairs.size, "train_pairs": len(train_idx), "load_seconds": round(time.time() - t0, 1)})

    def save(path, epoch):
        torch.save({"kind": net.kind, "model": net.state_dict(), "net_config": net.config, "epoch": epoch}, path)

    best, stale = np.inf, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        net.train()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        losses = []
        for start in range(0, len(perm), batch):
            idx = perm[start:start + batch]
            loss = criterion(net(*pairs.inputs(store, idx)), pairs, idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), float(sup["grad_clip"]))
            opt.step()
            sched.step()
            losses.append(loss.item())
        save(out / f"epoch{epoch}.pt", epoch)
        record = {"event": "epoch", "epoch": epoch, "train_loss": float(np.mean(losses)),
                  **metrics(net, store, pairs, metric_idx, args.label, args.loss, tau=args.tau)}
        if val:
            record.update({f"val_{k}": v for k, v in metrics(net, val[0], val[1], torch.arange(val[1].size, device=device),
                                                             args.label, args.loss, tau=args.tau).items()})
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
    net.load_state_dict(ckpt["model"])
    del store, pairs, val
    torch.cuda.empty_cache()
    d, n = load_labels(E07_DATA / f"b{args.budget}" / "test", args.label)
    store, pairs = RoundStore(d, device), PairData(d, device)
    acc_all, acc_pos = tie_accuracy(net, store, pairs)
    write({"event": "test", "checkpoint": chosen.name, "epoch": ckpt["epoch"], "test_maps": n,
           "test_acc_all": acc_all, "test_acc_pos": acc_pos,
           **{f"test_{k}": v for k, v in metrics(net, store, pairs, torch.arange(pairs.size, device=device),
                                                 args.label, args.loss, tau=args.tau).items()}})
    (out / "TRAIN_DONE").touch()


if __name__ == "__main__":
    main()
