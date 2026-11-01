"""E13 監督式訓練：Net13（E12 結構、新輸入）。標籤、loss、驗證早停、指標都沿用 E07 train_sup／E12 train_relflag。

資料（同檔名、逐回合對齊）：E13 results/data/b{預算}（collect13.py 收的新輸入）＋ E07 results/data/b{預算}（標籤 target、plain_delta）
    ＋ E08 results/data/b{預算}（②③⑤）。
用法（2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E13_ring_se_newinput/train13.py --budget 1000 --maps 140 \
        --label plain_delta --loss soft --tau 1 --val --epochs 20
輸出：E13 results/e13_b{預算}_m{N}_{label}{_soft{τ}…}/seed{seed}/（log.jsonl、config.json、best.pt、final.pt、TRAIN_DONE）。
評估分塊大小用 E12_CHUNK（預設 2048）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
for p in (EXP / "E07_ring_se_lifetime", EXP / "E08_ring_se_relations", EXP / "E12_ring_se_relflag", HERE):
    sys.path.insert(0, str(p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from common import DATA as E07_DATA  # noqa: E402
from net13 import Net13, Store13, norm_stats  # noqa: E402
from train_relflag import CHUNK, DATA as E08_DATA, LOSSES, RelPairs, load_extras, map_names, tie_accuracy  # noqa: E402
from train_sup import LABELS, loss_fn, metrics  # noqa: E402

DATA = HERE / "results" / "data"
RESULTS = HERE / "results"
INPUT_KEYS = ("S", "G", "sensor", "s_id", "g_id", "span", "cand", "glob")


def load_split(budget, split, label, maps=None):
    """E13 輸入＋E07 標籤＋E08 ②③⑤ → (data, extras, 張數)。"""
    names = map_names(E07_DATA / f"b{budget}" / split, maps)
    parts = []
    for name in names:
        with np.load(DATA / f"b{budget}" / split / f"{name}.npz") as z, \
                np.load(E07_DATA / f"b{budget}" / split / f"{name}.npz") as ref:
            if not np.array_equal(z["span"], ref["span"]):
                raise RuntimeError(f"{name}：E13 輸入和 E07 標籤沒對齊")
            part = {k: z[k] for k in INPUT_KEYS}
            part["plain_delta"] = ref["plain_delta"]
            part["target"] = ref[label]
        parts.append(part)
    data = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return data, load_extras(E08_DATA / f"b{budget}" / split, names), len(names)


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

    sup = json.loads((EXP / "E07_ring_se_lifetime" / "config.json").read_text())["supervised"]
    base = args.loss.removesuffix("_nz")
    suffix = {"huber": "", "cls": "_cls", "soft": f"_soft{args.tau:g}", "prop": "_prop"}[base]
    if args.loss.endswith("_nz"):
        suffix = (suffix or "_huber") + "_nz"
    out = RESULTS / f"e13_b{args.budget}_m{args.maps}_{args.label}{suffix}" / f"seed{args.seed}"
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
    data, extras, n_maps = load_split(args.budget, "train", args.label, args.maps)
    stats = norm_stats(data, seed=args.seed)
    swap = extras["swap"].reshape(-1, extras["swap"].shape[-1]).astype(np.float32)
    stats.update(swap_mean=swap.mean(0), swap_std=swap.std(0))
    store, pairs = Store13(data, device), RelPairs(data, extras, device)
    del data, extras, swap
    val = None
    if args.val:
        d, e, _ = load_split(args.budget, "val", args.label)
        val = (Store13(d, device), RelPairs(d, e, device))
        del d, e
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    metric_idx = torch.randperm(pairs.size, generator=gen)[:int(sup["metric_pairs"])].to(device)
    train_idx = (pairs.informative.nonzero().squeeze(1) if args.loss.endswith("_nz")
                 else torch.arange(pairs.size, device=device))

    net = Net13(stats).to(device)
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
                  **metrics(net, store, pairs, metric_idx, args.label, args.loss, tau=args.tau, chunk=CHUNK)}
        if val:
            record.update({f"val_{k}": v for k, v in metrics(net, val[0], val[1], torch.arange(val[1].size, device=device),
                                                             args.label, args.loss, tau=args.tau, chunk=CHUNK).items()})
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
    d, e, n = load_split(args.budget, "test", args.label)
    store, pairs = Store13(d, device), RelPairs(d, e, device)
    acc_all, acc_pos = tie_accuracy(net, store, pairs)
    write({"event": "test", "checkpoint": chosen.name, "epoch": ckpt["epoch"], "test_maps": n,
           "test_acc_all": acc_all, "test_acc_pos": acc_pos,
           **{f"test_{k}": v for k, v in metrics(net, store, pairs, torch.arange(pairs.size, device=device),
                                                 args.label, args.loss, tau=args.tau, chunk=CHUNK).items()}})
    (out / "TRAIN_DONE").touch()


if __name__ == "__main__":
    main()
