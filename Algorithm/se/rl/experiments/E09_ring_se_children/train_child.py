"""E09 監督式訓練：ChildNet（只看兩個子代的完整基因＋地圖常數，不用父代的解碼狀態）。

標籤與 loss 沿用 E07 train_sup 的 loss_fn（預設 plain_delta、soft τ＝1，全 0 的交易也訓練）。
用法（2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E09_ring_se_children/train_child.py --budget 1000 --maps 140 --epochs 20
輸出：E09 results/child_b{預算}_m{N}_{label}{loss}/seed{seed}/（log.jsonl、config.json、best.pt、final.pt、TRAIN_DONE）。
fp32（10/26 起；bf16 反而比較慢）。指標（同分算對）：acc_all＝全部交易、acc_pos＝正值交易選到最大，和 E07／E08 報告用的一樣。
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

from childdata import DATA, ChildData, map_files  # noqa: E402
from childnet import ChildNet  # noqa: E402
from train_sup import LABELS, loss_fn  # noqa: E402

RESULTS = HERE / "results"
LOSSES = tuple(f"{b}{n}" for b in ("huber", "cls", "soft", "prop") for n in ("", "_nz"))


class _Labels:
    """給 E07 loss_fn 用：只需要 .label。"""

    def __init__(self, data):
        self.label = data.label


def evaluate(net, data, criterion, chunk=512):
    net.eval()
    picks, losses = [], []
    with torch.no_grad():
        for s in range(0, data.size, chunk):
            idx = torch.arange(s, min(s + chunk, data.size), device=data.label.device)
            out = net(*data.inputs(idx)).float()
            losses.append(float(criterion(out, _Labels(data), idx)) * len(idx))
            picks.append(out.argmax(1))
    pick = torch.cat(picks)
    best = data.label.max(1).values
    ok = data.label.gather(1, pick[:, None]).squeeze(1) >= best
    pos = best > 0
    ratio = (data.label.gather(1, pick[:, None]).squeeze(1)[pos] / best[pos]).mean()
    net.train()
    return {"loss": sum(losses) / data.size, "acc_all": float(ok.float().mean()), "acc_pos": float(ok[pos].float().mean()),
            "label_ratio": float(ratio)}


def token_stats(data, n=4096, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(data.size, generator=gen)[:n].to(data.label.device)
    tokens, glob = data.inputs(idx)
    tok = tokens.reshape(-1, tokens.shape[-1])
    return {"tok_mean": tok.mean(0).cpu().numpy(), "tok_std": tok.std(0).cpu().numpy(),
            "glob_mean": glob.mean(0).cpu().numpy(), "glob_std": glob.std(0).cpu().numpy()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label", choices=LABELS, default="plain_delta")
    parser.add_argument("--loss", choices=LOSSES, default="soft")
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--maps", type=int, default=140)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--no-contrast", action="store_true", help="第一版：CLS 不減候選平均（會停在 ln4）")
    args = parser.parse_args()

    base = args.loss.removesuffix("_nz")
    suffix = {"huber": "", "cls": "_cls", "soft": f"_soft{args.tau:g}", "prop": "_prop"}[base]
    if args.loss.endswith("_nz"):
        suffix = (suffix or "_huber") + "_nz"
    out = (RESULTS / f"child_b{args.budget}_m{args.maps}_{args.label}{suffix}{'' if args.no_contrast else '_contrast'}"
           / f"seed{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    log = open(out / "log.jsonl", "a")

    def write(record):
        log.write(json.dumps(record) + "\n")
        log.flush()
        print(json.dumps(record), flush=True)

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    t0 = time.time()
    root = DATA / f"b{args.budget}"
    train = ChildData(map_files(root / "train", args.maps), args.label, device)
    val = ChildData(map_files(root / "val"), args.label, device)
    net = ChildNet(token_stats(train, seed=args.seed), n=train.n, contrast=not args.no_contrast).to(device)
    train_idx = (train.informative.nonzero().squeeze(1) if args.loss.endswith("_nz")
                 else torch.arange(train.size, device=device))
    criterion = loss_fn(args.label, args.loss, args.tau)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * ((len(train_idx) + args.batch - 1) // args.batch))
    write({"event": "start", "params": sum(p.numel() for p in net.parameters()), "train_pairs": len(train_idx),
           "val_pairs": val.size, "load_seconds": round(time.time() - t0, 1)})

    def save(path, epoch):
        torch.save({"kind": net.kind, "model": net.state_dict(), "net_config": net.config, "epoch": epoch}, path)

    labels = _Labels(train)
    best, stale = np.inf, 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        losses = []
        for s in range(0, len(perm), args.batch):
            idx = perm[s:s + args.batch]
            score = net(*train.inputs(idx))
            loss = criterion(score, labels, idx)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0)
            opt.step()
            sched.step()
            losses.append(loss.item())
        save(out / f"epoch{epoch}.pt", epoch)
        record = {"event": "epoch", "epoch": epoch, "train_loss": float(np.mean(losses)),
                  **{f"val_{k}": v for k, v in evaluate(net, val, criterion).items()},
                  "seconds": round(time.time() - t0, 1)}
        write(record)
        if record["val_loss"] < best - 1e-6:
            best, stale = record["val_loss"], 0
            save(out / "best.pt", epoch)
        else:
            stale += 1
            if stale >= args.patience:
                break
    save(out / "final.pt", epoch)
    ckpt = torch.load(out / "best.pt", map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model"])
    del train, val
    torch.cuda.empty_cache()
    test = ChildData(map_files(root / "test"), args.label, device)
    write({"event": "test", "epoch": ckpt["epoch"], **{f"test_{k}": v for k, v in evaluate(net, test, criterion).items()}})
    (out / "TRAIN_DONE").touch()


if __name__ == "__main__":
    main()
