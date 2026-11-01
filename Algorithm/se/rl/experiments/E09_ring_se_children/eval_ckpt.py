"""用某個 checkpoint（例如訓練中途的 epoch8.pt）算 test 指標，不影響正在跑的訓練。

用法（2027WSN 根目錄）：python Algorithm/se/rl/experiments/E09_ring_se_children/eval_ckpt.py <checkpoint.pt> [--split test]
指標同 train_child.evaluate（同分算對 acc_all、acc_pos、label_ratio），結果追加到同資料夾的 eval_ckpt.jsonl。
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "E07_ring_se_lifetime"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from childdata import DATA, ChildData, map_files  # noqa: E402
from childnet import load_childnet  # noqa: E402
from train_child import evaluate  # noqa: E402
from train_sup import loss_fn  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    args = parser.parse_args()
    ckpt = Path(args.ckpt)
    cfg = json.loads((ckpt.parent / "config.json").read_text())
    device = torch.device("cuda")
    net = load_childnet(ckpt, device)
    data = ChildData(map_files(DATA / f"b{cfg['budget']}" / args.split), cfg["label"], device)
    result = {"checkpoint": ckpt.name, "split": args.split,
              **evaluate(net, data, loss_fn(cfg["label"], cfg["loss"], cfg["tau"]))}
    with open(ckpt.parent / "eval_ckpt.jsonl", "a") as f:
        f.write(json.dumps(result) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
