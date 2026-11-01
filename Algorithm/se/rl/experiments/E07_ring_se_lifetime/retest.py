"""已訓練好的模型重算 test 指標（不重訓），追加一筆 "test" 到 log.jsonl（compare.py 讀最後一筆）。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E07_ring_se_lifetime/retest.py sup_b1000_m140_diff_cur sup_b3000_m140_diff_cur
模型的標籤、預算、checkpoint（best.pt，沒有就 final.pt）從 results/{tag}/seed0/config.json 讀。
"""
import json
import sys
from pathlib import Path

import torch

from common import RESULTS
from model import RoundStore, load_net
from train_sup import PairData, load_labels, metrics


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for tag in sys.argv[1:]:
        out = RESULTS / tag / "seed0"
        cfg = json.loads((out / "config.json").read_text())
        chosen = out / "best.pt" if (out / "best.pt").exists() else out / "final.pt"
        net = load_net(chosen, device)
        epoch = torch.load(chosen, map_location="cpu", weights_only=False)["epoch"]
        data, n = load_labels(Path(cfg["test"]), cfg["label"])
        store, pairs = RoundStore(data, device), PairData(data, device)
        record = {"event": "test", "checkpoint": chosen.name, "epoch": epoch, "test_maps": n, "retest": True,
                  **{f"test_{k}": v for k, v in metrics(net, store, pairs, torch.arange(pairs.size, device=device),
                                                       cfg["label"], cfg.get("loss", "huber"),
                                                       tau=cfg.get("tau", 0.3)).items()}}
        with open(out / "log.jsonl", "a") as log:
            log.write(json.dumps(record) + "\n")
        print(tag, json.dumps({k: round(v, 4) for k, v in record.items() if k in
                               ("test_label_ratio", "test_random_label_ratio", "test_best_hit", "test_agree")}))
        del store, pairs
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
