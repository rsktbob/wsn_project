"""整理 run_eval_maps.sh 的 lifetime 結果和 train_sup.py 的猜準度。

用法（在 2027WSN 根目錄）：
    python Algorithm/se/rl/experiments/E07_ring_se_lifetime/compare.py --split test --budget 1000 \
        sup_b1000_m30_diff_cur sup_b1000_m60_diff_cur
（不給模型名稱就列出這個 split／預算下所有跑過的模型。）
每個模型對同場原版 ring_se 做成對 Wilcoxon；另列 test 上的猜準度（選到的標籤 ÷ 最大標籤、
有改善的組裡選到最好的比例 best_hit（10/22 以前訓練的模型 log 裡沒有，顯示 —）、和作弊版一致的比例）。
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[5]
RESULTS = ROOT / "experiment_results"
MODELS = Path(__file__).resolve().parent / "results"


def load(exp, algo):
    fs = sorted((RESULTS / exp / algo).glob("*/summary.csv"))
    if not fs:
        return None
    return pd.concat([pd.read_csv(f).assign(map=f.parent.name) for f in fs]).set_index(["map", "seed"])["lifetime"].astype(float)


def test_metrics(tag):
    log = MODELS / tag / "seed0" / "log.jsonl"
    if not log.exists():
        return None
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    tests = [r for r in rows if r.get("event") == "test"]
    return tests[-1] if tests else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=("test", "val", "train"))
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("tags", nargs="*")
    args = parser.parse_args()
    prefix, suffix = f"{args.split}711_", f"_eval{args.budget}"
    base = load(f"{prefix}ring_se{suffix}", "ring_se")
    if base is None:
        raise SystemExit(f"還沒有 {prefix}ring_se{suffix}（先跑 run_eval_maps.sh）")
    tags = args.tags or sorted(p.name[len(prefix):-len(suffix)] for p in RESULTS.glob(f"{prefix}*{suffix}")
                               if p.name != f"{prefix}ring_se{suffix}")
    print(f"{args.split} 地圖、預算 {args.budget}、seed 7–11；原版 ring_se {base.mean():.1f}（{len(base)} 場）\n")
    print("| 模型 | lifetime | 對原版（勝/負，p） | 選到的標籤 ÷ 最大 | 選到最好 | 和作弊版一致 | best epoch |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for tag in tags:
        v = load(f"{prefix}{tag}{suffix}", "e07_net")
        t = test_metrics(tag)
        acc = f"{t['test_label_ratio']:.3f}（隨機 {t['test_random_label_ratio']:.3f}）" if t else "—"
        agree = f"{t['test_agree']:.3f}" if t else "—"
        hit = f"{t['test_best_hit']:.3f}" if t and "test_best_hit" in t else "—"
        epoch = t["epoch"] if t else "—"
        if v is None:
            print(f"| {tag} | 尚無結果 | | {acc} | {hit} | {agree} | {epoch} |")
            continue
        c = v.index.intersection(base.index)
        d = v[c] - base[c]
        p = wilcoxon(v[c], base[c]).pvalue
        print(f"| {tag} | {v[c].mean():.1f} | {100 * d.mean() / base[c].mean():+.2f}%（{(d > 0).sum()}/{(d < 0).sum()}，"
              f"p={p:.4f}） | {acc} | {hit} | {agree} | {epoch} |")


if __name__ == "__main__":
    main()
