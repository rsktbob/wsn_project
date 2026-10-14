"""Summarize completed fixed-state runs without re-evaluating fitness."""
import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


def summarize(folder):
    rows = []
    initial_by_seed = {}
    for file in sorted(folder.glob('MAP*_fixed_seed*.json')):
        report = json.loads(file.read_text())
        cfg = report['config']
        record = report['records'][0]
        metadata = record['metadata']
        history = record['history']
        initial_count = min(cfg['initial_size'], cfg['budget'])
        key = (cfg['map'], cfg['seed'])
        initial = history[:initial_count]
        if key in initial_by_seed and initial_by_seed[key] != initial:
            raise ValueError(f'unmatched initialization: {file}')
        initial_by_seed[key] = initial
        assert record['evaluations'] == cfg['budget'], f'incomplete budget: {file}'
        score = record['best_score']
        rows.append(dict(map=cfg['map'], seed=cfg['seed'], model=cfg['model'], order=cfg['order'],
            evaluations=record['evaluations'], feasible=record['feasible'], best_score=score,
            initial_best=initial[-1], gain=score-initial[-1], weighted_depletion=1-score,
            seconds=report['elapsed_seconds'], training_seconds=metadata['training_seconds'],
            feasible_candidate_fraction=metadata['feasible_evaluations']/record['evaluations'],
            unique_decoded_fraction=metadata['unique_decoded']/record['evaluations'],
            model_evaluations=metadata['source_evaluations']['model'],
            fallback_evaluations=metadata['source_evaluations']['fallback']))
    if not rows:
        raise ValueError('no fixed-state results')
    with (folder/'summary.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for row in rows:
        groups.setdefault((row['model'], row['order']), []).append(row)
    aggregate = []
    for (model, order), group in groups.items():
        aggregate.append(dict(model=model, order=order, runs=len(group),
            feasible_runs=sum(r['feasible'] for r in group),
            mean_best_score=mean(r['best_score'] for r in group),
            score_stdev=stdev(r['best_score'] for r in group) if len(group)>1 else None,
            mean_weighted_depletion=mean(r['weighted_depletion'] for r in group),
            mean_gain=mean(r['gain'] for r in group),
            mean_seconds=mean(r['seconds'] for r in group),
            mean_training_seconds=mean(r['training_seconds'] for r in group),
            mean_feasible_candidate_fraction=mean(r['feasible_candidate_fraction'] for r in group),
            mean_unique_decoded_fraction=mean(r['unique_decoded_fraction'] for r in group)))
    (folder/'aggregate.json').write_text(json.dumps(aggregate, indent=2))
    text = ['# MAP01 固定電量初步比較', '',
            '每次 1000 FE，seed=7/17/27；MPS rank=4，批次32，初始64，訓練樣本64，每批3步。',
            '使用相同初始樣本、v2 fitness、discrete 感測選項；不包含完整 lifetime。', '',
            '| 模型 | 排序 | 完成次數 | 平均最佳 fitness ↑ | 平均 1−fitness ↓ | 平均秒數 | 最終可行 |',
            '|---|---|---:|---:|---:|---:|---:|']
    for a in aggregate:
        text.append(f"| {a['model']} | {a['order']} | {a['runs']} | {a['mean_best_score']:.9f} | {a['mean_weighted_depletion']:.9f} | {a['mean_seconds']:.2f} | {a['feasible_runs']}/{a['runs']} |")
    text.extend(['', '1−fitness 僅在最終解可行時對應 v2 的加權能量耗損代理，不能解讀為實際壽命。',
                 '依序執行以減少互相干擾；耗時含模型初始化、抽樣、解碼、訓練與診斷，排除 Problem 建構與結果寫檔。',
                 '這是單圖三種子小型試驗，不能推論統計顯著、跨地圖泛化或优於 SETS。', '',
                 '逐次結果：summary.csv；原始設定、history 和 trace：各 MAP01_*.json。'])
    (folder/'REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    summarize(parser.parse_args().folder)
