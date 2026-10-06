"""Replay saved routes to separate scheduling, data routing and control costs.

Average successful slots within each map, then weight maps equally.
Failed attempts are excluded. Costs are whole-network mJ per slot.
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
INVOCATION_CWD = Path.cwd()
sys.path.insert(0, str(ROOT))
from experiments.problem_setup import build_problem
from State.State import State
os.chdir(INVOCATION_CWD)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    args = parser.parse_args()
    result_dir = args.result_dir.resolve()
    config = json.loads((result_dir / 'config.json').read_text())
    os.chdir(ROOT)
    per_map, per_slot = [], []
    for spec in config['map_specs']:
        p = build_problem(SimpleNamespace(_current_map=spec, sensing_mode='discrete'), config['seed'])
        for policy in config['policies']:
            doc = json.loads((result_dir / f"{spec['name']}_{policy}.json").read_text())
            costs = []
            for saved in doc['trace']:
                if saved['status'] != 'ok':
                    continue
                state = State.empty(p)
                state.levels[:] = config['level']
                state.next_hops[:] = saved['next_hops']
                # Source load must be added at every forwarding node, even
                # when generated_load differs between sources.
                for source in range(p.SENSOR_NUMBER):
                    node, seen = source, set()
                    while node != p.BSID:
                        if not 0 <= node < p.SENSOR_NUMBER or node in seen:
                            raise ValueError('invalid saved successful route')
                        seen.add(node)
                        state.tx_load[node] += p.generated_load[source]
                        node = int(state.next_hops[node])
                scheduling = float(p.energy_service.calculate_scheduling_cost(state).sum())
                routing = float(p.energy_service.calculate_routing_cost(state).sum())
                np.testing.assert_allclose(scheduling + routing, saved['data_and_sensing_energy'], rtol=1e-12, atol=1e-12)
                control = float(saved['control_energy'])
                costs.append([routing, scheduling, control])
                per_slot.append(dict(map=spec['name'], policy=policy, slot=saved['slot'],
                                     routing_mj=routing, scheduling_mj=scheduling, control_mj=control))
            if not costs:
                raise ValueError(f"no successful slots: {spec['name']} {policy}")
            assert len(costs) == doc['summary']['completed_slots']
            means = np.mean(costs, axis=0)
            per_map.append(dict(map=spec['name'], policy=policy, successful_slots=len(costs),
                                mean_routing_mj=float(means[0]), mean_scheduling_mj=float(means[1]),
                                mean_control_mj=float(means[2])))
    aggregate = []
    for policy in config['policies']:
        rows = [r for r in per_map if r['policy'] == policy]
        aggregate.append(dict(policy=policy, maps=len(rows), **{
            key: float(np.mean([r[key] for r in rows]))
            for key in ('mean_routing_mj', 'mean_scheduling_mj', 'mean_control_mj')}))
    write_csv(result_dir / 'costs_per_slot.csv', per_slot)
    write_csv(result_dir / 'costs_per_map.csv', per_map)
    write_csv(result_dir / 'costs_summary.csv', aggregate)
    report = ['# 平均路由與排程成本', '',
              '單位：全網 mJ/時槽。各圖先平均所有成功時槽，再對六張地圖等權平均；排除最後失敗嘗試。路由成本為資料收發/轉送，排程成本為感測，控制訊息另外列出。', '',
              '| 策略 | 平均路由成本 | 平均排程成本 | 平均控制成本 |',
              '| --- | ---: | ---: | ---: |']
    for r in aggregate:
        report.append(f"| {r['policy']} | {r['mean_routing_mj']:.6f} | {r['mean_scheduling_mj']:.6f} | {r['mean_control_mj']:.6f} |")
    report += ['', '所有感測器固定等級 5，故三策略排程成本相同。這不是感測排程最佳化的結果。每圖含 100 個 sensor；若改看每節點平均，上表各成本除以 100。', '',
               f'逐條重建 {len(per_slot)} 個成功時槽的路由和轉送負載，用既有 EnergyService 重算，並逐筆確認路由加排程成本等於原始 trace 的 data_and_sensing_energy。不同策略存活時槽數不同，因此各自平均期間不同。', '',
               'costs_per_map.csv 保存逐圖平均；costs_per_slot.csv 保存逐時槽拆分；costs_summary.csv 保存等權跨圖平均。']
    (result_dir / 'COSTS.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
