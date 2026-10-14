"""Audit and summarize paired GA/MEALPY shared-runner lifetime CSVs."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result_dir', type=Path)
    parser.add_argument('--methods', nargs='+', default=['ga', 'mealpy_ga'])
    options = parser.parse_args()
    root = options.result_dir
    methods = tuple(dict.fromkeys(options.methods))
    assert methods[0] == 'ga' and len(methods) >= 2, 'Start with ga and at least one comparison'
    cases = {}
    for method in methods:
        cases[method] = {}
        for summary in sorted((root / method).glob('*/summary.csv')):
            traces = list(csv.DictReader((summary.parent / 'trace.csv').open()))
            for row in csv.DictReader(summary.open()):
                key = (row['map'], int(row['seed']))
                assert key not in cases[method], 'Duplicate case'
                trace = [t for t in traces if int(t['seed']) == key[1]]
                budget = int(row['evaluate_budget'])
                assert trace and all(int(t['actual_evatime']) == budget for t in trace)
                assert sum(int(t['actual_evatime']) for t in trace) == int(row['actual_evatime'])
                assert sum(int(t['segment_length']) for t in trace) == int(row['lifetime'])
                assert all(int(t['life_end'])-int(t['life_start']) == int(t['segment_length']) for t in trace)
                cases[method][key] = row
    assert cases['ga'] and all(cases['ga'].keys() == cases[m].keys() for m in methods), 'Missing paired cases'
    keys = sorted(cases['ga'])
    for key in keys:
        for field in ('evaluate_budget', 'fitness_service', 'sensing_mode', 'routing_service',
                      'moving', 'file', 'full_energy', 'max_rounds', 'segment_slot_limit', 'lifetime_cap'):
            assert all(cases['ga'][key][field] == cases[m][key][field] for m in methods), (key, field)
    summary = {}
    for method in methods:
        rr = list(cases[method].values())
        summary[method] = dict(
            runs=len(rr), mean_lifetime=statistics.mean(int(r['lifetime']) for r in rr),
            median_lifetime=statistics.median(int(r['lifetime']) for r in rr),
            mean_seconds=statistics.mean(float(r['elapsed_seconds']) for r in rr),
            mean_evaluations=statistics.mean(int(r['actual_evatime']) for r in rr),
            stop_reasons=dict(Counter(r['stop_reason'] for r in rr)))
    pairs = [dict(map=mp, seed=seed, **{m:int(cases[m][mp,seed]['lifetime']) for m in methods})
             for mp,seed in keys]
    by_map = {}
    for mp in sorted({k[0] for k in keys}):
        by_map[mp] = {m: statistics.mean(r[m] for r in pairs if r['map']==mp) for m in methods}
    comparisons = {}
    for method in methods[1:]:
        for reference in methods:
            if reference == method: continue
            delta = [r[method]-r[reference] for r in pairs]
            base = summary[reference]['mean_lifetime']
            comparisons[f'{method}_vs_{reference}'] = dict(
                wins=sum(d>0 for d in delta), ties=sum(d==0 for d in delta), losses=sum(d<0 for d in delta),
                mean_difference=statistics.mean(delta),
                relative_mean_percent=100*(summary[method]['mean_lifetime']/base-1) if base else None)
    report = dict(summary=summary, by_map=by_map, pairs=pairs, comparisons=comparisons,
                  limitations=['Existing implementations and presets differ; not an isolated library comparison.',
                               'Equal real fitness calls per search, not equal new offspring or total calls across lifetime.',
                               'Project GA includes coverage-seeded initialization and cached decoded snapshots.',
                               'Timing is measured in parallel runs, not an isolated speed benchmark.',
                               'Service lifetime ends at search failure; no proof of physical infeasibility or statistical superiority.'])
    (root/'comparison.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# GA lifetime comparison', '', '| Map | '+' | '.join(methods)+' |',
           '|---|'+'---:|'*len(methods)]
    lines += ['| '+mp+' | '+' | '.join(f'{r[m]:.2f}' for m in methods)+' |' for mp,r in by_map.items()]
    lines += ['', 'See comparison.json and algorithm_parameters.json for settings, paired outcomes and limitations.']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,len(methods)-1,figsize=(6*(len(methods)-1),6),squeeze=False)
    high=max(r[m] for r in pairs for m in methods)*1.05
    for ax,method in zip(axes.flat,methods[1:]):
        for mp in by_map:
            rr=[r for r in pairs if r['map']==mp]
            ax.scatter([r['ga'] for r in rr], [r[method] for r in rr], label=mp, s=45)
        ax.plot([0,high],[0,high],'k--',alpha=.5,label='Equal lifetime')
        ax.set(xlim=(0,high),ylim=(0,high),aspect='equal',xlabel='Project GA lifetime (slots)',
               ylabel=f'{method} lifetime (slots)',title=f'{len(pairs)} paired cases: {method}')
        ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(root/'comparison.png',dpi=150);plt.close(fig)
    print(json.dumps({k:report[k] for k in ('summary','by_map','comparisons')},indent=2))


if __name__ == '__main__':
    main()
