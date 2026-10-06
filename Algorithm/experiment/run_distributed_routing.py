"""Run fixed-sensing distributed routing baselines on project map manifests."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

INVOCATION_CWD = Path.cwd()

from Algorithm.experiment.distributed_routing import DistributedRouting, LocalRoutingPolicy
from experiments.maps import load_map_specs
from experiments.problem_setup import build_problem

# experiments.__init__ changes cwd; preserve caller-relative output semantics.
os.chdir(INVOCATION_CWD)


def simulate(problem, router, levels, max_slots):
    """Stop before first unserviceable fixed-demand slot, or at explicit cap."""
    rows = []
    completed = 0
    for slot in range(1, max_slots + 1):
        epoch = router.build(problem, levels)
        uncovered = problem.find_uncovered_targets(epoch.state)
        data_cost = problem.energy_service.calculate_total_cost(epoch.state)
        total = data_cost + epoch.control_cost
        failed = np.flatnonzero(total > problem.energy)
        if np.any(epoch.control_cost > problem.energy):
            reason = "control_energy_shortfall"
        elif len(uncovered):
            reason = "uncovered_targets"
        elif epoch.disconnected_ids:
            reason = "disconnected_sensors"
        elif len(failed):
            reason = "data_energy_shortfall"
        else:
            reason = "ok"
        rows.append(dict(slot=slot, status=reason, control_messages=epoch.control_messages,
                         control_energy=float(epoch.control_cost.sum()),
                         data_and_sensing_energy=float(data_cost.sum()), waves=epoch.waves,
                         disconnected=len(epoch.disconnected_ids), uncovered=len(uncovered),
                         energy_failed=len(failed), max_load=int(epoch.state.tx_load.max()),
                         next_hops=epoch.state.next_hops.tolist()))
        if reason != "ok":
            break
        problem.energy -= total
        completed += 1
    reason = rows[-1]["status"]
    return dict(completed_slots=completed,
                stop_reason="max_slots" if reason == "ok" else reason,
                truncated=(reason == "ok"),
                # Failed attempts are diagnostic estimates, not paid slots.
                paid_control_messages=sum(r["control_messages"] for r in rows if r["status"] == "ok"),
                paid_control_energy=sum(r["control_energy"] for r in rows if r["status"] == "ok")), rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maps", default="maps/maps_100100100.json")
    parser.add_argument("--map", default=None, help="optional map name, otherwise all maps")
    parser.add_argument("--policies", nargs="+", choices=LocalRoutingPolicy.MODES,
                        default=list(LocalRoutingPolicy.MODES))
    parser.add_argument("--radio-range", type=float, default=30.0)
    parser.add_argument("--control-bits", type=int, default=128)
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--max-slots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_slots < 1:
        parser.error("--max-slots must be positive")
    args.output = args.output.resolve()
    args.maps = str((ROOT / args.maps).resolve())
    specs = load_map_specs(args.maps)
    if args.map:
        specs = [s for s in specs if s["name"] == args.map]
        if not specs:
            parser.error("unknown --map")
    # Legacy Problem loads dataset paths relative to project root.
    os.chdir(ROOT)
    args.output.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["output"] = str(args.output)
    config["map_specs"] = specs
    config["model"] = "fixed sensing; strict distance DAG; ideal reliable neighbor unicasts"
    (args.output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    results = []
    for spec in specs:
        for mode in args.policies:
            p = build_problem(SimpleNamespace(_current_map=spec, sensing_mode="discrete"), args.seed)
            router = DistributedRouting(mode, args.radio_range, args.control_bits)
            summary, trace = simulate(p, router, np.full(p.SENSOR_NUMBER, args.level, dtype=int), args.max_slots)
            summary.update(map=spec["name"], policy=mode)
            results.append(summary)
            (args.output / f'{spec["name"]}_{mode}.json').write_text(
                json.dumps(dict(summary=summary, trace=trace), indent=2), encoding="utf-8")
            print(json.dumps(summary), flush=True)
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
