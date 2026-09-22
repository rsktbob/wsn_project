"""CSV/JSON output: field layout, row assembly and the result folder."""

import csv
import json
import sys
import time
from pathlib import Path

from Problem.services import FitnessServiceV2, FitnessServiceV3
from Problem.services.evaluation_kernels import NUMBA_AVAILABLE, NUMBA_VERSION
from State.SensorEncoding import DEFAULT_SENSOR_ENCODING

from experiments import PROJECT_ROOT
from experiments.builder import algorithm_label, algorithm_params
from experiments.config import (
    IMPLEMENTATION_NAME,
    MOVING_TARGET_ENERGY_THRESHOLD,
    evaluate_budget,
    lifetime_cap,
    moving_enabled,
)
from experiments.evaluation import OBJECTIVE_FIELDS
from experiments.maps import current_map_spec
from experiments.presets import ALGORITHM_PRESETS

# Columns shared by summary.csv and trace.csv, in the order they are written.
_RUN_FIELDS = (
    "mode",
    "algorithm",
    "algorithm_label",
    "implementation",
    "run",
    "seed",
    "map",
    "map_seed",
)
_SETTING_FIELDS = (
    "max_rounds",
    "segment_slot_limit",
    "lifetime_cap",
    "moving",
    "sensing_mode",
    "fitness_service",
    "routing_service",
    "evaluation_backend",
    "numba_version",
    "moving_target_threshold",
)
_MAP_FIELDS = (
    "boundary",
    "sensors",
    "targets",
    "full_energy",
    "file",
)
# The objectives come first so a reader sees the fitness breakdown before the
# derived network metrics. Only objectives some fitness service can actually
# report are listed; "coverage" is kept last of them for historical column order.
_METRIC_FIELDS = (
    tuple(name for name in OBJECTIVE_FIELDS if name != "coverage")
    + (
        "coverage",
        "fitness",
        "mean_sensor_cost",
        "max_sensor_cost",
        "total_sensor_cost",
        "sensor_remaining_mean",
        "sensor_remaining_min",
        "sensor_remaining_variance",
        "target_remaining_min",
        "uncovered_count",
        "disconnected_count",
        "energy_failed_count",
        "active_sensor_count",
        "mean_sensing_radius",
        "max_sensing_radius",
    )
)

SUMMARY_FIELDNAMES = list(
    _RUN_FIELDS
    + ("lifetime",)
    + _SETTING_FIELDS
    + (
        "reoptimizations",
        "movement_events",
        "moved_sensor_count",
        "stop_reason",
        "elapsed_seconds",
        "evaluate_budget",
        "effective_evaluate_budget",
        "requested_evaluate_budget_total",
        "early_stop_count",
        "actual_evatime",
        "compute_device",
        "gpu_accelerated",
    )
    + _MAP_FIELDS
    + _METRIC_FIELDS
    + ("history_length",)
)

TRACE_FIELDNAMES = list(
    _RUN_FIELDS
    + ("segment", "life_start", "life_end", "segment_length")
    + _SETTING_FIELDS
    + (
        "movement_events",
        "moved_sensor_count",
        "stop_reason",
        "optimize_elapsed_seconds",
        "evaluate_budget",
        "effective_evaluate_budget",
        "early_stop_count",
        "actual_evatime",
    )
    + _MAP_FIELDS
    + _METRIC_FIELDS
)


def common_result_fields(name, run_id, seed, args):
    map_spec = current_map_spec(args)
    return {
        "algorithm": name,
        "algorithm_label": algorithm_label(name),
        "implementation": IMPLEMENTATION_NAME,
        "run": run_id,
        "seed": seed,
        "map": map_spec["name"],
        "map_seed": "" if map_spec["map_seed"] is None else map_spec["map_seed"],
        "evaluate_budget": evaluate_budget(args),
        "boundary": map_spec["boundary"],
        "sensors": map_spec["sensors"],
        "targets": map_spec["targets"],
        "full_energy": map_spec["full_energy"],
        "file": map_spec["file"],
        "max_rounds": getattr(args, "max_rounds", ""),
        "segment_slot_limit": getattr(args, "segment_slot_limit", ""),
        "lifetime_cap": "" if lifetime_cap(args) is None else lifetime_cap(args),
        "moving": moving_enabled(args),
        "sensing_mode": getattr(args, "sensing_mode", "discrete"),
        "fitness_service": getattr(args, "fitness_service", "v2"),
        "routing_service": getattr(args, "routing_service", "v1"),
        "sensor_ring_id": getattr(args, "sensor_ring_id", True),
        "evaluation_backend": "numba" if NUMBA_AVAILABLE else "python",
        "numba_version": NUMBA_VERSION if NUMBA_AVAILABLE else "",
        "moving_target_threshold": (
            MOVING_TARGET_ENERGY_THRESHOLD if moving_enabled(args) else ""
        ),
    }


def build_trace_row(
    name,
    run_id,
    seed,
    args,
    segment_id,
    life_start,
    life_end,
    segment_length,
    optimize_elapsed,
    actual_evatime,
    metrics,
    stop_reason,
    movement_events=0,
    moved_sensor_count=0,
    effective_evaluate_budget="",
    early_stop_count=0,
):
    row = common_result_fields(name, run_id, seed, args)
    row.update(
        {
            "mode": "lifetime",
            "segment": segment_id,
            "life_start": life_start,
            "life_end": life_end,
            "segment_length": segment_length,
            "optimize_elapsed_seconds": round(optimize_elapsed, 6),
            "actual_evatime": actual_evatime,
            "movement_events": movement_events,
            "moved_sensor_count": moved_sensor_count,
            "effective_evaluate_budget": effective_evaluate_budget,
            "early_stop_count": early_stop_count,
            "stop_reason": stop_reason,
        }
    )
    row.update(metrics)
    return row


def build_algorithm_parameter_report(algorithms, args):
    fitness_version = getattr(args, "fitness_service", "v2")
    fitness_parameters = {}
    if fitness_version == "v2":
        fitness_parameters = {
            "weights": list(FitnessServiceV2.DEFAULT_WEIGHTS),
            "formula": (
                "feasible fitness = 1 - (0.3 * global_depletion + "
                "0.7 * worst_target_depletion)"
            ),
            "incomplete_order": (
                "fitness = -uncovered_ratio - "
                "0.5 / target_count * weighted_depletion"
            ),
            "invalid_base_score": FitnessServiceV2.INVALID_BASE_SCORE,
        }
    elif fitness_version == "v3":
        fitness_parameters = {
            "weights": list(FitnessServiceV3.DEFAULT_WEIGHTS),
            "fully_feasible": "fitness = quality",
            "incomplete": (
                "fitness = -uncovered_count + 0.5 * quality"
            ),
            "infeasible": (
                "fitness = -uncovered_count - disconnected_ratio - "
                "energy_deficit_ratio"
            ),
        }
    return {
        "experiment_settings": {
            "mode": args.mode,
            "runs_per_map": args.runs,
            "sensor_encoding": DEFAULT_SENSOR_ENCODING,
            "map_count": len(args.map_specs),
            "total_cases_per_algorithm": args.runs * len(args.map_specs),
            "maps": args.map_specs,
            "implementation": IMPLEMENTATION_NAME,
            "evaluate_budget": evaluate_budget(args),
            "max_rounds": getattr(args, "max_rounds", ""),
            "segment_slot_limit": getattr(args, "segment_slot_limit", ""),
            "lifetime_cap": lifetime_cap(args),
            "moving": moving_enabled(args),
            "draw_save": getattr(args, "draw_save", False),
            "draw_fitness_history": getattr(
                args, "draw_fitness_history", False
            ),
            "sensing_mode": getattr(args, "sensing_mode", "discrete"),
            "fitness_service": getattr(args, "fitness_service", "v2"),
            "fitness_parameters": fitness_parameters,
            "routing_service": getattr(args, "routing_service", "v1"),
            "sensor_ring_id": getattr(args, "sensor_ring_id", True),
            "evaluation_backend": (
                "numba" if NUMBA_AVAILABLE else "python"
            ),
            "numba_version": NUMBA_VERSION if NUMBA_AVAILABLE else None,
            "moving_target_threshold": (
                MOVING_TARGET_ENERGY_THRESHOLD if moving_enabled(args) else None
            ),
        },
        "algorithms": [
            _algorithm_report_entry(algorithm, args)
            for algorithm in algorithms
        ],
    }


def _algorithm_report_entry(algorithm, args):
    preset = ALGORITHM_PRESETS[algorithm]
    entry = {
        "name": algorithm,
        "label": algorithm_label(algorithm),
        "implementation": IMPLEMENTATION_NAME,
        "params": algorithm_params(algorithm, args),
        "parameter_basis": preset.get("parameter_basis", "No recorded basis."),
    }
    # A chromosome description is documentation, not a constructor argument,
    # so only the algorithms that define one carry the field.
    if preset.get("encoding") is not None:
        entry["encoding"] = preset["encoding"]
    return entry


def write_algorithm_parameters(result_root, algorithms, args):
    output_path = result_root / "algorithm_parameters.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = build_algorithm_parameter_report(algorithms, args)
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2, sort_keys=True)
        json_file.write("\n")
    print("params", output_path)


def write_rows(rows, output, fieldnames):
    if output is None or str(output).lower() in ("", "none", "null"):
        return

    normalized_rows = [
        {fieldname: row.get(fieldname, "") for fieldname in fieldnames}
        for row in rows
    ]

    if output == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)
        return

    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(normalized_rows)
    print("csv", output_path)


def write_grouped_results(summary_rows, trace_rows, algorithms, args):
    result_root = resolved_result_root(args)
    write_algorithm_parameters(result_root, algorithms, args)
    for algorithm in algorithms:
        algorithm_dir = result_root / algorithm
        for map_spec in args.map_specs:
            map_name = map_spec["name"]
            map_dir = algorithm_dir / map_name
            map_summary = [
                row
                for row in summary_rows
                if row.get("algorithm") == algorithm
                and row.get("map") == map_name
            ]
            map_trace = [
                row
                for row in trace_rows
                if row.get("algorithm") == algorithm
                and row.get("map") == map_name
            ]
            write_rows(map_summary, map_dir / "summary.csv", SUMMARY_FIELDNAMES)
            if args.mode == "lifetime":
                write_rows(map_trace, map_dir / "trace.csv", TRACE_FIELDNAMES)

    print("result_dir", result_root)


def build_result_root(args):
    base_dir = Path(args.result_dir)
    if not base_dir.is_absolute():
        base_dir = PROJECT_ROOT / base_dir

    experiment_name = args.experiment_name
    if experiment_name is None:
        requested = args.algorithm
        if isinstance(requested, str):
            requested = [requested]
        if "all" in requested:
            selected = "all"
        elif len(requested) == 1:
            selected = requested[0]
        else:
            selected = "_".join(requested)
        experiment_name = time.strftime("%Y%m%d_%H%M%S") + "_" + selected

    result_root = base_dir / experiment_name
    if not result_root.exists():
        return result_root

    suffix = 2
    while True:
        candidate = base_dir / (experiment_name + "_" + str(suffix).zfill(2))
        if not candidate.exists():
            return candidate
        suffix += 1


def resolved_result_root(args):
    """Return one stable result folder shared by plots and final CSV output."""
    result_root = getattr(args, "_resolved_result_root", None)
    if result_root is None:
        result_root = build_result_root(args)
        args._resolved_result_root = result_root
    return Path(result_root)
