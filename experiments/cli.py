"""Command line for the experiment runner."""

import argparse
import json
import sys
from pathlib import Path



from experiments.builder import RANDOM_POLICY_ALGORITHMS, configured_algorithm_params
from experiments.config import (
    DEFAULT_FITNESS_SERVICE,
    DEFAULT_ROUTING_SERVICE,
    DEFAULT_SENSING_MODE,
    EVALUATE_OVERRIDES,
    EXPERIMENT_PRESET,
    FITNESS_SERVICES,
    ROUTING_SERVICES,
    parse_bool,
)
from experiments.maps import load_map_specs
from experiments.registry import (
    ALGORITHM_CHOICES,
    ALL_RUNNABLE_ALGORITHMS,
    RL_SETS_ALGORITHMS,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run selected WSN algorithms with fixed experiment presets."
    )
    parser.add_argument(
        "--algorithm",
        nargs="+",
        choices=ALGORITHM_CHOICES,
        default=["all"],
        help=(
            "Choose one or more algorithms, e.g. --algorithm ga nsga. "
            "all runs every algorithm that does not require an RL checkpoint."
        ),
    )
    parser.add_argument(
        "--rl-model",
        default=None,
        help=(
            "Required .npz checkpoint when an RL-SETS version is selected. The policy "
            "is always frozen in this experiment runner."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("lifetime", "single"),
        default="lifetime",
        help=(
            "lifetime (default) runs the complete WSN lifetime experiment; "
            "single runs one optimizer call per run."
        ),
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=EXPERIMENT_PRESET["runs"],
        help=(
            "Number of algorithm seeds per map. Seeds start at --seed; "
            "for example --runs 5 --seed 7 uses seeds 7 through 11 on every map."
        ),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--maps",
        default="maps/maps_100100100.json",
        metavar="JSON",
        help=(
            "JSON map manifest. Each entry supplies name/file and optional "
            "boundary, sensors, targets, full_energy, and map_seed. "
            "Default: maps/maps_100100100.json."
        ),
    )
    parser.add_argument(
        "--evaluate",
        type=int,
        default=None,
        help=(
            "Evaluation budget for each optimizer call "
            f"(default: {EXPERIMENT_PRESET['evaluate']}; "
            + ", ".join(
                f"{name}: {budget}"
                for name, budget in EVALUATE_OVERRIDES.items()
            )
            + " when run on their own)."
        ),
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=EXPERIMENT_PRESET["max_rounds"],
        help="Maximum re-optimization rounds, matching the old Round.py outer loop.",
    )
    parser.add_argument(
        "--segment-slot-limit",
        type=int,
        default=EXPERIMENT_PRESET["segment_slot_limit"],
        help="Maximum repeated lifetime slots for one optimized state.",
    )
    parser.add_argument(
        "--max-lifetime",
        type=int,
        default=None,
        help="Optional lifetime-slot cap for quick tests. Default is no lifetime cap.",
    )
    parser.add_argument(
        "--moving",
        type=parse_bool,
        default=False,
        metavar="{true,false}",
        help=(
            "Enable MobilityService repair in lifetime mode. "
            "Default is false; use --moving true to enable it."
        ),
    )
    parser.add_argument(
        "--sensor-ring-id",
        type=parse_bool,
        default=True,
        metavar="{true,false}",
        help=(
            "Sensor chromosome ordering. Default true sorts by BS-centered "
            "ring then clockwise angle (adjacent genes are spatially local). "
            "Use --sensor-ring-id false to sort purely by distance to the "
            "base station instead (no ring/angle grouping)."
        ),
    )
    parser.add_argument(
        "--draw-save",
        type=parse_bool,
        default=False,
        metavar="{true,false}",
        help=(
            "Save the initial and after-segment sensor maps, including energy, "
            "sensing ranges, and routing, in lifetime mode. Default is false; "
            "use --draw-save true to enable it."
        ),
    )
    parser.add_argument(
        "--show",
        type=parse_bool,
        default=False,
        metavar="{true,false}",
        help=(
            "Show the sensor energy map, summary, and color scale in one "
            "window after each optimizer segment. Default is false."
        ),
    )
    parser.add_argument(
        "--draw-fitness-history",
        type=parse_bool,
        default=False,
        metavar="{true,false}",
        help=(
            "Save one best-fitness curve for every optimizer call in lifetime "
            "mode. Default is false; use --draw-fitness-history true."
        ),
    )
    parser.add_argument(
        "--sensing-mode",
        choices=("discrete", "bucketed", "exact"),
        default=DEFAULT_SENSING_MODE,
        help=(
            "discrete uses the configured global levels; bucketed keeps the "
            "farthest new target-distance event in each level interval; "
            "exact keeps every per-sensor target-distance event."
        ),
    )
    parser.add_argument(
        "--fitness-service",
        choices=tuple(FITNESS_SERVICES),
        default=DEFAULT_FITNESS_SERVICE,
        help=(
            "Fitness evaluator used by every selected algorithm. "
            f"The experiment default is {DEFAULT_FITNESS_SERVICE}."
        ),
    )
    parser.add_argument(
        "--routing-service",
        choices=ROUTING_SERVICES,
        default=DEFAULT_ROUTING_SERVICE,
        help=(
            "Routing tree builder used by every selected algorithm. "
            "The currently supported service is v1."
        ),
    )
    parser.add_argument(
        "--execution",
        choices=("sequential", "parallel"),
        default="sequential",
        help=(
            "sequential (default) runs one case at a time. parallel runs "
            "several (map, run, algorithm) cases at once in separate "
            "processes. Each case is seeded from --seed plus its run id, so "
            "both modes produce identical results; a single-case experiment "
            "always runs in this process."
        ),
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help=(
            "How many cases --execution parallel runs at once. 0 (default) "
            "picks a value that leaves room for each case's own SE market "
            "workers. Ignored in sequential mode."
        ),
    )
    parser.add_argument(
        "--result-dir",
        default="experiment_results",
        help="Base directory for grouped experiment results.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional folder name under --result-dir. Default is timestamp + algorithm.",
    )
    return parser


def selected_algorithms(requested):
    """Expand the --algorithm value into the algorithms that will be run."""
    if isinstance(requested, str):
        requested = [requested]
    if "all" in requested:
        return list(ALL_RUNNABLE_ALGORITHMS)
    return list(requested)


def _resolve_evaluate_budget(args, explicit):
    """Apply the per-algorithm budget default when --evaluate was not given."""
    if explicit:
        return args.evaluate
    selection = set(args.algorithm)
    if len(selection) == 1:
        override = EVALUATE_OVERRIDES.get(next(iter(selection)))
        if override is not None:
            return override
    return EXPERIMENT_PRESET["evaluate"]


def parse_args(argv=None):
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args.evaluate = _resolve_evaluate_budget(args, args.evaluate is not None)

    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.jobs < 0:
        parser.error("--jobs must not be negative")
    try:
        args.map_specs = load_map_specs(args.maps)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        parser.error(str(error))

    selected = set(selected_algorithms(args.algorithm))
    selected_rl = selected & RL_SETS_ALGORITHMS
    if len(selected_rl) > 1:
        parser.error(
            "RL-SETS versions use incompatible checkpoints; "
            "run each version as a separate experiment"
        )
    for name in sorted(selected_rl):
        model = configured_algorithm_params(name, args)["model_path"]
        if not model and name not in RANDOM_POLICY_ALGORITHMS:
            parser.error(f"{name} requires --rl-model; train one with train_rl_sets.py")
        if model and not Path(model).expanduser().is_file():
            parser.error(f"{name} model does not exist: {model}")
    return args
