"""Train the RL-SETS D3QN policy separately from formal experiments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from Algorithm.se.RL_SETS import RL_SETS
from Algorithm.se.RL_SETSv2 import RL_SETSv2
from Algorithm.se.RL_SETSv3 import RL_SETSv3
from Algorithm.se.RL_SETSv4 import RL_SETSv4
from Algorithm.se.RL_SETSv5 import RL_SETSv5
from Algorithm.se.SA_SETS import SA_SETS
from CreateMapData import dataset_fingerprint
from environment_defaults import (
    DEFAULT_FITNESS_SERVICE, DEFAULT_ROUTING_SERVICE, DEFAULT_SENSING_MODE,
    DEFAULT_SENSOR_ENCODING, environment_contract,
)
from experiment_algorithms import (
    EXPERIMENT_PRESET,
    ALGORITHM_PRESETS,
    FITNESS_SERVICES,
    ROUTING_SERVICES,
    build_problem,
    load_map_specs,
    run_optimizer,
    prepare_result_state,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train RL-SETS with lifetime rollouts and save a D3QN checkpoint."
        )
    )
    parser.add_argument(
        "--rl-model",
        dest="model",
        default=None,
        help="Output .npz checkpoint used later by --rl-model.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume weights and optimizer state when --rl-model already exists."
        ),
    )
    parser.add_argument("--maps", default="maps/maps_100100100_train.json")
    parser.add_argument("--validation-maps", default=None,
                        help="Optional separate manifest; reads its maps list for validation.")
    parser.add_argument(
        "--algorithm",
        choices=(
            "rl_sets", "rl_setsv2", "rl_setsv3", "rl_setsv4", "rl_setsv5",
        ),
        default="rl_setsv5",
    )
    parser.add_argument("--episodes", "--episode", type=int, default=None,
                        help="Total lifetime episodes; default is passes * training map count.")
    parser.add_argument("--passes", type=int, default=None)
    parser.add_argument(
        "--map-limit",
        type=int,
        default=None,
        help="Use only the first N training maps; rl_sets defaults to 60.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--validation-every", type=int, default=None,
                        help="Validate every N episodes; default once per map pass, 0 disables.")
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[7007, 7008, 7009])
    parser.add_argument("--exploration-passes", type=float, default=None,
                        help="Decay epsilon over this many map passes, not early action steps.")
    parser.add_argument("--epsilon-schedule", choices=("maps", "steps"), default="maps")
    parser.add_argument(
        "--evaluate",
        type=int,
        default=EXPERIMENT_PRESET["evaluate"],
        help="Fitness-evaluation budget for each re-optimization.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=EXPERIMENT_PRESET["max_rounds"],
        help="Maximum re-optimizations in one lifetime episode.",
    )
    parser.add_argument(
        "--segment-slot-limit",
        type=int,
        default=EXPERIMENT_PRESET["segment_slot_limit"],
    )
    parser.add_argument(
        "--max-lifetime",
        type=int,
        default=None,
        help="Optional lifetime cap for short training checks.",
    )
    parser.add_argument(
        "--sensing-mode",
        choices=("discrete", "bucketed", "exact"),
        default=DEFAULT_SENSING_MODE,
    )
    parser.add_argument(
        "--fitness-service",
        choices=tuple(FITNESS_SERVICES),
        default=DEFAULT_FITNESS_SERVICE,
    )
    parser.add_argument(
        "--routing-service",
        choices=ROUTING_SERVICES,
        default=DEFAULT_ROUTING_SERVICE,
    )
    presets = ALGORITHM_PRESETS["rl_sets"]["params"]
    parser.add_argument("--n", type=int, default=presets["n"])
    parser.add_argument("--h", type=int, default=presets["h"])
    parser.add_argument("--w", type=int, default=presets["w"])
    parser.add_argument("--mu", type=float, default=presets["mu"])
    parser.add_argument("--replay-warmup", type=int, default=256)
    parser.add_argument("--replay-capacity", type=int, default=None)
    parser.add_argument("--replay-group-capacity", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9995)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--gradient-steps", type=int, default=1)
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    if "--evaluate" not in raw_argv and args.algorithm in ("rl_setsv4", "rl_setsv5"):
        args.evaluate = 5000
    if args.replay_capacity is None:
        args.replay_capacity = 200000 if args.algorithm == "rl_setsv5" else 20000
    if args.replay_group_capacity is None and args.algorithm == "rl_setsv5":
        args.replay_group_capacity = 1000
    if "--epsilon-end" not in raw_argv and args.algorithm == "rl_setsv5":
        args.epsilon_end = 0.10
    if args.model is None:
        args.model = f"Models/{args.algorithm}_generalized.npz"
    try:
        args.map_specs = load_map_specs(args.maps)
        if args.map_limit is None and args.algorithm in (
            "rl_sets", "rl_setsv2", "rl_setsv3", "rl_setsv4"
        ):
            args.map_limit = 60
        if args.map_limit is not None:
            if args.map_limit <= 0:
                raise ValueError("map-limit must be positive")
            args.map_specs = args.map_specs[: args.map_limit]
        if not args.map_specs:
            raise ValueError("training map selection is empty")
        if args.passes is None:
            args.passes = 1
        if args.exploration_passes is None:
            args.exploration_passes = 1.0
        if args.validation_maps:
            args.validation_specs = load_map_specs(args.validation_maps)
        elif args.validation_every == 0 or (
            args.validation_every is None
            and args.algorithm in (
                "rl_sets", "rl_setsv2", "rl_setsv3", "rl_setsv4"
            )
        ):
            args.validation_specs = []
        else:
            args.validation_specs = load_map_specs(args.maps, section="validation_maps")
        audit_training_maps(args.map_specs, args.validation_specs)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    if args.episodes is None:
        args.episodes = args.passes * len(args.map_specs)
    if args.validation_every is None:
        args.validation_every = (
            0 if args.algorithm in (
                "rl_sets", "rl_setsv2", "rl_setsv3", "rl_setsv4"
            ) else len(args.map_specs)
        )
    if args.validation_every < 0 or args.passes <= 0 or args.exploration_passes <= 0:
        parser.error("passes must be positive and validation-every non-negative")
    if not 0 < args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("epsilon requires 0 < end <= start <= 1")

    for name in ("episodes", "evaluate", "max_rounds", "segment_slot_limit"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.replay_capacity <= 0:
        parser.error("--replay-capacity must be positive")
    if args.replay_group_capacity is not None and args.replay_group_capacity <= 0:
        parser.error("--replay-group-capacity must be positive")
    if args.max_lifetime is not None and args.max_lifetime <= 0:
        parser.error("--max-lifetime must be positive")
    return args


def audit_training_maps(training, validation):
    """Reject shared coordinates, even if files were renamed or ids reordered."""
    reserved = load_map_specs("maps/maps_100100100.json") + load_map_specs(
        "maps/maps_100100100_a2.json"
    )
    seen = {dataset_fingerprint(spec): "reserved experiment map" for spec in reserved}
    reference = tuple(training[0][key] for key in ("boundary", "sensors", "targets", "full_energy"))
    for split, specs in (("train", training), ("validation", validation)):
        for spec in specs:
            dimensions = tuple(spec[key] for key in ("boundary", "sensors", "targets", "full_energy"))
            if dimensions != reference:
                raise ValueError("This training cohort requires a common scale and initial energy")
            fingerprint = dataset_fingerprint(spec)
            if spec.get("geometry_sha256") not in (None, fingerprint):
                raise ValueError(f"map coordinates changed: {spec['name']}")
            if fingerprint in seen:
                raise ValueError(f"{split} map {spec['name']} overlaps {seen[fingerprint]}")
            spec["geometry_sha256"] = fingerprint
            seen[fingerprint] = f"{split}/{spec['name']}"


def episode_plan(args):
    """Shuffled passes guarantee each map is visited before any is repeated."""
    rng = np.random.default_rng(np.random.SeedSequence([args.seed, 0x4D4150]))
    plan = []
    while len(plan) < args.episodes:
        for index in rng.permutation(len(args.map_specs)):
            if len(plan) == args.episodes:
                break
            plan.append((args.map_specs[int(index)], args.seed + len(plan)))
    return plan


def build_training_problem(args, seed, map_spec=None):
    """Build exactly the same WSN configuration as the experiment runner."""
    problem_args = SimpleNamespace(
        sensing_mode=args.sensing_mode,
        fitness_service=args.fitness_service,
        routing_service=args.routing_service,
        _current_map=map_spec if map_spec is not None else args.map_specs[0],
    )
    return build_problem(problem_args, seed)


def create_learner(problem, args, model_path):
    """Create a fresh learner or resume the requested checkpoint."""
    if args.resume and not model_path.is_file():
        raise FileNotFoundError(
            f"cannot resume because the checkpoint does not exist: {model_path}"
        )
    load_path = model_path if args.resume else None
    if not args.resume and model_path.exists():
        raise FileExistsError(f"checkpoint exists: {model_path}; use a new path or --resume")
    algorithm_types = {
        "rl_sets": RL_SETS,
        "rl_setsv2": RL_SETSv2,
        "rl_setsv3": RL_SETSv3,
        "rl_setsv4": RL_SETSv4,
        "rl_setsv5": RL_SETSv5,
    }
    algorithm_type = algorithm_types[args.algorithm]
    extra = dict(replay_group_capacity=args.replay_group_capacity)
    if args.algorithm == "rl_setsv5":
        extra["lifetime_normalization"] = args.segment_slot_limit
    algorithm = algorithm_type(
        problem,
        n=args.n,
        h=args.h,
        w=args.w,
        mu=args.mu,
        training=True,
        model_path=load_path,
        replay_capacity=args.replay_capacity,
        replay_warmup=args.replay_warmup,
        batch_size=args.batch_size,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay if args.epsilon_schedule == "steps" else 1.0,
        target_update_interval=args.target_update_interval,
        gradient_steps=args.gradient_steps,
        seed=args.seed,
        **extra,
    )
    # New runs use a distinct path; existing checkpoints require explicit resume.
    algorithm.model_path = model_path
    return algorithm


def run_lifetime_episode(algorithm, problem, args, episode, phase="train"):
    """Train over successive energy states until this WSN episode ends."""
    lifetime = 0
    segments = 0
    stop_reason = "max_rounds"
    actual_evaluations = 0
    started = time.perf_counter()
    episode_actions = {}
    reserved_sa_uses = 0
    reward_by_progress = {}

    while segments < args.max_rounds:
        result = run_optimizer(algorithm, problem, args.evaluate)
        actual_evaluations += int(result["evaluations"])
        if hasattr(algorithm, "policy_statistics"):
            policy = algorithm.policy_statistics()
            reserved_sa_uses += policy.get("reserved_sa_uses", 0)
            for name, counts in policy["actions"].items():
                total = episode_actions.setdefault(name, {"uses": 0, "successes": 0})
                total["uses"] += counts["uses"]
                total["successes"] += counts["successes"]
                # 成功率分母是投資數；reward 分布分母是區域動作決策數。
                for key in ("decisions", "reward_sum", "positive_rewards",
                            "zero_rewards", "negative_rewards", "improved_goods", "degraded_goods"):
                    total[key] = total.get(key, 0) + counts[key]
            for phase_name, counts in policy["reward_by_progress"].items():
                total = reward_by_progress.setdefault(phase_name, {})
                for key in ("decisions", "reward_sum", "positive_rewards",
                            "zero_rewards", "negative_rewards", "improved_goods", "degraded_goods"):
                    total[key] = total.get(key, 0) + counts[key]
        state = prepare_result_state(algorithm, problem, result["state"])
        if hasattr(algorithm, "policy_statistics"):
            if algorithm.agent.last_loss is not None and not np.isfinite(algorithm.agent.last_loss):
                raise FloatingPointError("non-finite training loss")
        if state is None:
            stop_reason = "no_state"
            break

        cost = problem.calculate_total_cost(state)
        if not problem.is_state_alive(state, cost):
            stop_reason = "infeasible_solution"
            break

        segment_length = 0
        for _ in range(args.segment_slot_limit):
            if not problem.is_state_alive(state, cost):
                break
            problem.energy = problem.calculate_remaining_energy(cost)
            lifetime += 1
            segment_length += 1
            if (
                args.max_lifetime is not None
                and lifetime >= args.max_lifetime
            ):
                stop_reason = "lifetime_cap"
                break

        if segment_length > 0:
            problem.prepare_coding_cache(cost * segment_length)

        statistics = (
            algorithm.policy_statistics()
            if hasattr(algorithm, "policy_statistics")
            else {}
        )
        mean_reward = (
            float(np.mean(algorithm.reward_history))
            if getattr(algorithm, "reward_history", [])
            else 0.0
        )
        if phase == "train":
            print(
                f"[{phase}]", f"episode={episode}", f"segment={segments}",
                f"lifetime={lifetime}",
                f"fitness={float(np.sum(result['objectives'])):.6f}",
                f"reward={mean_reward:.6f}",
                f"reward_pos={statistics['reward']['positive_rate']:.1%}",
                f"reward_zero={statistics['reward']['zero_rate']:.1%}",
                f"reward_neg={statistics['reward']['negative_rate']:.1%}",
                f"epsilon={statistics['epsilon']:.6f}",
                f"replay={statistics['replay_size']}",
                f"updates={statistics['training_steps']}",
            )
        segments += 1

        if stop_reason == "lifetime_cap":
            break
        if segment_length == 0:
            stop_reason = "energy_depleted"
            break
    return {
        "episode": int(episode),
        "lifetime": int(lifetime),
        "segments": int(segments),
        "stop_reason": stop_reason,
        "actual_evaluations": actual_evaluations,
        "episode_actions": episode_actions,
        "reserved_sa_uses": reserved_sa_uses,
        "reward_by_progress": reward_by_progress,
        "elapsed_seconds": time.perf_counter() - started,
        "disconnected_count": 0 if state is None else len(problem.find_disconnected_sensors(state)),
        "uncovered_count": 0 if state is None else len(problem.find_uncovered_targets(state)),
    }


def validate_checkpoint(args, model_path, baseline_cache):
    """Evaluate frozen policies and SA on paired, held-out lifetime cases."""
    rows = []
    algorithm_types = {
        "rl_sets": RL_SETS,
        "rl_setsv2": RL_SETSv2,
        "rl_setsv3": RL_SETSv3,
        "rl_setsv4": RL_SETSv4,
        "rl_setsv5": RL_SETSv5,
    }
    algorithm_type = algorithm_types[args.algorithm]
    for spec in args.validation_specs:
        for seed in args.validation_seeds:
            problem = build_training_problem(args, seed, spec)
            validation_extra = (
                {"lifetime_normalization": args.segment_slot_limit}
                if args.algorithm == "rl_setsv5" else {}
            )
            frozen = algorithm_type(
                problem, n=args.n, h=args.h, w=args.w, mu=args.mu,
                model_path=model_path, training=False, seed=seed,
                **validation_extra,
            )
            before = frozen.agent.training_steps
            result = run_lifetime_episode(frozen, problem, args, 0, phase="validation")
            if frozen.agent.epsilon != 0 or frozen.agent.training_steps != before or len(frozen.agent.replay):
                raise RuntimeError("validation unexpectedly changed the learner")
            key = f"{spec['name']}:{seed}"
            if key not in baseline_cache:
                baseline_problem = build_training_problem(args, seed, spec)
                baseline = SA_SETS(baseline_problem, n=args.n, h=args.h, w=args.w,
                                   mu=args.mu, seed=seed)
                baseline_cache[key] = run_lifetime_episode(
                    baseline, baseline_problem, args, 0, phase="baseline"
                )["lifetime"]
            baseline_life = baseline_cache[key]
            row = dict(result, map=spec["name"], seed=seed,
                       sa_lifetime=baseline_life,
                       lifetime_ratio=result["lifetime"] / max(1, baseline_life),
                       policy=frozen.policy_statistics())
            rows.append(row)
            print("[validation]", spec["name"], seed, "RL", result["lifetime"], "SA", baseline_life)
    return {
        "mean_lifetime": float(np.mean([row["lifetime"] for row in rows])),
        "mean_lifetime_ratio": float(np.mean([row["lifetime_ratio"] for row in rows])),
        "sa_win_rate": float(np.mean([row["lifetime"] > row["sa_lifetime"] for row in rows])),
        "zero_lifetime_cases": sum(row["lifetime"] == 0 for row in rows),
        "rows": rows,
    }


def write_training_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main(argv=None):
    args = parse_args(argv)
    model_path = Path(args.model).expanduser().resolve()
    report_path = model_path.with_suffix(".training.json")
    best_path = model_path.with_name(model_path.stem + ".best.npz")
    if not args.resume:
        for existing in (report_path, best_path):
            if existing.exists():
                raise FileExistsError(f"training output already exists: {existing}; use a new path")
    plan = episode_plan(args)
    first_problem = build_training_problem(args, plan[0][1], plan[0][0])
    algorithm = create_learner(first_problem, args, model_path)
    learner_protocol = {name: getattr(args, name) for name in (
        "replay_warmup", "batch_size", "gamma", "learning_rate",
        "target_update_interval", "gradient_steps",
    )}
    learner_protocol.update(
        replay_capacity=args.replay_capacity,
        replay_group_capacity=args.replay_group_capacity,
    )
    protocol = {
        "algorithm": args.algorithm, "seed": args.seed,
        "reward_version": algorithm.REWARD_VERSION,
        "maps": args.map_specs, "validation_maps": args.validation_specs,
        "validation_seeds": args.validation_seeds,
        "evaluate": args.evaluate, "max_rounds": args.max_rounds,
        "segment_slot_limit": args.segment_slot_limit, "max_lifetime": args.max_lifetime,
        "n": args.n, "h": args.h, "w": args.w, "mu": args.mu,
        "epsilon_schedule": args.epsilon_schedule,
        "exploration_passes": args.exploration_passes,
        "epsilon_start": args.epsilon_start, "epsilon_end": args.epsilon_end,
        "epsilon_decay": args.epsilon_decay,
        "validation_every": args.validation_every,
        "learner": learner_protocol,
        "environment": environment_contract(first_problem),
    }
    report = {
        "protocol": protocol, "episodes": [], "validations": [],
        "best_validation_mean_lifetime": None, "baseline_cache": {},
        "planned_episodes": args.episodes,
        "resume_note": (
            "Current checkpoints restore weights, optimizer, replay and learner RNG."
        ),
    }
    if args.resume:
        if not report_path.is_file():
            raise ValueError("resume requires the matching .training.json report")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["protocol"] != protocol:
            raise ValueError("resume protocol changed; start a new checkpoint")
        report["planned_episodes"] = args.episodes
        # Match the checkpoint/report boundary; a partial episode is retried.
        saved = getattr(algorithm.agent, "checkpoint_metadata", {})
        if saved.get("completed_episodes", len(report["episodes"])) != len(report["episodes"]):
            raise ValueError("checkpoint/report episode counts differ; inspect the interrupted run")
    start_episode = len(report["episodes"])

    def record_validation(after_episode):
        validation = validate_checkpoint(args, model_path, report["baseline_cache"])
        validation["after_episode"] = after_episode
        report["validations"].append(validation)
        score = validation["mean_lifetime"]
        best_score = report["best_validation_mean_lifetime"]
        if best_score is None or score > best_score:
            report["best_validation_mean_lifetime"] = score
            report["best_after_episode"] = after_episode
            # Saving a best snapshot must not redirect ongoing training.
            algorithm.agent.save(best_path)
        write_training_report(report_path, report)

    # Recover a validation interrupted after a completed episode checkpoint.
    if args.resume and start_episode > 0 and args.validation_every > 0:
        validation_due = start_episode % args.validation_every == 0 or start_episode >= args.episodes
        already_validated = any(row["after_episode"] == start_episode for row in report["validations"])
        if validation_due and not already_validated:
            record_validation(start_episode)
    if start_episode >= args.episodes:
        print("Already completed requested episodes:", start_episode)
        return report["episodes"]
    write_training_report(report_path, report)
    print("[training-plan]", args.algorithm, len(args.map_specs), "training maps;",
          len(args.validation_specs), "validation maps;", args.episodes, "episodes;",
          "fitness", args.fitness_service, "routing", args.routing_service,
          "decode", DEFAULT_SENSOR_ENCODING)
    try:
        replay_groups = {
            spec["name"]: index for index, spec in enumerate(args.map_specs)
        }
        for episode in range(start_episode, args.episodes):
            spec, seed = plan[episode]
            problem = first_problem if episode == 0 else build_training_problem(args, seed, spec)
            # Refresh search streams on the new map but retain learned network,
            # replay, optimizer and the learner's independent RNG across maps.
            algorithm._seed_random_streams(seed)
            algorithm.agent.set_replay_group(replay_groups[spec["name"]])
            if args.epsilon_schedule == "maps":
                progress = min(1.0, episode / max(1.0, args.exploration_passes * len(args.map_specs)))
                algorithm.agent.epsilon_start = args.epsilon_start * (
                    args.epsilon_end / args.epsilon_start
                ) ** progress
                algorithm.agent.epsilon_decay = 1.0
            print("[episode-map]", episode, spec["name"], "search_seed", seed,
                  "epsilon", algorithm.agent.epsilon)
            summary = run_lifetime_episode(algorithm, problem, args, episode)
            summary.update(
                map=spec["name"], map_seed=spec["map_seed"], seed=seed,
                policy=algorithm.policy_statistics(),
            )
            report["episodes"].append(summary)
            if hasattr(algorithm.agent, "checkpoint_metadata"):
                algorithm.agent.checkpoint_metadata["completed_episodes"] = episode + 1
            algorithm.save_model(model_path)
            write_training_report(report_path, report)
            validate_now = args.validation_every > 0 and (
                (episode + 1) % args.validation_every == 0 or episode + 1 == args.episodes
            )
            if validate_now:
                record_validation(episode + 1)
    except KeyboardInterrupt:
        print("Interrupted: the last completed episode is saved; partial episode discarded.")
        raise
    print("[training-complete]", len(report["episodes"]), "episodes;",
          "latest", model_path, "best", best_path if report["validations"] else "not validated")
    return report["episodes"]


if __name__ == "__main__":
    main()
