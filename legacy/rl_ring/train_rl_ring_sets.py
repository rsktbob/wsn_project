"""Train RL_Ring_SETS / RL_Ring_SETSpriority (per-pair D3QN segment choice).

訓練流程：
1. 一個 episode = 在一張訓練地圖上跑完整個 lifetime（每段重新最佳化，直到網路
   死亡）。地圖依洗牌後的順序輪流，200 張訓練地圖與 40 張驗證地圖都和
   MAP01–06、A2 的座標不重疊（沿用 train_rl_sets 的檢查）。
2. ε 依 episode 進度從 1.0 指數下降，到 ``exploration_fraction`` 的 episode 時
   降到 ``epsilon_end``，之後固定。
3. 每回合 n×w 筆 transition 進 replay，做 ``gradient_steps`` 次 Double-DQN 更新。
4. 每 ``validation_every`` 個 episode 存檔並凍結評估，和同樣編碼的隨機選區基準
   （Ring_SETS／Ring_SETSpriority）成對比較，保留最好的 checkpoint 為 ``*.best.npz``。
   * ``--validation-mode snapshot``（v2 預設）：訓練前先在驗證地圖上用 GA 推進
     lifetime，每隔 ``snapshot_stride`` 段存下完整的問題狀態（快照）。每次驗證在
     每個快照 × 每個驗證預算上跑一次，比較段末最好解的 fitness，指標是各預算
     「贏隨機的比例」的平均（平手算 0.5）。隨機基準只算一次並快取。
   * ``--validation-mode lifetime``（v1 預設）：驗證地圖上跑完整 lifetime 比較。
5. ``--budgets``：每段從清單中隨機抽一個評估預算（v2 用 1000 3000 10000）。

用法：
    python train_rl_ring_sets.py --algorithm rl_ring_sets --rl-model Models/rl_ring_sets.npz
短測：加上 ``--episodes 2 --max-lifetime 200 --validation-every 0``。
"""

from __future__ import annotations

import argparse
import copy
import gzip
import json
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from Algorithm.se.RL_Ring_SETS import RL_Ring_SETS
from Algorithm.se.RL_Ring_SETSpriority import RL_Ring_SETSpriority
from Algorithm.se.RL_Ring_SETSpriorityv2 import RL_Ring_SETSpriorityv2
from Algorithm.se.RL_Ring_SETSv2 import RL_Ring_SETSv2
from Algorithm.se.Ring_SETS import Ring_SETS
from Algorithm.se.Ring_SETSpriority import Ring_SETSpriority
from CreateMapData import dataset_fingerprint
from experiment_algorithms import (
    ALGORITHM_PRESETS,
    build_algorithm,
    build_problem,
    load_map_specs,
    prepare_result_state,
    run_optimizer,
)
from experiments.config import (
    DEFAULT_FITNESS_SERVICE,
    DEFAULT_ROUTING_SERVICE,
    DEFAULT_SENSING_MODE,
)

ALGORITHMS = {
    "rl_ring_sets": (RL_Ring_SETS, Ring_SETS),
    "rl_ring_setspriority": (RL_Ring_SETSpriority, Ring_SETSpriority),
    "rl_ring_setsv2": (RL_Ring_SETSv2, Ring_SETS),
    "rl_ring_setspriorityv2": (RL_Ring_SETSpriorityv2, Ring_SETSpriority),
}
V2_ALGORITHMS = ("rl_ring_setsv2", "rl_ring_setspriorityv2")


# ---- 地圖與問題（原本在已移到 legacy 的 train_rl_sets.py，內容不變） --------
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


def write_training_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--algorithm", choices=tuple(ALGORITHMS), default="rl_ring_sets")
    parser.add_argument("--rl-model", dest="model", default=None)
    parser.add_argument("--maps", default="maps/maps_100100100_train.json")
    parser.add_argument("--map-limit", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--evaluate", type=int, default=5000,
                        help="Evaluation budget of each re-optimization (when --budgets is not given).")
    parser.add_argument("--budgets", type=int, nargs="+", default=None,
                        help="Draw each segment's budget uniformly from these values.")
    parser.add_argument("--validation-mode", choices=("snapshot", "lifetime"), default=None)
    parser.add_argument("--validation-budgets", type=int, nargs="+", default=[1000, 3000, 10000])
    parser.add_argument("--snapshots", default="Models/validation_snapshots.pkl.gz")
    parser.add_argument("--snapshot-stride", type=int, default=2)
    parser.add_argument("--snapshot-max-segments", type=int, default=30)
    parser.add_argument("--snapshot-ga-evaluate", type=int, default=3000)
    parser.add_argument("--max-rounds", type=int, default=2000)
    parser.add_argument("--segment-slot-limit", type=int, default=5000)
    parser.add_argument("--max-lifetime", type=int, default=None,
                        help="Lifetime cap for short checks.")
    parser.add_argument("--validation-every", type=int, default=10)
    parser.add_argument("--validation-map-count", type=int, default=6)
    parser.add_argument("--validation-seeds", type=int, nargs="+", default=[7007])
    parser.add_argument("--exploration-fraction", type=float, default=0.6)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--gamma", type=float, default=None,
                        help="Default: the algorithm's own (v1 0.5, v2 0).")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-steps", type=int, default=4)
    parser.add_argument("--replay-capacity", type=int, default=200000)
    parser.add_argument("--replay-warmup", type=int, default=1024)
    parser.add_argument("--target-update-interval", type=int, default=500)
    parser.add_argument("--sensing-mode", default=DEFAULT_SENSING_MODE)
    parser.add_argument("--fitness-service", default=DEFAULT_FITNESS_SERVICE)
    parser.add_argument("--routing-service", default=DEFAULT_ROUTING_SERVICE)
    args = parser.parse_args(argv)

    preset = ALGORITHM_PRESETS[args.algorithm]["params"]
    for key in ("n", "h", "w", "mu", "hidden_size"):
        setattr(args, key, preset[key])
    if args.model is None:
        args.model = f"Models/{args.algorithm}.npz"
    if args.budgets is None:
        args.budgets = [args.evaluate]
    if args.validation_mode is None:
        args.validation_mode = "snapshot" if args.algorithm in V2_ALGORITHMS else "lifetime"
    args.map_specs = load_map_specs(args.maps)
    if args.map_limit is not None:
        args.map_specs = args.map_specs[: args.map_limit]
    validation = load_map_specs(args.maps, section="validation_maps")
    args.validation_specs = validation[: args.validation_map_count] if args.validation_every else []
    audit_training_maps(args.map_specs, args.validation_specs)
    if not 0 < args.epsilon_end <= args.epsilon_start <= 1:
        parser.error("epsilon requires 0 < end <= start <= 1")
    return args


def make_learner(args, problem):
    learner_type, _ = ALGORITHMS[args.algorithm]
    extra = {} if args.gamma is None else {"gamma": args.gamma}
    return learner_type(
        problem, n=args.n, h=args.h, w=args.w, mu=args.mu, seed=args.seed,
        training=True, hidden_size=args.hidden_size,
        replay_capacity=args.replay_capacity, replay_warmup=args.replay_warmup,
        batch_size=args.batch_size, **extra,
        learning_rate=args.learning_rate, epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end, epsilon_decay=1.0,
        target_update_interval=args.target_update_interval,
        gradient_steps=args.gradient_steps,
    )


def run_lifetime(algorithm, problem, args, log_prefix=None, budget_fn=None):
    """One full lifetime; returns lifetime, segments and per-segment policy stats."""
    lifetime, segments, stop_reason = 0, 0, "max_rounds"
    segment_stats = []
    started = time.perf_counter()
    while segments < args.max_rounds:
        if hasattr(algorithm, "reset_policy_statistics"):
            algorithm.reset_policy_statistics()
        budget = args.evaluate if budget_fn is None else budget_fn()
        result = run_optimizer(algorithm, problem, budget)
        state = prepare_result_state(algorithm, problem, result["state"])
        if state is None:
            stop_reason = "no_state"
            break
        cost = problem.calculate_total_cost(state)
        if not problem.is_state_alive(state, cost):
            stop_reason = "infeasible_solution"
            break
        length = 0
        for _ in range(args.segment_slot_limit):
            if not problem.is_state_alive(state, cost):
                break
            problem.energy = problem.calculate_remaining_energy(cost)
            lifetime += 1
            length += 1
            if args.max_lifetime is not None and lifetime >= args.max_lifetime:
                stop_reason = "lifetime_cap"
                break
        if length > 0:
            problem.prepare_coding_cache(cost * length)
        if hasattr(algorithm, "policy_statistics"):
            stats = algorithm.policy_statistics()
            stats.update(segment=segments, length=length, budget=budget,
                         fitness=float(np.sum(result["objectives"])))
            segment_stats.append(stats)
            if log_prefix:
                print(log_prefix, f"segment={segments}", f"budget={budget}", f"life={lifetime}",
                      f"fitness={stats['fitness']:.6f}",
                      f"reward_share={stats.get('reward_share')}",
                      f"reward_by_phase={stats['reward_by_phase']}",
                      f"effective_share={stats['effective_bucket_share']}",
                      f"eps={stats['epsilon']:.3f}", f"replay={stats['replay_size']}",
                      f"loss={stats['mean_loss']}", flush=True)
        segments += 1
        if stop_reason == "lifetime_cap":
            break
        if length == 0:
            stop_reason = "energy_depleted"
            break
    return {
        "lifetime": lifetime, "segments": segments, "stop_reason": stop_reason,
        "elapsed_seconds": time.perf_counter() - started, "segment_stats": segment_stats,
    }


def build_validation_snapshots(args):
    """GA-led lifetimes on the validation maps; keep every ``stride``-th segment's problem."""
    path = Path(args.snapshots)
    meta = {
        "maps": [spec["name"] for spec in args.validation_specs],
        "seed": args.validation_seeds[0], "stride": args.snapshot_stride,
        "max_segments": args.snapshot_max_segments, "ga_evaluate": args.snapshot_ga_evaluate,
        "fitness": args.fitness_service, "sensing": args.sensing_mode, "routing": args.routing_service,
        "segment_slot_limit": args.segment_slot_limit,
    }
    if path.is_file():
        with gzip.open(path, "rb") as handle:
            saved = pickle.load(handle)
        if saved["meta"] != meta:
            raise ValueError(f"{path} was built with different settings; use another --snapshots path")
        return saved["snapshots"]
    snapshots = []
    seed = args.validation_seeds[0]
    for spec in args.validation_specs:
        problem = build_training_problem(args, seed, spec)
        ga = build_algorithm("ga", problem, SimpleNamespace(rl_model=None, evaluate=args.snapshot_ga_evaluate), seed)
        for segment in range(args.snapshot_max_segments):
            if segment % args.snapshot_stride == 0:
                snapshots.append({"map": spec["name"], "segment": segment,
                                  "problem": pickle.dumps(copy.deepcopy(problem), protocol=pickle.HIGHEST_PROTOCOL)})
            result = run_optimizer(ga, problem, args.snapshot_ga_evaluate)
            state = prepare_result_state(ga, problem, result["state"])
            if state is None:
                break
            cost = problem.calculate_total_cost(state)
            length = 0
            for _ in range(args.segment_slot_limit):
                if not problem.is_state_alive(state, cost):
                    break
                problem.energy = problem.calculate_remaining_energy(cost)
                length += 1
            if length == 0:
                break
            problem.prepare_coding_cache(cost * length)
        print("[snapshots]", spec["name"], "total so far", len(snapshots), flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as handle:
        pickle.dump({"meta": meta, "snapshots": snapshots}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return snapshots


def validate_on_snapshots(args, model_path, snapshots, baseline_cache):
    """Frozen policy vs the random-segment baseline on identical snapshots and budgets."""
    learner_type, baseline_type = ALGORITHMS[args.algorithm]
    rows = []
    for index, snapshot in enumerate(snapshots):
        seed = 7007 + index
        for budget in args.validation_budgets:
            key = f"{snapshot['map']}:{snapshot['segment']}:{budget}"
            problem = pickle.loads(snapshot["problem"])
            frozen = learner_type(problem, n=args.n, h=args.h, w=args.w, mu=args.mu,
                                  seed=seed, model_path=model_path, hidden_size=args.hidden_size)
            frozen.run(problem, budget=budget)
            if key not in baseline_cache:
                base_problem = pickle.loads(snapshot["problem"])
                base = baseline_type(base_problem, n=args.n, h=args.h, w=args.w, mu=args.mu, seed=seed)
                base.run(base_problem, budget=budget)
                baseline_cache[key] = float(base.fitness)
            rows.append({"key": key, "budget": budget, "rl": float(frozen.fitness),
                         "baseline": baseline_cache[key]})
    per_budget = {}
    for budget in args.validation_budgets:
        diff = np.array([r["rl"] - r["baseline"] for r in rows if r["budget"] == budget])
        wins = np.where(diff > 0, 1.0, np.where(diff < 0, 0.0, 0.5))
        per_budget[str(budget)] = {"win_rate": float(wins.mean()), "median_diff": float(np.median(diff)),
                                   "n": int(len(diff))}
    score = float(np.mean([v["win_rate"] for v in per_budget.values()]))
    return {"rows": rows, "per_budget": per_budget, "score": score}


def validate(args, model_path, baseline_cache):
    learner_type, baseline_type = ALGORITHMS[args.algorithm]
    rows = []
    for spec in args.validation_specs:
        for seed in args.validation_seeds:
            problem = build_training_problem(args, seed, spec)
            frozen = learner_type(problem, n=args.n, h=args.h, w=args.w, mu=args.mu,
                                  seed=seed, model_path=model_path, hidden_size=args.hidden_size)
            life = run_lifetime(frozen, problem, args)["lifetime"]
            key = f"{spec['name']}:{seed}"
            if key not in baseline_cache:
                base_problem = build_training_problem(args, seed, spec)
                base = baseline_type(base_problem, n=args.n, h=args.h, w=args.w, mu=args.mu, seed=seed)
                baseline_cache[key] = run_lifetime(base, base_problem, args)["lifetime"]
            rows.append({"map": spec["name"], "seed": seed, "lifetime": life,
                         "baseline": baseline_cache[key]})
            print("[validation]", spec["name"], seed, "RL", life, "random", baseline_cache[key], flush=True)
    ratio = float(np.mean([r["lifetime"] / max(1, r["baseline"]) for r in rows]))
    return {"rows": rows, "mean_ratio": ratio,
            "win_rate": float(np.mean([r["lifetime"] > r["baseline"] for r in rows]))}


def main(argv=None):
    args = parse_args(argv)
    model_path = Path(args.model).expanduser().resolve()
    best_path = model_path.with_name(model_path.stem + ".best.npz")
    report_path = model_path.with_suffix(".training.json")
    for existing in (model_path, best_path, report_path):
        if existing.exists():
            raise FileExistsError(f"training output already exists: {existing}")
    plan = episode_plan(args)
    learner = make_learner(args, build_training_problem(args, plan[0][1], plan[0][0]))
    snapshots = []
    if args.validation_every and args.validation_mode == "snapshot":
        snapshots = build_validation_snapshots(args)
        print("[snapshots]", len(snapshots), "validation snapshots ×", len(args.validation_budgets), "budgets", flush=True)
    report = {
        "protocol": {key: getattr(args, key) for key in (
            "algorithm", "seed", "episodes", "evaluate", "budgets", "max_lifetime", "n", "h", "w", "mu",
            "hidden_size", "learning_rate", "batch_size", "gradient_steps",
            "replay_capacity", "replay_warmup", "target_update_interval",
            "exploration_fraction", "epsilon_start", "epsilon_end", "validation_every",
            "validation_mode", "validation_budgets", "validation_seeds", "snapshot_stride",
            "snapshot_max_segments", "snapshot_ga_evaluate")},
        "gamma": float(learner.agent.gamma),
        "training_maps": [spec["name"] for spec in args.map_specs],
        "validation_maps": [spec["name"] for spec in args.validation_specs],
        "reward_version": learner.REWARD_VERSION,
        "episodes": [], "validations": [], "baseline_cache": {}, "best": None,
    }
    print("[training-plan]", args.algorithm, len(args.map_specs), "training maps,",
          len(args.validation_specs), "validation maps,", args.episodes, "episodes, budgets", args.budgets,
          "gamma", learner.agent.gamma, "validation", args.validation_mode)
    for episode, (spec, seed) in enumerate(plan):
        problem = build_training_problem(args, seed, spec)
        learner._seed_random_streams(seed)
        progress = min(1.0, episode / max(1.0, args.exploration_fraction * args.episodes))
        learner.agent.epsilon_start = args.epsilon_start * (args.epsilon_end / args.epsilon_start) ** progress
        budget_rng = np.random.default_rng(np.random.SeedSequence([args.seed, episode, 0xB0D6E7]))
        summary = run_lifetime(learner, problem, args, log_prefix=f"[train] ep={episode} map={spec['name']}",
                               budget_fn=lambda: int(budget_rng.choice(args.budgets)))
        summary.update(episode=episode, map=spec["name"], seed=seed,
                       epsilon=float(learner.agent.epsilon))
        report["episodes"].append(summary)
        print("[episode]", episode, spec["name"], "lifetime", summary["lifetime"],
              "segments", summary["segments"], f"{summary['elapsed_seconds']:.0f}s", flush=True)
        learner.agent.checkpoint_metadata["completed_episodes"] = episode + 1
        learner.save_model(model_path)
        if args.validation_every and ((episode + 1) % args.validation_every == 0 or episode + 1 == args.episodes):
            if args.validation_mode == "snapshot":
                result = validate_on_snapshots(args, model_path, snapshots, report["baseline_cache"])
                score = result["score"]
                print("[validation-summary]", episode + 1, "score", round(score, 4),
                      {b: round(v["win_rate"], 3) for b, v in result["per_budget"].items()}, flush=True)
            else:
                result = validate(args, model_path, report["baseline_cache"])
                score = result["mean_ratio"]
                print("[validation-summary]", episode + 1, "mean ratio", round(score, 4),
                      "win rate", round(result["win_rate"], 3), flush=True)
            result["after_episode"] = episode + 1
            report["validations"].append(result)
            if report["best"] is None or score > report["best"]["score"]:
                report["best"] = {"after_episode": episode + 1, "score": score}
                learner.save_model(best_path)
        write_training_report(report_path, report)
    print("[training-complete]", model_path, "best", report["best"])


if __name__ == "__main__":
    main()
