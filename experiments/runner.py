"""The two experiment modes: one optimizer call, or a full WSN lifetime."""

import time

import numpy as np

from experiments.builder import algorithm_label, build_algorithm
from experiments.config import (
    effective_evaluate_budget,
    evaluate_budget,
    lifetime_cap,
    moving_enabled,
)
from experiments.evaluation import (
    prepare_result_state,
    run_optimizer,
    state_metrics,
)
from experiments.maps import current_map_spec
from experiments.plots import save_fitness_history_plot
from experiments.problem_setup import apply_mobility, build_problem
from experiments.reporting import (
    build_trace_row,
    common_result_fields,
    resolved_result_root,
)


def _case_output_dir(args, name, run_id, seed, kind):
    return (
        resolved_result_root(args)
        / name
        / current_map_spec(args)["name"]
        / kind
        / f"run_{run_id:02d}_seed_{seed}"
    )


def _optimize(algorithm, problem, budget):
    """One optimizer call, timed."""
    started_at = time.time()
    optimization = run_optimizer(algorithm, problem, budget)
    return optimization, time.time() - started_at


def _decode_and_measure(algorithm, problem, optimization):
    """Normalize the returned state and derive its reported metrics."""
    state = prepare_result_state(algorithm, problem, optimization["state"])
    cost = problem.calculate_total_cost(state)
    metrics = state_metrics(
        problem,
        state,
        cost,
        decode=False,
        objectives=optimization["objectives"],
    )
    return state, cost, metrics


def _as_evaluation_count(value):
    """Optimizers may report evaluations as something other than a number."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    return None


def run_single_case(name, run_id, args):
    seed = args.seed + run_id
    problem = build_problem(args, seed)
    algorithm = build_algorithm(name, problem, args, seed)
    case_evaluate_budget = effective_evaluate_budget(algorithm, args)

    optimization, elapsed = _optimize(algorithm, problem, case_evaluate_budget)
    if getattr(args, "draw_fitness_history", False):
        save_fitness_history_plot(
            optimization["history"],
            _case_output_dir(args, name, run_id, seed, "fitness_history")
            / "segment_0000_life_000000.png",
            algorithm_name=algorithm_label(name),
            segment_id=0,
            lifetime_start=0,
        )
    _, _, metrics = _decode_and_measure(algorithm, problem, optimization)

    row = common_result_fields(name, run_id, seed, args)
    row.update(
        {
            "mode": "single",
            "elapsed_seconds": round(elapsed, 6),
            "actual_evatime": optimization["evaluations"],
            "compute_device": getattr(algorithm, "device_name", ""),
            "gpu_accelerated": getattr(algorithm, "gpu_accelerated", ""),
            "effective_evaluate_budget": case_evaluate_budget,
            "requested_evaluate_budget_total": case_evaluate_budget,
            "early_stop_count": int(optimization["early_stopped"]),
            "history_length": len(optimization["history"]),
        }
    )
    row.update(metrics)
    print(
        "[single]",
        name,
        "run",
        run_id,
        "fitness",
        round(row["fitness"], 6),
        "time",
        round(elapsed, 3),
    )
    return row


def _advance_lifetime(problem, state, cost, lifetime, slot_limit, cap_lifetime):
    """Repeat one optimized state until it can no longer be sustained.

    Returns the new lifetime, how many slots this segment lasted, and why it
    stopped.
    """
    segment_length = 0
    stop_reason = "energy_depleted"
    for _ in range(slot_limit):
        if not problem.LifeCheck(state, cost):
            break
        problem.energy = problem.calculate_remaining_energy(cost)
        lifetime += 1
        segment_length += 1
        if cap_lifetime is not None and lifetime >= cap_lifetime:
            stop_reason = "lifetime_cap"
            break
    else:
        stop_reason = "segment_slot_limit"
    return lifetime, segment_length, stop_reason


def run_lifetime_case(name, run_id, args):
    seed = args.seed + run_id
    problem = build_problem(args, seed)
    algorithm = build_algorithm(name, problem, args, seed)
    max_rounds = int(args.max_rounds)
    segment_slot_limit = int(args.segment_slot_limit)
    cap_lifetime = lifetime_cap(args)
    draw_fitness_history = getattr(args, "draw_fitness_history", False)

    lifetime = 0
    reoptimizations = 0
    movement_events = 0
    moved_sensor_count = 0
    total_actual_evatime = 0
    total_requested_evaluate_budget = 0
    early_stop_count = 0
    last_history_length = 0
    previous_segment_length = None
    trace_rows = []
    last_metrics = {}
    stop_reason = "max_rounds"
    started_at = time.time()
    energy_frame = 0
    energy_dir = None
    show_energy = getattr(args, "show", False)
    if getattr(args, "draw_save", False):
        # Pillow and the drawing module are loaded only when explicitly enabled.
        from plotting.experiment_energy_plot import save_energy_snapshot

        energy_dir = _case_output_dir(args, name, run_id, seed, "energy")
        save_energy_snapshot(
            problem,
            energy_dir / f"{energy_frame:04d}_initial_life_{lifetime:06d}.png",
            algorithm=name,
            run_id=run_id,
            seed=seed,
            stage="initial",
            lifetime=lifetime,
        )
        print("energy_images", energy_dir)
        energy_frame += 1

    while reoptimizations < max_rounds:
        segment_id = reoptimizations
        segment_start_lifetime = lifetime
        segment_movement_events = 0
        segment_moved_sensor_count = 0
        prepare_reoptimization = getattr(
            algorithm, "prepare_reoptimization", None
        )
        if prepare_reoptimization is not None:
            prepare_reoptimization(previous_segment_length)
        segment_evaluate_budget = effective_evaluate_budget(algorithm, args)
        total_requested_evaluate_budget += segment_evaluate_budget
        optimization, optimize_elapsed = _optimize(
            algorithm, problem, segment_evaluate_budget
        )
        if draw_fitness_history:
            save_fitness_history_plot(
                optimization["history"],
                _case_output_dir(args, name, run_id, seed, "fitness_history")
                / (
                    f"segment_{segment_id:04d}"
                    f"_life_{segment_start_lifetime:06d}.png"
                ),
                algorithm_name=algorithm_label(name),
                segment_id=segment_id,
                lifetime_start=segment_start_lifetime,
            )
        last_history_length = len(optimization["history"])
        segment_early_stop_count = int(optimization["early_stopped"])
        early_stop_count += segment_early_stop_count

        def trace(life_end, segment_length, evatime, metrics, reason):
            return build_trace_row(
                name,
                run_id,
                seed,
                args,
                segment_id,
                segment_start_lifetime,
                life_end,
                segment_length,
                optimize_elapsed,
                evatime,
                metrics,
                reason,
                movement_events=segment_movement_events,
                moved_sensor_count=segment_moved_sensor_count,
                effective_evaluate_budget=segment_evaluate_budget,
                early_stop_count=segment_early_stop_count,
            )

        if optimization["state"] is None:
            stop_reason = "no_state"
            trace_rows.append(
                trace(lifetime, 0, optimization["evaluations"], {}, stop_reason)
            )
            break

        best_state, cost, metrics = _decode_and_measure(
            algorithm, problem, optimization
        )
        last_metrics = metrics
        segment_evatime = optimization["evaluations"]
        counted = _as_evaluation_count(segment_evatime)
        if counted is not None:
            total_actual_evatime += counted

        if moving_enabled(args):
            moved_problem, mobile_data = apply_mobility(problem, best_state, cost)
            if mobile_data:
                movement_events += 1
                segment_movement_events = 1
                moved_sensor_count += len(mobile_data)
                segment_moved_sensor_count = len(mobile_data)
                problem = moved_problem

                # 移動後拓樸已改變，必須重建演算法並重新最佳化排程與路由。
                algorithm = build_algorithm(
                    name, problem, args, seed + movement_events
                )
                movement_evaluate_budget = effective_evaluate_budget(
                    algorithm, args, after_movement=True
                )
                segment_evaluate_budget += movement_evaluate_budget
                total_requested_evaluate_budget += movement_evaluate_budget
                movement_optimization, movement_elapsed = _optimize(
                    algorithm, problem, movement_evaluate_budget
                )
                optimize_elapsed += movement_elapsed
                last_history_length = len(movement_optimization["history"])
                movement_early_stopped = int(
                    movement_optimization["early_stopped"]
                )
                segment_early_stop_count += movement_early_stopped
                early_stop_count += movement_early_stopped

                if movement_optimization["state"] is None:
                    stop_reason = "no_state_after_movement"
                    trace_rows.append(
                        trace(lifetime, 0, segment_evatime, {}, stop_reason)
                    )
                    break

                best_state, cost, metrics = _decode_and_measure(
                    algorithm, problem, movement_optimization
                )
                last_metrics = metrics
                movement_counted = _as_evaluation_count(
                    movement_optimization["evaluations"]
                )
                if movement_counted is not None:
                    total_actual_evatime += movement_counted
                    counted = _as_evaluation_count(segment_evatime)
                    segment_evatime = (
                        counted + movement_counted
                        if counted is not None
                        else movement_optimization["evaluations"]
                    )

        if not problem.LifeCheck(best_state, cost):
            stop_reason = "infeasible_solution"
            trace_rows.append(
                trace(lifetime, 0, segment_evatime, metrics, stop_reason)
            )
            break

        if show_energy:
            from plotting.experiment_energy_plot import save_energy_snapshot

            save_energy_snapshot(
                problem,
                output_path=None,
                algorithm=algorithm_label(name),
                run_id=run_id,
                seed=seed,
                stage="segment",
                lifetime=lifetime,
                segment_id=segment_id,
                state=best_state,
                display=True,
                window_name="WSN_live_energy",
            )

        lifetime, segment_length, segment_stop_reason = _advance_lifetime(
            problem,
            best_state,
            cost,
            lifetime,
            segment_slot_limit,
            cap_lifetime,
        )
        if segment_stop_reason == "lifetime_cap":
            stop_reason = "lifetime_cap"

        if segment_length > 0:
            problem.prepare_coding_cache(cost * segment_length)
            if energy_dir is not None:
                from plotting.experiment_energy_plot import save_energy_snapshot

                save_energy_snapshot(
                    problem,
                    energy_dir
                    / (
                        f"{energy_frame:04d}_after_segment_{segment_id:04d}"
                        f"_life_{lifetime:06d}.png"
                    ),
                    algorithm=name,
                    run_id=run_id,
                    seed=seed,
                    stage="after selected state finished",
                    lifetime=lifetime,
                    segment_id=segment_id,
                    segment_length=segment_length,
                    state=best_state,
                )
                energy_frame += 1

        trace_rows.append(
            trace(
                lifetime,
                segment_length,
                segment_evatime,
                metrics,
                segment_stop_reason,
            )
        )
        reoptimizations += 1
        previous_segment_length = segment_length

        print(
            "[segment]",
            name,
            "run",
            run_id,
            "segment",
            segment_id,
            "life",
            segment_start_lifetime,
            "->",
            lifetime,
            "fitness",
            round(metrics["fitness"], 6),
            "time",
            round(optimize_elapsed, 3),
        )

        if segment_length == 0:
            stop_reason = "no_lifetime_progress"
            break
        if stop_reason == "lifetime_cap":
            break

    elapsed = time.time() - started_at
    row = common_result_fields(name, run_id, seed, args)
    row.update(
        {
            "mode": "lifetime",
            "lifetime": lifetime,
            "reoptimizations": reoptimizations,
            "movement_events": movement_events,
            "moved_sensor_count": moved_sensor_count,
            "effective_evaluate_budget": evaluate_budget(args),
            "requested_evaluate_budget_total": total_requested_evaluate_budget,
            "early_stop_count": early_stop_count,
            "stop_reason": stop_reason,
            "elapsed_seconds": round(elapsed, 6),
            "actual_evatime": total_actual_evatime,
            "compute_device": getattr(algorithm, "device_name", ""),
            "gpu_accelerated": getattr(algorithm, "gpu_accelerated", ""),
            "history_length": last_history_length,
        }
    )
    row.update(last_metrics)
    print(
        "[lifetime]",
        name,
        "run",
        run_id,
        "lifetime",
        lifetime,
        "reopt",
        reoptimizations,
        "stop",
        stop_reason,
        "time",
        round(elapsed, 3),
    )
    return row, trace_rows


def run_case(name, run_id, args):
    if args.mode == "single":
        return run_single_case(name, run_id, args), []
    return run_lifetime_case(name, run_id, args)
