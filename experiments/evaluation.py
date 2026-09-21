"""Running one optimizer call and turning its result into comparable numbers."""

import numpy as np

from Algorithm.core import AlgorithmResult

from experiments.config import FITNESS_SERVICES

# Every objective any configured fitness service can report. A run fills only
# its own service's objectives; the rest stay blank so that one CSV can hold
# results from different fitness versions.
OBJECTIVE_FIELDS = tuple(
    dict.fromkeys(
        name
        for service in FITNESS_SERVICES.values()
        for name in service.OBJECTIVE_NAMES
    )
)


def run_optimizer(algorithm, problem, evaluate):
    """執行 optimizer，統一回傳實驗流程需要的資料。"""
    # console/history 的欄位名稱也必須跟目前啟用的 fitness 版本一致。
    objective_names = getattr(
        problem.fitness_service, "objective_names", None
    )
    if objective_names:
        algorithm.Fname = list(objective_names)
    result = algorithm.run(problem, budget=evaluate)
    if not isinstance(result, AlgorithmResult):
        raise TypeError(
            f"{algorithm.__class__.__name__}.run() 必須回傳 AlgorithmResult"
        )
    return {
        "state": result.best_state,
        "objectives": result.best_fitness,
        "evaluations": result.evaluations,
        "history": result.history,
        "early_stopped": bool(
            result.metadata.get("early_stopped", False)
        ),
        "metadata": result.metadata,
    }


def prepare_result_state(algorithm, problem, state):
    """Decode a result and apply algorithm-specific normalization/repair."""
    if state is None:
        return None

    # New algorithms already return the pure decoded State in AlgorithmResult.
    if all(
        hasattr(state, name)
        for name in ("levels", "next_hops", "tx_load")
    ) and not (
        hasattr(state, "code")
    ):
        return state

    normalize = getattr(algorithm, "normalize_state", None)
    if normalize is not None:
        state = normalize(problem, state)

    decode_state = getattr(algorithm, "decode_state", None)
    if decode_state is not None:
        state = decode_state(problem, state)
    else:
        decoder = getattr(state, "decode", None)
        if decoder is not None:
            decoded = decoder(problem)
            if decoded is not None:
                state = decoded
        else:
            state.Decode(problem)

    repair = getattr(algorithm, "repair_state", None)
    if repair is not None:
        state = repair(problem, state)
    return state


def state_metrics(
    problem,
    state,
    cost=None,
    decode=True,
    objectives=None,
):
    if decode:
        decoder = getattr(state, "decode", None)
        if decoder is not None:
            decoded = decoder(problem)
            if decoded is not None:
                state = decoded
        elif hasattr(state, "Decode"):
            state.Decode(problem)
    if cost is None:
        cost = problem.calculate_total_cost(state)
    fitness = (
        np.asarray(problem.evaluate_state(state), dtype=float)
        if objectives is None
        else np.asarray(objectives, dtype=float)
    )
    objective_names = tuple(
        getattr(problem.fitness_service, "objective_names", ())
    )
    if len(objective_names) != len(fitness):
        raise ValueError(
            "fitness objective_names and returned values have different lengths"
        )
    remaining_energy = problem.calculate_remaining_energy(cost)
    target_remaining_energy = problem.calculate_target_remaining_energy(
        remaining_energy
    )
    uncovered_count = len(problem.find_uncovered_targets(state))
    coverage_ratio = (
        1.0
        if problem.TARGET_NUMBER == 0
        else 1.0 - uncovered_count / problem.TARGET_NUMBER
    )
    # Objectives this fitness version does not report stay blank.
    metrics = {
        name: "" for name in OBJECTIVE_FIELDS if name != "coverage"
    }
    metrics.update(
        {
            # V1 overwrites this with its weighted coverage contribution;
            # V2/V3 leave the raw coverage ratio in place.
            "coverage": coverage_ratio,
            "fitness": float(np.sum(fitness)),
            "mean_sensor_cost": float(np.mean(cost)),
            "max_sensor_cost": float(np.max(cost)),
            "total_sensor_cost": float(np.sum(cost)),
            "sensor_remaining_mean": float(np.mean(remaining_energy)),
            "sensor_remaining_min": float(np.min(remaining_energy)),
            "sensor_remaining_variance": float(np.var(remaining_energy)),
            "target_remaining_min": float(np.min(target_remaining_energy)),
            "uncovered_count": uncovered_count,
            "disconnected_count": len(problem.find_disconnected(state)),
            "energy_failed_count": len(
                problem.find_energy_failed_sensors(state, cost)
            ),
            "active_sensor_count": int(
                np.count_nonzero(state.sensing_radii > 0)
            ),
            "mean_sensing_radius": float(np.mean(state.sensing_radii)),
            "max_sensing_radius": float(np.max(state.sensing_radii)),
        }
    )
    metrics.update(
        {
            name: float(value)
            for name, value in zip(objective_names, fitness)
        }
    )
    return metrics
