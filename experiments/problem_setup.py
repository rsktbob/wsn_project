"""Building the WSN Problem for a run, and repairing it after sensor movement."""

import numpy as np

from Problem.Problem import Problem
from Problem.services import FitnessServiceV2, FitnessServiceV3
from Problem.services.mobility_service import MobilityService
from environment_defaults import (
    DEFAULT_FITNESS_SERVICE,
    DEFAULT_ROUTING_SERVICE,
    DEFAULT_SENSING_MODE,
)

from experiments.config import (
    FITNESS_SERVICES,
    MOVING_TARGET_ENERGY_THRESHOLD,
    set_seed,
)
from experiments.maps import current_map_spec


def configure_fitness_service(problem, version):
    """替實驗 Problem 安裝指定版本的 fitness，並留下可追蹤的版本名稱。"""
    normalized_version = str(version).strip().lower()
    try:
        service_type = FITNESS_SERVICES[normalized_version]
    except KeyError as error:
        choices = ", ".join(FITNESS_SERVICES)
        raise ValueError(
            f"fitness service must be one of: {choices}"
        ) from error

    problem.fitness_service = service_type(problem)
    problem.fitness_service_name = normalized_version
    return problem


def build_problem(args, seed):
    set_seed(seed)
    map_spec = current_map_spec(args)
    problem = Problem(
        B=map_spec["boundary"],
        S=map_spec["sensors"],
        T=map_spec["targets"],
        F=map_spec["full_energy"],
        FILE=map_spec["file"],
        sensing_mode=getattr(args, "sensing_mode", DEFAULT_SENSING_MODE),
        routing_service=getattr(args, "routing_service", DEFAULT_ROUTING_SERVICE),
        sensor_ring_order=getattr(args, "sensor_ring_id", True),
    )
    return configure_fitness_service(
        problem,
        getattr(args, "fitness_service", DEFAULT_FITNESS_SERVICE),
    )


def rebuild_problem_after_movement(problem, positions, powers, mobile_data):
    """以移動後的位置與電量重建拓樸，並維持感測器和電量的索引一致。"""
    positions = np.asarray(positions, dtype=float)
    powers = np.asarray(powers, dtype=float)

    # Problem 會依基地台距離排序感測器，因此必須同步排序剩餘電量。
    distances = np.linalg.norm(positions - np.asarray(problem.BS), axis=1)
    order = np.argsort(distances, kind="stable")
    sorted_positions = positions[order]
    sorted_powers = powers[order]

    moved_problem = Problem(
        B=problem.BOUNDARY,
        S=problem.SENSOR_NUMBER,
        T=problem.TARGET_NUMBER,
        F=problem.initial_energy,
        FILE=None,
        update_position=sorted_positions,
        updated_power=sorted_powers,
        Target_position=np.asarray(problem.target).copy(),
        sensing_mode=problem.sensing_mode,
        sensing_levels=problem.DISCRETE_LEVEL_RANGE,
        maximum_sensing_radius=problem.MAX_SENSING_RADIUS,
        routing_service=getattr(problem, "routing_service_name", "v1"),
        sensor_ring_order=getattr(problem, "sensor_ring_order", True),
    )

    old_to_new = np.empty(problem.SENSOR_NUMBER, dtype=int)
    old_to_new[order] = np.arange(problem.SENSOR_NUMBER)
    remapped_mobile_data = []
    for record in mobile_data:
        remapped_record = list(record)
        remapped_record[0] = int(old_to_new[int(remapped_record[0])])
        remapped_mobile_data.append(remapped_record)
    moved_problem.mobile_id = remapped_mobile_data
    # 移動後會建立新的 Problem；必須沿用移動前的 fitness 版本，確保整場實驗一致。
    fitness_service = getattr(problem, "fitness_service_name", None)
    if fitness_service is None:
        if isinstance(problem.fitness_service, FitnessServiceV3):
            fitness_service = "v3"
        elif isinstance(problem.fitness_service, FitnessServiceV2):
            fitness_service = "v2"
        else:
            fitness_service = "v1"
    return configure_fitness_service(moved_problem, fitness_service)


def apply_mobility(problem, state, cost):
    """依 Round.py 的固定門檻移動修復弱區域與覆蓋盲區。"""
    remaining_energy = problem.calculate_remaining_energy(cost)
    target_remaining_energy = problem.calculate_target_remaining_energy(
        remaining_energy
    )
    failed_targets = list(problem.find_uncovered_targets(state))
    low_energy_targets = np.where(
        target_remaining_energy < MOVING_TARGET_ENERGY_THRESHOLD
    )[0].tolist()
    repair_targets = sorted(set(low_energy_targets).union(failed_targets))
    if not repair_targets:
        return problem, []

    mobility = MobilityService(problem, state)
    positions, powers, mobile_data = mobility.Move(
        problem,
        state,
        repair_targets,
        target_remaining_energy,
        failed_targets,
    )
    if not mobile_data:
        return problem, []

    moved_problem = rebuild_problem_after_movement(
        problem, positions, powers, mobile_data
    )
    return moved_problem, mobile_data
