import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.Scheduling.SRIME import SRIME
from Algorithm.gomea.GI_GOMEA import GI_GOMEA
from Algorithm.gomea.GPU_GOMEA import GPU_GOMEA
from Algorithm.gomea.GI_GOMEA_Target import GI_GOMEA_Target
from Algorithm.se.RL_SETS import RL_SETS
from Algorithm.se.RL_SETSv2 import RL_SETSv2
from Algorithm.se.RL_SETSv3 import RL_SETSv3
from Problem.Problem import Problem
from Problem.services import (
    FitnessService,
    FitnessServiceV2,
    RoutingService,
)
from experiment_algorithms import (
    ALGORITHM_NAMES,
    ALGORITHM_PRESETS,
    algorithm_params,
    build_algorithm,
    build_problem,
    configure_fitness_service,
    evaluate_budget,
    parse_args,
    prepare_result_state,
    state_metrics,
)
from plotting.experiment_energy_plot import CANVAS_SIZE, PLOT_BOX, _map_position, save_energy_snapshot
from PIL import Image


def main():
    default_args = parse_args(["--algorithm", "sa_sets"])
    assert default_args.algorithm == ["sa_sets"]
    assert default_args.mode == "lifetime"
    assert default_args.sensing_mode == "discrete"
    assert default_args.fitness_service == "v2"
    assert default_args.routing_service == "v1"
    assert default_args.draw_save is False
    assert not hasattr(default_args, "implementation")
    assert default_args.runs == 5
    assert evaluate_budget(default_args) == 10000
    default_problem = build_problem(default_args, seed=7)
    assert type(default_problem.fitness_service) is FitnessServiceV2
    assert type(default_problem.routing_service) is RoutingService
    draw_args = parse_args(
        ["--algorithm", "sa_sets", "--draw-save", "true"]
    )
    assert draw_args.draw_save is True
    bucketed_args = parse_args(
        ["--algorithm", "sa_sets", "--sensing-mode", "bucketed"]
    )
    assert bucketed_args.sensing_mode == "bucketed"
    # CUDA settings are preset-only now; there are no --cuda-device /
    # --require-cuda flags to override them.
    cuda_args = parse_args(["--algorithm", "gpu_gomea"])
    cuda_params = algorithm_params("gpu_gomea", cuda_args)
    assert cuda_params["cuda_device"] == 0
    assert cuda_params["require_cuda"] is False

    # RL-SETS is tested separately with a real checkpoint; the registry-wide
    # construction below may use small temporary ones. rl_setsv4/rl_setsv5
    # support a no-checkpoint uniform-random policy, so they need none.
    model_problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    model_algorithm = RL_SETS(
        model_problem,
        n=4,
        h=4,
        w=1,
        training=True,
        seed=7,
    )
    model_path = PROJECT_ROOT / "tests" / "_experiment_rl_sets.npz"
    model_algorithm.save_model(model_path)
    checkpoint_paths = {"rl_sets": model_path}
    for name, cls in (("rl_setsv2", RL_SETSv2), ("rl_setsv3", RL_SETSv3)):
        checkpoint_problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
        checkpoint_algorithm = cls(
            checkpoint_problem, n=4, h=4, w=1, training=True, seed=7,
        )
        checkpoint_path = PROJECT_ROOT / "tests" / f"_experiment_{name}.npz"
        checkpoint_algorithm.save_model(checkpoint_path)
        checkpoint_paths[name] = checkpoint_path
    args = SimpleNamespace(evaluate=100, rl_model=str(model_path))
    assert set(ALGORITHM_NAMES) == set(ALGORITHM_PRESETS)

    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    energy_plot = PROJECT_ROOT / "tests" / "_energy_snapshot_smoke.png"
    energy_stats = save_energy_snapshot(
        problem,
        energy_plot,
        algorithm="smoke",
        run_id=0,
        seed=7,
        stage="initial",
        lifetime=0,
    )
    assert energy_plot.is_file()
    assert energy_plot.stat().st_size > 0
    with Image.open(energy_plot) as snapshot:
        assert snapshot.size == CANVAS_SIZE
    assert energy_stats["total_energy"] == 3000.0
    assert energy_stats["weakest_target_id"] is not None
    assert _map_position((0, 0), problem.BOUNDARY) == PLOT_BOX[:2]
    energy_plot.unlink()
    algorithms = {}
    no_checkpoint_args = SimpleNamespace(evaluate=100, rl_model=None)
    for name in ALGORITHM_NAMES:
        if name in checkpoint_paths:
            name_args = SimpleNamespace(
                evaluate=100, rl_model=str(checkpoint_paths[name])
            )
        elif name in ("rl_setsv4", "rl_setsv5"):
            # No checkpoint trained for these: exercise their no-checkpoint
            # uniform-random policy fallback instead.
            name_args = no_checkpoint_args
        else:
            name_args = args
        algorithms[name] = build_algorithm(name, problem, name_args, seed=7)

    assert type(algorithms["gi_gomea"]) is GI_GOMEA
    assert type(algorithms["gpu_gomea"]) is GPU_GOMEA
    assert type(algorithms["gi_gomea_target"]) is GI_GOMEA_Target
    assert type(algorithms["srime"]) is SRIME
    assert type(algorithms["rl_sets"]) is RL_SETS
    assert type(algorithms["rl_setsv2"]) is RL_SETSv2
    assert type(algorithms["rl_setsv3"]) is RL_SETSv3
    assert "state_type" not in algorithm_params("gi_gomea", args)
    assert "state_type" not in algorithm_params("gi_gomea_target", args)
    assert algorithm_params("nsga", args)["generation"] == 1
    assert algorithm_params("srime", args)["generation"] == 2
    assert "setsv1" in ALGORITHM_NAMES
    assert "setsv2" in ALGORITHM_NAMES
    assert "setsv3" not in ALGORITHM_NAMES
    assert "setsv4" not in ALGORITHM_NAMES
    assert "wasa_sets" not in ALGORITHM_NAMES
    model_path.unlink()
    for checkpoint_path in checkpoint_paths.values():
        if checkpoint_path != model_path:
            checkpoint_path.unlink()

    # CS uses a 15-member preset with four workers. This exercises the
    # non-divisible population split and verifies that all 15 states count.
    np.random.seed(7)
    random.seed(7)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    cs = algorithms["cs"]
    best = cs.run(problem, budget=16).best_state
    assert best is not None
    assert cs.evatime == 16
    metrics = state_metrics(problem, best, decode=False)
    assert {
        "total_remaining_energy",
        "min_remaining_energy",
        "coverage",
    } <= set(metrics)

    configure_fitness_service(problem, "v2")
    v2_fitness = np.asarray(problem.evaluate_state(best), dtype=float)
    v2_metrics = state_metrics(problem, best, decode=False)
    assert problem.fitness_service_name == "v2"
    assert type(problem.fitness_service) is FitnessServiceV2
    assert len(v2_fitness) == 3
    assert v2_metrics["energy_efficiency"] == v2_fitness[0]
    assert v2_metrics["target_region_balance"] == v2_fitness[1]
    assert v2_metrics["constraint_score"] == v2_fitness[2]

    np.random.seed(7)
    random.seed(7)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    srime = SRIME(n=4, generation=2, route_selector=None)
    best_srime = srime.run(problem, budget=8).best_state
    assert best_srime.use_continuous_radius is False
    best_srime.Decode(problem)
    assert best_srime.use_continuous_radius is False
    prepare_result_state(srime, problem, best_srime)
    assert best_srime.use_continuous_radius is False
    assert best_srime.sensing_radii is not None

    exact_problem = Problem(
        B=50,
        S=30,
        T=9,
        F=100,
        FILE=None,
        sensing_mode="exact",
    )
    exact_srime = SRIME(n=4, generation=2, route_selector=None)
    exact_best = exact_srime.run(exact_problem, budget=8).best_state
    prepare_result_state(exact_srime, exact_problem, exact_best)
    assert exact_best.use_continuous_radius is True
    assert np.allclose(
        exact_best.sensing_radii,
        exact_problem.resolve_radius(exact_best.levels),
    )

    bucketed_problem = Problem(
        B=50,
        S=30,
        T=9,
        F=100,
        FILE=None,
        sensing_mode="bucketed",
    )
    bucketed_srime = SRIME(n=4, generation=2, route_selector=None)
    bucketed_best = bucketed_srime.run(
        bucketed_problem, budget=8
    ).best_state
    prepare_result_state(
        bucketed_srime,
        bucketed_problem,
        bucketed_best,
    )
    assert bucketed_best.use_continuous_radius is True
    assert np.allclose(
        bucketed_best.sensing_radii,
        bucketed_problem.resolve_radius(bucketed_best.levels),
    )

    print("smoke_experiment_algorithms_ok")


if __name__ == "__main__":
    main()
