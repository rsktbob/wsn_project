"""Entry point for the WSN algorithm experiments.

The implementation lives in the :mod:`experiments` package; this module keeps
the original public names so existing scripts and tests can keep importing
``experiment_algorithms``.

    python experiment_algorithms.py --algorithm sa_sets ga --runs 5
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from experiments import PROJECT_ROOT
from experiments.builder import (
    algorithm_label,
    algorithm_params,
    build_algorithm,
    configured_algorithm_params,
    nsga_generation,
    srime_generation,
)
from experiments.cli import parse_args, selected_algorithms
from experiments.config import (
    EVALUATE_OVERRIDES,
    EXPERIMENT_PRESET,
    FITNESS_SERVICES,
    IMPLEMENTATION_NAME,
    MOVING_TARGET_ENERGY_THRESHOLD,
    ROUTING_SERVICES,
    effective_evaluate_budget,
    evaluate_budget,
    lifetime_cap,
    moving_enabled,
    parse_bool,
    set_seed,
)
from experiments.evaluation import (
    OBJECTIVE_FIELDS,
    prepare_result_state,
    run_optimizer,
    state_metrics,
)
from experiments.maps import (
    current_map_spec,
    default_map_spec,
    load_map_specs,
    parse_file_name,
)
from experiments.plots import save_fitness_history_plot
from experiments.presets import ALGORITHM_PRESETS
from experiments.problem_setup import (
    apply_mobility,
    build_problem,
    configure_fitness_service,
    rebuild_problem_after_movement,
)
from experiments.registry import (
    ALGORITHM_CHOICES,
    ALGORITHM_IMPORTS,
    ALGORITHM_NAMES,
    ALL_RUNNABLE_ALGORITHMS,
    RL_SETS_ALGORITHMS,
    load_algorithm_type,
)
from experiments.reporting import (
    SUMMARY_FIELDNAMES,
    TRACE_FIELDNAMES,
    build_algorithm_parameter_report,
    build_result_root,
    build_trace_row,
    common_result_fields,
    resolved_result_root,
    write_algorithm_parameters,
    write_grouped_results,
    write_rows,
)
from experiments.runner import run_case, run_lifetime_case, run_single_case

__all__ = [
    "ALGORITHM_CHOICES",
    "ALGORITHM_IMPORTS",
    "ALGORITHM_NAMES",
    "ALGORITHM_PRESETS",
    "ALL_RUNNABLE_ALGORITHMS",
    "EVALUATE_OVERRIDES",
    "EXPERIMENT_PRESET",
    "FITNESS_SERVICES",
    "IMPLEMENTATION_NAME",
    "MOVING_TARGET_ENERGY_THRESHOLD",
    "OBJECTIVE_FIELDS",
    "PROJECT_ROOT",
    "RL_SETS_ALGORITHMS",
    "ROUTING_SERVICES",
    "SUMMARY_FIELDNAMES",
    "TRACE_FIELDNAMES",
    "algorithm_label",
    "algorithm_params",
    "apply_mobility",
    "build_algorithm",
    "build_algorithm_parameter_report",
    "build_problem",
    "build_result_root",
    "build_trace_row",
    "common_result_fields",
    "configure_fitness_service",
    "configured_algorithm_params",
    "current_map_spec",
    "default_map_spec",
    "effective_evaluate_budget",
    "evaluate_budget",
    "lifetime_cap",
    "load_algorithm_type",
    "load_map_specs",
    "main",
    "moving_enabled",
    "nsga_generation",
    "parse_args",
    "parse_bool",
    "parse_file_name",
    "prepare_result_state",
    "rebuild_problem_after_movement",
    "resolved_result_root",
    "run_case",
    "run_lifetime_case",
    "run_optimizer",
    "run_single_case",
    "save_fitness_history_plot",
    "selected_algorithms",
    "set_seed",
    "srime_generation",
    "state_metrics",
    "write_algorithm_parameters",
    "write_grouped_results",
    "write_rows",
]


# 每個 case 的亂數種子是 ``--seed`` 加上它自己的 run id，且 build_problem 會
# 在開跑前重設全域亂數，所以 case 之間沒有順序相依 —— 循序與平行兩種模式
# 產生的結果完全相同。
def _build_cases(args, algorithms):
    """Flatten the map/run/algorithm loops into an ordered case list."""
    return [
        (index, name, run_id, map_spec)
        for index, (map_spec, run_id, name) in enumerate(
            (map_spec, run_id, name)
            for map_spec in args.map_specs
            for run_id in range(args.runs)
            for name in algorithms
        )
    ]


def _run_one_case(case, args):
    """Run one case and tag the result with its position in the case list."""
    index, name, run_id, map_spec = case
    args._current_map = map_spec
    summary_row, trace_rows = run_case(name, run_id, args)
    return index, summary_row, trace_rows


def _parallel_job_count(args, case_count):
    """Pick how many cases run at once when --jobs was not given.

    直覺上「每個 case 自己已經開了一組 SE 市場 worker，不該再超額訂閱」是
    錯的：市場只從 4 個 worker 拿到約 3 倍，其餘時間都在等 pickle 與 pipe
    來回，因此工作負載是延遲受限而非吞吐受限，別的 case 正好填滿那些空檔。
    實測（A2、sa_sets、6 個 case、16 執行緒）：
    循序 120.5s、jobs=2 63.3s、jobs=4 50.4s、jobs=6 45.4s。
    所以預設不人為設限，只以核心數為上界避免 case 極多時開爆。
    """
    if args.jobs > 0:
        return min(args.jobs, case_count)
    cpu_count = os.cpu_count() or 1
    return max(1, min(case_count, cpu_count))


def _run_cases(args, algorithms):
    """Return one (index, summary_row, trace_rows) tuple per case."""
    cases = _build_cases(args, algorithms)
    if args.execution != "parallel" or len(cases) < 2:
        return [_run_one_case(case, args) for case in cases]

    # 繪圖後端與互動視窗無法安全地在多個子程序同時使用。
    if getattr(args, "draw_save", False) or getattr(args, "show", False):
        print(
            "[execution] --draw-save/--show 需要單一程序，已改用 sequential"
        )
        return [_run_one_case(case, args) for case in cases]

    jobs = _parallel_job_count(args, len(cases))
    if jobs < 2:
        return [_run_one_case(case, args) for case in cases]

    print(f"[execution] parallel: {len(cases)} 個 case，同時跑 {jobs} 個")
    # ProcessPoolExecutor 的 worker 不是 daemon，因此每個 case 仍能啟動
    # 自己的 SE 市場子程序；multiprocessing.Pool 的 daemon worker 不行。
    results = []
    with ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [
            executor.submit(_run_one_case, case, args) for case in cases
        ]
        for future in as_completed(futures):
            results.append(future.result())
    return results


def main():
    args = parse_args()
    algorithms = selected_algorithms(args.algorithm)

    summary_rows = []
    trace_rows = []
    # 依 case 原始順序輸出，平行模式的完成順序不影響報表。
    for _, summary_row, case_trace_rows in sorted(
        _run_cases(args, algorithms),
        key=lambda result: result[0],
    ):
        summary_rows.append(summary_row)
        trace_rows.extend(case_trace_rows)

    write_grouped_results(summary_rows, trace_rows, algorithms, args)


if __name__ == "__main__":
    main()
