import csv
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment_algorithms import (
    SUMMARY_FIELDNAMES,
    build_problem,
    common_result_fields,
    parse_args,
    write_grouped_results,
)


def main():
    args = parse_args(
        [
            "--algorithm",
            "sa_sets",
            "--runs",
            "5",
        ]
    )
    assert args.seed == 7
    assert args.maps == "maps/maps_100100100.json"
    assert len(args.map_specs) == 6
    assert [spec["name"] for spec in args.map_specs] == [
        "MAP01",
        "MAP02",
        "MAP03",
        "MAP04",
        "MAP05",
        "MAP06",
    ]
    assert [args.seed + run_id for run_id in range(args.runs)] == [7, 8, 9, 10, 11]
    assert args.runs * len(args.map_specs) == 30

    args._current_map = args.map_specs[0]
    problem = build_problem(args, seed=7)
    assert problem.sensor.shape == (100, 2)
    assert problem.target.shape == (100, 2)

    rows = []
    for map_spec in args.map_specs:
        args._current_map = map_spec
        for run_id in range(args.runs):
            rows.append(
                common_result_fields(
                    "sa_sets", run_id, args.seed + run_id, args
                )
            )

    with tempfile.TemporaryDirectory() as temp_dir:
        args.result_dir = temp_dir
        args.experiment_name = "map_layout"
        args.mode = "single"
        write_grouped_results(rows, [], ["sa_sets"], args)
        result_root = Path(temp_dir) / "map_layout" / "sa_sets"
        for index in range(1, 7):
            map_name = f"MAP{index:02d}"
            summary_path = result_root / map_name / "summary.csv"
            assert summary_path.is_file()
            with summary_path.open(newline="", encoding="utf-8") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))
            assert len(saved_rows) == 5
            assert [int(row["seed"]) for row in saved_rows] == [7, 8, 9, 10, 11]
            assert all(row["map"] == map_name for row in saved_rows)
            assert set(saved_rows[0]).issubset(SUMMARY_FIELDNAMES)

    print("experiment maps smoke passed")


if __name__ == "__main__":
    main()
