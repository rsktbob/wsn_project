import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


CASES = [
    ("SA-SETS", "tests/smoke_sa_sets.py"),
    ("SA-SETSv3", "tests/smoke_sa_setsv3.py"),
    ("GI-GOMEA-WSN", "tests/smoke_gi_gomea.py"),
    ("CMA-MAE", "tests/smoke_cma_mae.py"),
    (
        "Differential-MAP-Elites",
        "tests/smoke_differential_map_elites.py",
    ),
    ("Experiment Registry", "tests/smoke_experiment_algorithms.py"),
    ("Evaluation Budget", "tests/regression_evaluation_budget.py"),
    ("Lightweight Bases", "tests/smoke_lightweight_bases.py"),
    ("Classic Algorithms", "tests/smoke_classic_algorithms.py"),
    ("CodingSEv2", "tests/smoke_codingsev2.py"),
    ("NSOA", "tests/smoke_nsoa.py"),
    ("RIME", "tests/smoke_rime.py"),
    ("CPO", "tests/smoke_cpo.py"),
    ("CPOv2", "tests/smoke_cpov2.py"),
    ("GA", "tests/smoke_ga.py"),
    ("NSGA-II", "tests/smoke_nsga.py"),
    ("SNSGAII + RQLearning", "tests/smoke_snsga_rqlearning.py"),
    ("RL-SETS-D3QN", "tests/smoke_rl_sets.py"),
    ("RL-SETSv5-D3QN", "tests/smoke_rl_setsv5.py"),
    ("RL-SETS trainer", "tests/smoke_train_rl_sets.py"),
]


def main():
    for name, script in CASES:
        print("==", name, "==")
        subprocess.run(
            [sys.executable, "-B", str(PROJECT_ROOT / script)],
            cwd=str(PROJECT_ROOT),
            check=True,
        )

    print("smoke_selected_algorithms_ok")


if __name__ == "__main__":
    main()
