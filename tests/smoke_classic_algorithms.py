"""Smoke-test classic optimizers in isolated processes.

Several algorithms create multiprocessing workers.  Running every case in the
same interpreter can leave Windows multiprocessing resources interacting with
the next case, so the parent process launches one clean interpreter per case.
"""

import argparse
import math
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.misc.CS import CS
from Algorithm.se.CodingSE import CodingSE
from Algorithm.misc.EDA import EDA
from Algorithm.misc.PSO import PSO
from Problem.Problem import Problem


CASE_NAMES = ("pso", "codingse", "eda", "cs")


def assert_finite_fitness(fitness):
    assert len(fitness) == 3
    for value in fitness:
        assert math.isfinite(float(value))


def create_algorithm(name, problem):
    if name == "pso":
        return PSO(problem, n=4), 8
    if name == "codingse":
        return CodingSE(problem, n=3, h=2, w=1, seed=7), 8
    if name == "eda":
        return EDA(problem, n=4, alpha=0.8), 8
    if name == "cs":
        return CS(problem, n=4), 8
    raise ValueError(name)


def run_case(name):
    np.random.seed(7)
    random.seed(7)
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    algorithm, evaluate = create_algorithm(name, problem)
    if name == "codingse":
        # Exercise the branch that opens a closed CHS sensor for region bit 1.
        # This guards against mixing the loop names ``index`` and ``i``.
        algorithm.identity_sensors = algorithm._select_identity_sensors(
            problem
        )
        region_state = algorithm.create_candidate(problem)
        sensor_id = algorithm.identity_sensors[0]
        sensor_gene = sensor_id * 2
        region_state.code[sensor_gene] = 0
        algorithm.align_region(problem, region_state, region=1)
        assert 1 <= region_state.code[sensor_gene] < (
            problem.sensing_option_count(sensor_id)
        )

        # Keep the optimizer smoke run on its original deterministic seed.
        np.random.seed(7)
        random.seed(7)
    result = algorithm.run(problem, budget=evaluate)
    assert result.best_state is not None, name
    assert not hasattr(result, "best_coding"), name
    assert_finite_fitness(result.best_fitness)
    assert getattr(algorithm, "evatime", 0) > 0


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--case", choices=CASE_NAMES)
    args = parser.parse_args()

    if args.case is not None:
        run_case(args.case)
        return

    for name in CASE_NAMES:
        subprocess.run(
            [sys.executable, "-B", __file__, "--case", name],
            cwd=str(PROJECT_ROOT),
            check=True,
        )
    print("smoke_classic_algorithms_ok")


if __name__ == "__main__":
    main()
