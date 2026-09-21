"""Verify less frequently used optimizers return decoded State objects."""

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from Algorithm.Scheduling.SRIME import SRIME
from Algorithm.misc.GWO import GWO
from Algorithm.misc.JHC import JHC
from Problem.Problem import Problem


def assert_state_result(algorithm, problem, budget):
    result = algorithm.run(problem, budget=budget)
    assert result.best_state is not None
    assert all(
        hasattr(result.best_state, name)
        for name in ("levels", "next_hops", "tx_load")
    )
    assert not hasattr(result, "best_coding")


def main():
    problem = Problem(B=50, S=30, T=9, F=100, FILE=None)
    assert_state_result(JHC(problem), problem, 4)
    assert_state_result(GWO(problem, n=4), problem, 8)
    assert_state_result(SRIME(n=4), problem, 8)

    print("smoke_remaining_search_contract_ok")


if __name__ == "__main__":
    main()
