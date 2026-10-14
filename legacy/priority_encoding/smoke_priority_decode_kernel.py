"""PriorityEncoding 的 numba 解碼必須與 Python 版逐欄位相同。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import State.PriorityEncoding as priority_module
from Problem.Problem import Problem
from State.PriorityEncoding import PriorityEncoding


class EncodingC4(PriorityEncoding):
    ACTIVATION_BASE = "cost"
    ROUTING_BASE = "energy"


def decode_both(coding, problem):
    fast = coding.decode(problem)
    saved = priority_module.NUMBA_AVAILABLE
    priority_module.NUMBA_AVAILABLE = False
    try:
        slow = coding.decode(problem)
    finally:
        priority_module.NUMBA_AVAILABLE = saved
    return fast, slow


def assert_same(fast, slow, problem):
    for name in ("levels", "next_hops", "tx_load", "remaining_capacity"):
        assert np.array_equal(getattr(fast, name), getattr(slow, name)), name
    assert fast.paths == slow.paths
    assert np.array_equal(fast.sensing_radii, slow.sensing_radii)
    assert np.array_equal(
        np.asarray(problem.evaluate_state(fast)),
        np.asarray(problem.evaluate_state(slow)),
    )


def check(problem, cases, rng):
    for encoding_cls in (PriorityEncoding, EncodingC4):
        for split in (True, False):
            for _ in range(cases):
                coding = encoding_cls.random(
                    problem.SENSOR_NUMBER, rng=rng, split_priority=split
                )
                fast, slow = decode_both(coding, problem)
                assert_same(fast, slow, problem)


def main():
    if not priority_module.NUMBA_AVAILABLE:
        print("numba unavailable; kernel path not exercised")
        return
    rng = np.random.default_rng(3)
    check(Problem(B=50, S=30, T=9, F=100, FILE=None), 50, rng)
    check(Problem(B=100, S=100, T=100, F=10, FILE="MAP01"), 50, rng)

    # 電量不均時 cost／energy base 會改變順序，也要一致。
    problem = Problem(B=100, S=100, T=100, F=10, FILE="MAP02")
    problem.energy = problem.energy * rng.uniform(0.05, 1.0, problem.energy.shape)
    problem.energy_score = problem.energy / np.max(problem.energy)
    check(problem, 30, rng)
    print("smoke_priority_decode_kernel_ok")


if __name__ == "__main__":
    main()
