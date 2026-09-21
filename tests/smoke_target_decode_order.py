"""Verify TargetEncoding removes redundant sensors from low priority first."""

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from State.State import State
from State.TargetEncoding import TargetEncoding


class FixedPriorityTargetEncoding(TargetEncoding):
    def _target_priorities(self, problem):
        return np.arange(problem.TARGET_NUMBER, dtype=float)


class FakeProblem:
    SENSOR_NUMBER = 2
    TARGET_NUMBER = 2
    DEVICE_NUMBER = 3

    def __init__(self):
        # 兩顆 sensor 在 level 1 都可覆蓋兩個 target。
        self.coverage_table = np.zeros((2, 2, 2), dtype=int)
        self.coverage_table[:, :, 1] = 1
        self.cover_candidates = [
            [(0, 1), (1, 1)],
            [(0, 1), (1, 1)],
        ]


def main():
    problem = FakeProblem()
    # target 0 先選 sensor 0；target 1 後選 sensor 1。
    coding = FixedPriorityTargetEncoding(np.asarray([0, 5100]))
    state = State.empty(problem, with_target_assignment=True)

    retained = coding._decode_coverage(problem, state)

    # 從後往前刪除時，應刪掉後加入的 sensor 1，保留 sensor 0。
    assert retained == [[0, 1]]
    np.testing.assert_array_equal(state.levels, [1, 0])
    print("smoke_target_decode_order_ok")


if __name__ == "__main__":
    main()
