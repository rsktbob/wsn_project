"""確認 Numba decoder 與原本 Python 路徑逐值等價。"""

from __future__ import annotations

import sys
import importlib
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import Problem.services.evaluation_kernels as kernels
import Problem.services.routing_service as routing_module
from Problem.Problem import Problem
from Problem.services.fitness_servicev2 import FitnessServiceV2
from State.SensorEncoding import SensorEncoding


sensor_encoding_module = importlib.import_module("State.SensorEncoding")


def decode(problem, code, version, use_numba):
    """切換相同 decoder 的執行後端並回傳解碼結果。"""
    sensor_encoding_module.NUMBA_AVAILABLE = bool(use_numba)
    routing_module.NUMBA_AVAILABLE = bool(use_numba)
    candidate = SensorEncoding(code.copy())
    state = getattr(candidate, f"decode{version}")(problem)
    objectives = np.asarray(problem.evaluate_state(state), dtype=float)
    return candidate, state, objectives


def assert_same_result(reference, accelerated):
    """比較會影響搜尋決策與能量計算的所有 decoder 輸出。"""
    reference_candidate, reference_state, reference_objectives = reference
    accelerated_candidate, accelerated_state, accelerated_objectives = accelerated
    np.testing.assert_array_equal(
        reference_candidate.code,
        accelerated_candidate.code,
    )
    for name in (
        "levels",
        "next_hops",
        "tx_load",
        "remaining_capacity",
        "sensing_radii",
    ):
        np.testing.assert_array_equal(
            getattr(reference_state, name),
            getattr(accelerated_state, name),
        )
    assert reference_state.paths == accelerated_state.paths
    np.testing.assert_array_equal(reference_objectives, accelerated_objectives)


def main():
    """以 A2 固定 chromosome 驗證 v1、v2、v3 三種解碼模式。"""
    if not kernels.NUMBA_INSTALLED:
        print("smoke_numba_evaluation_skipped: numba is not installed")
        return

    problem = Problem(B=100, S=100, T=100, F=10, FILE="A2")
    problem.fitness_service = FitnessServiceV2(problem)
    rng = np.random.default_rng(20260909)
    codes = []
    for _ in range(30):
        code = np.empty(problem.SENSOR_NUMBER * 2, dtype=np.int64)
        code[0::2] = rng.integers(0, problem.radius_option_counts)
        code[1::2] = rng.integers(
            0,
            SensorEncoding.RANK_PRECISION,
            problem.SENSOR_NUMBER,
        )
        codes.append(code)

    original_sensor_flag = sensor_encoding_module.NUMBA_AVAILABLE
    original_routing_flag = routing_module.NUMBA_AVAILABLE
    try:
        for version in ("v1", "v2", "v3"):
            for code in codes:
                reference = decode(problem, code, version, False)
                accelerated = decode(problem, code, version, True)
                assert_same_result(reference, accelerated)
    finally:
        sensor_encoding_module.NUMBA_AVAILABLE = original_sensor_flag
        routing_module.NUMBA_AVAILABLE = original_routing_flag

    print("smoke_numba_evaluation_ok")


if __name__ == "__main__":
    main()
