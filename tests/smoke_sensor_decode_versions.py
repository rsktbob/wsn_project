"""Verify the three SensorEncoding decode policies."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from State.SensorEncoding import SensorEncoding


class FakeRouting:
    def __init__(self, problem):
        self.problem = problem
        self.reject_id = None

    def build_routes(
        self,
        state,
        routing_order,
        *,
        disable_failed=True,
        **kwargs,
    ):
        """模擬目前 RoutingService 介面與 v2／v3 的失敗處理。"""
        del kwargs
        routing_ids = np.asarray(routing_order, dtype=int)
        if routing_ids.ndim == 2:
            routing_ids = routing_ids[:, 0]
        for sensor_id in routing_ids:
            if sensor_id == self.reject_id:
                # 模擬正式 RoutingService 遇到容量不足時暫時關閉 level。
                if disable_failed:
                    state.levels[sensor_id] = 0
                continue
            state.next_hops[sensor_id] = self.problem.BSID
            state.paths[sensor_id] = [sensor_id, self.problem.BSID]
            state.tx_load[sensor_id] = 1
        failed = () if self.reject_id is None else (self.reject_id,)
        return SimpleNamespace(capacity_failed_ids=failed)


class FakeProblem:
    SENSOR_NUMBER = 3
    TARGET_NUMBER = 3
    DEVICE_NUMBER = 4
    BSID = 3

    def __init__(self):
        self.radius_option_counts = np.asarray([2, 2, 2], dtype=int)
        self.radius_table = np.asarray(
            [[0.0, 3.0], [0.0, 3.0], [0.0, 3.0]]
        )
        self.coverage_table = np.zeros((3, 3, 2), dtype=int)
        self.coverage_table[:, 0, 1] = [1, 0, 0]
        self.coverage_table[:, 1, 1] = [0, 1, 0]
        self.coverage_table[:, 2, 1] = [1, 1, 1]
        self.proximity_score = np.ones(3, dtype=float)
        self.energy_score = np.ones(3, dtype=float)
        self.routing_service = FakeRouting(self)

    def sensing_option_count(self, sensor_id):
        return int(self.radius_option_counts[int(sensor_id)])

    def resolve_state_radius(self, state):
        state.sensing_radii = self.radius_table[
            np.arange(self.SENSOR_NUMBER), state.levels
        ].copy()
        return state.sensing_radii


def main():
    problem = FakeProblem()
    # priority 依序為 sensor 0、1、2；sensor 2 最後加入並覆蓋全部 targets。
    code = np.asarray([1, 9, 1, 5, 1, 1], dtype=int)
    encoding = SensorEncoding(code)

    np.testing.assert_array_equal(
        encoding.decodev1(problem).levels,
        [1, 1, 1],
    )
    np.testing.assert_array_equal(
        encoding.decodev2(problem).levels,
        [0, 0, 1],
    )
    np.testing.assert_array_equal(
        encoding.decodev3(problem).levels,
        [0, 0, 1],
    )
    # 公開 decode 現在選擇 v3；無路由拒絕時與 v2 排程相同。
    np.testing.assert_array_equal(
        encoding.decode(problem).levels,
        [0, 0, 1],
    )
    np.testing.assert_array_equal(encoding.code, code)

    # 舊連續演算法可能輸出共同 FMAX 值；v3 與 v2 使用相同排程規則，
    # sensing level 正規化仍只作用於 State。
    legacy_code = np.asarray([3, 9, 1, 5, 1, 1], dtype=int)
    legacy = SensorEncoding(legacy_code.copy())
    legacy_state = legacy.decodev3(problem)
    np.testing.assert_array_equal(legacy.code, legacy_code)
    np.testing.assert_array_equal(legacy_state.levels, [0, 0, 1])

    # v2 關閉路由失敗的 State sensor 並回寫 chromosome；v3 讓 sensor
    # 保留在排程中，但不加入路由，也不修改 chromosome。
    problem.routing_service.reject_id = 2
    route_code = np.asarray([1, 9, 1, 5, 1, 1], dtype=int)
    routed_v2 = SensorEncoding(route_code.copy())
    routed_v3 = SensorEncoding(route_code.copy())
    state_v2 = routed_v2.decodev2(problem)
    state_v3 = routed_v3.decodev3(problem)
    np.testing.assert_array_equal(state_v2.levels, [0, 0, 0])
    np.testing.assert_array_equal(state_v3.levels, [0, 0, 1])
    assert routed_v2.code[4] == 0
    np.testing.assert_array_equal(routed_v3.code, route_code)
    assert state_v3.next_hops[2] == -1

    print("smoke_sensor_decode_versions_ok")


if __name__ == "__main__":
    main()
