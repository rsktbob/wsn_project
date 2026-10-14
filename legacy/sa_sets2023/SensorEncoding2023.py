"""學長 2023 版的 sensor 解碼（codes/2023以恩MWSN/State/CodingState.py 的 Decode）。

染色體格式和 :class:`SensorEncoding` 相同（[s0, p0, s1, p1, ...]），只換解碼規則：

1. 路由優先值 = (電量分數 + 距離分數) × gene / 9，沒有 0.5 偏移（RankDecoding）。
2. 排程用 v1：依優先值只開有新增覆蓋的 sensor，沒有冗餘移除（SchDecoding）。
3. 選父節點用原版打分（RouDecoding）：
   a = min(⌊連線容量⌋ + 路徑容量 / 最大路徑容量, ⌊路徑容量⌋ + 連線容量 / 最大連線容量) − 封包量，
   BS 的路徑容量為 0x3F3F3F3F；a 最大者 < 0 視為容量不足。
4. 連線容量全為 0 的 sensor 只在解碼結果關閉；容量不足的 sensor 除了關閉，
   還會把染色體的感測等級基因寫回 0（原版的副作用，刻意保留）。
"""

from __future__ import annotations

import numpy as np

from Problem.services.evaluation_kernels import (
    decode_sensor_schedule_kernel,
    njit,
)
from State.SensorEncoding import SensorEncoding
from State.State import State

# 原版 CodingState.RouDecoding 給 BS 的路徑容量。
BS_CAPACITY_2023 = float(0x3F3F3F3F)


@njit(cache=True, nogil=True)
def build_routes_2023_kernel(
    levels,
    routing_ids,
    link_capacity,
    generated_load,
    sensor_count,
    bs_id,
    next_hops,
    tx_load,
    remaining_capacity,
):
    """原版選父節點打分的貪婪路由；回傳 (已連線 id, 容量不足 id)。"""
    next_hops.fill(-1)
    tx_load.fill(0)
    remaining_capacity.fill(-1.0)
    remaining_capacity[bs_id] = BS_CAPACITY_2023

    routed_nodes = np.empty(sensor_count + 1, dtype=np.int64)
    routed_nodes[0] = bs_id
    routed_count = 1
    capacity_failed = np.empty(routing_ids.shape[0], dtype=np.int64)
    capacity_failed_count = 0

    for route_id in range(routing_ids.shape[0]):
        sensor_id = int(routing_ids[route_id])
        level = int(levels[sensor_id])
        required_load = generated_load[sensor_id]

        max_link = -np.inf
        max_path = -np.inf
        for parent_index in range(routed_count):
            parent_id = routed_nodes[parent_index]
            link_value = link_capacity[sensor_id, level, parent_id]
            if link_value > max_link:
                max_link = link_value
            if remaining_capacity[parent_id] > max_path:
                max_path = remaining_capacity[parent_id]
        if max_link <= 0.0:
            # 原版：sch 設 0、不改 code。
            levels[sensor_id] = 0
            continue

        # np.argmax 在相同最大值時選第一個，因此只在嚴格較大時更新。
        parent_index = 0
        best_score = -np.inf
        for candidate_index in range(routed_count):
            parent_id = routed_nodes[candidate_index]
            link_value = link_capacity[sensor_id, level, parent_id]
            path_value = remaining_capacity[parent_id]
            score = (
                np.floor(link_value) + path_value / max_path - required_load
            )
            other = (
                np.floor(path_value) + link_value / max_link - required_load
            )
            if other < score:
                score = other
            if score > best_score:
                best_score = score
                parent_index = candidate_index

        if best_score < 0.0:
            levels[sensor_id] = 0
            capacity_failed[capacity_failed_count] = sensor_id
            capacity_failed_count += 1
            continue

        parent_id = routed_nodes[parent_index]
        next_hops[sensor_id] = parent_id
        routed_nodes[routed_count] = sensor_id
        routed_count += 1

        forwarded_id = sensor_id
        while forwarded_id != bs_id:
            tx_load[forwarded_id] += required_load
            forwarded_id = next_hops[forwarded_id]

        # 同原版 CalCap：依接入順序重算每個已連線節點的瓶頸容量。
        for connected_index in range(1, routed_count):
            connected_id = routed_nodes[connected_index]
            connected_parent = next_hops[connected_id]
            direct_capacity = (
                link_capacity[
                    connected_id,
                    levels[connected_id],
                    connected_parent,
                ]
                - tx_load[connected_id]
            )
            parent_capacity = remaining_capacity[connected_parent]
            if parent_capacity < direct_capacity:
                direct_capacity = parent_capacity
            remaining_capacity[connected_id] = direct_capacity

    return routed_nodes[1:routed_count], capacity_failed[:capacity_failed_count]


class SensorEncoding2023(SensorEncoding):
    """學長 2023 版解碼：無 0.5 偏移、v1 排程、原版選父節點、容量不足寫回染色體。"""

    def decode(self, problem):
        if self.code is None:
            raise ValueError("SensorEncoding2023.code has not been initialized")
        code = np.asarray(self.code, dtype=np.int64)
        expected_shape = (problem.SENSOR_NUMBER * 2,)
        if code.shape != expected_shape:
            raise ValueError(
                f"sensor code must have shape {expected_shape}, "
                f"got {code.shape}"
            )

        schedule_genes = np.mod(
            code[0::2],
            np.asarray(problem.radius_option_counts, dtype=np.int64),
        )
        priority_genes = code[1::2] % self.RANK_PRECISION
        priority = (problem.energy_score + problem.proximity_score) * (
            priority_genes / (self.RANK_PRECISION - 1)
        )
        sensor_order = np.argsort(priority)[::-1]

        state = State.empty(problem)
        state.levels = np.ascontiguousarray(state.levels, dtype=np.int64)
        state.next_hops = np.ascontiguousarray(state.next_hops, dtype=np.int64)
        state.tx_load = np.ascontiguousarray(state.tx_load, dtype=np.int64)
        state.remaining_capacity = np.ascontiguousarray(
            state.remaining_capacity, dtype=np.float64
        )

        routing_ids = decode_sensor_schedule_kernel(
            np.asarray(problem.coverage_table),
            schedule_genes,
            np.ascontiguousarray(sensor_order, dtype=np.int64),
            False,
            state.levels,
        )
        connected_ids, capacity_failed_ids = build_routes_2023_kernel(
            state.levels,
            routing_ids,
            np.asarray(problem.link_capacity, dtype=np.float64),
            np.asarray(problem.generated_load, dtype=np.int64),
            int(problem.SENSOR_NUMBER),
            int(problem.BSID),
            state.next_hops,
            state.tx_load,
            state.remaining_capacity,
        )

        bs_id = int(problem.BSID)
        state.paths = {bs_id: [bs_id]}
        for sensor_id in connected_ids:
            sensor_id = int(sensor_id)
            state.paths[sensor_id] = (
                [sensor_id] + state.paths[int(state.next_hops[sensor_id])]
            )

        # 原版副作用：容量不足的 sensor 把感測等級基因寫回 0。
        for sensor_id in capacity_failed_ids:
            self.code[int(sensor_id) * 2] = 0

        problem.resolve_state_radius(state)
        return state


__all__ = ["SensorEncoding2023", "build_routes_2023_kernel"]
