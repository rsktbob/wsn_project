"""WSN 評估流程的 Numba 純陣列核心。"""

from __future__ import annotations

import os

import numpy as np

try:
    import numba as _numba

    _numba_njit = _numba.njit
    NUMBA_INSTALLED = True
    NUMBA_VERSION = _numba.__version__
except ImportError:  # pragma: no cover - 供尚未更新環境時維持原功能
    _numba_njit = None
    NUMBA_INSTALLED = False
    NUMBA_VERSION = ""

NUMBA_AVAILABLE = NUMBA_INSTALLED and os.environ.get(
    "WSN_DISABLE_NUMBA", ""
).strip().lower() not in {"1", "true", "yes", "on"}

if NUMBA_AVAILABLE:
    njit = _numba_njit
else:
    def njit(*args, **kwargs):
        """Numba 缺失時保留相同函式介面的循序後備。"""
        del args, kwargs

        def decorate(function):
            return function

        return decorate


@njit(cache=True, nogil=True)
def count_covered_targets_kernel(coverage_table, schedule):
    """計算每個 target 被目前排程覆蓋的次數。"""
    target_count = coverage_table.shape[0]
    sensor_count = coverage_table.shape[1]
    counts = np.zeros(target_count, dtype=np.int64)
    for target_id in range(target_count):
        count = 0
        for sensor_id in range(sensor_count):
            level = schedule[sensor_id]
            if coverage_table[target_id, sensor_id, level] == 1:
                count += 1
        counts[target_id] = count
    return counts


@njit(cache=True, nogil=True)
def find_uncovered_targets_kernel(coverage_table, schedule):
    """直接找出未覆蓋 target，找到第一顆可覆蓋 sensor 後立即停止。"""
    target_count = coverage_table.shape[0]
    sensor_count = coverage_table.shape[1]
    uncovered = np.empty(target_count, dtype=np.int64)
    uncovered_count = 0
    for target_id in range(target_count):
        covered = False
        for sensor_id in range(sensor_count):
            level = schedule[sensor_id]
            if coverage_table[target_id, sensor_id, level] == 1:
                covered = True
                break
        if not covered:
            uncovered[uncovered_count] = target_id
            uncovered_count += 1
    return uncovered[:uncovered_count]


@njit(cache=True, nogil=True)
def decode_sensor_schedule_kernel(
    coverage_table,
    schedule_genes,
    sensor_order,
    remove_redundant,
    levels,
):
    """將排程寫入既有 levels，並回傳需要建立路由的 sensor id。"""
    target_count = coverage_table.shape[0]
    sensor_count = coverage_table.shape[1]
    levels.fill(0)
    uncovered = np.ones(target_count, dtype=np.uint8)
    open_ids = np.empty(sensor_count, dtype=np.int64)
    open_count = 0

    # 完整保留 SensorEncoding 的規則：priority 高者先處理，只有能新增
    # 覆蓋的 sensor 才會進入候選路由順序。
    for order_id in range(sensor_order.shape[0]):
        sensor_id = int(sensor_order[order_id])
        level = int(schedule_genes[sensor_id])
        adds_coverage = False
        for target_id in range(target_count):
            if (
                uncovered[target_id] == 1
                and coverage_table[target_id, sensor_id, level] != 0
            ):
                adds_coverage = True
                break
        if not adds_coverage:
            continue

        for target_id in range(target_count):
            if coverage_table[target_id, sensor_id, level] != 0:
                uncovered[target_id] = 0
        levels[sensor_id] = level
        open_ids[open_count] = sensor_id
        open_count += 1

    if not remove_redundant:
        return open_ids[:open_count]

    coverage_count = np.zeros(target_count, dtype=np.int64)
    for open_id in range(open_count):
        sensor_id = open_ids[open_id]
        level = levels[sensor_id]
        for target_id in range(target_count):
            if coverage_table[target_id, sensor_id, level] != 0:
                coverage_count[target_id] += 1

    retained = np.ones(open_count, dtype=np.uint8)
    for open_id in range(open_count - 1, -1, -1):
        sensor_id = open_ids[open_id]
        level = levels[sensor_id]
        has_target = False
        removable = True
        for target_id in range(target_count):
            if coverage_table[target_id, sensor_id, level] == 0:
                continue
            has_target = True
            if coverage_count[target_id] < 2:
                removable = False
                break
        if not has_target or not removable:
            continue
        retained[open_id] = 0
        levels[sensor_id] = 0
        for target_id in range(target_count):
            if coverage_table[target_id, sensor_id, level] != 0:
                coverage_count[target_id] -= 1

    retained_count = 0
    for open_id in range(open_count):
        if retained[open_id] == 0:
            continue
        open_ids[retained_count] = open_ids[open_id]
        retained_count += 1
    return open_ids[:retained_count]


@njit(cache=True, nogil=True)
def build_routes_kernel(
    levels,
    routing_ids,
    link_capacity,
    generated_load,
    sensor_count,
    device_count,
    bs_id,
    disable_failed,
    next_hops,
    tx_load,
    remaining_capacity,
):
    """將 greedy bottleneck 路由寫入既有 State 陣列。"""
    next_hops.fill(-1)
    tx_load.fill(0)
    remaining_capacity.fill(-1.0)
    remaining_capacity[bs_id] = np.inf

    routed_nodes = np.empty(sensor_count + 1, dtype=np.int64)
    routed_nodes[0] = bs_id
    routed_count = 1
    rejected = np.empty(routing_ids.shape[0], dtype=np.int64)
    unavailable = np.empty(routing_ids.shape[0], dtype=np.int64)
    capacity_failed = np.empty(routing_ids.shape[0], dtype=np.int64)
    rejected_count = 0
    unavailable_count = 0
    capacity_failed_count = 0

    for route_id in range(routing_ids.shape[0]):
        sensor_id = int(routing_ids[route_id])
        level = int(levels[sensor_id])
        required_load = generated_load[sensor_id]

        max_link = -np.inf
        for parent_index in range(routed_count):
            parent_id = routed_nodes[parent_index]
            link_value = link_capacity[sensor_id, level, parent_id]
            if link_value > max_link:
                max_link = link_value
        if max_link <= 0.0:
            if disable_failed:
                levels[sensor_id] = 0
            rejected[rejected_count] = sensor_id
            rejected_count += 1
            unavailable[unavailable_count] = sensor_id
            unavailable_count += 1
            continue

        # np.argmax 在相同最大值時選第一個，因此只在嚴格較大時更新。
        parent_index = 0
        best_score = -np.inf
        for candidate_index in range(routed_count):
            parent_id = routed_nodes[candidate_index]
            link_value = link_capacity[sensor_id, level, parent_id]
            path_value = remaining_capacity[parent_id]
            bottleneck = link_value
            if path_value < bottleneck:
                bottleneck = path_value
            score = bottleneck - required_load
            if score > best_score:
                best_score = score
                parent_index = candidate_index

        parent_id = routed_nodes[parent_index]
        if best_score < 0.0:
            if disable_failed:
                levels[sensor_id] = 0
            rejected[rejected_count] = sensor_id
            rejected_count += 1
            capacity_failed[capacity_failed_count] = sensor_id
            capacity_failed_count += 1
            continue

        next_hops[sensor_id] = parent_id
        routed_nodes[routed_count] = sensor_id
        routed_count += 1

        # 新 sensor 的資料量會流經自己與所有上游 sensor，但不計入 BS。
        forwarded_id = sensor_id
        while forwarded_id != bs_id:
            tx_load[forwarded_id] += required_load
            forwarded_id = next_hops[forwarded_id]

        # 任一上游負載改變後，所有已接入節點的 bottleneck 都需重算。
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

    return (
        routed_nodes[1:routed_count],
        rejected[:rejected_count],
        unavailable[:unavailable_count],
        capacity_failed[:capacity_failed_count],
    )


@njit(cache=True, nogil=True)
def first_out_of_range_kernel(schedule, option_count):
    """回傳第一個越界 sensor 的索引；全部合法時回傳 -1。

    核心本身關閉邊界檢查，越界的 level 會讀到界外記憶體並靜默回傳
    看似合理的錯誤值，因此排程必須在進入任何核心前先通過這道檢查。
    """
    for sensor_id in range(schedule.shape[0]):
        value = schedule[sensor_id]
        if value < 0 or value >= option_count:
            return sensor_id
    return -1


@njit(cache=True, nogil=True)
def first_out_of_range_per_sensor_kernel(schedule, option_counts):
    """同上，但每顆 sensor 各自比對自己的 option 數。

    ``bucketed``／``exact`` 模式下各 sensor 的 option 數不同，且
    ``coverage_table`` 會把多餘欄位補成最後一個合法半徑的結果，
    因此只比對全域上界不足以擋下越界。
    """
    for sensor_id in range(schedule.shape[0]):
        value = schedule[sensor_id]
        if value < 0 or value >= option_counts[sensor_id]:
            return sensor_id
    return -1


@njit(cache=True, nogil=True)
def find_disconnected_kernel(
    next_hops,
    sensing_radii,
    base_station_id,
    seed_status,
):
    """回傳啟用但轉送鏈無法抵達基地台的 sensor。

    沿著 ``next_hops`` 逐跳前進並記錄每個節點的結果，因此每條鏈最多
    只走一次；原本的 Python 版以遞迴建出完整 path 再看結尾，但呼叫端
    一律只用 ``len(...)`` 或 id 本身，path 本身是中間產物。

    ``seed_status`` 帶入 ``state.paths`` 這份解碼期快取（0 未知、1 連通、
    2 斷線）。快取的優先序必須保留：原實作對已在 ``paths`` 裡的 sensor
    直接讀結尾，不會重走 ``next_hops``，兩者不一致時以快取為準。

    判定與 ``trace_path`` 的結尾規則一致：走到基地台為連通；走到負值
    或其他非法節點視為斷線；繞回已在目前鏈上的節點（迴圈）同樣斷線。
    """
    sensor_count = next_hops.shape[0]
    # 0 = 尚未判定，1 = 連通，2 = 斷線
    status = seed_status.copy()
    on_chain = np.zeros(sensor_count, dtype=np.uint8)
    chain = np.empty(sensor_count, dtype=np.int64)
    disconnected = np.empty(sensor_count, dtype=np.int64)
    found = 0

    for start in range(sensor_count):
        if sensing_radii[start] <= 0.0:
            continue
        if status[start] == 0:
            depth = 0
            node = start
            result = 2
            while True:
                if node == base_station_id:
                    result = 1
                    break
                if node < 0 or node >= sensor_count:
                    # 既不是基地台也不是合法 sensor：等同 trace_path
                    # 以 [sensor, next_hop] 收尾而結尾不是 BSID。
                    result = 2
                    break
                if status[node] != 0:
                    result = status[node]
                    break
                if on_chain[node] == 1:
                    # 迴圈：trace_path 會把 path 截在重複節點，結尾不是
                    # BSID，因此整條鏈都算斷線。
                    result = 2
                    break
                chain[depth] = node
                on_chain[node] = 1
                depth += 1
                node = next_hops[node]
            for index in range(depth):
                node_id = chain[index]
                status[node_id] = result
                on_chain[node_id] = 0
        if status[start] == 2:
            disconnected[found] = start
            found += 1
    return disconnected[:found]


# 不要在這裡加 fastmath=True。實測過：它對速度沒有可量測的貢獻
# （本檔的核心只佔評估時間的一小部分），卻會讓浮點重新結合改變路由
# 成本的末位，進而改變整條 lifetime 軌跡 —— sa_sets seed 9 的重最佳化
# 次數從 27 變成 23。其餘三個核心是純整數運算，加了本來就沒有作用。
@njit(cache=True, nogil=True)
def calculate_routing_cost_kernel(
    sensing_radii,
    next_hops,
    tx_load,
    distances,
    generated_load,
    first_amp_factor,
    second_amp_factor,
    circuit_cost,
    packet_energy,
    switch_distance,
):
    """以 JIT 迴圈計算所有 sensor 的資料傳送與轉送成本。"""
    sensor_count = sensing_radii.shape[0]
    device_count = distances.shape[1]
    cost = np.zeros(sensor_count, dtype=np.float64)
    for sensor_id in range(sensor_count):
        if tx_load[sensor_id] <= 0.0:
            continue
        next_hop = int(next_hops[sensor_id])
        if next_hop < 0 or next_hop >= device_count:
            continue

        distance = distances[sensor_id, next_hop]
        if distance <= switch_distance:
            distance_cost = (
                first_amp_factor * packet_energy * distance * distance
            )
        else:
            distance_cost = (
                second_amp_factor
                * packet_energy
                * distance
                * distance
                * distance
                * distance
            )
        own_load = 0.0
        if sensing_radii[sensor_id] > 0.0:
            own_load = generated_load[sensor_id]
        cost[sensor_id] = (
            tx_load[sensor_id] * (distance_cost + circuit_cost * 2.0)
            - circuit_cost * own_load
        )
    return cost


_KERNELS_WARMED = False


@njit(cache=True, nogil=True)
def target_remaining_energy_kernel(target_sensor_mask, remaining_energy):
    """各 target 周圍 sensor 的剩餘能量總和。

    取代 ``sum(mask * remaining_energy, axis=1)``：後者每次評估都要配置一個
    (TARGET_NUMBER, SENSOR_NUMBER) 的暫存陣列再歸約，這裡直接融合成一趟迴圈。
    """
    target_count = target_sensor_mask.shape[0]
    sensor_count = target_sensor_mask.shape[1]
    totals = np.zeros(target_count, dtype=np.float64)
    for target_id in range(target_count):
        total = 0.0
        for sensor_id in range(sensor_count):
            total += (
                target_sensor_mask[target_id, sensor_id]
                * remaining_energy[sensor_id]
            )
        totals[target_id] = total
    return totals


def warm_evaluation_kernels():
    """在建立 worker 前產生快取，避免 Windows 子程序同時編譯。"""
    global _KERNELS_WARMED
    if _KERNELS_WARMED or not NUMBA_AVAILABLE:
        return

    coverage = np.zeros((1, 1, 1), dtype=np.int64)
    schedule = np.zeros(1, dtype=np.int64)
    target_remaining_energy_kernel(
        np.zeros((1, 1), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
    )
    count_covered_targets_kernel(coverage, schedule)
    first_out_of_range_kernel(schedule, 1)
    first_out_of_range_per_sensor_kernel(schedule, schedule)
    find_disconnected_kernel(
        np.full(1, -1, dtype=np.int64),
        np.zeros(1, dtype=np.float64),
        1,
        np.zeros(1, dtype=np.uint8),
    )
    find_uncovered_targets_kernel(coverage, schedule)
    decode_sensor_schedule_kernel(
        coverage,
        schedule,
        schedule,
        True,
        schedule.copy(),
    )
    build_routes_kernel(
        schedule.copy(),
        schedule,
        np.zeros((1, 1, 2), dtype=np.float64),
        schedule,
        1,
        2,
        1,
        True,
        np.full(1, -1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        np.full(2, -1.0, dtype=np.float64),
    )
    calculate_routing_cost_kernel(
        np.zeros(1, dtype=np.float64),
        np.full(1, -1, dtype=np.int64),
        np.zeros(1, dtype=np.float64),
        np.zeros((2, 2), dtype=np.float64),
        np.zeros(1, dtype=np.float64),
        0.0,
        0.0,
        0.0,
        0.0,
        87.0,
    )
    _KERNELS_WARMED = True


__all__ = [
    "NUMBA_AVAILABLE",
    "NUMBA_INSTALLED",
    "NUMBA_VERSION",
    "calculate_routing_cost_kernel",
    "build_routes_kernel",
    "count_covered_targets_kernel",
    "decode_sensor_schedule_kernel",
    "find_disconnected_kernel",
    "find_uncovered_targets_kernel",
    "first_out_of_range_kernel",
    "first_out_of_range_per_sensor_kernel",
    "target_remaining_energy_kernel",
    "warm_evaluation_kernels",
]
