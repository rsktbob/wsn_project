"""WSN 問題定義與環境服務入口。

演算法與 State 應把 ``Problem`` 視為環境介面：State 負責把基因解碼成
``levels``、``next_hops``、``tx_load``，Problem 再統一處理覆蓋、路由、能量與 fitness。
"""

import numpy as np
import pandas as pd

from Problem.Cal import *
from Problem.services import (
    CoverageService,
    EnergyService,
    FitnessService,
    MobilityService,
    RoutingService,
)
from Problem.services.evaluation_kernels import (
    NUMBA_AVAILABLE,
    first_out_of_range_per_sensor_kernel,
)


class Problem:
    """WSN 網路環境與所有物理模型的統一入口。

    重要 State 欄位：
        ``state.levels[sensor_id]``
            感測器的開啟 level；0 表示關閉，1 以上表示感測範圍索引。
        ``state.next_hops[sensor_id]``
            感測器選擇的下一跳 device id；BS 的 id 是 ``BSID``。
        ``state.tx_load[sensor_id]``
            感測器目前需要傳送或轉送的資料量。

    重要 Problem 快取：
        ``coverage_table[target, sensor, level]``
            指定 sensor/level 是否能覆蓋 target。
        ``link_capacity[sensor, level, next_hop]``
            扣除感測消耗後，sensor 到 next_hop 的可傳輸容量估計。
        ``cover_candidates[target]``
            可覆蓋該 target 的 ``[sensor_id, level]`` 候選清單。

    Problem 不解讀演算法的原始 chromosome。CodingState、TargetCodingState
    可以使用不同編碼，只要最後產生相同的 ``levels``、
    ``next_hops``、``tx_load`` 即可。
    """

    DATA_GENERATION_MIN_COVERAGE = 3

    def __init__(
        self,
        B=200,
        S=300,
        T=64,
        F=500,
        FILE=None,
        sch=None,
        update_position=None,
        updated_power=None,
        Target_position=None,
        sensing_mode="discrete",
        sensing_levels=None,
        maximum_sensing_radius=None,
        sensor_ring_width=None,
        sensor_ring_order=True,
        routing_service="v1",
    ):
        """建立一個 WSN 問題。

        Args:
            B: 正方形部署區域的邊長。
            S: sensor 數量。
            T: target 數量。
            F: 每個 sensor 的初始能量。
            FILE: 資料集代號；None 表示隨機產生 sensor/target。
            sch: 舊版呼叫介面保留參數，目前建構流程未使用。
            update_position: 移動後的 sensor 座標；None 表示建立新資料。
            updated_power: 移動或 lifetime 更新後的 sensor 剩餘能量。
            Target_position: 搭配 update_position 使用的既有 target 座標。
            sensor_ring_width: sensor chromosome 的空間排序環寬；未指定時
                使用 8，讓同一 ring 內的 sensor 依角度排列。
            sensor_ring_order: True(預設)時 chromosome index 依「距離環→
                環內角度」排序，讓相鄰基因具有局部空間意義；False 時改為
                單純依離 base station 的距離排序，不分環、不看角度。
            routing_service: 路由服務版本；目前僅支援 ``v1``。
        """
        # 幾何與資料產生工具，負責距離、隨機座標及資料可行性檢查。
        self.G = Cal()

        # FILE/MFILE 分別是輸入資料與舊版處理結果的目錄。
        if FILE is None:
            print("New")
            self.FILE = None
        else:
            self.MFILE = "process/" + str(B) + str(S) + str(T) + "_" + str(FILE) + "/"
            self.FILE = "data/" + str(B) + str(S) + str(T) + "_" + str(FILE) + "/"
            print("Read:" + self.FILE)

        # 基本拓樸大小與 id 陣列。sensor/target 會在 create_map_data 中填入座標。
        self.BOUNDARY = B
        self.SENSOR_NUMBER = S
        self.SENSOR_ID = np.arange(self.SENSOR_NUMBER)
        self.sensor = None
        self.TARGET_NUMBER = T
        self.TARGET_ID = np.arange(self.TARGET_NUMBER)
        self.target = None

        # Base Station 位置由資料集類型決定。
        print(FILE)
        if FILE == "C" or FILE == "DSC" or FILE == "DS6":
            self.BS = np.array([B // 2, B // 2])
        elif FILE == "B" or FILE == "DSB" or FILE == "DS7":
            self.BS = np.array([3, B // 2])
        else:
            self.BS = np.array([3, 3])
        print("BS position:", self.BS)

        # 離散感測範圍。levels 儲存的是索引，不是公尺值：
        # level=0 關閉；level=1..5 對應 3、6、9、12、15。
        if sensing_levels is None:
            sensing_levels = [0, 3, 6, 9, 12, 15]
        sensing_levels = np.asarray(sensing_levels, dtype=float)
        if (
            sensing_levels.ndim != 1
            or len(sensing_levels) < 2
            or not np.isclose(sensing_levels[0], 0.0)
            or np.any(np.diff(sensing_levels) <= 0)
        ):
            raise ValueError(
                "sensing_levels must be strictly increasing and start at 0"
            )

        normalized_sensing_mode = str(sensing_mode).strip().lower()
        if normalized_sensing_mode == "continuous":
            normalized_sensing_mode = "exact"
        if normalized_sensing_mode == "calibrated":
            normalized_sensing_mode = "bucketed"
        if normalized_sensing_mode not in (
            "discrete",
            "bucketed",
            "exact",
        ):
            raise ValueError(
                "sensing_mode must be 'discrete', 'bucketed', or 'exact'"
            )

        self.sensing_mode = normalized_sensing_mode
        self.DISCRETE_LEVEL_RANGE = sensing_levels.copy()
        self.LEVEL_RANGE = sensing_levels.copy()
        self.MAX_SENSING_RADIUS = (
            float(sensing_levels[-1])
            if maximum_sensing_radius is None
            else float(maximum_sensing_radius)
        )
        if self.MAX_SENSING_RADIUS <= 0:
            raise ValueError("maximum_sensing_radius must be positive")
        self.sensor_ring_width = (
            8.0
            if sensor_ring_width is None
            else float(sensor_ring_width)
        )
        if self.sensor_ring_width <= 0:
            raise ValueError("sensor_ring_width must be positive")
        self.sensor_ring_order = bool(sensor_ring_order)
        if (
            self.sensing_mode == "bucketed"
            and not np.isclose(
                self.MAX_SENSING_RADIUS,
                self.DISCRETE_LEVEL_RANGE[-1],
            )
        ):
            raise ValueError(
                "bucketed mode uses sensing_levels as interval boundaries; "
                "maximum_sensing_radius must equal the last boundary"
            )
        self.LEVEL = len(self.LEVEL_RANGE)
        self.radius_options = None
        self.radius_option_counts = None
        self.radius_table = None
        self.sensing_costs = None

        # 將不同物理責任交給 service；Problem 對外提供統一轉接方法。
        self.coverage_service = CoverageService(self)
        normalized_routing_service = str(routing_service).strip().lower()
        routing_services = {"v1": RoutingService}
        if normalized_routing_service not in routing_services:
            raise ValueError("routing_service must be 'v1'")
        self.routing_service_name = normalized_routing_service
        self.routing_service = routing_services[normalized_routing_service](self)
        self.energy_service = EnergyService(self)
        self.fitness_service = FitnessService(self)

        # 建立或載入座標，再產生 coverage_table 與 target_sensor_mask。
        self.create_map_data(self.FILE, update_position, Target_position)
        self.configure_sensing_ranges()
        self.coverage_table, self.target_sensor_mask = (
            self.calculate_coverage_matrix()
        )

        # device = 所有 sensors 加上最後一個 BS；因此 BSID 永遠是最後一個 id。
        self.device = np.concatenate((self.sensor, [self.BS]))
        self.DEVICE_NUMBER = len(self.device)
        self.BSID = self.DEVICE_NUMBER - 1

        # distances[source, destination] 是 device 間距離。
        # blocked_distance 表示禁止作為下一跳；對角線也設為此值。
        self.distances = self.G.CalDistance(self.device, self.device)
        self.blocked_distance = 0x3F3F3F3F
        self.distances[
            np.arange(self.DEVICE_NUMBER), np.arange(self.DEVICE_NUMBER)
        ] = self.blocked_distance

        # 移除「sensor 到候選節點」比「sensor 直接到 BS」還長的連線。
        # 注意：這只比較兩段連線的長度，不代表候選節點本身一定更靠近 BS。
        # 被設為 inf 的候選節點之後不會被 routing decoder 選為下一跳。
        for sensor_id in range(self.SENSOR_NUMBER):
            farther_nodes = np.where(
                self.distances[sensor_id, :]
                > self.distances[sensor_id, self.SENSOR_NUMBER]
            )[0]
            self.distances[sensor_id, farther_nodes] = self.blocked_distance
        self.distances[self.BSID][self.BSID] = 0

        # 能量與通訊參數：initial_energy 是初始能量，energy 是當前能量；
        # generated_load 是自產資料量，circuit_cost 是電路收發單位成本。
        # amp_factors/d0 決定放大器模型，sensing_factor 是感測半徑平方係數。
        self.initial_energy = F
        self.generated_load = np.repeat(400, self.SENSOR_NUMBER)
        self.circuit_cost = 5 * nj
        self.amp_factors = [10, 0.0013]
        self.sensing_factor = 100 * nj
        self.d0 = 87
        self.energy = np.zeros(self.SENSOR_NUMBER)

        # 各離散感測 level 的成本：LEVEL_RANGE ** 2 * sensing_factor。
        self.level_costs = self.LEVEL_RANGE**2 * self.sensing_factor
        self.sensing_costs = self.radius_table**2 * self.sensing_factor
        if updated_power is None:
            self.energy.fill(self.initial_energy)
        else:
            updated_power = np.asarray(updated_power, dtype=float)
            if updated_power.shape != (self.SENSOR_NUMBER,):
                raise ValueError(
                    "updated_power must have one value per sensor"
                )
            # update_position 與 updated_power 都以重排前的同一組 id
            # 傳入；sensor 座標改成 ring order 後，能量也必須套用完全相同
            # 的 permutation，才能讓 priority 的距離與剩餘能量對應同一顆。
            self.energy = updated_power[self.sensor_sort_order].copy()
        self.mobile_id = []

        # 先建立 link_capacity/amp_cost，再準備 coverage 候選與
        # routing priority 快取。
        self.cover_candidates = None
        self.build_capacities()
        self.prepare_coding_cache()
        self.mobility_service = MobilityService(self)

        # 部分 SE/SETS/ALNS 演算法選擇或修復 sensor 時使用的最低能量門檻。
        # 這不是全域死亡判定；真正的能量失敗條件是執行後 J - cost < 0。
        # 目前 GI-GOMEA 系列不使用 liveJ。
        self.liveJ = 0.5

    # ------------------------------------------------------------------
    # 舊欄位名稱相容層。核心服務使用上方的完整名稱；尚未遷移的論文
    # 演算法仍可透過這些 property 讀寫同一份資料，不會產生副本。
    # ------------------------------------------------------------------
    @property
    def J(self):
        return self.energy

    @J.setter
    def J(self, value):
        self.energy = value

    @property
    def FULL(self):
        return self.initial_energy

    @FULL.setter
    def FULL(self, value):
        self.initial_energy = value

    @property
    def pack(self):
        return self.generated_load

    @pack.setter
    def pack(self, value):
        self.generated_load = value

    @property
    def Te(self):
        return self.circuit_cost

    @Te.setter
    def Te(self, value):
        self.circuit_cost = value

    @property
    def amp(self):
        return self.amp_factors

    @amp.setter
    def amp(self, value):
        self.amp_factors = value

    @property
    def sch_cof(self):
        return self.sensing_factor

    @sch_cof.setter
    def sch_cof(self, value):
        self.sensing_factor = value

    @property
    def COST_RANGE(self):
        return self.level_costs

    @COST_RANGE.setter
    def COST_RANGE(self, value):
        self.level_costs = value

    @property
    def SENSING_COST(self):
        return self.sensing_costs

    @SENSING_COST.setter
    def SENSING_COST(self, value):
        self.sensing_costs = value

    @property
    def dis(self):
        return self.distances

    @dis.setter
    def dis(self, value):
        self.distances = value

    @property
    def inf(self):
        return self.blocked_distance

    @inf.setter
    def inf(self, value):
        self.blocked_distance = value

    @property
    def d3(self):
        return self.amp_cost

    @d3.setter
    def d3(self, value):
        self.amp_cost = value

    @property
    def cap(self):
        return self.link_capacity

    @cap.setter
    def cap(self, value):
        self.link_capacity = value

    @property
    def connect(self):
        return self.coverage_table

    @connect.setter
    def connect(self, value):
        self.coverage_table = value

    @property
    def sensor_weight(self):
        return self.target_sensor_mask

    @sensor_weight.setter
    def sensor_weight(self, value):
        self.target_sensor_mask = value

    @property
    def CCS(self):
        return self.cover_candidates

    @CCS.setter
    def CCS(self, value):
        self.cover_candidates = value

    @property
    def long(self):
        return self.inverse_bs_distance

    @long.setter
    def long(self, value):
        self.inverse_bs_distance = value

    @property
    def bs_rank(self):
        return self.proximity_score

    @bs_rank.setter
    def bs_rank(self, value):
        self.proximity_score = value

    @property
    def red_rank(self):
        return self.energy_score

    @red_rank.setter
    def red_rank(self, value):
        self.energy_score = value

    @property
    def coverage(self):
        return self.coverage_service

    @coverage.setter
    def coverage(self, value):
        self.coverage_service = value

    @property
    def routing(self):
        return self.routing_service

    @routing.setter
    def routing(self, value):
        self.routing_service = value

    @property
    def fitness(self):
        return self.fitness_service

    @fitness.setter
    def fitness(self, value):
        self.fitness_service = value

    @property
    def mobility(self):
        return self.mobility_service

    @mobility.setter
    def mobility(self, value):
        self.mobility_service = value

    def configure_sensing_ranges(self):
        """Build the sensor-specific radius options addressed by ``levels``.

        ``discrete`` repeats the configured global levels for every sensor.
        ``bucketed`` keeps at most one coverage event per interval between
        consecutive discrete levels: the farthest newly reachable target in
        that interval.  Empty intervals create no option.  ``exact`` keeps
        every target-distance event within the maximum range.
        """
        if self.sensing_mode == "discrete":
            options = [
                self.DISCRETE_LEVEL_RANGE.copy()
                for _ in range(self.SENSOR_NUMBER)
            ]
        else:
            distances = self.G.CalDistance(self.target, self.sensor)
            options = []
            for sensor_id in range(self.SENSOR_NUMBER):
                sensor_distances = np.asarray(
                    distances[:, sensor_id], dtype=float
                )
                if self.sensing_mode == "exact":
                    valid = sensor_distances[
                        sensor_distances
                        <= self.MAX_SENSING_RADIUS + 1e-12
                    ]
                    values = np.unique(np.concatenate(([0.0], valid)))
                    options.append(np.sort(values))
                    continue

                # Compact interval-based encoding.  For (lower, upper], add
                # one option only when that interval contains newly reachable
                # targets, and use the farthest such target as its radius.
                values = [0.0]
                for lower, upper in zip(
                    self.DISCRETE_LEVEL_RANGE[:-1],
                    self.DISCRETE_LEVEL_RANGE[1:],
                ):
                    interval_distances = sensor_distances[
                        (sensor_distances > lower + 1e-12)
                        & (sensor_distances <= upper + 1e-12)
                    ]
                    if len(interval_distances) > 0:
                        values.append(float(np.max(interval_distances)))
                options.append(np.asarray(values, dtype=float))

        self.radius_options = tuple(
            np.asarray(row, dtype=float) for row in options
        )
        self.radius_option_counts = np.asarray(
            [len(row) for row in self.radius_options], dtype=int
        )
        self.LEVEL = int(max(self.radius_option_counts, default=1))
        self.radius_table = np.zeros(
            (self.SENSOR_NUMBER, self.LEVEL), dtype=float
        )
        for sensor_id, row in enumerate(self.radius_options):
            self.radius_table[sensor_id, : len(row)] = row
            if len(row) < self.LEVEL:
                self.radius_table[sensor_id, len(row) :] = row[-1]

        self.target_sensor_distance = self.G.CalDistance(
            self.target, self.sensor
        )
        if hasattr(self, "sensing_factor"):
            self.sensing_costs = self.radius_table**2 * self.sensing_factor

    def sensing_option_count(self, sensor_id):
        return int(self.radius_option_counts[int(sensor_id)])

    def sensing_radius(self, sensor_id, option_id):
        sensor_id = int(sensor_id)
        option_id = int(option_id)
        if not 0 <= sensor_id < self.SENSOR_NUMBER:
            raise IndexError("sensor_id is outside the sensing option table")
        if not 0 <= option_id < self.sensing_option_count(sensor_id):
            raise IndexError(
                "sensing option is outside this sensor's configured domain"
            )
        return float(self.radius_table[sensor_id, option_id])

    def sensing_cost(self, sensor_id, option_id):
        sensor_id = int(sensor_id)
        option_id = int(option_id)
        self.sensing_radius(sensor_id, option_id)
        return float(self.sensing_costs[sensor_id, option_id])

    def validate_schedule_bounds(self, schedule):
        """確認每顆 sensor 的 level 落在它自己的 option 範圍內。

        這是整個評估流程唯一的邊界檢查點：解碼結束時由
        ``resolve_state_radius`` 呼叫一次，之後的核心都假設排程已合法。
        核心以 ``njit`` 編譯且不做邊界檢查，越界的 level 會讀到界外記憶體
        並靜默回傳看似合理的錯誤覆蓋值，因此這道檢查不能省略。
        """
        schedule = np.asarray(schedule, dtype=np.int64)
        if NUMBA_AVAILABLE:
            offender = int(
                first_out_of_range_per_sensor_kernel(
                    schedule,
                    np.asarray(self.radius_option_counts, dtype=np.int64),
                )
            )
        elif np.any(schedule < 0) or np.any(
            schedule >= self.radius_option_counts
        ):
            offender = int(
                np.flatnonzero(
                    (schedule < 0)
                    | (schedule >= self.radius_option_counts)
                )[0]
            )
        else:
            offender = -1
        if offender >= 0:
            raise ValueError(
                "schedule contains an invalid sensing option: "
                f"sensor {offender} has level {int(schedule[offender])} "
                f"but only {int(self.radius_option_counts[offender])} options"
            )
        return schedule

    def resolve_radius(self, schedule):
        """Translate a sensor option-index vector into physical radii."""
        schedule = np.asarray(schedule, dtype=int)
        expected_shape = (self.SENSOR_NUMBER,)
        if schedule.shape != expected_shape:
            raise ValueError(
                f"schedule must have shape {expected_shape}, got {schedule.shape}"
            )
        self.validate_schedule_bounds(schedule)
        return self.radius_table[self.SENSOR_ID, schedule].copy()

    def resolve_state_radius(self, state):
        """Synchronize and return a decoded state's physical radii."""
        radius = self.resolve_radius(state.levels)
        state.sensing_radii = radius
        # Retained only for old callers.  Evaluation no longer branches on it.
        state.use_continuous_radius = self.sensing_mode != "discrete"
        # Track the schedule and table used for this conversion. Direct level
        # edits and a different Problem must not reuse obsolete radii.
        state._radius_levels = np.asarray(state.levels).copy()
        state._radius_table_id = id(self.radius_table)
        # 蓋章：這個 levels 陣列物件已通過邊界檢查，評估路徑不必重驗。
        # 保存物件本身而不是 id()，避免陣列被釋放後位址重用造成誤判。
        # 解碼後仍會就地修改 levels 的只有路由失敗的關閉動作（一律設 0，
        # 恆為合法）；任何重新指派 levels 或 State.copy() 都會失去章，
        # 自動退回重新驗證。
        state._validated_levels = state.levels
        return radius

    def state_radius(self, state):
        """Read decoded radii; resolve only if the schedule or table changed."""
        radius = getattr(state, "sensing_radii", None)
        if (
            radius is not None
            and getattr(state, "_radius_table_id", None) == id(self.radius_table)
            and np.array_equal(getattr(state, "_radius_levels", None), state.levels)
        ):
            return radius
        return self.resolve_state_radius(state)

    def level_for_radius(self, sensor_id, radius):
        """Return the smallest sensor-specific option that covers ``radius``."""
        radius = float(radius)
        if radius <= 0:
            return 0
        sensor_id = int(sensor_id)
        options = self.radius_options[sensor_id]
        level = int(np.searchsorted(options, radius, side="left"))
        return min(max(1, level), len(options) - 1)

    def maximum_radius(self, sensor_id=None):
        if sensor_id is None:
            return float(np.max(self.radius_table[:, -1]))
        return float(self.radius_options[int(sensor_id)][-1])

    # ------------------------------------------------------------------
    # 主要 API：新程式應優先使用這一組易讀名稱。
    # ------------------------------------------------------------------
    def calculate_coverage_matrix(self):
        """建立 coverage 快取。

        Returns:
            coverage_table: shape=(T, S, LEVEL) 的 0/1 覆蓋矩陣。
            target_sensor_mask: shape=(T, S)，表示最大半徑下的 target-sensor
                鄰接關係，fitness 的 target energy 聚合會使用它。
        """
        return self.coverage_service.calculate_coverage_matrix()

    def count_covered_targets(self, schedule):
        """計算每個 target 被多少個開啟 sensor 覆蓋。

        Args:
            schedule: shape=(S,) 的 level 索引陣列，通常就是 state.levels。
        """
        return self.coverage_service.count_covered_targets(schedule)

    def find_uncovered_targets(self, schedule):
        """回傳完全沒有被覆蓋的 target id 陣列。"""
        return self.coverage_service.find_uncovered_targets(schedule)

    def build_cover_candidate_sets(self):
        """建立每個 target 可選的 ``[sensor_id, level]`` 清單。"""
        self.cover_candidates = self.coverage_service.build_cover_candidate_sets()
        return self.cover_candidates

    def compute_priorities(self):
        """依 sensor 到 BS 的距離與目前剩餘能量更新 routing priority。"""
        return self.routing_service.compute_priorities()

    def build_capacities(self, schedule=None):
        """依目前剩餘能量建立連線容量與放大器成本表。"""
        return self.routing_service.build_capacities(schedule)

    def update_capacities(self, cost, schedule=None):
        """把已消耗能量反映到現有連線容量表。"""
        return self.routing_service.update_capacities(cost, schedule)

    def update_capacity(self, state, routed_nodes):
        """選定一條路由後，更新該 state 的路徑剩餘容量。"""
        return self.routing_service.update_capacity(state, routed_nodes)

    def trace_path(self, sensor_id, next_hops, paths, search):
        """追蹤單一 sensor 到 BS 的完整路徑，並偵測循環路由。"""
        return self.routing_service.trace_path(
            sensor_id, next_hops, paths, search
        )

    def trace_paths(self, state):
        """追蹤 state 中所有開啟 sensor 的路徑。"""
        return self.routing_service.trace_paths(state)

    def calculate_loads(self, paths, state=None):
        """根據所有路徑計算每個 sensor 需產生與轉送的封包量。"""
        return self.routing_service.calculate_loads(paths, state)

    def find_disconnected(self, state, paths=None, sensing_radii=None):
        """回傳路徑無法抵達 BS 的開啟 sensor id。"""
        return self.routing_service.find_disconnected(
            state,
            paths,
            sensing_radii=sensing_radii,
        )

    # 舊方法名的相容層；新程式應使用上方的簡短名稱。
    def calculate_routing_priority(self):
        return self.compute_priorities()

    def precompute_route_capacity(self, cost=None, schedule=None, first=False):
        """舊版「建立／更新容量」合併入口。

        Args:
            cost: lifetime 已消耗的每個 sensor 成本；更新 cap 時使用。
            schedule: 可選的 sensor level，用來略過未開啟 sensor。
            first: True 表示首次建立 cap 與單次傳輸成本 d3；False 表示
                以 cost 更新既有 cap。
        """
        if first:
            return self.build_capacities(schedule)
        return self.update_capacities(cost, schedule)

    def update_state_route_capacity(self, state, connected_nodes):
        return self.update_capacity(state, connected_nodes)

    def trace_route(self, sensor_id, route_schedule, path, search):
        return self.trace_path(
            sensor_id, route_schedule, path, search
        )

    def trace_all_routes(self, state):
        return self.trace_paths(state)

    def calculate_transmission_load(self, path, state=None):
        return self.calculate_loads(path, state)

    def find_disconnected_sensors(self, state, path=None):
        return self.find_disconnected(state, path)

    def calculate_scheduling_cost(self, state, sensing_radii=None):
        """計算各 sensor 的感測範圍能量成本。"""
        return self.energy_service.calculate_scheduling_cost(
            state,
            sensing_radii=sensing_radii,
        )

    def calculate_routing_cost(self, state, test=False, sensing_radii=None):
        """計算各 sensor 的資料傳送與轉送能量成本。"""
        return self.energy_service.calculate_routing_cost(
            state,
            test,
            sensing_radii=sensing_radii,
        )

    def calculate_total_cost(self, state, test=False, sensing_radii=None):
        """回傳各 sensor 的總成本：感測成本 + routing 成本。"""
        return self.energy_service.calculate_total_cost(
            state,
            test,
            sensing_radii=sensing_radii,
        )

    def evaluate_state(self, state, debug=False):
        """計算 state 的三個原版加權 fitness 分量。

        預設 ``FitnessService`` 依序回傳 total target remaining energy、
        minimum target remaining energy 與 coverage。實驗用的 V5 evaluator
        不會由 Problem 自動啟用。
        """
        return self.fitness_service.evaluate_state(state, debug)

    def calculate_remaining_energy(self, cost):
        """以目前 energy 扣除 cost，回傳執行一次後的剩餘能量。"""
        return self.energy_service.calculate_remaining_energy(cost)

    def calculate_target_remaining_energy(self, remaining_energy):
        """依 target_sensor_mask 聚合每個 target 可使用的剩餘能量。"""
        return self.energy_service.calculate_target_remaining_energy(
            remaining_energy
        )

    def find_energy_failed_sensors(self, state, cost):
        """回傳扣除 cost 後能量小於 0 的 sensor id。

        ``state`` 是舊介面保留參數；目前判斷只需要 cost 與 energy。
        """
        return self.energy_service.find_energy_failed_sensors(cost)

    def prepare_coding_cache(self, cost=None):
        """準備 State 解碼會使用的 coverage 與 routing 快取。

        如果提供 cost，會先把 lifetime 已使用的能量反映到容量表，再重建
        cover_candidates 與依目前 energy 計算的 routing priority。
        """
        if cost is not None:
            self.update_capacities(cost)
        self.build_cover_candidate_sets()
        self.compute_priorities()

    def validate_problem(self, min_coverage=None):
        """確認最大感測範圍下所有 targets 都達到指定覆蓋要求。

        未指定時使用新資料的三重覆蓋規範。既有資料與移動後位置只需
        確認基本 coverage，避免把歷史 A2 或運行中拓樸當成新生成資料。
        """
        if min_coverage is None:
            min_coverage = self.DATA_GENERATION_MIN_COVERAGE
        min_coverage = int(min_coverage)
        if min_coverage <= 0:
            raise ValueError("min_coverage must be positive")
        schedule = np.zeros(self.SENSOR_NUMBER).astype(int)
        schedule.fill(self.LEVEL - 1)
        self.coverage_table, self.target_sensor_mask = (
            self.calculate_coverage_matrix()
        )

        if len(self.G.CheckFailTargetInKSensor(
            self, schedule, k=min_coverage
        )) == 0:
            print("Sucess")
            return True
        return False

    def sort_nodes_by_bs_distance(self):
        """建立距離索引，並以 BS-centered ring 座標排列 sensor。

        target 維持距離排序。當 ``sensor_ring_order`` 為 True(預設)時，
        sensor 的 chromosome index 排成「由內到外的 ring、同一 ring 由
        正上方順時針掃到正下方」；ring 寬度為 ``sensor_ring_width``。這讓
        相鄰基因具有局部空間意義。當 ``sensor_ring_order`` 為 False 時，
        改為單純依離 base station 的距離排序，不分環也不看角度。兩種模式
        都保留 ``sensor_ids_by_bs_distance`` 給 routing/priority 與舊演算法
        查詢。
        """
        target_distance = self.G.CalDistance(self.target, np.array([self.BS]))[:, 0]
        sensor_distance = self.G.CalDistance(self.sensor, np.array([self.BS]))[:, 0]
        sort_target_ids = np.argsort(target_distance, kind="stable")

        old_sensor_ids = np.arange(self.SENSOR_NUMBER, dtype=int)
        if self.sensor_ring_order:
            # 以 BS 為原點：0 度在正上方，角度順時針增加。先分 ring，再依
            # 角度排序；最後以舊 id 打破完全相同座標的平手，確保可重現性。
            delta = np.asarray(self.sensor, dtype=float) - np.asarray(self.BS)
            sensor_angles = np.mod(
                np.arctan2(delta[:, 0], -delta[:, 1]), 2.0 * np.pi
            )
            sensor_rings = np.floor(
                sensor_distance / self.sensor_ring_width
            ).astype(int)
            sort_sensor_ids = np.lexsort(
                (old_sensor_ids, sensor_angles, sensor_rings)
            )
        else:
            # 純距離排序：不分環、不看角度，只用舊 id 打破平手。
            sensor_angles = np.zeros(self.SENSOR_NUMBER, dtype=float)
            sensor_rings = np.zeros(self.SENSOR_NUMBER, dtype=int)
            sort_sensor_ids = np.lexsort((old_sensor_ids, sensor_distance))
        self.sensor_sort_order = sort_sensor_ids
        self.target = self.target[sort_target_ids]
        self.sensor = self.sensor[sort_sensor_ids]

        # 原始資料 id 只用於追蹤；演算法一律使用目前的 chromosome id。
        self.sensor_original_ids = old_sensor_ids[sort_sensor_ids]
        sorted_distances = sensor_distance[sort_sensor_ids]
        sorted_rings = sensor_rings[sort_sensor_ids]
        sorted_angles = sensor_angles[sort_sensor_ids]
        self.sensor_ring_by_id = sorted_rings
        self.sensor_angle_by_id = sorted_angles
        self.sensor_ids_by_ring = np.arange(self.SENSOR_NUMBER, dtype=int)
        self.sensor_ids_by_bs_distance = np.argsort(
            sorted_distances, kind="stable"
        ).astype(int)
        self.sensor_distance_rank_by_id = np.empty(
            self.SENSOR_NUMBER, dtype=int
        )
        self.sensor_distance_rank_by_id[self.sensor_ids_by_bs_distance] = (
            np.arange(self.SENSOR_NUMBER, dtype=int)
        )
        if self.radius_table is not None:
            self.configure_sensing_ranges()
        self.coverage_table, self.target_sensor_mask = (
            self.calculate_coverage_matrix()
        )

    def create_map_data(self, FILE, update_position=None, Target_position=None):
        """建立、載入或重建 sensor/target 座標。

        FILE=None 時會隨機產生 sensor；有 FILE 時從 sensor.csv 載入。
        update_position 不為 None 時代表沿用移動後位置與既有 targets。
        每組資料都必須通過 validate_problem 才會被採用。
        """
        if update_position is None:
            self.target = self.G.CreateTestTargetSample(self)
            if FILE is None:
                for attempt in range(10000):
                    self.sensor = self.G.CreateTestSensor(self)
                    if self.validate_problem():
                        self.generation_attempts = attempt + 1
                        self.sort_nodes_by_bs_distance()
                        return True
            else:
                sensor_data = pd.read_csv(FILE + "sensor.csv")
                self.sensor = sensor_data.values[:, 1:3]
                if self.validate_problem(min_coverage=1):
                    self.generation_attempts = 0
                    self.sort_nodes_by_bs_distance()
                    return True
            print("Fail")
        else:
            self.sensor = update_position
            self.target = Target_position
            if self.validate_problem(min_coverage=1):
                self.sort_nodes_by_bs_distance()
                return True
        return False

    def is_state_alive(self, state, cost, test=False):
        """判斷 state 是否能繼續執行一個 lifetime slot。

        必須同時滿足：沒有 sensor 能量失敗、所有開啟 sensor 都連到 BS、
        所有 targets 都被覆蓋。
        """
        if (
            len(self.find_energy_failed_sensors(state, cost)) == 0
            and len(self.find_disconnected_sensors(state)) == 0
            and len(self.find_uncovered_targets(state)) == 0
        ):
            return True

        if test:
            print("energy failed sensors", self.find_energy_failed_sensors(state, cost))
            print("disconnected sensors", self.find_disconnected_sensors(state))
            print("uncovered targets", self.find_uncovered_targets(state))
        return False

    # ------------------------------------------------------------------
    # 舊名稱相容層：既有論文演算法仍在呼叫這些名稱。
    # 新程式應優先使用上方 snake_case API；兩者目前行為相同。
    # ------------------------------------------------------------------
    def CalAllDistance(self):
        """舊名：calculate_coverage_matrix。"""
        return self.calculate_coverage_matrix()

    def CheckTargetInSensor(self, schedule):
        """舊名：count_covered_targets。"""
        return self.count_covered_targets(schedule)

    def CheckFailTargetInSensor(self, schedule):
        """舊名：find_uncovered_targets。"""
        return self.find_uncovered_targets(schedule)

    def Find(self, sensor_id, route_schedule, path, search):
        """舊名：trace_route。"""
        return self.trace_route(sensor_id, route_schedule, path, search)

    def FindForward(self, state):
        """舊名：trace_all_routes。"""
        return self.trace_all_routes(state)

    def Findtb(self, path, s=None):
        """舊名：calculate_transmission_load。"""
        return self.calculate_transmission_load(path, s)

    def RoutingCost(self, state, test=False):
        """舊名：calculate_routing_cost。"""
        return self.calculate_routing_cost(state, test)

    def SchedulingCost(self, state):
        """舊名：calculate_scheduling_cost。"""
        return self.calculate_scheduling_cost(state)

    def CalCost(self, state, test=False):
        """舊名：calculate_total_cost。"""
        return self.calculate_total_cost(state, test)

    def EvaluateState(self, state, debug=False):
        """舊名：evaluate_state。"""
        return self.evaluate_state(state, debug)

    def CalRedCap(self, cost):
        """舊名：calculate_remaining_energy。"""
        return self.calculate_remaining_energy(cost)

    def CheckConnect(self, state, path=None):
        """舊名：find_disconnected_sensors。"""
        return self.find_disconnected_sensors(state, path)

    def FindLossRedCap(self, state, cost):
        """舊名：find_energy_failed_sensors。"""
        return self.find_energy_failed_sensors(state, cost)

    def LifeCheck(self, state, cost, test=False):
        """舊名：is_state_alive。"""
        return self.is_state_alive(state, cost, test)

    def ProblemCheck(self):
        """舊名：validate_problem。"""
        return self.validate_problem()

    def SortData(self):
        """舊名：sort_nodes_by_bs_distance。"""
        return self.sort_nodes_by_bs_distance()

    def CreateTestData(self, FILE, update_position=None, Target_position=None):
        """舊名：create_map_data。"""
        return self.create_map_data(FILE, update_position, Target_position)

    def PreCal(self, cost=None):
        """舊名：prepare_coding_cache。"""
        return self.prepare_coding_cache(cost)

    def CalRank(self):
        """舊名：calculate_routing_priority。"""
        return self.calculate_routing_priority()

    def CalRouPath(self, cost=None, sch=None, first=False):
        """舊名：precompute_route_capacity。"""
        return self.precompute_route_capacity(cost, sch, first)

    def CalCap(self, state, connected_nodes):
        """舊名：update_state_route_capacity。"""
        return self.update_state_route_capacity(state, connected_nodes)

    def CalTargetRed(self, remaining_energy):
        """舊名：calculate_target_remaining_energy。"""
        return self.calculate_target_remaining_energy(remaining_energy)

    def CalCCS(self):
        """舊名：build_cover_candidate_sets。"""
        return self.build_cover_candidate_sets()
