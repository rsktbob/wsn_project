import numpy as np

from Problem.Cal import pj
from Problem.services.evaluation_kernels import (
    NUMBA_AVAILABLE,
    calculate_routing_cost_kernel,
    target_remaining_energy_kernel,
    warm_evaluation_kernels,
)


class EnergyService:
    """Energy, cost, and remaining-power calculations."""

    def __init__(self, problem):
        self.problem = problem
        self._mask_source = None
        self._mask_f64 = None
        warm_evaluation_kernels()

    def _contiguous_mask(self):
        """回傳 float64 連續版的 target_sensor_mask，並快取。

        ``problem.target_sensor_mask`` 是 int 陣列，而且在地圖重建時
        （Problem.py:754、:822）會被重新指派。這裡用物件身分判斷是否
        換過，換過才重建，避免每次評估都複製一份 (T, S)。
        """
        mask = self.problem.target_sensor_mask
        if self._mask_source is not mask:
            self._mask_f64 = np.ascontiguousarray(mask, dtype=np.float64)
            self._mask_source = mask
        return self._mask_f64

    def calculate_scheduling_cost(self, state, sensing_radii=None):
        """Calculate sensing-range scheduling cost."""
        problem = self.problem
        if sensing_radii is None:
            sensing_radii = problem.state_radius(state)
        return (
            np.maximum(sensing_radii, 0.0) ** 2
            * problem.sensing_factor
        )

    def calculate_routing_cost(self, state, test=False, sensing_radii=None):
        """Calculate data forwarding cost."""
        problem = self.problem
        cost = np.zeros(problem.SENSOR_NUMBER)
        if sensing_radii is None:
            sensing_radii = problem.state_radius(state)
        if state.tx_load is not None:
            tx_load = state.tx_load.copy()
        else:
            paths = problem.routing_service.trace_paths(state)
            tx_load = problem.routing_service.calculate_loads(paths, state)
        routing_sensors = np.where(np.asarray(tx_load) > 0)[0]

        if NUMBA_AVAILABLE and not test:
            return calculate_routing_cost_kernel(
                np.asarray(sensing_radii, dtype=float),
                np.asarray(state.next_hops, dtype=np.int64),
                np.asarray(tx_load, dtype=float),
                np.asarray(problem.distances, dtype=float),
                np.asarray(problem.generated_load, dtype=float),
                float(problem.amp_factors[0]),
                float(problem.amp_factors[1]),
                float(problem.circuit_cost),
                float(pj),
                float(problem.d0),
            )

        for sensor_id in routing_sensors:
            next_hop = int(state.next_hops[sensor_id])
            # v3 保留 encoding 指定但無法接入路由樹的 sensor。這類候選
            # 由 disconnected constraint 判為不可行；此處不可用 -1 當成
            # 最後一個 device 索引，否則會產生虛假的傳輸耗能。
            if not 0 <= next_hop < problem.DEVICE_NUMBER:
                continue
            distance = problem.distances[sensor_id][next_hop]
            if distance <= 87:
                distance_cost = (
                    problem.amp_factors[0] * pj * distance * distance
                )
            else:
                distance_cost = (
                    problem.amp_factors[1]
                    * pj
                    * distance
                    * distance
                    * distance
                    * distance
                )

            own_load = (
                problem.generated_load[sensor_id]
                if sensing_radii[sensor_id] > 0
                else 0
            )
            cost[sensor_id] += (
                tx_load[sensor_id]
                * (distance_cost + problem.circuit_cost * 2)
                - problem.circuit_cost
                * own_load
            )
            if test:
                print(
                    sensor_id,
                    "route distance",
                    round(
                        problem.distances[sensor_id][
                            state.next_hops[sensor_id]
                        ],
                        2,
                    ),
                    "distance cost",
                    distance_cost,
                    "forward load",
                    tx_load[sensor_id],
                    "routing cost",
                    cost[sensor_id],
                )

        return cost

    def calculate_total_cost(self, state, test=False, sensing_radii=None):
        """Calculate total scheduling and routing cost."""
        if sensing_radii is None:
            sensing_radii = self.problem.state_radius(state)
        scheduling_cost = self.calculate_scheduling_cost(
            state,
            sensing_radii=sensing_radii,
        )
        routing_cost = self.calculate_routing_cost(
            state,
            test,
            sensing_radii=sensing_radii,
        )
        return scheduling_cost + routing_cost

    def calculate_remaining_energy(self, cost):
        """Calculate remaining energy after cost is paid."""
        problem = self.problem
        return problem.energy - cost

    def calculate_target_remaining_energy(self, remaining_energy):
        """Aggregate remaining energy around each target."""
        problem = self.problem
        if not NUMBA_AVAILABLE:
            return np.sum(
                problem.target_sensor_mask * remaining_energy, axis=1
            )
        return target_remaining_energy_kernel(
            self._contiguous_mask(),
            np.ascontiguousarray(remaining_energy, dtype=np.float64),
        )

    def find_energy_failed_sensors(self, cost):
        """Return sensors whose remaining energy is negative."""
        return np.where(self.calculate_remaining_energy(cost) < 0)[0]
