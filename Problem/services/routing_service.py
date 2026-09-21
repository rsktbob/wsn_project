import time
from dataclasses import dataclass

import numpy as np

from Problem.Cal import pj
from Problem.services.evaluation_kernels import (
    find_disconnected_kernel,
    NUMBA_AVAILABLE,
    build_routes_kernel,
)


@dataclass(frozen=True, eq=False)
class RoutingResult:
    """Summary of one routing-tree construction.

    ``unavailable_ids`` had no positive-capacity next hop.
    ``capacity_failed_ids`` had candidates, but every exact bottleneck margin
    was negative.  Keeping the two cases separate preserves the
    historical SensorEncoding repair behaviour without coupling this service
    to any chromosome type.

    The four id fields stay as ``int64`` arrays straight from the routing
    kernel.  Converting them to tuples of Python ints cost ~77k generator
    calls per 3k evaluations while every consumer only iterates or takes
    ``len``.  ``eq=False`` keeps the dataclass from building an ``__eq__``
    that would compare arrays elementwise and then fail on ``bool()``.
    """

    connected_ids: np.ndarray
    rejected_ids: np.ndarray
    unavailable_ids: np.ndarray
    capacity_failed_ids: np.ndarray

    # Compatibility aliases for callers that still use the former verbose
    # result-field names.  They reference the same immutable tuple values.
    @property
    def connected_sensor_ids(self):
        return self.connected_ids

    @property
    def rejected_sensor_ids(self):
        return self.rejected_ids

    @property
    def unavailable_sensor_ids(self):
        return self.unavailable_ids

    @property
    def insufficient_capacity_sensor_ids(self):
        return self.capacity_failed_ids


class RoutingService:
    """Routing priority, forwarding path, and route-capacity calculations."""

    def __init__(self, problem):
        self.problem = problem

    @staticmethod
    def _parse_routing_order(routing_order):
        """同時支援新的 sensor-id 陣列與舊的 [sensor, level] 介面。"""
        routing_array = np.asarray(routing_order, dtype=np.int64)
        if routing_array.size == 0:
            return np.empty(0, dtype=np.int64), None
        if routing_array.ndim == 1:
            return routing_array.reshape(-1), None
        if routing_array.ndim == 2 and routing_array.shape[1] == 2:
            return routing_array[:, 0], routing_array[:, 1]
        raise ValueError(
            "routing_order must contain sensor ids or [sensor, level] pairs"
        )

    def compute_priorities(self):
        """Calculate distance-to-BS and remaining-energy routing priorities."""
        problem = self.problem
        distance_to_bs = problem.G.CalDistance(problem.sensor, np.array([problem.BS]))[
            :, 0
        ]
        problem.inverse_bs_distance = np.divide(
            1.0,
            distance_to_bs,
            out=np.full_like(distance_to_bs, np.inf, dtype=float),
            where=distance_to_bs != 0,
        )
        finite_distance_rank = problem.inverse_bs_distance[
            np.isfinite(problem.inverse_bs_distance)
        ]
        if np.isinf(problem.inverse_bs_distance).any():
            replacement = (
                np.max(finite_distance_rank) if len(finite_distance_rank) > 0 else 1.0
            )
            problem.inverse_bs_distance[
                ~np.isfinite(problem.inverse_bs_distance)
            ] = replacement

        max_distance_rank = max(problem.inverse_bs_distance)
        if max_distance_rank == 0:
            problem.proximity_score = np.zeros_like(
                problem.inverse_bs_distance
            )
        else:
            problem.proximity_score = (
                problem.inverse_bs_distance / max_distance_rank
            )

        max_energy = max(problem.energy)
        if max_energy == 0:
            problem.energy_score = np.zeros_like(problem.energy)
        else:
            problem.energy_score = problem.energy / max_energy

    def _choose_parent(
        self,
        state,
        route_selector,
        sensor_id,
        level,
        routed_nodes,
        parent_scores,
    ):
        """Select a candidate next-hop index, with greedy fallback."""
        if route_selector is None:
            return int(np.argmax(parent_scores))

        problem = self.problem
        callback = getattr(route_selector, "select_next_hop", None)
        if callback is None:
            callback = route_selector.SelectNextHop
        parent_id = callback(
            problem,
            state,
            sensor_id,
            level,
            routed_nodes,
            parent_scores,
        )

        if parent_id is None:
            return int(np.argmax(parent_scores))
        matches = np.where(routed_nodes == int(parent_id))[0]
        if len(matches) == 0:
            return int(np.argmax(parent_scores))
        return int(matches[0])

    def _notify_selector(
        self,
        state,
        route_selector,
        sensor_id,
        level,
        parent_id,
        routed_nodes,
        parent_scores,
        parent_score,
        success,
    ):
        """Let an optional adaptive selector learn from a routing decision."""
        if route_selector is None:
            return
        callback = getattr(route_selector, "learn_from_route", None)
        if callback is None:
            callback = getattr(route_selector, "LearnFromRoute", None)
        if callback is not None:
            callback(
                self.problem,
                state,
                sensor_id,
                level,
                parent_id,
                routed_nodes,
                parent_scores,
                parent_score,
                success,
            )

    def build_routes(
        self,
        state,
        routing_order,
        *,
        route_selector=None,
        test=False,
        disable_failed=True,
    ):
        """Build a forwarding tree by maximizing remaining bottleneck capacity."""
        problem = self.problem
        routing_ids, routing_levels = self._parse_routing_order(
            routing_order
        )
        if NUMBA_AVAILABLE and route_selector is None and not test:
            # State.empty 已配置好這些陣列；核心直接寫入，
            # 避免每個候選解另外建立一組相同尺寸的陣列。
            state.levels = np.ascontiguousarray(
                state.levels,
                dtype=np.int64,
            )
            state.next_hops = np.ascontiguousarray(
                state.next_hops,
                dtype=np.int64,
            )
            state.tx_load = np.ascontiguousarray(
                state.tx_load,
                dtype=np.int64,
            )
            state.remaining_capacity = np.ascontiguousarray(
                state.remaining_capacity,
                dtype=np.float64,
            )
            (
                connected_ids,
                rejected_ids,
                unavailable_ids,
                capacity_failed_ids,
            ) = build_routes_kernel(
                state.levels,
                routing_ids,
                np.asarray(problem.link_capacity, dtype=np.float64),
                np.asarray(problem.generated_load, dtype=np.int64),
                int(problem.SENSOR_NUMBER),
                int(problem.DEVICE_NUMBER),
                int(problem.BSID),
                bool(disable_failed),
                state.next_hops,
                state.tx_load,
                state.remaining_capacity,
            )
            # path 是 Python dict，數值核心完成後才由 parent chain 還原；
            # connected_ids 維持建立順序，故 parent 一定已經存在。
            state.paths = {int(problem.BSID): [int(problem.BSID)]}
            for sensor_id in connected_ids:
                sensor_id = int(sensor_id)
                parent_id = int(state.next_hops[sensor_id])
                state.paths[sensor_id] = [sensor_id] + state.paths[parent_id]
            # 核心已經回傳 int64 陣列，直接沿用，不再逐一轉成 Python int。
            return RoutingResult(
                connected_ids=connected_ids,
                rejected_ids=rejected_ids,
                unavailable_ids=unavailable_ids,
                capacity_failed_ids=capacity_failed_ids,
            )

        state.remaining_capacity[problem.BSID] = np.inf
        state.paths = {problem.BSID: [problem.BSID]}
        routed_nodes = np.array([problem.BSID], dtype=int)
        rejected_ids = []
        unavailable_ids = []
        capacity_failed_ids = []
        start_time = time.time() if test else None

        if routing_levels is None:
            routing_levels = np.asarray(
                state.levels[routing_ids],
                dtype=np.int64,
            )
        for sensor_id, level in zip(routing_ids, routing_levels):
            sensor_id = int(sensor_id)
            level = int(level)
            required_load = problem.generated_load[sensor_id]
            path_caps = state.remaining_capacity[routed_nodes]

            link_caps = problem.link_capacity[
                sensor_id, level, routed_nodes
            ]
            if np.max(link_caps) <= 0:
                if disable_failed:
                    state.levels[sensor_id] = 0
                rejected_ids.append(sensor_id)
                unavailable_ids.append(sensor_id)
                continue

            # A route is feasible only when both the source-to-parent link and
            # the parent's complete path can carry this sensor's packets.
            # The score is the exact bottleneck margin; no flooring or
            # cross-normalized tie breaker can turn an insufficient link into
            # an apparently feasible one.
            parent_scores = (
                np.minimum(link_caps, path_caps) - required_load
            )
            parent_index = self._choose_parent(
                state,
                route_selector,
                sensor_id,
                level,
                routed_nodes,
                parent_scores,
            )
            parent_id = routed_nodes[parent_index]
            parent_score = parent_scores[parent_index]

            if parent_score >= 0:
                state.next_hops[sensor_id] = parent_id
                routed_nodes = np.append(routed_nodes, sensor_id)
                state.paths[sensor_id] = [sensor_id] + state.paths[
                    state.next_hops[sensor_id]
                ]
                state.tx_load[
                    np.asarray(state.paths[sensor_id][:-1], dtype=int)
                ] += problem.generated_load[sensor_id]
                self.update_capacity(state, routed_nodes)
                self._notify_selector(
                    state,
                    route_selector,
                    sensor_id,
                    level,
                    parent_id,
                    routed_nodes,
                    parent_scores,
                    parent_score,
                    True,
                )
            else:
                if disable_failed:
                    state.levels[sensor_id] = 0
                rejected_ids.append(sensor_id)
                capacity_failed_ids.append(sensor_id)
                self._notify_selector(
                    state,
                    route_selector,
                    sensor_id,
                    level,
                    parent_id,
                    routed_nodes,
                    parent_scores,
                    parent_score,
                    False,
                )

        if test:
            print(0, 0, time.time() - start_time)
        # 非 numba 後備路徑：同樣統一回傳 int64 陣列。
        return RoutingResult(
            connected_ids=np.asarray(routed_nodes[1:], dtype=np.int64),
            rejected_ids=np.asarray(rejected_ids, dtype=np.int64),
            unavailable_ids=np.asarray(unavailable_ids, dtype=np.int64),
            capacity_failed_ids=np.asarray(
                capacity_failed_ids, dtype=np.int64
            ),
        )

    def build_capacities(self, schedule=None):
        """Build link-capacity and amplifier-cost tables."""
        problem = self.problem
        problem.link_capacity = np.zeros(
            [problem.SENSOR_NUMBER, problem.LEVEL, problem.DEVICE_NUMBER]
        )
        problem.amp_cost = np.zeros(
            [problem.SENSOR_NUMBER, problem.DEVICE_NUMBER]
        )
        # Preserve the historical first=True behaviour when a schedule is
        # supplied: callers currently build the full table with schedule=None.
        if schedule is not None:
            return

        for sensor_id in range(problem.SENSOR_NUMBER):
            for parent_id in range(problem.DEVICE_NUMBER):
                if (
                    problem.distances[sensor_id][parent_id]
                    >= problem.blocked_distance / 100
                ):
                    continue

                distance = problem.distances[sensor_id][parent_id]
                if distance <= 87:
                    problem.amp_cost[sensor_id, parent_id] = (
                        problem.amp_factors[0] * pj * distance * distance
                    )
                else:
                    problem.amp_cost[sensor_id, parent_id] = (
                        problem.amp_factors[1]
                        * pj
                        * distance
                        * distance
                        * distance
                        * distance
                    )

                for level in range(1, problem.LEVEL):
                    energy_budget = (
                        problem.energy[sensor_id]
                        - problem.sensing_costs[sensor_id, level]
                        + problem.generated_load[sensor_id]
                        * problem.circuit_cost
                    )
                    problem.link_capacity[sensor_id, level, parent_id] = (
                        energy_budget
                        / (
                            problem.amp_cost[sensor_id, parent_id]
                            + 2 * problem.circuit_cost
                        )
                    )

    def update_capacities(self, cost, schedule=None):
        """Apply consumed energy to the existing link-capacity table."""
        problem = self.problem
        for sensor_id in range(problem.SENSOR_NUMBER):
            if schedule is not None and schedule[sensor_id] == 0:
                continue
            for parent_id in range(problem.DEVICE_NUMBER):
                for level in range(1, problem.LEVEL):
                    problem.link_capacity[sensor_id, level, parent_id] -= (
                        cost[sensor_id]
                        / (
                            problem.amp_cost[sensor_id, parent_id]
                            + 2 * problem.circuit_cost
                        )
                    )

    def precompute_route_capacity(self, cost=None, schedule=None, first=False):
        """Compatibility wrapper for the former combined capacity method."""
        if first:
            return self.build_capacities(schedule)
        return self.update_capacities(cost, schedule)

    def update_capacity(self, state, routed_nodes):
        """Update the current state's route-capacity values."""
        problem = self.problem
        routed_nodes = routed_nodes[1:]
        state.remaining_capacity[routed_nodes] = (
            problem.link_capacity[
                routed_nodes,
                state.levels[routed_nodes],
                state.next_hops[routed_nodes],
            ]
            - state.tx_load[routed_nodes]
        )

        for index in range(len(routed_nodes)):
            sensor_id = routed_nodes[index]
            state.remaining_capacity[sensor_id] = min(
                state.remaining_capacity[state.next_hops[sensor_id]],
                state.remaining_capacity[sensor_id],
            )

    def trace_path(self, sensor_id, next_hops, paths, search):
        """Trace a sensor's forwarding path to the base station."""
        problem = self.problem

        if sensor_id in paths:
            return paths[sensor_id].copy()

        if (
            next_hops[sensor_id] == problem.SENSOR_NUMBER
            or next_hops[sensor_id] < 0
        ):
            paths[sensor_id] = [sensor_id, next_hops[sensor_id]]
            return paths[sensor_id].copy()

        if next_hops[sensor_id] in search:
            index = search.index(next_hops[sensor_id])
            paths[sensor_id] = [sensor_id] + search[index:] + [sensor_id]
            return paths[sensor_id].copy()

        search.append(sensor_id)
        traced_path = self.trace_path(
            next_hops[sensor_id], next_hops, paths, search
        )
        if sensor_id in traced_path:
            index = traced_path.index(sensor_id)
            traced_path = traced_path[: index + 1]

        paths[sensor_id] = [sensor_id] + traced_path
        return paths[sensor_id].copy()

    def trace_paths(self, state):
        """Trace forwarding paths for all active sensors."""
        problem = self.problem
        sensing_radii = problem.state_radius(state)
        paths = {}
        for sensor_id in range(problem.SENSOR_NUMBER):
            if sensing_radii[sensor_id] <= 0:
                continue
            paths[sensor_id] = self.trace_path(
                sensor_id, state.next_hops, paths, []
            )
        return paths

    def calculate_loads(self, paths, state=None):
        """Calculate how much traffic each sensor forwards."""
        problem = self.problem
        tx_load = np.zeros(problem.SENSOR_NUMBER).astype("int")
        sensing_radii = (
            problem.state_radius(state)
            if state is not None
            else None
        )

        for sensor_id in range(problem.SENSOR_NUMBER):
            if sensing_radii is not None and sensing_radii[sensor_id] <= 0:
                tx_load[sensor_id] = problem.generated_load[sensor_id]
                continue

            paths[sensor_id] = np.array(paths[sensor_id])
            tx_load[paths[sensor_id][:-1]] += problem.generated_load[
                paths[sensor_id][:-1]
            ]

        return tx_load

    def find_disconnected(self, state, paths=None, sensing_radii=None):
        """Return active sensors whose route does not end at the base station."""
        problem = self.problem
        if sensing_radii is None:
            sensing_radii = problem.state_radius(state)

        if NUMBA_AVAILABLE and paths is None:
            # 只讀 state.paths 每條路徑的結尾，不複製整份 dict 與其中的
            # list；核心再沿 next_hops 補走沒有快取的 sensor。呼叫端一律
            # 只用得到 id 本身，完整 path 是原實作的中間產物。
            sensor_count = int(problem.SENSOR_NUMBER)
            base_station_id = int(problem.BSID)
            seed_status = np.zeros(sensor_count, dtype=np.uint8)
            for sensor_id, path in state.paths.items():
                index = int(sensor_id)
                if 0 <= index < sensor_count and len(path):
                    seed_status[index] = (
                        1 if int(path[-1]) == base_station_id else 2
                    )
            return [
                int(sensor_id)
                for sensor_id in find_disconnected_kernel(
                    np.ascontiguousarray(state.next_hops, dtype=np.int64),
                    np.ascontiguousarray(sensing_radii, dtype=np.float64),
                    base_station_id,
                    seed_status,
                )
            ]

        if paths is None:
            # Decoding already built paths for connected sensors. Copy that
            # cache and trace only active sensors that routing rejected.
            paths = {
                int(sensor_id): list(path)
                for sensor_id, path in state.paths.items()
            }

        disconnected_ids = []
        for sensor_id in range(problem.SENSOR_NUMBER):
            if sensing_radii[sensor_id] <= 0:
                continue
            if sensor_id not in paths:
                paths[sensor_id] = self.trace_path(
                    sensor_id,
                    state.next_hops,
                    paths,
                    [],
                )
            if (
                paths[sensor_id][len(paths[sensor_id]) - 1]
                != problem.BSID
            ):
                disconnected_ids.append(sensor_id)
        return disconnected_ids

    # Compatibility methods for algorithms that have not migrated yet.
    def calculate_routing_priority(self):
        return self.compute_priorities()

    def update_state_route_capacity(self, state, connected_nodes):
        return self.update_capacity(state, connected_nodes)

    def trace_route(self, sensor_id, route_schedule, path, search):
        return self.trace_path(sensor_id, route_schedule, path, search)

    def trace_all_routes(self, state):
        return self.trace_paths(state)

    def calculate_transmission_load(self, path, state=None):
        return self.calculate_loads(path, state)

    def find_disconnected_sensors(self, state, path=None):
        return self.find_disconnected(state, path)


__all__ = ["RoutingResult", "RoutingService"]
