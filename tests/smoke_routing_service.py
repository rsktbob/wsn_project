import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Problem.services.routing_service import RoutingService
from State.State import State


def make_problem(link_capacity, generated_load):
    sensor_count = len(generated_load)
    return SimpleNamespace(
        SENSOR_NUMBER=sensor_count,
        DEVICE_NUMBER=sensor_count + 1,
        BSID=sensor_count,
        generated_load=np.asarray(generated_load, dtype=int),
        link_capacity=np.asarray(link_capacity, dtype=float),
    )


def check_fractional_shortfall_is_rejected():
    capacities = np.zeros((1, 2, 2), dtype=float)
    capacities[0, 1, 1] = 399.9
    problem = make_problem(capacities, [400])
    state = State.empty(problem)
    state.levels[0] = 1

    result = RoutingService(problem).build_routes(state, [[0, 1]])

    assert result.connected_ids.tolist() == []
    assert result.capacity_failed_ids.tolist() == [0]
    assert state.levels[0] == 0


def check_largest_bottleneck_margin_is_selected():
    capacities = np.zeros((2, 2, 3), dtype=float)
    capacities[0, 1, 2] = 1000
    capacities[1, 1, 0] = 900
    capacities[1, 1, 2] = 500
    problem = make_problem(capacities, [400, 400])
    state = State.empty(problem)
    state.levels[:] = 1

    result = RoutingService(problem).build_routes(
        state,
        [[0, 1], [1, 1]],
    )

    assert result.connected_ids.tolist() == [0, 1]
    assert state.next_hops[0] == problem.BSID
    assert state.next_hops[1] == 0


def main():
    check_fractional_shortfall_is_rejected()
    check_largest_bottleneck_margin_is_selected()
    print("smoke_routing_service_ok")


if __name__ == "__main__":
    main()
