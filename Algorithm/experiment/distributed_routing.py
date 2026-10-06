"""Local next-hop policies and an ideal synchronous message-passing simulator.

The policy has no Problem, global state, fitness, or other nodes' tables.
All energies use the existing project's units (mJ).
"""
from dataclasses import dataclass
import numpy as np

from Problem.Cal import pj
from State.State import State


@dataclass(frozen=True)
class Advertisement:
    node_id: int
    rank: float
    energy: float
    reachable: bool
    path_metric: float
    hops: int


@dataclass(frozen=True)
class Neighbor:
    advertisement: Advertisement
    amplifier_cost: float


@dataclass(frozen=True)
class LocalObservation:
    node_id: int
    rank: float
    energy: float  # after control and own sensing costs
    circuit_cost: float
    sink_id: int
    neighbors: tuple[Neighbor, ...]


class LocalRoutingPolicy:
    """Choose from received neighbor advertisements only; deterministic ties."""
    MODES = ("energy_balanced", "min_energy", "min_hop")

    def __init__(self, mode="energy_balanced"):
        if mode not in self.MODES:
            raise ValueError(f"unknown policy: {mode}")
        self.mode = mode

    def choose(self, observation):
        o = observation
        if o.energy <= 0:
            return None
        choices = []
        for neighbor in o.neighbors:
            a = neighbor.advertisement
            if not a.reachable or not a.rank < o.rank or a.energy <= 0:
                continue
            tx = o.circuit_cost + neighbor.amplifier_cost
            rx = 0.0 if a.node_id == o.sink_id else o.circuit_cost
            if self.mode == "energy_balanced":
                metric = tx / o.energy + rx / a.energy + a.path_metric
            elif self.mode == "min_energy":
                metric = tx + rx + a.path_metric
            else:
                metric = float(a.hops + 1)
            choices.append((metric, tx, a.node_id, a.hops + 1))
        return min(choices) if choices else None


@dataclass
class RoutingEpoch:
    state: State
    control_cost: np.ndarray
    control_messages: int
    waves: int
    disconnected_ids: list[int]


class DistributedRouting:
    """One epoch with neighbor HELLO + route advertisements.

    Ideal reliable unicast, known neighbor links, static positions per epoch.
    Physical control neighbors are symmetric; data edges also obey the legacy
    link filter AND a new strictly decreasing distance-to-sink rank rule.
    """
    def __init__(self, mode="energy_balanced", radio_range=30.0, control_bits=128):
        self.policy = LocalRoutingPolicy(mode)
        if not np.isfinite(radio_range) or radio_range <= 0:
            raise ValueError("radio_range must be finite and positive")
        if isinstance(control_bits, bool) or int(control_bits) != control_bits or control_bits < 0:
            raise ValueError("control_bits must be a nonnegative integer")
        self.radio_range = float(radio_range)
        self.control_bits = int(control_bits)

    def build(self, problem, levels):
        p = problem
        levels = np.asarray(levels)
        n, sink = p.SENSOR_NUMBER, p.BSID
        if levels.shape != (n,) or not np.issubdtype(levels.dtype, np.integer):
            raise ValueError("levels must be one integer option per sensor")
        if np.any(levels <= 0) or np.any(levels >= np.asarray(p.radius_option_counts)):
            raise ValueError("v1 requires a fixed positive legal sensing option at every node")
        state = State.empty(p)
        state.levels[:] = levels
        geometry = np.linalg.norm(p.device[:, None] - p.device[None, :], axis=2)
        rank = geometry[:, sink].copy()
        rank[sink] = -1.0  # root below even a sensor colocated with the sink
        physical = (geometry <= self.radio_range) & ~np.eye(n + 1, dtype=bool)
        amplifier = np.where(geometry <= p.d0,
                             p.amp_factors[0] * pj * geometry**2,
                             p.amp_factors[1] * pj * geometry**4)
        # Two unicast messages in each direction: HELLO and route status.
        # Include the sink's sends in message counts; its energy is unlimited.
        control = np.zeros(n)
        edges = np.argwhere(physical)
        for source, destination in edges:
            if source != sink:
                control[source] += 2 * self.control_bits * (p.circuit_cost + amplifier[source, destination])
            if destination != sink:
                control[destination] += 2 * self.control_bits * p.circuit_cost
        available = p.energy - control - p.sensing_costs[np.arange(n), levels]
        data_edges = physical[:n] & (p.distances[:n] < p.blocked_distance / 100)
        data_edges &= rank[None, :] < rank[:n, None]
        advertisements = {sink: Advertisement(sink, -1.0, float("inf"), True, 0.0, 0)}
        pending = set(range(n))
        waves = 0
        while pending:
            emitted = {}
            # Synchronous waves: only announcements from PREVIOUS waves read.
            for node in sorted(pending):
                lower = np.flatnonzero(data_edges[node])
                if any(int(parent) not in advertisements for parent in lower):
                    continue
                neighbors = tuple(Neighbor(advertisements[int(j)], float(amplifier[node, j])) for j in lower)
                observation = LocalObservation(node, float(rank[node]), float(available[node]),
                                               float(p.circuit_cost), sink, neighbors)
                choice = self.policy.choose(observation)
                if choice is None:
                    emitted[node] = Advertisement(node, float(rank[node]), float(available[node]), False, 0.0, 0)
                else:
                    metric, _, parent, hops = choice
                    state.next_hops[node] = parent
                    emitted[node] = Advertisement(node, float(rank[node]), float(available[node]), True, metric, hops)
            if not emitted:
                raise RuntimeError("strict rank DAG should always allow progress")
            advertisements.update(emitted)
            pending.difference_update(emitted)
            waves += 1
        # Observer reconstructs paths/loads AFTER local decisions; never feeds
        # global path/load/fitness information back into a node's policy.
        state.paths = {sink: [sink]}
        for node in sorted(range(n), key=lambda i: rank[i]):
            parent = int(state.next_hops[node])
            if parent >= 0:
                state.paths[node] = [node] + state.paths[parent]
                state.tx_load[state.paths[node][:-1]] += int(p.generated_load[node])
        disconnected = [i for i in range(n) if state.next_hops[i] < 0]
        return RoutingEpoch(state, control, 2 * len(edges), waves, disconnected)
