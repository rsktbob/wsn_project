"""Event-driven local next-hop selection with hop-by-hop load reservation.

A centralized sensing scheduler supplies levels ONLY. No original next-hop
or chromosome priority is consumed. Joining nodes query already joined peers.
"""
from dataclasses import dataclass
import numpy as np
from State.State import State
from Problem.Cal import pj


@dataclass(frozen=True)
class Offer:
    parent: int
    own_capacity: float
    downstream_capacity: float
    own_slots: float = float("inf")
    downstream_slots: float = float("inf")


def choose_offer(offers, load):
    """Pure local decision from link estimates and replies (not global state)."""
    legal = [(min(o.own_slots, o.downstream_slots), min(o.own_capacity, o.downstream_capacity) - load, -o.parent, o)
             for o in offers if min(o.own_capacity, o.downstream_capacity) >= load]
    return max(legal, key=lambda x: x[:3])[3] if legal else None


@dataclass
class Node:
    parent: int
    residual: float
    energy: float = float("inf")
    fixed_cost: float = 0.0
    unit_cost: float = 0.0
    load: float = 0.0


@dataclass
class BuildResult:
    state: State
    control: np.ndarray
    messages: int


class ReservedRouting:
    def __init__(self, control_bits=128):
        if control_bits < 0:
            raise ValueError('control_bits must be nonnegative')
        self.control_bits = control_bits

    def build(self, p, levels):
        state = State.empty(p)
        levels = np.asarray(levels)
        if levels.shape != state.levels.shape or not np.issubdtype(levels.dtype, np.integer):
            raise ValueError('one integer level required per node')
        if np.any(levels < 0) or np.any(levels >= p.radius_option_counts):
            raise ValueError('invalid level')
        state.levels[:] = levels
        active = np.flatnonzero(levels > 0)
        nodes = {int(p.BSID): Node(-1, float('inf'))}
        control = np.zeros(p.SENSOR_NUMBER)
        messages = 0

        def send(a, b):
            nonlocal messages
            messages += 1
            distance = float(np.linalg.norm(p.device[a] - p.device[b]))
            amp = (p.amp_factors[0] * pj * distance**2 if distance <= p.d0
                   else p.amp_factors[1] * pj * distance**4)
            if a != p.BSID:
                control[a] += self.control_bits * (p.circuit_cost + amp)
            if b != p.BSID:
                control[b] += self.control_bits * p.circuit_cost

        def query(node, extra):
            # Each relay sees only its own counter and its parent's reply.
            if node == p.BSID:
                return float('inf'), float('inf')
            parent = nodes[node].parent
            send(node, parent)
            downstream, downstream_slots = query(parent, extra)
            send(parent, node)
            record = nodes[node]
            slots = record.energy / (record.fixed_cost + (record.load + extra) * record.unit_cost)
            return min(record.residual, downstream), min(slots, downstream_slots)

        def reserve(node, load):
            if node == p.BSID:
                return
            parent = nodes[node].parent
            send(node, parent)
            reserve(parent, load)
            nodes[node].residual -= load
            nodes[node].load += load
            send(parent, node)

        # Ideal collision-free contention ordered by locally known remaining energy, then distance,
        # then id. It controls join timing, not the parent's selected route.
        order = sorted(active, key=lambda i: (-float(p.energy[i]), float(np.linalg.norm(p.device[i]-p.BS)), int(i)))
        for i in order:
            i = int(i)
            offers = []
            for parent in nodes:
                own_cap = float(p.link_capacity[i, levels[i], parent])
                if own_cap <= 0:
                    continue
                send(i, parent)
                downstream, downstream_slots = query(parent, float(p.generated_load[i]))
                send(parent, i)
                unit = float(p.amp_cost[i, parent] + 2 * p.circuit_cost)
                fixed = float(p.sensing_costs[i, levels[i]] - p.generated_load[i] * p.circuit_cost)
                own_slots = float(p.energy[i]) / (fixed + p.generated_load[i] * unit)
                offers.append(Offer(parent, own_cap, downstream, own_slots, downstream_slots))
            chosen = choose_offer(offers, float(p.generated_load[i]))
            if chosen is None:
                continue
            parent = chosen.parent
            send(i, parent)
            reserve(parent, float(p.generated_load[i]))
            send(parent, i)
            nodes[i] = Node(parent, chosen.own_capacity - p.generated_load[i],
                            float(p.energy[i]),
                            float(p.sensing_costs[i, levels[i]] - p.generated_load[i] * p.circuit_cost),
                            float(p.amp_cost[i, parent] + 2 * p.circuit_cost),
                            float(p.generated_load[i]))
            state.next_hops[i] = parent
        state.paths = {int(p.BSID): [int(p.BSID)]}
        for i in nodes:
            if i == p.BSID:
                continue
            state.paths[i] = [i] + state.paths[nodes[i].parent]
            state.tx_load[state.paths[i][:-1]] += p.generated_load[i]
        p.resolve_state_radius(state)
        return BuildResult(state, control, messages)
