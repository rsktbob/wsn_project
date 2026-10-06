"""Frozen coverage coordination from v1 with an instance-local policy hook.

Only the propose(view) call changes; original local_scheduling.py is untouched.
"""
import numpy as np
from Problem.Cal import pj
from Algorithm.experiment.local_scheduling import SchedulingView, ScheduleResult

class ProbabilisticScheduling:
    def __init__(self, policy, charge_control=True):
        self.policy = policy
        self.charge_control = charge_control

    def build(self, p):
        self.policy.start_epoch()
        n = p.SENSOR_NUMBER
        levels = np.zeros(n, dtype=int)
        observed = [set() for _ in range(n)]
        cover = [tuple(frozenset(np.flatnonzero(p.coverage_table[:, i, l]))
                       for l in range(int(p.radius_option_counts[i]))) for i in range(n)]
        # Overlap peers are logical coordination neighbors. v1 explicitly
        # assumes reliable direct control unicast over their geometric link.
        peers = [[j for j in range(n) if j != i and cover[i][-1] & cover[j][-1]] for i in range(n)]
        control = np.zeros(n)
        claims, messages = [], 0
        # Conservative own data estimate: direct BS transmission, no relay
        # reception. Can reject a node that would be feasible via multi-hop.
        routing = p.generated_load * (p.circuit_cost + p.amp_cost[:, p.BSID])

        def amp_between(a, b):
            d = float(np.linalg.norm(p.device[a] - p.device[b]))
            return p.amp_factors[0] * pj * d*d if d <= p.d0 else p.amp_factors[1] * pj * d**4

        # Each node reserves a conservative bound for all possible claim/ACK
        # exchanges in this epoch. Peer max option count/claim size is static
        # discovery metadata, not remote energy or optimization information.
        reserve = np.zeros(n)
        # Keep participation policy identical in the zero-charge ablation.
        for i in range(n):
            for j in peers[i]:
                unit = p.circuit_cost + amp_between(i, j)
                own_bits = 64 + 16 * len(cover[i][-1])
                peer_bits = 64 + 16 * len(cover[j][-1])
                reserve[i] += (len(cover[i])-1) * (own_bits * unit + 64 * p.circuit_cost)
                reserve[i] += (len(cover[j])-1) * (peer_bits * p.circuit_cost + 64 * unit)
        participating = p.energy > reserve + np.array([min(p.sensing_costs[i,1:len(cover[i])]) for i in range(n)]) + routing
        # Ideal wake roster known at epoch start, as part of assumed neighbor
        # discovery. Ineligible nodes sleep, neither receive nor acknowledge.
        peers = [[j for j in peers[i] if participating[j]] if participating[i] else [] for i in range(n)]

        def send(a, b, bits):
            nonlocal messages
            messages += 1
            if not self.charge_control:
                return
            d = float(np.linalg.norm(p.device[a] - p.device[b]))
            amp = p.amp_factors[0] * pj * d*d if d <= p.d0 else p.amp_factors[1] * pj * d**4
            control[a] += bits * (p.circuit_cost + amp)
            control[b] += bits * p.circuit_cost

        for _ in range(sum(int(x)-1 for x in p.radius_option_counts)):
            candidates = []
            for i in range(n):
                if not participating[i]:
                    continue
                view = SchedulingView(i, float(p.energy[i]-reserve[i]), float(p.initial_energy),
                                      int(levels[i]), cover[i], tuple(p.sensing_costs[i, :len(cover[i])]),
                                      float(routing[i]), frozenset(observed[i]))
                proposal = self.policy.propose(view)
                if proposal is not None:
                    candidates.append(proposal)
            if not candidates:
                break
            # Event simulator executes the earliest reciprocal-score timer.
            # The winner only receives its own timer expiry, not other scores.
            claim = max(candidates, key=lambda x: (x.score, -x.node_id))
            i = claim.node_id
            levels[i] = claim.level
            observed[i].update(claim.targets)
            for j in peers[i]:
                send(i, j, 64 + 16 * len(claim.targets))
                observed[j].update(claim.targets)
                send(j, i, 64)  # acknowledgment before another claim commits
            claims.append(dict(node=i, level=claim.level, targets=sorted(map(int, claim.targets)), score=float(claim.score)))
        # No global coverage result is supplied to propose(). Global coverage
        # is checked only by the experiment's observer after this process.
        if np.any(control > reserve + 1e-12):
            raise RuntimeError("coordination energy exceeded reserved bound")
        return ScheduleResult(levels, control, messages, claims)

