"""Quantum-inspired local categorical exploration; no quantum hardware.

Online scalar local feedback, NOT original QEA/QEA3 and no global elite.
"""
from dataclasses import dataclass
import math
import numpy as np
from Algorithm.experiment.local_scheduling import Claim, JointResult
from Algorithm.experiment.probabilistic_scheduling import ProbabilisticScheduling
from Algorithm.experiment.reserved_routing import ReservedRouting


@dataclass(frozen=True)
class LocalFeedback:
    level: int
    route_ack: bool
    own_energy: float
    own_slot_cost: float
    own_control_cost: float


class LevelModel:
    def __init__(self, count, mode='quantum', step=0.01*math.pi, rate=0.05, floor=0.01):
        if count < 1 or mode not in ('quantum','classical','random','greedy'):
            raise ValueError('invalid level count or mode')
        if not 0 < floor < .5 or step <= 0 or not 0 < rate <= 1:
            raise ValueError('invalid update parameter')
        self.count, self.mode = count, mode
        self.step, self.rate, self.floor = step, rate, min(floor,1/(count+1))
        # Uniform positive levels, using count-1 conditional Bernoulli gates.
        self.probabilities = np.array([(count-j-1)/(count-j) for j in range(count-1)],dtype=float)
        self.angles = np.arcsin(np.sqrt(self.probabilities))
        self.baseline = .5
        self.updates = 0

    def masses(self):
        result = np.empty(self.count)
        prefix = 1.
        for j,p in enumerate(self.probabilities):
            result[j] = prefix*(1-p)
            prefix *= p
        result[-1] = prefix
        return result

    def update(self, index, reward):
        if not 0 <= index < self.count or not 0 <= reward <= 1:
            raise ValueError('invalid local outcome')
        advantage = reward-self.baseline
        if self.mode in ('quantum','classical'):
            offsets = np.arange(self.count-1)
            target = (offsets < index).astype(float)
            mask = offsets <= index  # first zero plus preceding ones only
            if self.mode == 'quantum':
                self.angles += self.step*advantage*(2*target-1)*mask
                lower = math.asin(math.sqrt(self.floor))
                self.angles = np.clip(self.angles,lower,math.pi/2-lower)
                self.probabilities = np.sin(self.angles)**2
            else:
                self.probabilities += self.rate*advantage*(target-self.probabilities)*mask
                self.probabilities = np.clip(self.probabilities,self.floor,1-self.floor)
                self.angles = np.arcsin(np.sqrt(self.probabilities))
        self.baseline = .8*self.baseline + .2*reward
        self.updates += 1
        return float(advantage)


class LocalLevelPolicy:
    def __init__(self, mode='quantum', seed=7, exploration=.15):
        if mode not in ('quantum','classical','random','greedy') or not 0 <= exploration <= 1:
            raise ValueError('invalid policy')
        self.mode, self.exploration = mode, exploration
        self.seed = int(seed)
        self.node_rngs = {}
        self.models = {}
        self.cache = {}

    def start_epoch(self):
        self.cache.clear()  # beliefs persist; stale neighbor proposals do not

    def propose(self, view):
        count = len(view.coverage)-1
        if view.node_id not in self.models:
            self.models[view.node_id] = LevelModel(count,self.mode)
            self.node_rngs[view.node_id] = np.random.default_rng(np.random.SeedSequence([self.seed,view.node_id]))
        model = self.models[view.node_id]
        rng = self.node_rngs[view.node_id]
        if model.count != count:
            raise ValueError('changing level domains within a run is unsupported')
        key = (view.node_id,view.current_level,view.energy,view.claimed_targets)
        if key in self.cache:
            return self.cache[key]
        candidates = []
        for level in range(view.current_level+1,len(view.coverage)):
            gain = view.coverage[level]-view.claimed_targets
            if not gain or view.sensing_costs[level]+view.own_routing_estimate > view.energy:
                continue
            extra = view.sensing_costs[level]-view.sensing_costs[view.current_level]
            if view.current_level == 0:
                extra += view.own_routing_estimate
            score = len(gain)*(view.energy/view.initial_energy)/max(extra,1e-15)
            candidates.append(Claim(view.node_id,level,frozenset(gain),float(score)))
        if not candidates:
            chosen = None
        else:
            chosen = max(candidates,key=lambda c:(c.score,-c.level))
            if self.mode != 'greedy' and rng.random() < self.exploration:
                masses = model.masses()[[c.level-1 for c in candidates]]
                masses /= masses.sum()
                chosen = candidates[int(rng.choice(len(candidates),p=masses))]
        self.cache[key] = chosen
        return chosen

    def feedback(self, node_id, feedback):
        if feedback.level <= 0 or node_id not in self.models:
            return None
        available = feedback.own_energy-feedback.own_control_cost
        feasible = feedback.route_ack and available >= feedback.own_slot_cost and available > 0
        # Local prospective horizon only. 100 slots is a fixed normalization
        # scale, not the achieved network lifetime or a global reward.
        horizon = available/max(feedback.own_slot_cost,1e-15) if feasible else 0.
        reward = horizon/(horizon+100.)
        model = self.models[node_id]
        advantage = model.update(feedback.level-1,float(reward))
        return dict(node=int(node_id),level=int(feedback.level),reward=float(reward),
                    advantage=advantage,baseline=model.baseline,
                    probabilities=model.probabilities.tolist(),angles=model.angles.tolist())


class QuantumLocalV2:
    def __init__(self, mode='quantum', seed=7, charge_control=True, exploration=.15):
        self.policy = LocalLevelPolicy(mode,seed,exploration)
        self.scheduler = ProbabilisticScheduling(self.policy,charge_control)
        self.router = ReservedRouting(128 if charge_control else 0)
        self.last_feedback = []

    def build(self, p):
        schedule = self.scheduler.build(p)
        routed = self.router.build(p,schedule.levels)
        return JointResult(routed.state,schedule.control+routed.control,
                           schedule.messages+routed.messages,schedule)

    def observe_local_outcomes(self, levels, next_hops, own_energies, own_costs, own_controls):
        # Environment delivers separate node-local records; no coverage,
        # network LifeCheck result or achieved lifetime is used by the policy.
        self.last_feedback = []
        for i,level in enumerate(levels):
            record = self.policy.feedback(i,LocalFeedback(int(level),int(next_hops[i])>=0,
                  float(own_energies[i]),float(own_costs[i]),float(own_controls[i])))
            if record is not None:
                self.last_feedback.append(record)
