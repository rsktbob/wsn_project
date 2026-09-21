import random

import numpy as np

class RQLearning:
    """Q-learning next-hop selector for WSN routing decode."""

    def __init__(
        self,
        alpha=0.2,
        gamma=0.2,
        epsilon=0.1,
        heuristic_weight=0.35,
        learn=True,
        seed=None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.heuristic_weight = heuristic_weight
        self.learn = learn
        self.q_table = {}
        self.rng = random.Random(seed)
        self.name = "RQLearning"

    def StateKey(self, P, state, sensor_id, level):
        return (int(sensor_id), int(level))

    def QActionKey(self, P, state, sensor_id, level, next_hop):
        return (self.StateKey(P, state, sensor_id, level), int(next_hop))

    def GetQValue(self, P, state, sensor_id, level, next_hop):
        return self.q_table.get(
            self.QActionKey(P, state, sensor_id, level, next_hop), 0.0
        )

    def SetQValue(self, P, state, sensor_id, level, next_hop, value):
        self.q_table[self.QActionKey(P, state, sensor_id, level, next_hop)] = float(
            value
        )

    def DistanceToBaseStation(self, P, node_id):
        if int(node_id) == P.BSID:
            return 0.0
        return float(P.distances[int(node_id), P.BSID])

    def NormalizeRouteScore(self, route_scores, route_score):
        finite_scores = np.array(route_scores)[np.isfinite(route_scores)]
        if len(finite_scores) == 0:
            return 0.0

        scale = np.max(np.abs(finite_scores))
        if scale == 0:
            return 0.0
        return float(route_score) / float(scale)

    def ProgressScore(self, P, sensor_id, next_hop):
        sensor_distance = self.DistanceToBaseStation(P, sensor_id)
        if sensor_distance <= 0 or not np.isfinite(sensor_distance):
            return 0.0

        next_distance = self.DistanceToBaseStation(P, next_hop)
        if not np.isfinite(next_distance):
            return -1.0

        return (sensor_distance - next_distance) / sensor_distance

    def EnergyScore(self, P, next_hop):
        if int(next_hop) == P.BSID:
            return 1.0
        max_energy = max(P.energy)
        if max_energy == 0:
            return 0.0
        return float(P.energy[int(next_hop)] / max_energy)

    def LoadPenalty(self, P, state, next_hop):
        if int(next_hop) == P.BSID:
            return 0.0

        max_load = max(1, P.SENSOR_NUMBER) * max(
            1, int(P.generated_load[int(next_hop)])
        )
        return float(state.tx_load[int(next_hop)] / max_load)

    def EstimateReward(
        self,
        P,
        state,
        sensor_id,
        level,
        next_hop,
        route_scores,
        route_score,
        success=True,
    ):
        if not success:
            return -1.0

        route_score = self.NormalizeRouteScore(route_scores, route_score)
        energy_score = self.EnergyScore(P, next_hop)
        progress_score = self.ProgressScore(P, sensor_id, next_hop)
        load_penalty = self.LoadPenalty(P, state, next_hop)
        return (
            0.45 * route_score
            + 0.25 * energy_score
            + 0.20 * progress_score
            - 0.10 * load_penalty
        )

    def CandidateIndexes(self, route_scores):
        scores = np.array(route_scores)
        candidates = np.where(np.isfinite(scores) & (scores >= 0))[0]
        return candidates.astype(int)

    def SelectNextHop(
        self,
        P,
        state,
        sensor_id,
        level,
        connected_nodes,
        route_scores,
    ):
        candidates = self.CandidateIndexes(route_scores)
        if len(candidates) == 0:
            return None

        if self.rng.random() < self.epsilon:
            route_index = int(self.rng.choice(list(candidates)))
            return int(connected_nodes[route_index])

        best_index = int(candidates[0])
        best_score = -float("inf")
        for route_index in candidates:
            next_hop = int(connected_nodes[int(route_index)])
            q_value = self.GetQValue(P, state, sensor_id, level, next_hop)
            reward_hint = self.EstimateReward(
                P,
                state,
                sensor_id,
                level,
                next_hop,
                route_scores,
                route_scores[int(route_index)],
                success=True,
            )
            score = q_value + self.heuristic_weight * reward_hint
            if score > best_score:
                best_score = score
                best_index = int(route_index)

        return int(connected_nodes[best_index])

    def FutureQValue(self, P, state, next_hop):
        if int(next_hop) == P.BSID:
            return 0.0
        if state.levels[int(next_hop)] <= 0:
            return 0.0

        routed_next_hop = int(state.next_hops[int(next_hop)])
        if routed_next_hop < 0:
            return 0.0
        return self.GetQValue(
            P,
            state,
            int(next_hop),
            int(state.levels[int(next_hop)]),
            routed_next_hop,
        )

    def LearnFromRoute(
        self,
        P,
        state,
        sensor_id,
        level,
        next_hop,
        connected_nodes,
        route_scores,
        route_score,
        success,
    ):
        if not self.learn or next_hop is None:
            return None

        reward = self.EstimateReward(
            P,
            state,
            sensor_id,
            level,
            next_hop,
            route_scores,
            route_score,
            success,
        )
        old_q = self.GetQValue(P, state, sensor_id, level, next_hop)
        future_q = self.FutureQValue(P, state, next_hop)
        new_q = old_q + self.alpha * (reward + self.gamma * future_q - old_q)
        self.SetQValue(P, state, sensor_id, level, next_hop, new_q)
        return new_q

    def Reset(self):
        self.q_table = {}

    def select_next_hop(self, *args, **kwargs):
        return self.SelectNextHop(*args, **kwargs)

    def learn_from_route(self, *args, **kwargs):
        return self.LearnFromRoute(*args, **kwargs)

    def reset(self):
        return self.Reset()


RQlearning = RQLearning
RQLEARNING = RQLearning
