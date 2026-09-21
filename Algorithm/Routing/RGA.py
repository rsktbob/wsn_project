from Algorithm.ga.BaseGA import BaseGA


class RGA(BaseGA):
    """Routing GA based on the shared GA template."""

    def __init__(self, DP, n=50, seed=None):
        super().__init__(n, l=DP.SENSOR_NUMBER, seed=seed)
        self.Fname = ["routing_distance"]

    def _create_state(self, P, sch_s):
        state = sch_s.Copy()
        state.CreteRouStateByDirect(P)
        return state

    def evaluate(self, P, state):
        self.evatime += 1
        distance = 0
        for sensor_id in range(P.SENSOR_NUMBER):
            distance += P.distances[sensor_id][state.next_hops[sensor_id]]
        return [1 / distance]

    def _crossover_states(self, father, mother, index1, index2):
        father.next_hops, mother.next_hops = self.crossover_segment(
            father.next_hops, mother.next_hops, index1, index2
        )
        return father, mother

    def _mutate_state(self, P, state):
        sensor_id = self.random.randint(0, P.SENSOR_NUMBER - 1)
        state.CandRouting(P, sensor_id)
        return state
