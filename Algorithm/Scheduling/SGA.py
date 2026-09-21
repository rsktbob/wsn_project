import numpy as np

from Algorithm.ga.BaseGA import BaseGA


class ScheduleState:
    """Minimal state used by schedule-only algorithms."""

    def __init__(self):
        self.levels = None
        self.next_hops = None
        self.tx_load = None
        self.remaining_capacity = None
        self.code = None
        self.sensing_radii = None
        self.use_continuous_radius = False

    def CreateSchState(self, P, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        self.levels = np.asarray(
            [
                rng.integers(P.sensing_option_count(sensor_id))
                for sensor_id in range(P.SENSOR_NUMBER)
            ],
            dtype=int,
        )
        self.next_hops = np.repeat(-1, P.SENSOR_NUMBER)
        self.tx_load = np.zeros(P.SENSOR_NUMBER).astype(int)
        return self

    def Copy(self):
        copied_state = ScheduleState()
        if self.levels is not None:
            copied_state.levels = self.levels.copy()
        if self.next_hops is not None:
            copied_state.next_hops = self.next_hops.copy()
        if self.tx_load is not None:
            copied_state.tx_load = self.tx_load.copy()
        if self.remaining_capacity is not None:
            copied_state.remaining_capacity = self.remaining_capacity.copy()
        if self.code is not None:
            copied_state.code = self.code.copy()
        if self.sensing_radii is not None:
            copied_state.sensing_radii = self.sensing_radii.copy()
        copied_state.use_continuous_radius = self.use_continuous_radius
        return copied_state

    def copy(self):
        return self.Copy()

    # 過渡用舊名稱；演算法內部不再使用。
    sch = property(
        lambda self: self.levels,
        lambda self, value: setattr(self, "levels", value),
    )
    rou = property(
        lambda self: self.next_hops,
        lambda self, value: setattr(self, "next_hops", value),
    )
    use = property(
        lambda self: self.tx_load,
        lambda self, value: setattr(self, "tx_load", value),
    )
    cap = property(
        lambda self: self.remaining_capacity,
        lambda self, value: setattr(self, "remaining_capacity", value),
    )
    radius = property(
        lambda self: self.sensing_radii,
        lambda self, value: setattr(self, "sensing_radii", value),
    )


class SGA(BaseGA):
    """Scheduling GA based on the shared GA template."""

    def __init__(self, DP, n=50, seed=None):
        super().__init__(n, l=DP.SENSOR_NUMBER, seed=seed)
        self.Fname = ["total_remaining_energy", "sleep_ratio", "coverage"]

    def _create_state(self, P, sch_s=None):
        state = ScheduleState()
        state.CreateSchState(P, self.rng)
        return state

    def evaluate(self, P, state):
        self.evatime += 1
        cost = P.calculate_scheduling_cost(state)
        remaining_energy = P.energy - cost

        total_energy_score = np.sum(remaining_energy) / np.sum(
            P.initial_energy * P.SENSOR_NUMBER
        )
        uncovered_targets = P.find_uncovered_targets(state.levels)
        sleep_ratio = np.sum(state.levels == 0) / P.SENSOR_NUMBER
        coverage_score = (P.TARGET_NUMBER - len(uncovered_targets)) / P.TARGET_NUMBER
        return total_energy_score, sleep_ratio, coverage_score

    def _crossover_states(self, father, mother, index1, index2):
        father.levels, mother.levels = self.crossover_segment(
            father.levels, mother.levels, index1, index2
        )
        return father, mother

    def _mutate_state(self, P, state):
        sensor_id = self.random.randint(0, P.SENSOR_NUMBER - 1)
        level = self.random.randint(0, P.sensing_option_count(sensor_id) - 1)
        state.levels[sensor_id] = level
        return state
