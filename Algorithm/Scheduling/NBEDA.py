import numpy as np

from Algorithm.misc.BaseEDA import BaseEDA
from Algorithm.Scheduling.SGA import ScheduleState


class NBEDA(BaseEDA):
    """Schedule-only EDA."""

    def __init__(self, DP, n=50):
        super().__init__(n)
        self.Fname = ["total_remaining_energy", "sleep_ratio", "coverage"]

    def _evaluate_state(self, P, state):
        cost = P.calculate_scheduling_cost(state)
        remaining_energy = P.energy - cost

        total_energy_score = np.sum(remaining_energy) / np.sum(
            P.initial_energy * P.SENSOR_NUMBER
        )
        uncovered_targets = P.find_uncovered_targets(state.levels)
        sleep_ratio = np.sum(state.levels == 0) / P.SENSOR_NUMBER
        coverage_score = (P.TARGET_NUMBER - len(uncovered_targets)) / P.TARGET_NUMBER
        return total_energy_score, sleep_ratio, coverage_score

    def _build_states(self, P, code_population=None):
        # Accept the historical one-argument form as well as BaseEDA's
        # CreateCompleteState(problem, codes) hook.
        if code_population is None:
            code_population = P
        states = []
        for code in code_population:
            state = ScheduleState()
            state.levels = code.copy()
            states.append(state)
        return states
