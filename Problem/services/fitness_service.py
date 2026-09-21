import numpy as np


class FitnessService:
    """Original three-objective fitness calculation for coded WSN states."""

    OBJECTIVE_NAMES = (
        "total_remaining_energy",
        "min_remaining_energy",
        "coverage",
    )

    def __init__(
        self,
        problem,
        w1=0.15,
        w2=0.25,
        w3=0.60,
    ):
        self.problem = problem
        self.w1 = float(w1)
        self.w2 = float(w2)
        self.w3 = float(w3)
        self.objective_names = list(self.OBJECTIVE_NAMES)

    def evaluate_state(self, state, debug=False):
        """Evaluate a decoded state and return the original three objectives."""
        problem = self.problem
        cost = problem.calculate_total_cost(state)
        remaining_energy = problem.energy - cost
        target_remaining_energy = problem.calculate_target_remaining_energy(
            remaining_energy
        )
        problem.target_red = target_remaining_energy

        total_target_remaining_energy = np.sum(target_remaining_energy)
        total_initial_target_energy = np.sum(
            problem.initial_energy * problem.target_sensor_mask
        )
        if total_initial_target_energy == 0:
            energy_balance_score = 0
        else:
            energy_balance_score = (
                total_target_remaining_energy / total_initial_target_energy
            )

        if len(target_remaining_energy) > 0:
            min_target_remaining_energy = np.min(target_remaining_energy)
        else:
            min_target_remaining_energy = 0

        min_initial_target_energy = problem.initial_energy * np.min(
            np.sum(problem.target_sensor_mask, axis=1)
        )
        min_energy_score = 0
        if min_target_remaining_energy >= 0 and min_initial_target_energy != 0:
            min_energy_score = (
                min_target_remaining_energy / min_initial_target_energy
            )

        energy_failed_sensors = np.where(remaining_energy < 0)[0]
        if len(energy_failed_sensors) > 0:
            if debug:
                print("energy failed sensors", energy_failed_sensors)
                for sensor_id in energy_failed_sensors:
                    print(
                        sensor_id,
                        state.levels[sensor_id],
                        state.next_hops[sensor_id],
                        remaining_energy[sensor_id],
                    )
            return -1, -5, -1

        uncovered_targets = problem.find_uncovered_targets(state.levels)
        if len(problem.target) == 0:
            coverage_score = 0
        else:
            coverage_score = (
                len(problem.target) - len(uncovered_targets)
            ) / len(problem.target)

        return (
            self.w1 * energy_balance_score,
            self.w2 * min_energy_score,
            self.w3 * coverage_score,
        )


__all__ = ["FitnessService"]
