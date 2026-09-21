import numpy as np


class MobilityService:
    """Relocate inactive sensors to recover weak or uncovered targets."""

    def __init__(self, problem, state=None, verbose=False):
        self.problem = problem
        self.state = state
        self.verbose = verbose
        self._reset(problem)

    def _reset(self, problem):
        self.sensor_weight = problem.target_sensor_mask.copy()
        self.update_power = problem.energy.copy()
        self.update_position = problem.sensor.copy()
        self.long = []
        self.bs_rank = []
        self.red_rank = []
        self.mobile_id = []

    def _log(self, *values):
        if self.verbose:
            print(*values)

    def _target_target_distances(self, problem):
        target_points = np.array([complex(t[0], t[1]) for t in problem.target])
        x, y = np.meshgrid(target_points, target_points)
        return abs(x - y)

    def _target_sensor_distances(self, problem):
        target_points = np.array([complex(t[0], t[1]) for t in problem.target])
        sensor_points = np.array([complex(s[0], s[1]) for s in problem.sensor])
        x, y = np.meshgrid(sensor_points, target_points)
        return abs(x - y)

    def _record_move(self, problem, sensor_id, new_position, remaining_power):
        self.update_position[sensor_id] = new_position
        self.update_power[sensor_id] = remaining_power
        self.mobile_id.append(
            [
                sensor_id,
                problem.sensor[sensor_id][0],
                problem.sensor[sensor_id][1],
                self.update_position[sensor_id][0],
                self.update_position[sensor_id][1],
            ]
        )

    def relocate_sensors(
        self,
        problem=None,
        state=None,
        low_energy_targets=None,
        target_remaining_energy=None,
        failed_targets=None,
    ):
        """Move inactive sensors toward weak targets and return updated data."""
        if problem is None:
            problem = self.problem
        if state is None:
            state = self.state
        if problem is not self.problem:
            self.problem = problem
            self._reset(problem)
        if state is not None:
            self.state = state

        if low_energy_targets is None:
            low_energy_targets = []
        else:
            low_energy_targets = list(low_energy_targets)
        if failed_targets is None:
            failed_targets = []
        else:
            failed_targets = list(failed_targets)
        if target_remaining_energy is None:
            target_remaining_energy = np.zeros(problem.TARGET_NUMBER)

        self.mobile_id = []
        target_ids = list(range(problem.TARGET_NUMBER))
        target_target_dist = self._target_target_distances(problem)
        target_sensor_dist = self._target_sensor_distances(problem)

        self._move_uncovered_sensors(problem, low_energy_targets, target_sensor_dist)

        low_energy_targets = list(
            set(low_energy_targets).difference(set(failed_targets))
        )
        self._move_inactive_sensors_to_failed_targets(
            problem, state, failed_targets, target_sensor_dist
        )

        backup_targets = self._select_backup_targets(
            problem,
            target_ids,
            low_energy_targets,
            target_remaining_energy,
            target_target_dist,
        )
        mobile_sensor_ids = self._select_mobile_sensors(
            problem, state, backup_targets, target_sensor_dist
        )
        self._move_selected_sensors(
            problem, low_energy_targets, mobile_sensor_ids, target_sensor_dist
        )
        return self.update_position, self.update_power, self.mobile_id

    def _move_uncovered_sensors(self, problem, low_energy_targets, target_sensor_dist):
        if len(low_energy_targets) == 0:
            return

        max_level = problem.LEVEL - 1
        for sensor_id in range(problem.SENSOR_NUMBER):
            cover_count = np.sum(
                problem.coverage_table[:, sensor_id, max_level]
            )
            if cover_count != 0:
                continue

            distances = np.zeros(len(low_energy_targets)).astype(float)
            for index, target_id in enumerate(low_energy_targets):
                distances[index] = target_sensor_dist[target_id][sensor_id]

            nearest_index = np.argmin(distances)
            nearest_target_id = low_energy_targets[nearest_index]
            distance = np.min(distances)
            move_distance = distance / 2
            move_cost = move_distance * 0.1
            if self.update_power[sensor_id] - move_cost > 0:
                new_position = (
                    problem.target[nearest_target_id] + problem.sensor[sensor_id]
                ) / 2
                remaining_power = problem.energy[sensor_id] - move_cost
                self._record_move(problem, sensor_id, new_position, remaining_power)

    def _move_inactive_sensors_to_failed_targets(
        self, problem, state, failed_targets, target_sensor_dist
    ):
        if state is None:
            return

        for target_id in failed_targets:
            moved_count = 0
            for sensor_id in range(problem.SENSOR_NUMBER - 1, 0, -1):
                distance = target_sensor_dist[target_id][sensor_id]
                move_cost = distance * 0.1
                if (
                    state.levels[sensor_id] == 0
                    and problem.energy[sensor_id] - move_cost > 1
                ):
                    if moved_count > 1:
                        break
                    moved_count += 1
                    self._record_move(
                        problem,
                        sensor_id,
                        problem.target[target_id],
                        problem.energy[sensor_id] - move_cost,
                    )

    def _select_backup_targets(
        self,
        problem,
        target_ids,
        low_energy_targets,
        target_remaining_energy,
        target_target_dist,
    ):
        backup_targets = []
        if len(low_energy_targets) == 0:
            return backup_targets

        max_target_remaining_energy = max(target_remaining_energy)
        if max_target_remaining_energy == 0:
            target_energy_rank = np.zeros_like(target_remaining_energy)
        else:
            target_energy_rank = target_remaining_energy / max_target_remaining_energy

        for target_id in low_energy_targets:
            target_target_dist[target_id][target_id] = 0.0000000000000001
            inverse_distance = 1 / target_target_dist[target_id]
            inverse_distance[np.isinf(inverse_distance)] = 0
            max_inverse_distance = max(inverse_distance)
            if max_inverse_distance == 0:
                distance_rank = np.zeros_like(inverse_distance)
            else:
                distance_rank = inverse_distance / max_inverse_distance

            best_fitness = 0
            best_target = 0
            for candidate_target in target_ids:
                is_valid_candidate = (
                    target_target_dist[target_id][candidate_target] != 0
                    and np.sum(
                        problem.target_sensor_mask[target_id] == 1
                    )
                    > 2
                    and candidate_target not in low_energy_targets
                    and candidate_target not in backup_targets
                )
                if not is_valid_candidate:
                    continue

                current_fitness = (
                    distance_rank[candidate_target] * 0.5
                    + target_energy_rank[candidate_target] * 0.5
                )
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_target = candidate_target
            backup_targets.append(best_target)

        return backup_targets

    def _select_mobile_sensors(
        self, problem, state, backup_targets, target_sensor_dist
    ):
        mobile_sensor_ids = []
        if state is None:
            return mobile_sensor_ids

        for target_id in backup_targets:
            candidate_sensors = np.where(
                problem.target_sensor_mask[target_id] == 1
            )[0]
            if len(candidate_sensors) == 0:
                continue

            inverse_distance = 1 / target_sensor_dist[target_id][candidate_sensors]
            inverse_distance[np.isinf(inverse_distance)] = 0
            max_inverse_distance = max(inverse_distance)
            if max_inverse_distance == 0:
                distance_rank = np.zeros_like(inverse_distance)
            else:
                distance_rank = inverse_distance / max_inverse_distance

            max_energy = max(problem.energy[candidate_sensors])
            if max_energy == 0:
                energy_rank = np.zeros_like(
                    problem.energy[candidate_sensors]
                )
            else:
                energy_rank = problem.energy[candidate_sensors] / max_energy

            best_fitness = 0
            best_sensor = -1
            for index, sensor_id in enumerate(candidate_sensors):
                if (
                    sensor_id in mobile_sensor_ids
                    or state.levels[sensor_id] != 0
                ):
                    continue

                current_fitness = energy_rank[index] * 0.3 + distance_rank[index] * 0.7
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_sensor = sensor_id
            if best_sensor != -1:
                mobile_sensor_ids.append(best_sensor)

        return mobile_sensor_ids

    def _move_selected_sensors(
        self, problem, low_energy_targets, mobile_sensor_ids, target_sensor_dist
    ):
        move_count = min(len(low_energy_targets), len(mobile_sensor_ids))
        for index in range(move_count):
            target_id = low_energy_targets[index]
            sensor_id = mobile_sensor_ids[index]
            distance = target_sensor_dist[target_id][sensor_id]

            half_distance_cost = distance / 2 * 0.1
            if self.update_power[sensor_id] - half_distance_cost > 0:
                new_position = (problem.target[target_id] + problem.sensor[sensor_id]) / 2
                remaining_power = (
                    problem.energy[sensor_id] - half_distance_cost
                )
                self._record_move(problem, sensor_id, new_position, remaining_power)
                continue

            third_distance_cost = distance / 3 * 0.1
            if self.update_power[sensor_id] - third_distance_cost > 0:
                new_position = (problem.target[target_id] + problem.sensor[sensor_id]) / 3
                remaining_power = (
                    problem.energy[sensor_id] - third_distance_cost
                )
                self._record_move(problem, sensor_id, new_position, remaining_power)

    def Move(self, P, s, min_tid, target_red, fail_target):
        """Backward-compatible alias for older experiment scripts."""
        return self.relocate_sensors(P, s, min_tid, target_red, fail_target)
