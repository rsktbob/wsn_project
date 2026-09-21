import numpy as np

from Problem.services.evaluation_kernels import (
    NUMBA_AVAILABLE,
    count_covered_targets_kernel,
    find_uncovered_targets_kernel,
    first_out_of_range_kernel,
)


class CoverageService:
    """Coverage-related calculations for a WSN problem."""

    def __init__(self, problem):
        self.problem = problem

    def calculate_coverage_matrix(self):
        """Build target-sensor-level coverage and target-sensor reachability."""
        problem = self.problem
        distances = problem.G.CalDistance(problem.target, problem.sensor)

        radius_table = getattr(problem, "radius_table", None)
        if radius_table is None:
            # Problem construction validates candidate positions before the
            # final sensor-specific table can be built.
            provisional_levels = np.asarray(
                problem.LEVEL_RANGE, dtype=float
            ).copy()
            provisional_levels[-1] = problem.MAX_SENSING_RADIUS
            radius_table = np.repeat(
                provisional_levels[None, :],
                problem.SENSOR_NUMBER,
                axis=0,
            )

        coverage_table = (
            distances[:, :, None]
            <= np.asarray(radius_table, dtype=float)[None, :, :] + 1e-12
        ).astype(int)
        # Option zero always means that the sensor is switched off, including
        # the degenerate case where a target is colocated with a sensor.
        coverage_table[:, :, 0] = 0
        target_sensor_mask = np.any(coverage_table > 0, axis=2).astype(int)

        return coverage_table, target_sensor_mask

    def count_covered_targets(self, schedule):
        """Count how many selected sensors cover each target."""
        problem = self.problem
        schedule = self._prepare_schedule(schedule)
        if not NUMBA_AVAILABLE:
            detect_count = np.zeros(problem.TARGET_NUMBER).astype(int)
            target_ids, sensor_ids = np.where(
                problem.coverage_table[:, problem.SENSOR_ID, schedule] == 1
            )
            detect_count[target_ids] += 1
            return detect_count
        return count_covered_targets_kernel(
            np.asarray(problem.coverage_table), schedule
        )

    def find_uncovered_targets(self, schedule):
        """Return target ids that are not covered by the selected schedule."""
        problem = self.problem
        schedule = self._prepare_schedule(schedule)
        if not NUMBA_AVAILABLE:
            coverage_count = self.count_covered_targets(schedule)
            return np.where(coverage_count == 0)[0]
        return find_uncovered_targets_kernel(
            np.asarray(problem.coverage_table), schedule
        )

    def _prepare_schedule(self, source):
        """取出排程陣列，只在它尚未通過解碼期檢查時才驗證一次。

        解碼結束時 ``Problem.resolve_state_radius`` 會對 ``state.levels``
        蓋章（見 ``_validated_levels``）。評估路徑傳進來的都是蓋過章的
        State，因此這裡直接跳過；外部傳入的裸陣列或手工組裝的 State
        沒有章，仍會走完整驗證。
        """
        problem = self.problem
        state = source if hasattr(source, "levels") else None
        schedule = state.levels if state is not None else source
        validated = (
            state is not None
            and getattr(state, "_validated_levels", None) is schedule
        )
        schedule = np.asarray(schedule, dtype=int)
        if schedule.shape != (problem.SENSOR_NUMBER,):
            raise ValueError("schedule has the wrong sensor dimension")
        if not validated:
            self._validate_schedule_options(schedule)
        return schedule

    def _validate_schedule_options(self, schedule):
        """在進入無邊界檢查的 JIT 核心前驗證 sensing option。"""
        option_count = int(self.problem.coverage_table.shape[2])
        if NUMBA_AVAILABLE:
            offender = int(
                first_out_of_range_kernel(
                    np.asarray(schedule, dtype=np.int64),
                    option_count,
                )
            )
            if offender < 0:
                return
        elif not (
            np.any(schedule < 0) or np.any(schedule >= option_count)
        ):
            return
        else:
            offender = int(
                np.flatnonzero(
                    (schedule < 0) | (schedule >= option_count)
                )[0]
            )
        raise ValueError(
            "schedule contains an invalid sensing option: "
            f"sensor {offender} has level {int(schedule[offender])} "
            f"but the coverage table has {option_count} levels"
        )

    def build_cover_candidate_sets(self):
        """Build candidate sensor-level pairs for each target."""
        problem = self.problem
        cover_candidate_sets = []

        for target_id in range(problem.TARGET_NUMBER):
            cover_candidate_sets.append([])
            sensor_ids, levels = np.where(
                problem.coverage_table[target_id, :, :]
            )
            target_candidates = {}

            for index in range(len(sensor_ids)):
                sensor_id = sensor_ids[index]
                level = levels[index]
                if sensor_id in target_candidates:
                    target_candidates[sensor_id] = min(
                        target_candidates[sensor_id], level
                    )
                else:
                    target_candidates[sensor_id] = level

            for sensor_id, level in target_candidates.items():
                if (
                    np.sum(
                        problem.link_capacity[sensor_id, level, :] >= 50
                    )
                    > 0
                ):
                    cover_candidate_sets[target_id].append([sensor_id, level])

        return cover_candidate_sets

