"""Extremal Optimization for joint WSN scheduling and routing."""

from __future__ import annotations

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from State.SensorEncoding import SensorEncoding


class EOPT(Algorithm):
    """以感測器局部劣度驅動的極值最佳化演算法。

    每次將目前解中的 active sensors 依耗能比例、轉送負載與覆蓋冗餘
    排名，再以 ``rank ** -tau`` 抽出一顆較差的 sensor。鄰域操作會關閉
    該 sensor，並在覆蓋不足時最多啟用一顆替代 sensor，因此不會重現
    SA-SETS-AOS 因 active sensor 數持續增加而造成的高耗能問題。
    """

    def __init__(
        self,
        problem,
        tau=1.5,
        initial_samples=32,
        depletion_weight=0.55,
        routing_weight=0.25,
        redundancy_weight=0.20,
        replacement_pool_size=5,
        seed=None,
    ):
        super().__init__(seed=seed)
        if tau <= 0:
            raise ValueError("tau must be positive")
        if initial_samples < 1:
            raise ValueError("initial_samples must be at least 1")
        if replacement_pool_size < 1:
            raise ValueError("replacement_pool_size must be at least 1")

        weights = np.asarray(
            [depletion_weight, routing_weight, redundancy_weight],
            dtype=float,
        )
        if np.any(weights < 0) or float(np.sum(weights)) <= 0:
            raise ValueError("component weights must be non-negative")

        self.tau = float(tau)
        self.initial_samples = int(initial_samples)
        self.component_weights = weights / np.sum(weights)
        self.replacement_pool_size = int(replacement_pool_size)
        self.name = "EOPT"
        problem.prepare_coding_cache()

    def search(self, problem, budget, state=None):
        """在 evaluation budget 內執行 constrained extremal search。"""
        evaluation_limit = max(1, int(budget))
        # Lifetime 實驗會重複呼叫同一個 optimizer。每一段都必須取得自己
        # 完整的 evaluation budget，不能沿用上一段的累積計數。
        self.evatime = 0
        history = []

        current = self._initialize(problem, evaluation_limit, state, history)
        current_coding, current_state, current_score, current_coverage = current
        current_active = self._active_count(current_state)

        while self.evatime < evaluation_limit:
            candidate = self._create_neighbor(
                problem,
                current_coding,
                current_state,
            )
            candidate_state, objectives, candidate_score, candidate_coverage = (
                self._evaluate_candidate(problem, candidate)
            )
            candidate_active = self._active_count(candidate_state)

            # EO 可以接受較差 fitness 以跳離局部最佳；但完整覆蓋一旦取得
            # 就不退回不完整解，且鄰域不能增加 active sensor 數。
            if current_coverage < 1.0:
                # 可行解建立階段允許逐顆加入 sensor，但不能接受覆蓋更差
                # 的狀態；覆蓋相同時只接受 fitness 改善。
                accepted = (
                    candidate_coverage > current_coverage
                    or (
                        candidate_coverage == current_coverage
                        and candidate_score >= current_score
                    )
                )
            else:
                # 取得完整覆蓋後才套用 cardinality protection。EO 仍可
                # 接受較差 fitness，以保留跳離局部最佳的能力。
                accepted = (
                    candidate_coverage >= 1.0
                    and candidate_active <= current_active
                )
            if accepted:
                current_coding = candidate
                current_state = candidate_state
                current_score = candidate_score
                current_coverage = candidate_coverage
                current_active = candidate_active

            history.append(float(self.fitness))
            self.on_iteration_finish(
                problem=problem,
                state=self.best_state,
                iteration=self.iteration,
                Name=self.name,
                best_value=self.best_objectives.copy(),
                fitness=float(self.fitness),
                current_fitness=float(current_score),
                current_coverage=float(current_coverage),
                active_sensors=int(current_active),
            )

        self.history = np.asarray(history, dtype=float)
        return None if self.best_state is None else self.best_state.copy()

    def _initialize(self, problem, budget, initial_state, history):
        """從少量均勻樣本中選擇最佳起點，不額外超出 budget。"""
        best = None
        samples = min(self.initial_samples, budget)
        for sample_id in range(samples):
            if sample_id == 0 and initial_state is not None:
                candidate = self._encode_state(problem, initial_state)
            else:
                candidate = self._create_seed_candidate(problem)
            state, _, score, coverage = self._evaluate_candidate(
                problem,
                candidate,
            )
            coding = candidate.copy()
            rank = (coverage >= 1.0, coverage, score)
            if best is None or rank > best[0]:
                best = (rank, coding, state.copy(), score, coverage)
            history.append(float(self.fitness))

        _, coding, decoded, score, coverage = best
        return coding, decoded, score, coverage

    def _create_seed_candidate(self, problem):
        """以隨機染色體為底，再用 CCS 注入每個 target 的存活候選。"""
        candidate = SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
            rng=self.rng,
        )
        target_ids = list(range(problem.TARGET_NUMBER))
        # 先處理可選 sensor 較少的 target，降低關鍵 target 遺漏機率。
        self.random.shuffle(target_ids)
        target_ids.sort(
            key=lambda target_id: sum(
                problem.energy[int(sensor_id)] >= problem.liveJ
                for sensor_id, _ in problem.cover_candidates[target_id]
            )
        )
        for target_id in target_ids:
            choices = [
                (int(sensor_id), int(level))
                for sensor_id, level in problem.cover_candidates[target_id]
                if problem.energy[int(sensor_id)] >= problem.liveJ
            ]
            if not choices:
                continue
            sensor_id, level = self.random.choice(choices)
            gene_id = sensor_id * 2
            option_count = problem.sensing_option_count(sensor_id)
            current_level = int(candidate.code[gene_id]) % option_count
            candidate.code[gene_id] = max(current_level, level)
            # 被選來補覆蓋的 sensor 優先解碼，避免低優先權使其被略過。
            candidate.code[gene_id + 1] = SensorEncoding.RANK_PRECISION - 1
        return candidate

    def _encode_state(self, problem, state):
        """將可選的 warm-start State 轉成 EOPT 使用的 sensor coding。"""
        if isinstance(state, SensorEncoding):
            return state.copy()
        if not all(
            hasattr(state, name)
            for name in ("levels", "next_hops", "tx_load")
        ):
            raise TypeError("initial state must be a State or SensorEncoding")
        code = np.empty(problem.SENSOR_NUMBER * 2, dtype=int)
        code[0::2] = np.asarray(state.levels, dtype=int)
        code[1::2] = self.rng.integers(
            0,
            SensorEncoding.RANK_PRECISION,
            problem.SENSOR_NUMBER,
        )
        return SensorEncoding(code)

    def _evaluate_candidate(self, problem, coding):
        """只解碼一次並透過共用 Algorithm.evaluate 消耗一次評估。"""
        state = self.decode_candidate(problem, coding)
        objectives = self.evaluate(
            problem,
            state,
            source_candidate=coding,
        )
        score = float(np.sum(objectives))
        target_count = int(problem.TARGET_NUMBER)
        uncovered = len(problem.find_uncovered_targets(state.levels))
        coverage = (
            1.0
            if target_count == 0
            else (target_count - uncovered) / target_count
        )
        return state, objectives, score, float(coverage)

    def _create_neighbor(self, problem, coding, state):
        """移除一個極端劣質 sensor，必要時最多加入一個替代者。"""
        uncovered = np.asarray(
            problem.find_uncovered_targets(state.levels),
            dtype=int,
        )
        if len(uncovered):
            # 尚未可行時先做 coverage repair。這是唯一允許 active sensor
            # 數增加的階段，每次最多加入一顆。
            neighbor = coding.copy()
            replacement = self._select_replacement(
                problem,
                state,
                uncovered,
                excluded_sensor=-1,
            )
            if replacement is not None:
                sensor_id, level = replacement
                neighbor.code[sensor_id * 2] = int(level)
                neighbor.code[sensor_id * 2 + 1] = (
                    SensorEncoding.RANK_PRECISION - 1
                )
                return neighbor
            return self._create_seed_candidate(problem)

        active = np.flatnonzero(np.asarray(state.levels) > 0)
        if len(active) == 0:
            return self._create_seed_candidate(problem)

        coverage_count, covered_targets = self._coverage_details(
            problem,
            state,
            active,
        )
        removable = np.asarray(
            [
                sensor_id
                for sensor_id in active
                if not np.any(
                    coverage_count[covered_targets[int(sensor_id)]] == 1
                )
            ],
            dtype=int,
        )
        ranked_pool = removable if len(removable) else active
        selected = self._select_extreme_sensor(
            problem,
            state,
            ranked_pool,
            coverage_count,
            covered_targets,
        )

        neighbor = coding.copy()
        neighbor.code[selected * 2] = 0
        deficits = covered_targets[selected][
            coverage_count[covered_targets[selected]] == 1
        ]
        if len(deficits):
            replacement = self._select_replacement(
                problem,
                state,
                deficits,
                excluded_sensor=selected,
            )
            if replacement is None:
                # 沒有單一替代者能修復覆蓋時，只擾動該 sensor 的 routing
                # priority；仍是一個元件層級變動且不增加 active 數量。
                neighbor.code[selected * 2] = coding.code[selected * 2]
                neighbor.code[selected * 2 + 1] = int(
                    self.rng.integers(SensorEncoding.RANK_PRECISION)
                )
            else:
                sensor_id, level = replacement
                neighbor.code[sensor_id * 2] = int(level)
                neighbor.code[sensor_id * 2 + 1] = (
                    SensorEncoding.RANK_PRECISION - 1
                )
        return neighbor

    def _select_extreme_sensor(
        self,
        problem,
        state,
        sensors,
        coverage_count,
        covered_targets,
    ):
        """依局部劣度排序，再使用 EO power-law rank 抽樣。"""
        cost = np.asarray(problem.calculate_total_cost(state), dtype=float)
        energy = np.maximum(
            np.asarray(problem.energy, dtype=float), 1e-15
        )
        depletion = np.clip(cost / energy, 0.0, None)
        load = np.asarray(state.tx_load, dtype=float)
        max_load = max(float(np.max(load[sensors])), 1.0)

        badness = []
        for sensor_id in sensors:
            targets = covered_targets[int(sensor_id)]
            redundancy = (
                0.0
                if len(targets) == 0
                else float(np.mean(coverage_count[targets] > 1))
            )
            value = float(
                self.component_weights[0] * depletion[sensor_id]
                + self.component_weights[1] * load[sensor_id] / max_load
                + self.component_weights[2] * redundancy
            )
            badness.append((value, int(sensor_id)))

        ranked = [sensor_id for _, sensor_id in sorted(badness, reverse=True)]
        ranks = np.arange(1, len(ranked) + 1, dtype=float)
        probabilities = ranks ** (-self.tau)
        probabilities /= np.sum(probabilities)
        return int(ranked[int(self.rng.choice(len(ranked), p=probabilities))])

    def _select_replacement(
        self,
        problem,
        state,
        deficits,
        excluded_sensor,
    ):
        """挑選可修復最多缺口且剩餘能量較佳的一顆 dormant sensor。"""
        active = set(
            np.flatnonzero(np.asarray(state.levels) > 0).tolist()
        )
        required_levels = {}
        gains = {}
        deficit_set = set(int(target_id) for target_id in deficits)
        for target_id in deficit_set:
            for sensor_id, level in problem.cover_candidates[target_id]:
                sensor_id = int(sensor_id)
                level = int(level)
                if (
                    sensor_id == excluded_sensor
                    or sensor_id in active
                    or problem.energy[sensor_id] < problem.liveJ
                ):
                    continue
                required_levels[sensor_id] = max(
                    required_levels.get(sensor_id, 0),
                    level,
                )
                gains.setdefault(sensor_id, set()).add(target_id)

        candidates = []
        max_energy = max(float(np.max(problem.energy)), 1e-15)
        for sensor_id, level in required_levels.items():
            covered = set(
                np.flatnonzero(
                    problem.coverage_table[:, sensor_id, level] > 0
                ).tolist()
            )
            gain = len(deficit_set.intersection(covered))
            sensing_cost = problem.sensing_cost(sensor_id, level)
            energy_ratio = float(problem.energy[sensor_id]) / max_energy
            score = (gain, energy_ratio, -sensing_cost)
            candidates.append((score, sensor_id, level))

        if not candidates:
            return None
        candidates.sort(reverse=True)
        pool = candidates[: self.replacement_pool_size]
        _, sensor_id, level = pool[int(self.rng.integers(len(pool)))]
        return int(sensor_id), int(level)

    @staticmethod
    def _coverage_details(problem, state, active):
        coverage_count = np.zeros(problem.TARGET_NUMBER, dtype=int)
        covered_targets = {}
        for sensor_id in active:
            targets = np.flatnonzero(
                problem.coverage_table[
                    :, sensor_id, int(state.levels[sensor_id])
                ]
                > 0
            )
            covered_targets[int(sensor_id)] = targets
            coverage_count[targets] += 1
        return coverage_count, covered_targets

    @staticmethod
    def _active_count(state):
        return int(np.count_nonzero(np.asarray(state.levels) > 0))


__all__ = ["EOPT"]
