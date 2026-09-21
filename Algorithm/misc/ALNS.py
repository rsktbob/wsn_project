"""適用於聯合感測排程與路由問題的自適應大型鄰域搜尋。"""

import math

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from State.SensorEncoding import SensorEncoding


class ALNS(Algorithm):
    """以 WSN 專用破壞與修復操作搜尋 ``CodingState``。

    演算法每次只評估一個完整候選解，並依操作近期帶來的改善程度更新
    選擇權重。模擬退火接受準則只允許在相同可行性等級內接受較差解，
    避免為了短期探索而失去完整覆蓋或產生無法傳輸的路由。
    """

    DESTROY_OPERATORS = (
        "random",
        "low_energy",
        "coverage_overlap",
        "route_bottleneck",
    )
    REPAIR_OPERATORS = (
        "coverage_efficiency",
        "energy_balance",
        "route_aware",
    )

    def __init__(
        self,
        P,
        initial_samples=8,
        min_destroy_ratio=0.05,
        max_destroy_ratio=0.20,
        reaction_factor=0.20,
        segment_length=25,
        initial_temperature=0.05,
        final_temperature=0.001,
        score_global_best=10.0,
        score_improved=5.0,
        score_accepted=1.0,
        feasibility_first=True,
        seed=None,
    ):
        super().__init__()
        if initial_samples < 1:
            raise ValueError("initial_samples 必須至少為 1")
        if not 0 < min_destroy_ratio <= max_destroy_ratio <= 1:
            raise ValueError("破壞比例必須滿足 0 < min <= max <= 1")
        if not 0 < reaction_factor <= 1:
            raise ValueError("reaction_factor 必須位於 (0, 1]")
        if segment_length < 1:
            raise ValueError("segment_length 必須至少為 1")
        if not 0 < final_temperature <= initial_temperature:
            raise ValueError("溫度必須滿足 0 < final <= initial")

        P.prepare_coding_cache()
        self.initial_samples = int(initial_samples)
        self.min_destroy_ratio = float(min_destroy_ratio)
        self.max_destroy_ratio = float(max_destroy_ratio)
        self.reaction_factor = float(reaction_factor)
        self.segment_length = int(segment_length)
        self.initial_temperature = float(initial_temperature)
        self.final_temperature = float(final_temperature)
        self.score_global_best = float(score_global_best)
        self.score_improved = float(score_improved)
        self.score_accepted = float(score_accepted)
        self.feasibility_first = bool(feasibility_first)
        self.rng = np.random.default_rng(seed)

        self.name = "ALNS"
        self.evatime = 0
        self.history = np.array([], dtype=float)
        self.destroy_weights = np.ones(len(self.DESTROY_OPERATORS), dtype=float)
        self.repair_weights = np.ones(len(self.REPAIR_OPERATORS), dtype=float)
        self.operator_history = []

    def search(self, P, budget, state=None):
        """在嚴格適應值評估次數限制內執行 ALNS。"""
        evaluate = max(1, int(budget))
        run = 1
        self.history = np.zeros(evaluate, dtype=float)

        all_run_best_state = None
        all_run_best_rank = None

        for run_id in range(run):
            self._reset_run()
            current_state, current_values, current_score, current_rank = (
                self._initial_solution(P, evaluate)
            )
            best_state = current_state.copy()
            best_values = current_values.copy()
            best_score = current_score
            best_rank = current_rank
            history_end = self._append_history(0, best_score)

            segment_destroy_score = np.zeros(len(self.DESTROY_OPERATORS))
            segment_repair_score = np.zeros(len(self.REPAIR_OPERATORS))
            segment_destroy_use = np.zeros(len(self.DESTROY_OPERATORS))
            segment_repair_use = np.zeros(len(self.REPAIR_OPERATORS))
            iteration = 0

            while self.evatime < evaluate:
                iteration += 1
                destroy_id = self._roulette(self.destroy_weights)
                repair_id = self._roulette(self.repair_weights)
                destroy_name = self.DESTROY_OPERATORS[destroy_id]
                repair_name = self.REPAIR_OPERATORS[repair_id]

                candidate = current_state.copy()
                decoded_candidate = candidate.decode(P)
                destroyed = self._destroy(
                    P,
                    candidate,
                    decoded_candidate,
                    destroy_name,
                )
                candidate = self._repair(P, candidate, destroyed, repair_name)
                values, score, rank = self._evaluate(P, candidate)

                previous_score = current_score
                accepted = self._accept(
                    current_rank,
                    current_score,
                    rank,
                    score,
                    self._temperature(self.evatime, evaluate),
                )
                new_global_best = rank > best_rank
                improved_current = rank > current_rank

                if accepted:
                    current_state = candidate.copy()
                    current_values = values.copy()
                    current_score = score
                    current_rank = rank

                if new_global_best:
                    best_state = candidate.copy()
                    best_values = values.copy()
                    best_score = score
                    best_rank = rank

                reward = 0.0
                if new_global_best:
                    reward = self.score_global_best
                elif improved_current:
                    reward = self.score_improved
                elif accepted:
                    reward = self.score_accepted

                segment_destroy_score[destroy_id] += reward
                segment_repair_score[repair_id] += reward
                segment_destroy_use[destroy_id] += 1
                segment_repair_use[repair_id] += 1
                self.operator_history.append(
                    {
                        "iteration": iteration,
                        "destroy": destroy_name,
                        "repair": repair_name,
                        "destroyed": len(destroyed),
                        "accepted": bool(accepted),
                        "improved": bool(score > previous_score),
                        "global_best": bool(new_global_best),
                        "score": float(score),
                    }
                )

                if iteration % self.segment_length == 0:
                    self.destroy_weights = self._update_weights(
                        self.destroy_weights,
                        segment_destroy_score,
                        segment_destroy_use,
                    )
                    self.repair_weights = self._update_weights(
                        self.repair_weights,
                        segment_repair_score,
                        segment_repair_use,
                    )
                    segment_destroy_score.fill(0)
                    segment_repair_score.fill(0)
                    segment_destroy_use.fill(0)
                    segment_repair_use.fill(0)

                history_end = self._append_history(history_end, best_score)
                self.on_iteration_finish(
                    problem=P,
                    state=best_state,
                    iteration=iteration - 1,
                    run=run_id,
                    Name=self.name,
                    time_cost=None,
                    stop=False,
                    scale=3,
                    best_value=best_values.copy(),
                    fitness=best_score,
                    destroy_operator=destroy_name,
                    repair_operator=repair_name,
                    accepted=accepted,
                )

            if all_run_best_rank is None or best_rank > all_run_best_rank:
                all_run_best_rank = best_rank
                all_run_best_state = best_state.copy()

        self.history /= run
        return (
            None
            if all_run_best_state is None
            else all_run_best_state.decode(P)
        )

    def _evaluate_state(self, P, state):
        values, _, _ = self._evaluate(P, state)
        return values

    # ------------------------------------------------------------------
    # 破壞操作
    # ------------------------------------------------------------------
    def _destroy(self, P, coding, state, operator_name):
        active = np.flatnonzero(np.asarray(state.levels) > 0)
        candidates = active if len(active) else np.arange(P.SENSOR_NUMBER)
        count = self._destroy_count(len(candidates))

        if operator_name == "random":
            selected = self.rng.choice(candidates, size=count, replace=False)
        elif operator_name == "low_energy":
            cost = P.calculate_total_cost(state)
            remaining = np.asarray(P.energy, dtype=float) - np.asarray(
                cost, dtype=float
            )
            selected = candidates[np.argsort(remaining[candidates])[:count]]
        elif operator_name == "coverage_overlap":
            selected = self._overlap_sensors(P, state, candidates, count)
        elif operator_name == "route_bottleneck":
            loads = np.asarray(state.tx_load)
            selected = candidates[np.argsort(loads[candidates])[::-1][:count]]
        else:
            raise ValueError("未知的破壞操作：" + operator_name)

        selected = np.asarray(selected, dtype=int)
        for sensor_id in selected:
            coding.code[sensor_id * 2] = 0
            coding.code[sensor_id * 2 + 1] = int(
                self.rng.integers(
                    max(P.LEVEL, SensorEncoding.RANK_PRECISION)
                )
            )
        return selected.tolist()

    def _overlap_sensors(self, P, state, candidates, count):
        coverage_count = np.zeros(P.TARGET_NUMBER, dtype=int)
        sensor_targets = {}
        for sensor_id in candidates:
            level = int(state.levels[sensor_id])
            targets = np.flatnonzero(
                P.coverage_table[:, sensor_id, level] > 0
            )
            sensor_targets[int(sensor_id)] = targets
            coverage_count[targets] += 1

        scores = []
        for sensor_id in candidates:
            targets = sensor_targets[int(sensor_id)]
            redundant = int(np.sum(np.maximum(coverage_count[targets] - 1, 0)))
            scores.append((redundant, int(sensor_id)))
        scores.sort(reverse=True)
        return np.asarray([sensor_id for _, sensor_id in scores[:count]], dtype=int)

    # ------------------------------------------------------------------
    # 修復操作
    # ------------------------------------------------------------------
    def _repair(self, P, coding, destroyed, operator_name):
        gene_domain = max(P.LEVEL, SensorEncoding.RANK_PRECISION)
        for sensor_id in destroyed:
            coding.code[sensor_id * 2 + 1] = int(
                self.rng.integers(gene_domain)
            )

        attempted = set()
        max_attempts = max(1, P.TARGET_NUMBER * 3)
        for _ in range(max_attempts):
            state = coding.decode(P)
            uncovered = list(P.find_uncovered_targets(state.levels))
            if not uncovered:
                break

            target_id = min(
                uncovered,
                key=lambda target: len(P.cover_candidates[int(target)]),
            )
            choices = []
            for sensor_id, level in P.cover_candidates[int(target_id)]:
                key = (int(target_id), int(sensor_id), int(level))
                if key in attempted or P.energy[int(sensor_id)] < P.liveJ:
                    continue
                choices.append((int(sensor_id), int(level)))

            if not choices:
                # 沒有新的可用候選時，保留隨機擾動以避免無限修復迴圈。
                break

            sensor_id, level = self._choose_repair_candidate(
                P, state, uncovered, choices, operator_name
            )
            attempted.add((int(target_id), sensor_id, level))
            coding.code[sensor_id * 2] = max(
                int(coding.code[sensor_id * 2])
                % P.sensing_option_count(sensor_id),
                level,
            )
            coding.code[sensor_id * 2 + 1] = int(
                self.rng.integers(gene_domain)
            )

        self._remove_redundant_sensors(P, coding)
        coding.decode(P)
        return coding

    def _choose_repair_candidate(self, P, state, uncovered, choices, operator_name):
        uncovered_set = set(int(target) for target in uncovered)
        max_energy = max(float(np.max(P.energy)), 1e-12)
        max_distance = max(
            float(np.max(P.distances[:, P.BSID])), 1e-12
        )
        scored = []

        for sensor_id, level in choices:
            covered = np.flatnonzero(
                P.coverage_table[:, sensor_id, level] > 0
            )
            gain = len(uncovered_set.intersection(int(target) for target in covered))
            energy = float(P.energy[sensor_id]) / max_energy
            near_base = (
                1.0
                - float(P.distances[sensor_id, P.BSID]) / max_distance
            )

            if operator_name == "coverage_efficiency":
                value = gain / max(1, level)
            elif operator_name == "energy_balance":
                value = 2.0 * energy + gain / max(1, level)
            elif operator_name == "route_aware":
                value = gain + energy + near_base
            else:
                raise ValueError("未知的修復操作：" + operator_name)
            scored.append((value, energy, -level, sensor_id, level))

        _, _, _, sensor_id, level = max(scored)
        return int(sensor_id), int(level)

    def _remove_redundant_sensors(self, P, coding):
        state = coding.decode(P)
        active = np.flatnonzero(np.asarray(state.levels) > 0)
        if len(active) <= 1:
            return

        coverage_count = np.zeros(P.TARGET_NUMBER, dtype=int)
        covered_by_sensor = {}
        for sensor_id in active:
            level = int(state.levels[sensor_id])
            covered = P.coverage_table[:, sensor_id, level] > 0
            covered_by_sensor[int(sensor_id)] = covered
            coverage_count += covered.astype(int)

        # 優先嘗試移除負載高且覆蓋貢獻小的感測器。
        loads = np.asarray(state.tx_load)
        order = sorted(
            (int(sensor_id) for sensor_id in active),
            key=lambda sensor_id: (
                np.sum(covered_by_sensor[sensor_id]),
                -loads[sensor_id],
            ),
        )
        for sensor_id in order:
            covered = covered_by_sensor[sensor_id]
            if np.any(covered) and np.all(coverage_count[covered] >= 2):
                coverage_count[covered] -= 1
                coding.code[sensor_id * 2] = 0

    # ------------------------------------------------------------------
    # 自適應選擇、接受準則與評估
    # ------------------------------------------------------------------
    def _initial_solution(self, P, evaluate):
        sample_count = min(self.initial_samples, evaluate)
        best = None
        for _ in range(sample_count):
            coding = SensorEncoding.random(
                P.SENSOR_NUMBER,
                P.radius_option_counts,
            )
            values, score, rank = self._evaluate(P, coding)
            if best is None or rank > best[3]:
                best = (coding.copy(), values.copy(), score, rank)
        return best

    def _evaluate(self, P, coding):
        state = coding.decode(P)
        values = np.asarray(P.evaluate_state(state), dtype=float)
        values = np.nan_to_num(values, nan=-5.0, neginf=-5.0, posinf=5.0)
        score = float(np.sum(values))
        rank = self._selection_rank(P, state, score)
        self.evatime += 1
        return values, score, rank

    def _selection_rank(self, P, state, score):
        if not self.feasibility_first:
            return (float(score),)
        cost = P.calculate_total_cost(state)
        energy_failures = len(P.find_energy_failed_sensors(state, cost))
        disconnected = len(P.find_disconnected(state))
        uncovered = len(P.find_uncovered_targets(state.levels))
        total_violations = energy_failures + disconnected + uncovered
        return (
            int(total_violations == 0),
            -total_violations,
            -uncovered,
            -disconnected,
            -energy_failures,
            float(score),
        )

    def _accept(
        self,
        current_rank,
        current_score,
        candidate_rank,
        candidate_score,
        temperature,
    ):
        if candidate_rank >= current_rank:
            return True
        # 不跨越可行性等級接受較差解；只對相同限制狀態使用模擬退火。
        if tuple(candidate_rank[:-1]) != tuple(current_rank[:-1]):
            return False
        scale = max(abs(current_score), 1.0)
        exponent = (candidate_score - current_score) / max(temperature * scale, 1e-12)
        return bool(self.rng.random() < math.exp(max(-700.0, exponent)))

    def _temperature(self, evaluations, evaluation_limit):
        progress = min(1.0, evaluations / max(1, evaluation_limit))
        ratio = self.final_temperature / self.initial_temperature
        return self.initial_temperature * (ratio**progress)

    def _roulette(self, weights):
        probabilities = np.asarray(weights, dtype=float)
        probabilities = probabilities / np.sum(probabilities)
        return int(self.rng.choice(len(probabilities), p=probabilities))

    def _update_weights(self, weights, scores, uses):
        updated = np.asarray(weights, dtype=float).copy()
        for index in range(len(updated)):
            if uses[index] <= 0:
                continue
            observed = scores[index] / uses[index]
            updated[index] = (
                (1.0 - self.reaction_factor) * updated[index]
                + self.reaction_factor * observed
            )
        return np.maximum(updated, 0.05)

    def _destroy_count(self, candidate_count):
        ratio = self.rng.uniform(self.min_destroy_ratio, self.max_destroy_ratio)
        return min(candidate_count, max(1, int(math.ceil(candidate_count * ratio))))

    def _append_history(self, history_end, best_score):
        new_end = min(self.evatime, len(self.history))
        self.history[history_end:new_end] += best_score
        return new_end

    def _reset_run(self):
        self.evatime = 0
        self.destroy_weights = np.ones(len(self.DESTROY_OPERATORS), dtype=float)
        self.repair_weights = np.ones(len(self.REPAIR_OPERATORS), dtype=float)
        self.operator_history = []
