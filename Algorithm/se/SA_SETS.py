"""Legacy multi-process SA-SETS implemented on the shared SE phase flow."""

from __future__ import annotations

import math

import numpy as np
from scipy.special import betainc as scipy_betainc

from Algorithm.se.BaseSE import BaseSE
from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding


def beta_cdf(a, b, x):
    """Calculate the regularized Beta CDF through the shared SciPy wrapper."""
    return float(scipy_betainc(float(a), float(b), float(x)))


class SA_SETS(BaseSE):
    """Search Economics with adaptive region beliefs and sensor coding.

    ``BaseSE`` owns the four stable SE phases.  This class owns everything
    specific to SA-SETS: identity-sensor classification, sensor chromosome
    creation, transition operators, and the Beta-based region probabilities.
    """

    def __init__(self, problem, n=5, h=4, w=2, mu=0.2, seed=None):
        code_length = problem.SENSOR_NUMBER * 2
        super().__init__(
            problem,
            n=n,
            h=h,
            w=w,
            mu=mu,
            code_length=code_length,
            seed=seed,
        )
        self.identity_bit_count = int(math.log2(self.h))
        if 2**self.identity_bit_count != self.h:
            raise ValueError("SA-SETS requires h to be a power of two")

        self.identity_sensors = []
        self.adaptive_step = 0.001
        self.current_adaptive_step = self.adaptive_step

    def select_identity_sensors(self, problem):
        """Select the two high-energy sensors near the base station."""
        distances = np.asarray(
            problem.distances[: problem.SENSOR_NUMBER, problem.BSID],
            dtype=float,
        )
        distance_order = getattr(
            problem,
            "sensor_ids_by_bs_distance",
            np.argsort(distances, kind="stable"),
        )
        boundary_id = int(distance_order[min(1, len(distance_order) - 1)])
        distance_limit = distances[boundary_id] + 15
        candidate_ids = distance_order[
            distances[distance_order] <= distance_limit
        ]

        selected = []
        energy = np.asarray(problem.energy[candidate_ids], dtype=float).copy()
        for _ in range(self.identity_bit_count):
            if not len(energy):
                break
            candidate_index = int(np.argmax(energy))
            if energy[candidate_index] < problem.liveJ:
                break
            selected.append(int(candidate_ids[candidate_index]))
            energy[candidate_index] = -np.inf

        if len(selected) == self.identity_bit_count:
            return selected

        # Preserve the original fallback: use the first live sensors when the
        # near-base classification cannot provide enough identities.
        return [
            int(sensor_id)
            for sensor_id in distance_order
            if problem.energy[sensor_id] >= problem.liveJ
        ][: self.identity_bit_count]

    def initialize_market(self, problem, initial_state=None):
        """Choose the classification sensors, then initialize the SE market."""
        self.identity_sensors = self.select_identity_sensors(problem)
        if len(self.identity_sensors) != self.identity_bit_count:
            raise RuntimeError("not enough live sensors for SA-SETS regions")
        super().initialize_market(problem, initial_state)

    def create_candidate(self, problem, region=None):
        """以隨機 chromosome 為底，依 CCS 為每個 target 放入一組覆蓋。"""
        candidate = SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
            rng=self.rng,
        )

        # 原始 SA-SETS 初始化：每個 target 從其 CCS 候選表隨機抽一組
        # (sensor, level)，直接寫入該 sensor 的 sensing gene。若後面的
        # target 再選到同一 sensor，則以後一次抽樣覆寫，保留原本的行為。
        # 路由 priority 仍由 SensorEncoding.random() 隨機產生。
        for target_candidates in problem.cover_candidates:
            if target_candidates:
                sensor_id, level = self.random.choice(target_candidates)
                candidate.code[int(sensor_id) * 2] = int(level)

        if region is not None:
            self.align_region(problem, candidate, region)
        return candidate

    def align_region(self, problem, candidate, region):
        """Force identity-sensor bits to represent one SA-SETS region."""
        bits = int(region)
        for bit_id, sensor_id in enumerate(self.identity_sensors):
            must_open = (bits >> bit_id) & 1
            gene_id = int(sensor_id) * 2
            option_count = problem.sensing_option_count(sensor_id)
            if must_open:
                if int(candidate.code[gene_id]) % option_count == 0:
                    candidate.code[gene_id] = self.random.randrange(
                        1, option_count
                    )
            else:
                candidate.code[gene_id] = 0
        return candidate

    def invest(self, problem, searcher, good):
        """Use one regional good to make an SA-SETS investment."""
        difference = 2
        midpoint = self.code_length // 2
        if midpoint <= difference or self.code_length - difference <= midpoint:
            investment = searcher.copy()
        else:
            left = self.random.randint(difference, midpoint - difference)
            right = self.random.randint(
                midpoint - difference,
                self.code_length - difference,
            )
            first, second = swap_segment(searcher, good, left, right)
            investment = (
                first if self.random.random() < 0.5 else second
            )
        if self.random.random() < self.mutation_rate:
            self.mutate_candidate(problem, investment)
        return investment

    def mutate_candidate(self, problem, candidate):
        """使用目前選定的 v1 規則突變；切換版本只需改這一行。"""
        return self.mutatev1(problem, candidate)

    def mutatev1(self, problem, candidate):
        """舊版：隨機重抽一至三組 sensing／priority 基因。"""
        mutation_count = self.random.choice([1] * 90 + [2] * 5 + [3] * 5)
        for _ in range(mutation_count):
            gene_id = self.random.randrange(self.code_length)
            sensing_id = gene_id if gene_id % 2 == 0 else gene_id - 1
            priority_id = sensing_id + 1
            operation = self.random.randint(0, 2)
            if operation in (0, 1):
                sensing_bound = problem.sensing_option_count(
                    sensing_id // 2
                )
                candidate.code[sensing_id] = self.random.randrange(
                    sensing_bound
                )
            if operation in (0, 2):
                candidate.code[priority_id] = self.random.randrange(
                    SensorEncoding.RANK_PRECISION
                )
        return candidate

    def mutatev2(self, problem, candidate):
        """新版：執行必定有效的 sensor 操作，縮小範圍後以 CCS 修補。"""
        mutation_count = self.random.choice([1] * 90 + [2] * 5 + [3] * 5)
        for _ in range(mutation_count):
            sensor_id = self.random.randrange(problem.SENSOR_NUMBER)
            level_id = sensor_id * 2
            priority_id = level_id + 1
            option_count = problem.sensing_option_count(sensor_id)
            current_level = int(candidate.code[level_id]) % option_count
            candidate.code[level_id] = current_level

            # 開啟中的 sensor 可改路由優先值或 sensing level；關閉中的
            # sensor 沒有可觀察的 priority，因此直接以非零 level 開啟。
            change_priority = (
                current_level > 0 and self.random.randrange(2) == 0
            )
            if change_priority:
                current_priority = (
                    int(candidate.code[priority_id])
                    % SensorEncoding.RANK_PRECISION
                )
                candidate.code[priority_id] = self._other_value(
                    current_priority,
                    SensorEncoding.RANK_PRECISION,
                )
                continue

            if option_count <= 1:
                continue
            new_level = self._other_value(current_level, option_count)
            candidate.code[level_id] = new_level
            if current_level == 0:
                # 新開啟的 sensor 不沿用可能無效的零 priority。
                candidate.code[priority_id] = self.random.randrange(
                    1,
                    SensorEncoding.RANK_PRECISION,
                )

            # 增大 sensing level 不可能失去覆蓋；只有縮小或關閉才修補。
            if new_level < current_level:
                self._repair_coverage(
                    problem,
                    candidate,
                    excluded_sensor=sensor_id,
                )
        return candidate

    def _other_value(self, current, option_count):
        """從 domain 中均勻抽出一個不同於目前值的選項。"""
        value = self.random.randrange(option_count - 1)
        return value + (value >= current)

    def _repair_coverage(self, problem, candidate, excluded_sensor):
        """以 target CCS 修補 sensing level 降低後出現的覆蓋缺口。"""
        sensor_ids = np.arange(problem.SENSOR_NUMBER)
        option_counts = np.asarray(problem.radius_option_counts, dtype=int)
        levels = np.asarray(candidate.code[0::2], dtype=int) % option_counts
        coverage_count = np.sum(
            problem.coverage_table[:, sensor_ids, levels] > 0,
            axis=1,
        )

        for target_id in np.where(coverage_count == 0)[0]:
            # 前一次修補可能已同時覆蓋目前 target。
            if coverage_count[target_id] > 0:
                continue
            candidates = [
                (int(sensor_id), int(level))
                for sensor_id, level in problem.CCS[int(target_id)]
                if int(level) > 0
            ]
            alternatives = [
                item for item in candidates if item[0] != excluded_sensor
            ]
            if alternatives:
                candidates = alternatives
            if not candidates:
                continue

            replacement_id, required_level = self.random.choice(candidates)
            replacement_gene = replacement_id * 2
            replacement_domain = problem.sensing_option_count(replacement_id)
            old_level = (
                int(candidate.code[replacement_gene]) % replacement_domain
            )
            new_level = max(old_level, required_level)
            if new_level == old_level:
                continue

            old_coverage = (
                problem.coverage_table[:, replacement_id, old_level] > 0
            )
            new_coverage = (
                problem.coverage_table[:, replacement_id, new_level] > 0
            )
            candidate.code[replacement_gene] = new_level
            coverage_count += new_coverage.astype(int)
            coverage_count -= old_coverage.astype(int)

            if old_level == 0:
                candidate.code[replacement_gene + 1] = self.random.randrange(
                    1,
                    SensorEncoding.RANK_PRECISION,
                )
        return candidate

    def vision_search(self, problem):
        """Update the adaptive learning step before the normal SE search."""
        progress = min(
            1.0,
            self.evatime / max(1, self.evaluation_limit),
        )
        self.current_adaptive_step = self.adaptive_step * (1.0 - progress)
        super().vision_search(problem)

    def region_probabilities(self, goods_fitness, investment_fitness):
        """Calculate the SA-SETS Beta probability for each region/searcher."""
        average_investment = np.mean(investment_fitness, axis=2)
        average_goods = np.mean(goods_fitness, axis=1)
        best_goods = np.max(goods_fitness, axis=1)
        total = float(np.sum(average_goods))
        if abs(total) <= np.finfo(float).eps:
            region_share = np.full(self.h, 1.0 / self.h)
        else:
            region_share = average_goods / total

        expected_value = (
            best_goods[:, None]
            * average_investment
            * region_share[:, None]
        )
        probability = np.empty((self.h, self.n), dtype=float)
        for region in range(self.h):
            for searcher_id in range(self.n):
                probability[region, searcher_id] = beta_cdf(
                    self.ta[region],
                    self.tb[region],
                    expected_value[region, searcher_id],
                )
        return probability

    def update_search_memory(self, problem, evaluation_start):
        """Adapt Beta beliefs and retain the best searcher of this round."""
        step = self.current_adaptive_step
        for selected_region in self.selected_regions:
            selected_region = int(selected_region)
            self.ta[selected_region] += step
            for region in range(self.h):
                if region != selected_region:
                    self.tb[region] += step

        history_end = min(int(self.evatime), len(self.history))
        if history_end > evaluation_start:
            self.history[evaluation_start:history_end] = self.fitness


__all__ = ["SA_SETS", "beta_cdf"]
