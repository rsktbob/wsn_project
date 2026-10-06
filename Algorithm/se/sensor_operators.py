"""SETS 系列共用的 sensor 編碼、分區對齊與交配突變。

這些元件只描述「解長什麼樣、怎麼變」，和市場流程無關，因此做成 mixin，
讓流程不同的家族（SETS／SA_SETS 的全區投資、SI_SETS 的選區投資）
不必互相繼承也能共用：

* ``SensorOperators``：identity sensor 分區、sensor chromosome、交配與突變。
* ``NearBaseIdentitySensors``：改選靠近 BS、高電量的 identity sensor。
* ``CCSInitialization``：依 target 的 CCS 候選表產生初始解。
"""

from __future__ import annotations

import math

import numpy as np

from State.Encoding import swap_segment
from State.SensorEncoding import SensorEncoding

# 每次突變改動的 sensor 數：90% 一組、5% 兩組、5% 三組。
MUTATION_COUNTS = [1] * 90 + [2] * 5 + [3] * 5


def draw_mutation_count(random):
    """抽出這次突變要改幾組基因。"""
    return random.choice(MUTATION_COUNTS)


class SensorOperators:
    """Identity-sensor regions plus the sensor chromosome's operators.

    Identity sensors are the first live sensors and initial chromosomes are
    uniformly random; the mixins below replace those two choices.
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
            raise ValueError("SETS requires h to be a power of two")

        self.identity_sensors = []

    def select_identity_sensors(self, problem):
        """Use the first live sensors as region identity sensors."""
        return [
            sensor_id
            for sensor_id in range(problem.SENSOR_NUMBER)
            if problem.energy[sensor_id] >= problem.liveJ
        ][: self.identity_bit_count]

    def initialize_market(self, problem, initial_state=None):
        """Choose the classification sensors, then initialize the SE market."""
        self.identity_sensors = self.select_identity_sensors(problem)
        if len(self.identity_sensors) != self.identity_bit_count:
            raise RuntimeError("not enough live sensors for SETS regions")
        super().initialize_market(problem, initial_state)

    def create_candidate(self, problem, region=None):
        """Uniformly random sensor chromosome aligned to ``region``."""
        candidate = SensorEncoding.random(
            problem.SENSOR_NUMBER,
            problem.radius_option_counts,
            rng=self.rng,
        )
        if region is not None:
            self.align_region(problem, candidate, region)
        return candidate

    def align_region(self, problem, candidate, region):
        """Force identity-sensor bits to represent one SETS region."""
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

    def crossover_points(self):
        """Pick one cut left of and one right of the chromosome midpoint.

        Returns ``None`` when the chromosome is too short to cut.
        """
        difference = 2
        midpoint = self.code_length // 2
        if midpoint <= difference or self.code_length - difference <= midpoint:
            return None
        left = self.random.randint(difference, midpoint - difference)
        right = self.random.randint(
            midpoint - difference,
            self.code_length - difference,
        )
        return left, right

    def crossover(self, searcher, good):
        """Swap one midpoint-spanning segment; return both children."""
        points = self.crossover_points()
        if points is None:
            return searcher.copy(), good.copy()
        return swap_segment(searcher, good, *points)

    def invest(self, problem, searcher, good):
        """Use one regional good to make a single-child investment."""
        points = self.crossover_points()
        if points is None:
            investment = searcher.copy()
        else:
            first, second = swap_segment(searcher, good, *points)
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
        mutation_count = draw_mutation_count(self.random)
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
        mutation_count = draw_mutation_count(self.random)
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
                for sensor_id, level in problem.cover_candidates[int(target_id)]
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


class NearBaseIdentitySensors:
    """Pick identity sensors among high-energy sensors near the base station."""

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


class CCSInitialization:
    """Seed each initial chromosome with one CCS cover per target."""

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


__all__ = [
    "CCSInitialization",
    "MUTATION_COUNTS",
    "NearBaseIdentitySensors",
    "SensorOperators",
    "draw_mutation_count",
]
