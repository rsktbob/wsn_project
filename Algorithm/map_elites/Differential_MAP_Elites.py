"""Differential MAP-Elites 在 WSN 排程與路由問題上的實作。

演算法核心依照 Choi 與 Togelius（GECCO 2021）：

* Algorithm 1：Differential MAP-Elites。
* Algorithm 2：以隨機樣本和 Lloyd 更新近似 CVT centroids。
* Algorithm 3：每個 CVT cell 只保留該行為區域中 fitness 最佳的解。
* reproduction：由四個不同非空 cell 取出 target、r1、r2、r3，
  執行 DE/rand/1 與 binomial crossover。

WSN 的連續 chromosome 位於 [0, 1]^T，T 為 target 數量，再以 random
key 轉為 TargetCodingState 的 0 到 9999 整數基因。
"""

import math
from dataclasses import dataclass

import numpy as np

try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - wsnenv 的 environment.yml 已包含 SciPy
    cKDTree = None

from Algorithm.core.Algorithm import Algorithm
from State.TargetEncoding import TargetEncoding


@dataclass
class _DMEElite:
    """CVT archive cell 中保存的 elite。"""

    vector: np.ndarray
    coding: TargetEncoding
    objective: float
    measures: np.ndarray
    fitness: np.ndarray
    feasible: bool
    violations: int

    def copy(self):
        return _DMEElite(
            vector=self.vector.copy(),
            coding=self.coding.copy(),
            objective=float(self.objective),
            measures=self.measures.copy(),
            fitness=self.fitness.copy(),
            feasible=bool(self.feasible),
            violations=int(self.violations),
        )


class _CVTArchive:
    """Differential MAP-Elites 使用的 CVT archive。"""

    def __init__(
        self,
        centroid_count,
        behavior_dim,
        sample_count,
        iterations,
        rng,
    ):
        self.centroid_count = max(1, int(centroid_count))
        self.behavior_dim = max(1, int(behavior_dim))
        self.sample_count = max(self.centroid_count, int(sample_count))
        self.iterations = max(1, int(iterations))
        self.rng = rng
        self.centroids = self._cvt_approximation()
        self._tree = (
            cKDTree(self.centroids) if cKDTree is not None else None
        )
        self.elites = {}
        self.additions = 0
        self.replacements = 0

    def _nearest_indices(self, points, centroids):
        points = np.asarray(points, dtype=float)
        if cKDTree is not None:
            tree = cKDTree(centroids)
            return np.asarray(tree.query(points, k=1)[1], dtype=int)

        # 無 SciPy 時採分批暴力搜尋，只供小型 smoke test 使用。
        result = np.empty(len(points), dtype=int)
        chunk_size = max(1, 100000 // max(1, len(centroids)))
        for start in range(0, len(points), chunk_size):
            stop = min(len(points), start + chunk_size)
            difference = (
                points[start:stop, np.newaxis, :]
                - centroids[np.newaxis, :, :]
            )
            distances = np.sum(difference * difference, axis=2)
            result[start:stop] = np.argmin(distances, axis=1)
        return result

    def _cvt_approximation(self):
        """實作論文 Algorithm 2 的 CVT approximation。"""

        centroids = self.rng.random(
            (self.centroid_count, self.behavior_dim)
        )
        samples = self.rng.random((self.sample_count, self.behavior_dim))
        for _ in range(self.iterations):
            assignments = self._nearest_indices(samples, centroids)
            sums = np.zeros_like(centroids)
            counts = np.bincount(
                assignments, minlength=self.centroid_count
            ).astype(float)
            for dimension in range(self.behavior_dim):
                sums[:, dimension] = np.bincount(
                    assignments,
                    weights=samples[:, dimension],
                    minlength=self.centroid_count,
                )
            occupied = counts > 0
            centroids[occupied] = (
                sums[occupied] / counts[occupied, np.newaxis]
            )
        return centroids

    def index_of(self, measures):
        measures = np.clip(
            np.asarray(measures, dtype=float), 0.0, 1.0
        )
        if self._tree is not None:
            return int(self._tree.query(measures, k=1)[1])
        distances = np.sum((self.centroids - measures) ** 2, axis=1)
        return int(np.argmin(distances))

    def add(self, elite):
        """實作 Algorithm 3：空 cell 或 fitness 較好時取代。"""

        index = self.index_of(elite.measures)
        incumbent = self.elites.get(index)
        if incumbent is None:
            self.elites[index] = elite.copy()
            self.additions += 1
            return True
        if elite.objective > incumbent.objective:
            self.elites[index] = elite.copy()
            self.replacements += 1
            return True
        return False

    def select_four(self, rng):
        if len(self.elites) < 4:
            return None
        indices = np.asarray(tuple(self.elites), dtype=int)
        selected = rng.choice(indices, size=4, replace=False)
        return [self.elites[int(index)] for index in selected]

    def best(self):
        if not self.elites:
            return None
        feasible = [elite for elite in self.elites.values() if elite.feasible]
        candidates = feasible or list(self.elites.values())
        return max(
            candidates,
            key=lambda elite: (
                int(elite.feasible),
                -elite.violations,
                elite.objective,
            ),
        ).copy()


class Differential_MAP_Elites(Algorithm):
    """以 CVT archive 和 DE/rand/1/bin 搜尋 WSN 解。"""

    def __init__(
        self,
        P,
        centroid_count=25000,
        initial_sample_ratio=0.01,
        minimum_initial_samples=100,
        scaling_factor=0.5,
        crossover_rate=0.9,
        cvt_sample_count=100000,
        cvt_iterations=5,
        seed=None,
    ):
        super().__init__()
        self.centroid_count = max(4, int(centroid_count))
        self.initial_sample_ratio = float(
            np.clip(initial_sample_ratio, 0.0, 1.0)
        )
        self.minimum_initial_samples = max(
            4, int(minimum_initial_samples)
        )
        self.scaling_factor = float(scaling_factor)
        self.crossover_rate = float(
            np.clip(crossover_rate, 0.0, 1.0)
        )
        self.cvt_sample_count = max(
            self.centroid_count, int(cvt_sample_count)
        )
        self.cvt_iterations = max(1, int(cvt_iterations))
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.length = int(P.TARGET_NUMBER)
        self.behavior_dim = 2
        self.name = "DIFFERENTIAL_MAP_ELITES_WSN"
        self.archive = None
        self.evatime = 0
        self.history = np.array([], dtype=float)
        self.iteration = 0
        self.best_elite = None
        self.device_name = "cpu"
        self.gpu_accelerated = False
        P.prepare_coding_cache()

    def _coding_from_vector(self, P, vector):
        vector = np.clip(np.asarray(vector, dtype=float), 0.0, 1.0)
        code = np.clip(
            np.floor(vector * 10000.0).astype(int), 0, 9999
        )
        return TargetEncoding(code)

    @staticmethod
    def _behavior_measures(P, state, cost):
        """與 CMA-MAE 相同的兩個行為描述子，以利公平比較。"""

        sensor_count = max(1, int(P.SENSOR_NUMBER))
        active_ratio = (
            float(np.count_nonzero(state.levels > 0)) / sensor_count
        )
        positive_cost = np.maximum(np.asarray(cost, dtype=float), 0.0)
        total_cost = float(np.sum(positive_cost))
        bottleneck_ratio = (
            0.0
            if total_cost <= 1e-15
            else float(np.max(positive_cost) / total_cost)
        )
        return np.clip(
            np.asarray([active_ratio, bottleneck_ratio], dtype=float),
            0.0,
            1.0,
        )

    def _evaluate(self, P, vector):
        coding = self._coding_from_vector(P, vector)
        state = coding.decode(P)
        fitness = np.asarray(P.evaluate_state(state), dtype=float)
        cost = np.asarray(P.calculate_total_cost(state), dtype=float)
        uncovered = len(P.find_uncovered_targets(state.levels))
        disconnected = len(P.find_disconnected(state))
        energy_failed = len(P.find_energy_failed_sensors(state, cost))
        violations = int(uncovered + disconnected + energy_failed)
        feasible = violations == 0

        # Differential MAP-Elites 本身是無限制式演算法。WSN 採用 scalar
        # feasibility-first：所有可行解都優於不可行解，不可行解之間則以
        # 較少違規者為佳。
        objective = (
            float(np.sum(fitness))
            if feasible
            else -1000.0 - float(violations)
        )
        elite = _DMEElite(
            vector=np.asarray(vector, dtype=float).copy(),
            coding=coding,
            objective=objective,
            measures=self._behavior_measures(P, state, cost),
            fitness=fitness,
            feasible=feasible,
            violations=violations,
        )
        self.evatime += 1
        self._update_best(elite)
        history_value = (
            self.best_elite.objective
            if self.best_elite is not None and self.best_elite.feasible
            else 0.0
        )
        self.history[self.evatime - 1] = history_value
        return elite

    def _update_best(self, elite):
        if self.best_elite is None:
            self.best_elite = elite.copy()
            return
        candidate = (
            int(elite.feasible),
            -elite.violations,
            elite.objective,
        )
        incumbent = (
            int(self.best_elite.feasible),
            -self.best_elite.violations,
            self.best_elite.objective,
        )
        if candidate > incumbent:
            self.best_elite = elite.copy()

    def _initial_sample_count(self, evaluate):
        # 論文 CEC 設定 G=n*100，且總預算為 10000*n，初始化占 1%。
        # 本專案固定 10000 evaluations，因此以相同比例縮放。
        ratio_count = int(math.ceil(evaluate * self.initial_sample_ratio))
        return min(
            evaluate,
            max(self.minimum_initial_samples, ratio_count, 4),
        )

    def _trial_vector(self, selected):
        target, donor_1, donor_2, donor_3 = selected
        mutant = donor_1.vector + self.scaling_factor * (
            donor_2.vector - donor_3.vector
        )
        mutant = np.clip(mutant, 0.0, 1.0)
        crossover_mask = self.rng.random(self.length) <= self.crossover_rate
        forced_dimension = int(self.rng.integers(0, self.length))
        crossover_mask[forced_dimension] = True
        return np.where(crossover_mask, mutant, target.vector)

    def search(self, P, budget, state=None):
        evaluate = max(1, int(budget))
        self.evatime = 0
        self.history = np.zeros(evaluate, dtype=float)
        self.iteration = 0
        self.best_elite = None
        self.archive = _CVTArchive(
            centroid_count=self.centroid_count,
            behavior_dim=self.behavior_dim,
            sample_count=self.cvt_sample_count,
            iterations=self.cvt_iterations,
            rng=self.rng,
        )

        # Algorithm 1 initialization：G 個均勻隨機 chromosome。
        initial_count = self._initial_sample_count(evaluate)
        for _ in range(initial_count):
            vector = self.rng.random(self.length)
            self.archive.add(self._evaluate(P, vector))

        # 若行為空間暫時少於四個非空 cell，繼續隨機初始化，直到可以
        # 按論文選出四個相異 cell 或評估預算耗盡。
        while self.evatime < evaluate and len(self.archive.elites) < 4:
            vector = self.rng.random(self.length)
            self.archive.add(self._evaluate(P, vector))

        # Algorithm 1 reproduction：每次產生一個 DE/rand/1/bin trial。
        while self.evatime < evaluate:
            selected = self.archive.select_four(self.rng)
            if selected is None:
                break
            trial = self._trial_vector(selected)
            self.archive.add(self._evaluate(P, trial))
            self.iteration += 1
            if self.iteration % 100 == 0 or self.evatime == evaluate:
                state = (
                    self.best_elite.coding.decode(P)
                    if self.best_elite is not None
                    else None
                )
                self.on_iteration_finish(
                    problem=P,
                    state=state,
                    iteration=self.iteration,
                    run=0,
                    Name="Differential-MAP-Elites",
                    archive_size=len(self.archive.elites),
                    additions=self.archive.additions,
                    replacements=self.archive.replacements,
                    fitness=(
                        self.best_elite.objective
                        if self.best_elite is not None
                        else -math.inf
                    ),
                )

        if self.evatime < evaluate:
            previous = self.history[max(0, self.evatime - 1)]
            self.history[self.evatime :] = previous

        final_elite = self.archive.best() or self.best_elite
        if final_elite is None:
            raise RuntimeError("Differential MAP-Elites 未產生候選解")
        print(
            "Differential-MAP-Elites end",
            "iterations=",
            self.iteration,
            "archive=",
            len(self.archive.elites),
            "additions=",
            self.archive.additions,
            "replacements=",
            self.archive.replacements,
            "evatime=",
            self.evatime,
        )
        return final_elite.coding.decode(P)


DifferentialMAPElites = Differential_MAP_Elites
DME = Differential_MAP_Elites
