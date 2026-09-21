"""CMA-MAE 在可調式感測範圍 WSN 上的實作。

核心流程依照 Fontaine 與 Nikolaidis（GECCO 2023）的 CMA-MAE：

1. 多個獨立 CMA-ES emitter 共用一個 soft archive。
2. 候選解以 ``f(theta) - t_e`` 排名，並採用 μ-selection。
3. 每個 archive cell 具有獨立接受門檻 ``t_e``。
4. 門檻依論文附錄 H 的批次、順序不變公式更新。
5. emitter 僅依 CMA-ES 的 basic convergence rule 重啟，重啟中心由
   soft archive 中隨機選取。

WSN 適配只發生在解碼、目標函數與行為描述子。搜尋空間仍是連續高斯
向量，經固定常態 CDF 映射為 TargetCodingState 的 0 到 9999 整數基因。
"""

import math
from statistics import NormalDist
from dataclasses import dataclass

import numpy as np

try:
    from scipy.special import ndtr as _scipy_ndtr
    from scipy.special import ndtri as _scipy_ndtri
except ImportError:  # pragma: no cover - wsnenv 已包含 SciPy
    _scipy_ndtr = None
    _scipy_ndtri = None

from Algorithm.core.Algorithm import Algorithm
from State.TargetEncoding import TargetEncoding


def _normal_cdf(values):
    """標準常態 CDF；沒有 SciPy 時使用向量化近似式。"""

    values = np.asarray(values, dtype=float)
    if _scipy_ndtr is not None:
        return _scipy_ndtr(values)

    # Abramowitz-Stegun 7.1.26，最大誤差約 7.5e-8。
    absolute = np.abs(values) / math.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * absolute)
    polynomial = (
        (
            (
                (
                    1.061405429 * t
                    - 1.453152027
                )
                * t
                + 1.421413741
            )
            * t
            - 0.284496736
        )
        * t
        + 0.254829592
    ) * t
    erf_approximation = np.sign(values) * (
        1.0 - polynomial * np.exp(-(absolute**2))
    )
    return 0.5 * (1.0 + erf_approximation)


def _normal_ppf(probabilities):
    """標準常態反 CDF；fallback 僅用於 warm start。"""

    probabilities = np.asarray(probabilities, dtype=float)
    if _scipy_ndtri is not None:
        return _scipy_ndtri(probabilities)
    normal = NormalDist()
    return np.asarray(
        [normal.inv_cdf(float(value)) for value in probabilities], dtype=float
    )


@dataclass
class _ArchiveElite:
    """Archive cell 中的解與其評估資料。"""

    theta: np.ndarray
    coding: TargetEncoding
    objective: float
    measures: np.ndarray
    fitness: np.ndarray
    feasible: bool
    violations: int

    def copy(self):
        return _ArchiveElite(
            theta=self.theta.copy(),
            coding=self.coding.copy(),
            objective=float(self.objective),
            measures=self.measures.copy(),
            fitness=self.fitness.copy(),
            feasible=bool(self.feasible),
            violations=int(self.violations),
        )


class _SoftGridArchive:
    """具有退火接受門檻的稀疏 GridArchive。"""

    def __init__(self, dims, learning_rate, threshold_min):
        self.dims = tuple(max(1, int(value)) for value in dims)
        if len(self.dims) != 2:
            raise ValueError("CMA-MAE 的 WSN archive 必須具有兩個維度")
        self.learning_rate = float(learning_rate)
        self.threshold_min = float(threshold_min)
        self.thresholds = {}
        self.elites = {}

    def index_of(self, measures):
        values = np.clip(np.asarray(measures, dtype=float), 0.0, 1.0)
        return tuple(
            min(size - 1, int(math.floor(value * size)))
            for value, size in zip(values, self.dims)
        )

    def threshold_of(self, index):
        return float(self.thresholds.get(index, self.threshold_min))

    def add_batch(self, entries):
        """依 Appendix H 同時更新一整批候選解。

        所有 improvement 都使用批次開始前的門檻。若同一 cell 有 c 個
        候選超過舊門檻，則：

        t'_e = (1-alpha)^c t_e
               + mean(f) [1-(1-alpha)^c]
        """

        indices = [self.index_of(entry["measures"]) for entry in entries]
        old_thresholds = np.asarray(
            [self.threshold_of(index) for index in indices], dtype=float
        )
        objectives = np.asarray(
            [float(entry["objective"]) for entry in entries], dtype=float
        )
        improvements = objectives - old_thresholds

        accepted_by_cell = {}
        for position, (index, objective, threshold) in enumerate(
            zip(indices, objectives, old_thresholds)
        ):
            if objective > threshold:
                accepted_by_cell.setdefault(index, []).append(position)

        for index, positions in accepted_by_cell.items():
            old_threshold = self.threshold_of(index)
            accepted_objectives = objectives[positions]
            count = len(positions)
            decay = (1.0 - self.learning_rate) ** count
            self.thresholds[index] = float(
                decay * old_threshold
                + (1.0 - decay) * float(np.mean(accepted_objectives))
            )

            # Soft archive 不保證保留歷史最佳解；同一批次則保留該 cell
            # 中 objective 最大的可接受候選，避免批次順序造成差異。
            best_position = max(
                positions, key=lambda position: entries[position]["objective"]
            )
            self.elites[index] = entries[best_position]["elite"].copy()

        return improvements

    def random_theta(self, rng, fallback):
        if not self.elites:
            return np.asarray(fallback, dtype=float).copy()
        indices = tuple(self.elites)
        index = indices[int(rng.integers(0, len(indices)))]
        return self.elites[index].theta.copy()


class _ResultGridArchive:
    """獨立保存每個 cell 的歷史最佳解，供最終比較使用。"""

    def __init__(self, dims):
        self.dims = tuple(max(1, int(value)) for value in dims)
        self.elites = {}

    def index_of(self, measures):
        values = np.clip(np.asarray(measures, dtype=float), 0.0, 1.0)
        return tuple(
            min(size - 1, int(math.floor(value * size)))
            for value, size in zip(values, self.dims)
        )

    def add_batch(self, entries):
        for entry in entries:
            elite = entry["elite"]
            if not elite.feasible:
                continue
            index = self.index_of(elite.measures)
            incumbent = self.elites.get(index)
            if incumbent is None or elite.objective > incumbent.objective:
                self.elites[index] = elite.copy()

    def best(self):
        if not self.elites:
            return None
        return max(self.elites.values(), key=lambda elite: elite.objective).copy()


class _CMAESEmitter:
    """CMA-MAE 使用的標準 rank-μ CMA-ES emitter。"""

    def __init__(self, mean, sigma, batch_size, rng):
        self.initial_mean = np.asarray(mean, dtype=float).copy()
        self.initial_sigma = float(sigma)
        self.batch_size = max(2, int(batch_size))
        self.rng = rng
        self.dimension = len(self.initial_mean)
        self.mu = max(1, self.batch_size // 2)

        raw_weights = np.log(self.mu + 0.5) - np.log(
            np.arange(1, self.mu + 1, dtype=float)
        )
        self.weights = raw_weights / np.sum(raw_weights)
        self.mu_eff = 1.0 / float(np.sum(self.weights**2))

        n = float(self.dimension)
        self.c_sigma = (self.mu_eff + 2.0) / (n + self.mu_eff + 5.0)
        self.d_sigma = (
            1.0
            + 2.0 * max(0.0, math.sqrt((self.mu_eff - 1.0) / (n + 1.0)) - 1.0)
            + self.c_sigma
        )
        self.c_c = (4.0 + self.mu_eff / n) / (
            n + 4.0 + 2.0 * self.mu_eff / n
        )
        self.c_1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(
            1.0 - self.c_1,
            2.0
            * (self.mu_eff - 2.0 + 1.0 / self.mu_eff)
            / ((n + 2.0) ** 2 + self.mu_eff),
        )
        self.chi_n = math.sqrt(n) * (
            1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n)
        )
        self.reset(self.initial_mean)

    def reset(self, mean):
        self.mean = np.asarray(mean, dtype=float).copy()
        self.sigma = self.initial_sigma
        self.covariance = np.eye(self.dimension, dtype=float)
        self.path_sigma = np.zeros(self.dimension, dtype=float)
        self.path_c = np.zeros(self.dimension, dtype=float)
        self.eigenvectors = np.eye(self.dimension, dtype=float)
        self.axis_scales = np.ones(self.dimension, dtype=float)
        self.inverse_sqrt_covariance = np.eye(self.dimension, dtype=float)
        self.generation = 0
        self.condition_number = 1.0

    def ask(self, count=None):
        count = self.batch_size if count is None else max(1, int(count))
        normal_samples = self.rng.standard_normal((count, self.dimension))
        transformed = (
            normal_samples * self.axis_scales
        ) @ self.eigenvectors.T
        return self.mean + self.sigma * transformed

    def tell(self, solutions, improvements):
        """依 improvement ranking 執行 μ-selection 與 CMA-ES 更新。"""

        solutions = np.asarray(solutions, dtype=float)
        improvements = np.asarray(improvements, dtype=float)
        if len(solutions) < self.mu:
            return

        order = np.argsort(-improvements, kind="stable")
        selected = solutions[order[: self.mu]]
        old_mean = self.mean.copy()
        normalized_steps = (selected - old_mean) / max(self.sigma, 1e-300)
        weighted_step = np.sum(
            self.weights[:, np.newaxis] * normalized_steps, axis=0
        )
        self.mean = old_mean + self.sigma * weighted_step

        path_scale = math.sqrt(
            self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff
        )
        self.path_sigma = (
            (1.0 - self.c_sigma) * self.path_sigma
            + path_scale * (self.inverse_sqrt_covariance @ weighted_step)
        )

        self.generation += 1
        path_correction = math.sqrt(
            max(1e-300, 1.0 - (1.0 - self.c_sigma) ** (2 * self.generation))
        )
        normalized_path = np.linalg.norm(self.path_sigma) / path_correction
        h_sigma = float(
            normalized_path
            < (1.4 + 2.0 / (self.dimension + 1.0)) * self.chi_n
        )
        self.path_c = (
            (1.0 - self.c_c) * self.path_c
            + h_sigma
            * math.sqrt(self.c_c * (2.0 - self.c_c) * self.mu_eff)
            * weighted_step
        )

        rank_mu = np.einsum(
            "i,ij,ik->jk", self.weights, normalized_steps, normalized_steps
        )
        old_factor = (
            1.0
            - self.c_1
            - self.c_mu
            + (1.0 - h_sigma) * self.c_1 * self.c_c * (2.0 - self.c_c)
        )
        self.covariance = (
            old_factor * self.covariance
            + self.c_1 * np.outer(self.path_c, self.path_c)
            + self.c_mu * rank_mu
        )
        self.covariance = 0.5 * (
            self.covariance + self.covariance.T
        )
        self.sigma *= math.exp(
            (self.c_sigma / self.d_sigma)
            * (np.linalg.norm(self.path_sigma) / self.chi_n - 1.0)
        )
        self._update_eigensystem()

    def _update_eigensystem(self):
        eigenvalues, eigenvectors = np.linalg.eigh(self.covariance)
        eigenvalues = np.maximum(eigenvalues, 1e-30)
        self.eigenvectors = eigenvectors
        self.axis_scales = np.sqrt(eigenvalues)
        self.inverse_sqrt_covariance = (
            eigenvectors
            @ np.diag(1.0 / self.axis_scales)
            @ eigenvectors.T
        )
        self.condition_number = float(
            np.max(eigenvalues) / np.min(eigenvalues)
        )

    def has_converged(self, improvements):
        """CMA-ES basic convergence rule，不使用 no-improvement restart。"""

        if not np.isfinite(self.sigma) or self.sigma <= 0.0:
            return True
        if self.condition_number > 1e14:
            return True
        if self.sigma * float(np.max(self.axis_scales)) < 1e-11:
            return True

        # no-effect coordinate 與 no-effect axis 是標準 CMA-ES 停止條件。
        coordinate_step = 0.2 * self.sigma * np.sqrt(
            np.maximum(np.diag(self.covariance), 0.0)
        )
        if np.any(self.mean == self.mean + coordinate_step):
            return True
        axis_id = self.generation % self.dimension
        axis_step = (
            0.1
            * self.sigma
            * self.axis_scales[axis_id]
            * self.eigenvectors[:, axis_id]
        )
        if np.all(self.mean == self.mean + axis_step):
            return True

        values = np.asarray(improvements, dtype=float)
        return bool(
            len(values) > 1
            and np.all(np.isfinite(values))
            and float(np.max(values) - np.min(values)) < 1e-12
        )


class CMA_MAE(Algorithm):
    """以 TargetCodingState 搜尋 WSN 排程與路由的 CMA-MAE。"""

    def __init__(
        self,
        P,
        num_emitters=15,
        batch_size=36,
        sigma=0.2,
        archive_dims=(100, 100),
        learning_rate=0.01,
        threshold_min=0.0,
        seed=None,
    ):
        super().__init__()
        self.num_emitters = max(1, int(num_emitters))
        self.batch_size = max(2, int(batch_size))
        self.sigma = max(float(sigma), 1e-12)
        self.archive_dims = tuple(int(value) for value in archive_dims)
        self.learning_rate = float(learning_rate)
        self.threshold_min = float(threshold_min)
        if not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError("learning_rate 必須介於 0 與 1 之間")

        self.seed = seed
        self.master_rng = np.random.default_rng(seed)
        self.length = int(P.TARGET_NUMBER)
        self.name = "CMA_MAE_WSN"
        self.evatime = 0
        self.history = np.array([], dtype=float)
        self.generation = 0
        self.restart_count = 0
        self.archive = None
        self.result_archive = None
        self.emitters = []
        self.best_entry = None
        self.device_name = "cpu"
        self.gpu_accelerated = False
        P.prepare_coding_cache()

    def _initial_mean(self, sch_s=None):
        if sch_s is None or getattr(sch_s, "code", None) is None:
            return np.zeros(self.length, dtype=float)
        code = np.asarray(sch_s.code)
        if len(code) != self.length:
            return np.zeros(self.length, dtype=float)
        probability = (np.clip(code.astype(float), 0.0, 9999.0) + 0.5) / 10000.0
        return self.sigma * _normal_ppf(probability)

    def _coding_from_theta(self, P, theta):
        # 除以初始 σ 後套用常態 CDF，使第一代 N(0, σ²I) 可均勻探索
        # TargetCodingState 的整數 random-key 範圍。
        probability = _normal_cdf(np.asarray(theta, dtype=float) / self.sigma)
        code = np.floor(probability * 10000.0).astype(int)
        return TargetEncoding(np.clip(code, 0, 9999))

    @staticmethod
    def _behavior_measures(P, state, cost):
        """WSN 的兩個 QD 行為描述子，皆正規化至 [0, 1]。"""

        sensor_count = max(1, int(P.SENSOR_NUMBER))
        active_ratio = (
            float(np.count_nonzero(state.levels > 0)) / sensor_count
        )
        positive_cost = np.maximum(np.asarray(cost, dtype=float), 0.0)
        total_cost = float(np.sum(positive_cost))
        if total_cost <= 1e-15:
            bottleneck_ratio = 0.0
        else:
            bottleneck_ratio = float(np.max(positive_cost) / total_cost)
        return np.clip(
            np.asarray([active_ratio, bottleneck_ratio], dtype=float), 0.0, 1.0
        )

    def _evaluate(self, P, theta):
        coding = self._coding_from_theta(P, theta)
        state = coding.decode(P)
        fitness = np.asarray(P.evaluate_state(state), dtype=float)
        cost = np.asarray(P.calculate_total_cost(state), dtype=float)
        uncovered = len(P.find_uncovered_targets(state.levels))
        disconnected = len(P.find_disconnected(state))
        energy_failed = len(P.find_energy_failed_sensors(state, cost))
        violations = int(uncovered + disconnected + energy_failed)
        feasible = violations == 0

        # 專案三項 fitness 皆為最大化且可行時非負。不可行解低於
        # threshold_min，故不會進入 archive，但違規數仍能引導 CMA 排名。
        objective = (
            float(np.sum(fitness))
            if feasible
            else self.threshold_min - float(violations)
        )
        measures = self._behavior_measures(P, state, cost)
        elite = _ArchiveElite(
            theta=np.asarray(theta, dtype=float).copy(),
            coding=coding,
            objective=objective,
            measures=measures,
            fitness=fitness,
            feasible=feasible,
            violations=violations,
        )
        entry = {
            "elite": elite,
            "objective": objective,
            "measures": measures,
        }
        self.evatime += 1
        self._update_best_entry(elite)
        best_score = (
            self.best_entry.objective
            if self.best_entry is not None and self.best_entry.feasible
            else 0.0
        )
        self.history[self.evatime - 1] = best_score
        return entry

    def _update_best_entry(self, elite):
        if self.best_entry is None:
            self.best_entry = elite.copy()
            return
        candidate_key = (
            int(elite.feasible),
            -elite.violations,
            elite.objective,
        )
        incumbent_key = (
            int(self.best_entry.feasible),
            -self.best_entry.violations,
            self.best_entry.objective,
        )
        if candidate_key > incumbent_key:
            self.best_entry = elite.copy()

    def _create_emitters(self, initial_mean):
        self.emitters = []
        for _ in range(self.num_emitters):
            emitter_seed = int(
                self.master_rng.integers(0, np.iinfo(np.uint32).max)
            )
            self.emitters.append(
                _CMAESEmitter(
                    mean=initial_mean,
                    sigma=self.sigma,
                    batch_size=self.batch_size,
                    rng=np.random.default_rng(emitter_seed),
                )
            )

    def search(self, P, budget, state=None):
        evaluate = max(1, int(budget))
        self.evatime = 0
        self.history = np.zeros(evaluate, dtype=float)
        self.generation = 0
        self.restart_count = 0
        self.best_entry = None
        self.archive = _SoftGridArchive(
            self.archive_dims, self.learning_rate, self.threshold_min
        )
        self.result_archive = _ResultGridArchive(self.archive_dims)
        initial_mean = self._initial_mean(None)
        self._create_emitters(initial_mean)

        while self.evatime < evaluate:
            solutions_by_emitter = []
            batch_entries = []
            remaining = evaluate - self.evatime
            for emitter in self.emitters:
                if remaining <= 0:
                    solutions_by_emitter.append(np.empty((0, self.length)))
                    continue
                count = min(self.batch_size, remaining)
                solutions = emitter.ask(count)
                solutions_by_emitter.append(solutions)
                for theta in solutions:
                    batch_entries.append(self._evaluate(P, theta))
                remaining -= count

            if not batch_entries:
                break

            improvements = self.archive.add_batch(batch_entries)
            self.result_archive.add_batch(batch_entries)
            offset = 0
            for emitter, solutions in zip(self.emitters, solutions_by_emitter):
                count = len(solutions)
                if count == 0:
                    continue
                emitter_improvements = improvements[offset : offset + count]
                emitter.tell(solutions, emitter_improvements)
                if emitter.has_converged(emitter_improvements):
                    restart_mean = self.archive.random_theta(
                        emitter.rng, initial_mean
                    )
                    emitter.reset(restart_mean)
                    self.restart_count += 1
                offset += count

            self.generation += 1
            state = (
                self.best_entry.coding.decode(P)
                if self.best_entry is not None
                else None
            )
            self.on_iteration_finish(
                problem=P,
                state=state,
                iteration=self.generation,
                run=0,
                Name="CMA-MAE",
                archive_size=len(self.archive.elites),
                result_archive_size=len(self.result_archive.elites),
                restarts=self.restart_count,
                fitness=(
                    self.best_entry.objective
                    if self.best_entry is not None
                    else self.threshold_min
                ),
            )

        result_best = self.result_archive.best()
        final_elite = result_best or self.best_entry
        if final_elite is None:
            raise RuntimeError("CMA-MAE 未產生任何候選解")
        if self.evatime < evaluate:
            self.history[self.evatime :] = self.history[max(0, self.evatime - 1)]

        print(
            "CMA-MAE end",
            "generation=",
            self.generation,
            "archive=",
            len(self.archive.elites),
            "result_archive=",
            len(self.result_archive.elites),
            "restarts=",
            self.restart_count,
            "evatime=",
            self.evatime,
        )
        return final_elite.coding.decode(P)


CMAMAE = CMA_MAE
