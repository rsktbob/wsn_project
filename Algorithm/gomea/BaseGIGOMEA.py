"""用於聯合感測排程與路由的 Gene-Invariant GOMEA。

本模組獨立於 SA-SETS。核心變異依照 GI-GOMEA 論文，直接交換族群中
兩個解的 linkage set；接受或拒絕都以一整對解為單位，因此每個基因位置
的等位值數量不會因演化而消失。
"""

import heapq
import math
import numpy as np

from Algorithm.core.Algorithm import Algorithm
from Algorithm.gomea.cuda_nmi import CudaNMI, CudaUnavailableError
from State.SensorEncoding import SensorEncoding


class BaseGIGOMEA(Algorithm):
    """圖結構引導的多值離散 GI-GOMEA。"""

    def __init__(
        self,
        P,
        population_size=24,
        elite_fraction=0.5,
        statistical_weight=0.7,
        graph_weight=0.3,
        max_linkage_size=16,
        max_linkage_sets=160,
        seed=None,
        device="cpu",
        cuda_device=0,
        require_cuda=False,
    ):
        """初始化 GI-GOMEA 參數、基因定義與 WSN 結構關聯矩陣。

        ``statistical_weight`` 與 ``graph_weight`` 分別控制從目前族群學到的
        統計關聯，以及由覆蓋/通訊拓樸事先建立的結構關聯。
        """
        super().__init__(seed=seed)
        self.population_size = max(4, int(population_size))
        self.elite_fraction = float(np.clip(elite_fraction, 0.1, 1.0))
        self.statistical_weight = max(0.0, float(statistical_weight))
        self.graph_weight = max(0.0, float(graph_weight))
        # 將兩種關聯來源正規化，避免使用者給定的權重總和影響相似度尺度。
        weight_sum = self.statistical_weight + self.graph_weight
        if weight_sum <= 0.0:
            self.statistical_weight = 1.0
            self.graph_weight = 0.0
        else:
            self.statistical_weight /= weight_sum
            self.graph_weight /= weight_sum
        self.max_linkage_size = max(2, int(max_linkage_size))
        self.max_linkage_sets = max(4, int(max_linkage_sets))
        self.state_type = "coding"

        # 子類別可覆寫以下方法，以支援 Sensor、Target 或 Critical 編碼。
        self.length = self._encoding_length(P)
        self.schedule_domain = int(P.LEVEL)
        self.schedule_domains = np.asarray(
            P.radius_option_counts, dtype=int
        ).copy()
        self.routing_domain = int(
            getattr(SensorEncoding, "RANK_PRECISION", 10)
        )
        self.target_domain = 10000
        self.name = "GI_GOMEA_WSN"
        self.evatime = 0
        self.history = np.array([], dtype=float)
        self.population = []
        self.last_linkage_model = []
        self.generation = 0
        self.accepted_swaps = 0
        self.rejected_swaps = 0
        self.no_change_swaps = 0
        self.device_requested = str(device).strip().lower()
        self.cuda_device = int(cuda_device)
        self.require_cuda = bool(require_cuda)
        self.device_name = "cpu"
        self.device_detail = "NumPy CPU"
        self.gpu_accelerated = False
        self.gpu_scope = None
        self.cuda_error = None
        self.nmi_gpu_calls = 0
        self.nmi_cpu_calls = 0
        self._nmi_backend = None
        self._best_quality = None
        self._last_evaluation_details = None
        self._configure_device()

        # 先完成 Problem 的幾何/連通預計算，才可建立固定的圖結構 DSM。
        P.prepare_coding_cache()
        self.graph_dsm = self._build_graph_dsm(P)

    def _configure_device(self):
        """依設定啟用 CUDA NMI；無 GPU 時可安全退回原本的 CPU 路徑。"""
        if self.device_requested not in {"cpu", "cuda", "auto"}:
            raise ValueError("device 必須是 'cpu'、'cuda' 或 'auto'。")
        if self.device_requested == "cpu":
            return

        try:
            self._nmi_backend = CudaNMI(self.cuda_device)
        except CudaUnavailableError as error:
            self.cuda_error = str(error)
            if self.require_cuda:
                raise
            self.device_detail = "NumPy CPU (CUDA unavailable)"
            return

        self.device_name = "cuda"
        self.device_detail = (
            f"cuda:{self._nmi_backend.info.device_id} "
            f"({self._nmi_backend.info.name})"
        )
        self.gpu_accelerated = True
        self.gpu_scope = "linkage_nmi"

    def _disable_cuda(self, error):
        """記錄執行期 CUDA 錯誤，非嚴格模式改由 CPU 完成本次與後續計算。"""
        self.cuda_error = str(error)
        if self.require_cuda:
            raise RuntimeError(
                f"CUDA NMI 執行失敗：{self.cuda_error}"
            ) from error
        self._nmi_backend = None
        self.device_name = "cpu"
        self.device_detail = "NumPy CPU (CUDA fallback)"
        self.gpu_accelerated = False
        self.gpu_scope = None

    def _new_coding(self, P, genes):
        """把已正規化的基因陣列包裝成 SensorEncoding。"""
        return SensorEncoding(genes)

    @staticmethod
    def _encoding_length(P):
        """回傳 Sensor 編碼長度：每顆感測器各有排程與路由兩個基因。"""
        return int(P.SENSOR_NUMBER) * 2

    def _domain_size(self, gene_id):
        """回傳指定基因的合法等位值數量。"""
        if gene_id % 2 == 0:
            return int(self.schedule_domains[gene_id // 2])
        return self.routing_domain

    def _canonicalize(self, code):
        """以各基因自己的 domain 將任意整數映射回合法範圍。"""
        canonical = np.asarray(code, dtype=int).copy()
        for gene_id in range(len(canonical)):
            canonical[gene_id] %= self._domain_size(gene_id)
        return canonical

    def _coding_from_code(self, P, code):
        """先修正基因範圍，再建立對應的編碼物件。"""
        return self._new_coding(P, self._canonicalize(code))

    def _probabilistically_complete_population(self, P, size):
        """以 PC sampling 建立每個多值基因皆近似均勻的初始族群。

        當 domain 不大於族群時，各等位值會循環出現；domain 太大時則採
        不重複抽樣，降低初始族群在單一數值附近聚集的機率。
        """
        codes = np.empty((size, self.length), dtype=int)
        for gene_id in range(self.length):
            domain = self._domain_size(gene_id)
            if domain <= size:
                values = np.arange(size, dtype=int) % domain
            else:
                # 大型 target gene domain 無法在小族群中列舉完全，改採不重複分層取樣。
                values = self.rng.choice(domain, size=size, replace=False)
            self.rng.shuffle(values)
            codes[:, gene_id] = values
        return [self._coding_from_code(P, code) for code in codes]

    @staticmethod
    def _entropy(values):
        """計算單一基因在族群中的 Shannon entropy。"""
        _, counts = np.unique(values, return_counts=True)
        probabilities = counts.astype(float) / max(1, np.sum(counts))
        return float(-np.sum(probabilities * np.log(probabilities)))

    @staticmethod
    def _joint_entropy(first, second):
        """計算兩個基因聯合分布的 Shannon entropy。"""
        pairs = np.column_stack((first, second))
        _, counts = np.unique(pairs, axis=0, return_counts=True)
        probabilities = counts.astype(float) / max(1, np.sum(counts))
        return float(-np.sum(probabilities * np.log(probabilities)))

    def _normalized_mutual_information_cpu(self, codes):
        """以原本 NumPy 流程建立 NMI，並作為 CUDA 的正確性基準。"""
        gene_count = codes.shape[1]
        entropies = np.array(
            [self._entropy(codes[:, gene_id]) for gene_id in range(gene_count)]
        )
        similarity = np.eye(gene_count, dtype=float)
        for first in range(gene_count):
            for second in range(first + 1, gene_count):
                joint = self._joint_entropy(codes[:, first], codes[:, second])
                # 聯合熵接近零代表樣本沒有變化，此時不宣稱兩基因有關聯。
                if joint <= 1e-15:
                    nmi = 0.0
                else:
                    mutual_information = entropies[first] + entropies[second] - joint
                    nmi = max(0.0, mutual_information / joint)
                similarity[first, second] = nmi
                similarity[second, first] = nmi
        return similarity

    def _normalized_mutual_information(self, codes):
        """依目前 backend，以 CUDA 或 NumPy 建立基因關聯矩陣。"""
        if self._nmi_backend is not None:
            try:
                similarity = self._nmi_backend.similarity(codes)
                self.nmi_gpu_calls += 1
                return similarity
            except Exception as error:
                self._disable_cuda(error)

        self.nmi_cpu_calls += 1
        return self._normalized_mutual_information_cpu(codes)

    def _build_graph_dsm(self, P):
        """由目標覆蓋重疊與可通訊關係建立 WSN 圖結構 DSM。

        排程基因的關聯來自共同可覆蓋目標，路由基因的關聯來自感測器間
        是否可通訊；同一感測器的排程與路由基因則設為最強關聯。
        """
        sensor_count = P.SENSOR_NUMBER
        graph = np.zeros((self.length, self.length), dtype=float)

        # 忽略「關閉」半徑（索引 0），只判斷兩感測器是否可能覆蓋同一目標。
        coverage = np.any(
            np.asarray(P.coverage_table)[:, :, 1:], axis=2
        ).astype(float)
        overlap = coverage.T @ coverage
        overlap_scale = max(1.0, float(np.max(overlap)))
        overlap /= overlap_scale

        finite_distance = (
            np.asarray(P.distances[:sensor_count, :sensor_count])
            < P.blocked_distance
        )
        finite_distance = finite_distance.astype(float)

        for first in range(sensor_count):
            schedule_first = first * 2
            route_first = schedule_first + 1
            graph[schedule_first, route_first] = 1.0
            graph[route_first, schedule_first] = 1.0
            for second in range(first + 1, sensor_count):
                schedule_second = second * 2
                route_second = schedule_second + 1
                coverage_strength = float(overlap[first, second])
                route_strength = float(
                    max(finite_distance[first, second], finite_distance[second, first])
                )
                graph[schedule_first, schedule_second] = coverage_strength
                graph[schedule_second, schedule_first] = coverage_strength
                graph[route_first, route_second] = route_strength
                graph[route_second, route_first] = route_strength
                # 跨類型關聯較弱，避免 linkage model 過度綁定排程與異感測器路由。
                cross_strength = 0.5 * max(coverage_strength, route_strength)
                graph[schedule_first, route_second] = cross_strength
                graph[route_second, schedule_first] = cross_strength
                graph[schedule_second, route_first] = cross_strength
                graph[route_first, schedule_second] = cross_strength
        np.fill_diagonal(graph, 1.0)
        return graph

    @staticmethod
    def _cluster_similarity(similarity, first, second):
        """以兩群基因間所有配對相似度的平均值作為群間相似度。"""
        return float(np.mean(similarity[np.ix_(first, second)]))

    def _linkage_tree(self, similarity):
        """以平均連結階層分群建立 Family of Subsets（FOS）。"""
        clusters = {gene_id: (gene_id,) for gene_id in range(self.length)}
        active = set(clusters)
        heap = []
        for first in range(self.length):
            for second in range(first + 1, self.length):
                heapq.heappush(heap, (-similarity[first, second], first, second))

        # 單基因集合永遠保留，讓演化仍可進行細粒度的局部修改。
        linkage_sets = [(gene_id,) for gene_id in range(self.length)]
        next_cluster_id = self.length
        while len(active) > 1:
            while heap:
                _, first_id, second_id = heapq.heappop(heap)
                if first_id in active and second_id in active:
                    break
            else:
                break

            merged = tuple(sorted(clusters[first_id] + clusters[second_id]))
            active.remove(first_id)
            active.remove(second_id)
            clusters[next_cluster_id] = merged
            active.add(next_cluster_id)

            # 不加入完整染色體，並限制一次交換的最大範圍。
            if len(merged) < self.length and len(merged) <= self.max_linkage_size:
                linkage_sets.append(merged)

            for other_id in active:
                if other_id == next_cluster_id:
                    continue
                score = self._cluster_similarity(
                    similarity, merged, clusters[other_id]
                )
                low, high = sorted((next_cluster_id, other_id))
                heapq.heappush(heap, (-score, low, high))
            next_cluster_id += 1
        return linkage_sets

    def _limit_linkage_model(self, linkage_sets, P):
        """在固定評估預算下抽樣 FOS，並保留感測器的排程/路由配對。"""
        singleton_sets = [item for item in linkage_sets if len(item) == 1]
        learned_sets = [item for item in linkage_sets if len(item) > 1]
        sensor_pairs = [(sensor_id * 2, sensor_id * 2 + 1) for sensor_id in range(P.SENSOR_NUMBER)]

        self.random.shuffle(singleton_sets)
        self.random.shuffle(sensor_pairs)
        self.random.shuffle(learned_sets)
        # 預算約分給單基因、同感測器配對與統計學到的多基因集合。
        quotas = (
            max(1, self.max_linkage_sets // 3),
            max(1, self.max_linkage_sets // 3),
        )
        selected = singleton_sets[: quotas[0]] + sensor_pairs[: quotas[1]]
        remaining = self.max_linkage_sets - len(selected)
        selected.extend(learned_sets[: max(0, remaining)])

        unique = []
        seen = set()
        for item in selected:
            normalized = tuple(sorted(item))
            if normalized not in seen and len(normalized) < self.length:
                unique.append(normalized)
                seen.add(normalized)
        self.random.shuffle(unique)
        return unique

    def _learn_linkage_model(self, P, entries):
        """從菁英解學習統計 DSM，與圖 DSM 融合後產生本代 FOS。"""
        elite_count = max(2, int(math.ceil(len(entries) * self.elite_fraction)))
        elites = sorted(entries, key=lambda entry: entry["quality"], reverse=True)[:elite_count]
        codes = np.asarray(
            [entry["coding"].code for entry in elites],
            dtype=int,
        )
        statistical_dsm = self._normalized_mutual_information(codes)
        # 固定拓樸提供先驗，互資訊則隨族群演化反映當前的基因依賴。
        combined = (
            self.statistical_weight * statistical_dsm
            + self.graph_weight * self.graph_dsm
        )
        linkage_sets = self._linkage_tree(combined)
        return self._limit_linkage_model(linkage_sets, P)

    @staticmethod
    def _projected_duration(P, cost, feasible):
        """由各使用中感測器的剩餘能量/耗能，估計解可維持的最短時間。"""
        if not feasible:
            return -1.0
        used = np.asarray(cost) > 1e-15
        if not np.any(used):
            return 0.0
        return float(
            np.min(np.asarray(P.energy)[used] / np.asarray(cost)[used])
        )

    def reset_best(self):
        """Reset the common result plus GI-GOMEA's feasibility ordering."""
        super().reset_best()
        self._best_quality = None
        self._last_evaluation_details = None

    def update_best(
        self,
        problem,
        state,
        objectives,
        fitness,
        coverage=None,
        *,
        candidate=None,
    ):
        """Keep GI-GOMEA's feasibility/violation/fitness/duration order."""
        cost = np.asarray(
            problem.calculate_total_cost(state), dtype=float
        )
        target_count = int(problem.TARGET_NUMBER)
        if coverage is None:
            coverage = (
                (target_count - len(problem.find_uncovered_targets(state)))
                / target_count if target_count else 0.0
            )
        covered = int(round(float(coverage) * target_count))
        uncovered = max(0, target_count - covered)
        disconnected = len(problem.find_disconnected(state))
        energy_failed = len(
            problem.find_energy_failed_sensors(state, cost)
        )
        violations = uncovered + disconnected + energy_failed
        feasible = violations == 0
        duration = self._projected_duration(problem, cost, feasible)
        quality = (
            int(feasible),
            -int(violations),
            float(fitness),
            duration,
        )
        self._last_evaluation_details = {
            "cost": cost,
            "feasible": feasible,
            "violations": violations,
            "duration": duration,
            "quality": quality,
        }
        if self._best_quality is not None and quality <= self._best_quality:
            return False

        self._best_quality = quality
        self.best_state = state.copy()
        self.best_objectives = np.asarray(objectives, dtype=float).copy()
        self.best_state.objectives = self.best_objectives.copy()
        self.fitness = float(fitness)
        self.coverage = float(coverage)
        copier = getattr(candidate, "copy", None)
        self.best_candidate = copier() if callable(copier) else None
        return True

    def _evaluate(self, P, coding):
        """解碼並完整評估一個候選解，回傳比較與輸出所需的資料。

        ``quality`` 採字典序比較：先可行性，再違規數、fitness 總和，最後
        才比較預估生命期。因此不可行但高 fitness 的解不會勝過可行解。
        """
        state = coding.decode(P)
        fitness = self.evaluate(P, state, source_candidate=coding)
        details = self._last_evaluation_details
        return {
            "coding": coding,
            "state": state,
            "fitness": fitness,
            "cost": details["cost"],
            "feasible": details["feasible"],
            "violations": details["violations"],
            "duration": details["duration"],
            "quality": details["quality"],
        }

    def _record_history(self, best_score):
        """把目前最佳 fitness 寫入與評估次數對齊的歷史陣列。"""
        if 0 < self.evatime <= len(self.history):
            self.history[self.evatime - 1] = best_score

    def _evaluate_and_record(self, P, state, best_score):
        """評估候選解並同步更新全域最佳分數與收斂歷史。"""
        entry = self._evaluate(P, state)
        best_score = max(best_score, float(entry["quality"][2]))
        self._record_history(best_score)
        return entry, best_score

    def _best_of_two_mates(self, entries, parent_id):
        """以大小為二的 tournament，替指定親代選出交換對象。"""
        candidates = [index for index in range(len(entries)) if index != parent_id]
        first, second = self.random.sample(candidates, 2)
        if entries[first]["quality"] >= entries[second]["quality"]:
            return first
        return second

    def _gi_gom(self, P, entries, parent_id, mate_id, linkage_set, best_score, limit):
        """對一組親代做保持基因數量不變的雙向 linkage-set 交換。

        先評估原本品質較好的那一側；若它因交換而退步，就直接拒絕整組
        交換。接受時兩側必須一起替換，才能維持每個位置的等位值直方圖。
        """
        parent = entries[parent_id]
        mate = entries[mate_id]
        mask = np.asarray(linkage_set, dtype=int)
        parent_code = parent["coding"].code.copy()
        mate_code = mate["coding"].code.copy()
        # 交換片段完全相同時不需浪費 fitness 評估。
        if np.array_equal(parent_code[mask], mate_code[mask]):
            self.no_change_swaps += 1
            return best_score

        # 接受時必須同時替換一對解；預留兩次評估以維持 gene invariance。
        if self.evatime + 2 > limit:
            return best_score
        parent_child_code = parent_code.copy()
        mate_child_code = mate_code.copy()
        parent_child_code[mask] = mate_code[mask]
        mate_child_code[mask] = parent_code[mask]

        # 先檢查較優的親代，讓明顯不利的交換只消耗一次評估。
        if parent["quality"] > mate["quality"]:
            checked_id = parent_id
            other_id = mate_id
            checked_code = parent_child_code
            other_code = mate_child_code
        else:
            checked_id = mate_id
            other_id = parent_id
            checked_code = mate_child_code
            other_code = parent_child_code

        checked_entry, best_score = self._evaluate_and_record(
            P, self._coding_from_code(P, checked_code), best_score
        )
        if checked_entry["quality"] < entries[checked_id]["quality"]:
            self.rejected_swaps += 1
            return best_score

        other_entry, best_score = self._evaluate_and_record(
            P, self._coding_from_code(P, other_code), best_score
        )
        entries[checked_id] = checked_entry
        entries[other_id] = other_entry
        self.accepted_swaps += 1
        return best_score

    def _population_histogram(self, states):
        """統計族群中每個基因位置的各等位值數量。"""
        codes = np.asarray([state.code for state in states], dtype=int)
        histogram = []
        for gene_id in range(self.length):
            domain = self._domain_size(gene_id)
            histogram.append(tuple(np.bincount(codes[:, gene_id], minlength=domain)))
        return tuple(histogram)

    def gene_invariance_error(self):
        """回傳目前族群相對搜尋起點的最大等位值數量誤差；正常應為零。"""
        if not self.population or not hasattr(self, "initial_histogram"):
            return 0
        current = self._population_histogram(self.population)
        return max(
            abs(int(now) - int(initial))
            for current_gene, initial_gene in zip(current, self.initial_histogram)
            for now, initial in zip(current_gene, initial_gene)
        )

    def search(self, P, budget, state=None):
        """在評估預算內反覆學習 FOS、進行 GI-GOM 並回傳最佳編碼。

        ``initial_state`` 依循 Algorithm 介面保留；此實作以 PC sampling 或
        上一輪保存的族群啟動，確保 GI-GOMEA 的族群式等位值不變特性。
        """
        evaluate = max(1, int(budget))
        self.evatime = 0
        self.history = np.zeros(evaluate, dtype=float)
        self.accepted_swaps = 0
        self.rejected_swaps = 0
        self.no_change_swaps = 0
        self.nmi_gpu_calls = 0
        self.nmi_cpu_calls = 0

        size = min(self.population_size, evaluate)
        # 族群尺寸改變時重新均勻取樣；否則沿用前一輪族群繼續搜尋。
        if len(self.population) != size:
            states = self._probabilistically_complete_population(P, size)
        else:
            states = [
                self._coding_from_code(P, coding.code)
                for coding in self.population
            ]
        self.initial_histogram = self._population_histogram(states)

        entries = []
        best_score = -math.inf
        for state in states:
            entry, best_score = self._evaluate_and_record(P, state, best_score)
            entries.append(entry)
        best_entry = max(entries, key=lambda entry: entry["quality"])

        # 每次有效的 GI 交換最多需要評估兩個子代，因此至少預留兩次預算。
        while self.evatime + 2 <= evaluate and len(entries) >= 3:
            self.last_linkage_model = self._learn_linkage_model(P, entries)
            # 每位親代都嘗試本代 FOS；隨機打散可降低固定順序偏誤。
            items = [
                (parent_id, linkage_set)
                for parent_id in range(len(entries))
                for linkage_set in self.last_linkage_model
            ]
            self.random.shuffle(items)
            evaluations_before = self.evatime

            for parent_id, linkage_set in items:
                if self.evatime + 2 > evaluate:
                    break
                mate_id = self._best_of_two_mates(entries, parent_id)
                best_score = self._gi_gom(
                    P,
                    entries,
                    parent_id,
                    mate_id,
                    linkage_set,
                    best_score,
                    evaluate,
                )
                candidate = max(
                    (entries[parent_id], entries[mate_id]),
                    key=lambda entry: entry["quality"],
                )
                if candidate["quality"] > best_entry["quality"]:
                    best_entry = candidate

            self.generation += 1
            self.on_iteration_finish(
                problem=P,
                state=best_entry["state"],
                iteration=self.generation,
                run=0,
                Name=self.name.replace("_", "-"),
                linkage_sets=len(self.last_linkage_model),
                fitness=float(best_entry["quality"][2]),
            )
            # 全部片段都相同或預算不足時，繼續下一代也不會產生新評估。
            if self.evatime == evaluations_before:
                break

        self.population = [entry["coding"].copy() for entry in entries]
        if self.evatime < evaluate:
            self.history[self.evatime :] = best_score if math.isfinite(best_score) else 0.0

        # 交換的另一側可能退步；保留搜尋期間的全域 elitist 作為輸出候選。
        output_entries = entries + [best_entry]
        feasible_entries = [entry for entry in output_entries if entry["feasible"]]
        final_entry = max(
            feasible_entries or output_entries,
            key=lambda entry: entry["quality"],
        )
        print(
            self.name.replace("_", "-") + " end",
            "generation=",
            self.generation,
            "linkage_sets=",
            len(self.last_linkage_model),
            "accepted=",
            self.accepted_swaps,
            "rejected=",
            self.rejected_swaps,
            "gene_error=",
            self.gene_invariance_error(),
            "evatime=",
            self.evatime,
            "device=",
            self.device_detail,
            "gpu_nmi_calls=",
            self.nmi_gpu_calls,
        )
        self.final_coding = final_entry["coding"].copy()
        return final_entry["state"].copy()
