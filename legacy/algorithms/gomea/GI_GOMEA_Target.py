"""以目標優先序（TargetEncoding）求解 WSN 的 GI-GOMEA 版本。"""

import numpy as np

from Algorithm.gomea.BaseGIGOMEA import BaseGIGOMEA
from State.TargetEncoding import TargetEncoding


class GI_GOMEA_Target(BaseGIGOMEA):
    """每個目標使用一個優先序基因的 GI-GOMEA。"""

    def __init__(
        self,
        P,
        population_size=16,
        elite_fraction=0.5,
        statistical_weight=0.7,
        graph_weight=0.3,
        max_linkage_size=16,
        max_linkage_sets=120,
        seed=None,
    ):
        """初始化 Target 版本，並沿用共用 GI-GOMEA 搜尋流程。"""
        super().__init__(
            P,
            population_size=population_size,
            elite_fraction=elite_fraction,
            statistical_weight=statistical_weight,
            graph_weight=graph_weight,
            max_linkage_size=max_linkage_size,
            max_linkage_sets=max_linkage_sets,
            seed=seed,
        )
        self.state_type = "target"
        self.name = "GI_GOMEA_TARGET_WSN"

    def _new_coding(self, P, genes):
        """把基因陣列包裝成 TargetEncoding。"""
        return TargetEncoding(genes)

    @staticmethod
    def _encoding_length(P):
        """回傳 Target 編碼長度：每個目標各有一個優先序基因。"""
        return int(P.TARGET_NUMBER)

    def _domain_size(self, gene_id):
        """回傳目標優先序基因的固定離散範圍大小。"""
        return self.target_domain

    def _canonicalize(self, code):
        """將所有目標優先序基因映射回合法範圍。"""
        canonical = np.asarray(code, dtype=int).copy()
        canonical %= self.target_domain
        return canonical

    def _build_graph_dsm(self, P):
        """以兩目標候選感測器集合的 Jaccard 相似度建立圖 DSM。

        可由相同感測器覆蓋的目標較可能互相影響，因此應較常被放入同一個
        linkage set 一起交換。
        """
        graph = np.zeros((self.length, self.length), dtype=float)
        # CCS[target] 記錄可覆蓋該目標的感測器及其所需半徑。
        candidate_sets = [
            {
                int(sensor_id)
                for sensor_id, _ in P.cover_candidates[target_id]
            }
            for target_id in range(P.TARGET_NUMBER)
        ]

        for first in range(P.TARGET_NUMBER):
            graph[first, first] = 1.0
            for second in range(first + 1, P.TARGET_NUMBER):
                union = candidate_sets[first] | candidate_sets[second]
                shared = candidate_sets[first] & candidate_sets[second]
                similarity = len(shared) / len(union) if union else 0.0
                graph[first, second] = similarity
                graph[second, first] = similarity
        return graph

    def _limit_linkage_model(self, linkage_sets, P):
        """限制 Target 版本的 FOS 數量，保留單基因與學得的目標群組。"""
        singleton_sets = [item for item in linkage_sets if len(item) == 1]
        learned_sets = [item for item in linkage_sets if len(item) > 1]

        # 這些 sensor_pairs 不會加入 Target FOS；仍建立並打散它們，是為了保留
        # 舊版 state_type="target" 的亂數呼叫順序，使固定種子的實驗結果不變。
        sensor_pairs = [
            (sensor_id * 2, sensor_id * 2 + 1)
            for sensor_id in range(P.SENSOR_NUMBER)
        ]
        self.random.shuffle(singleton_sets)
        self.random.shuffle(sensor_pairs)
        self.random.shuffle(learned_sets)

        # 一半配額優先保留細粒度的單目標修改，其餘交給學到的多目標群組。
        singleton_quota = max(1, self.max_linkage_sets // 2)
        selected = singleton_sets[:singleton_quota]
        remaining = self.max_linkage_sets - len(selected)
        selected.extend(learned_sets[: max(0, remaining)])

        # 去除重複集合及等同完整染色體的集合，避免無效或過大的交換。
        unique = []
        seen = set()
        for item in selected:
            normalized = tuple(sorted(item))
            if normalized not in seen and len(normalized) < self.length:
                unique.append(normalized)
                seen.add(normalized)
        self.random.shuffle(unique)
        return unique


GIGOMEATarget = GI_GOMEA_Target
