"""以 TargetEncoding 實作的 target-oriented SA-SETS。"""

import math
import numpy as np

from Algorithm.se.SA_SETS import SA_SETS
from State.Encoding import swap_segment
from State.TargetEncoding import TargetEncoding


class SA_SETS_Target(SA_SETS):
    """將 SA-SETS 市場搜尋流程套用到 target-oriented chromosome。

    每個 target 使用一個 0 到 9999 的 gene。高位部分選擇該 target
    的 CCS candidate，低兩位則參與 target priority 的計算。
    """

    TARGET_GENE_DOMAIN = 10000

    def __init__(self, P, n=8, h=4, w=2, mu=0.4, seed=None):
        """初始化 Target 編碼版本，並依目前網路狀態選出區域關鍵目標。"""
        # 先沿用 SA-SETS 的市場參數，再把 chromosome 長度改為 target 數。
        super().__init__(P, n=n, h=h, w=w, mu=mu, seed=seed)
        self.code_length = P.TARGET_NUMBER
        self.name = (
            "SA_SETS_TARGET_" + str(n) + str(h) + str(w) + str(mu)
        )

        # h 個區域最多需要 ceil(log2(h)) 個關鍵 target bit。
        self.identity_bit_count = max(
            1, int(math.ceil(math.log(max(2, self.h), 2)))
        )
        self.critical_targets = []

    @staticmethod
    def _normalize(values):
        """將陣列線性正規化到 [0, 1]；常數陣列回傳全零。"""
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            return values
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if maximum - minimum <= 1e-12:
            return np.zeros_like(values)
        return (values - minimum) / (maximum - minimum)

    def select_identity_sensors(self, P):
        """依能量風險、候選稀缺度與 BS 距離選擇關鍵 targets。"""
        # target 周圍剩餘能量越少，energy_risk 越高。
        target_energy = P.calculate_target_remaining_energy(P.energy)
        energy_risk = 1.0 - self._normalize(target_energy)

        # CCS candidates 越少，該 target 越難替代，scarcity 越高。
        candidate_count = np.asarray(
            [len(candidates) for candidates in P.cover_candidates],
            dtype=float,
        )
        scarcity = 1.0 - self._normalize(candidate_count)

        # 越靠近 BS 的 target 得到越高的 near_base 分數。
        target_distance = P.G.CalDistance(
            P.target,
            np.asarray([P.BS]),
        )[:, 0]
        near_base = 1.0 - self._normalize(target_distance)

        # 這是專案對 TargetEncoding 的延伸分類方式，並非 2023 論文公式 (44)。
        score = 0.5 * energy_risk + 0.3 * scarcity + 0.2 * near_base
        valid_targets = np.where(candidate_count > 0)[0]
        if len(valid_targets) == 0:
            valid_targets = np.arange(P.TARGET_NUMBER)

        order = valid_targets[
            np.argsort(score[valid_targets], kind="stable")[::-1]
        ].tolist()
        critical_targets = order[: self.identity_bit_count]

        # target 數不足時循環補齊，維持既有 region bit 數量。
        while len(critical_targets) < self.identity_bit_count and order:
            critical_targets.append(
                order[len(critical_targets) % len(order)]
            )
        self.critical_targets = critical_targets
        return critical_targets

    @staticmethod
    def _candidate_layout(P, target_id):
        """回傳 target gene 可表示的 candidate 分段方式。

        Returns:
            ``(candidate_count, board, reachable_count)``。board 是每個
            candidate 在 schedule-code 區段中占用的寬度。
        """
        candidate_count = len(P.cover_candidates[target_id])
        if candidate_count == 0:
            return 0, 1, 0
        board = (
            TargetEncoding.RANK_PRECISION // candidate_count + 1
        )
        reachable_count = min(
            candidate_count,
            (TargetEncoding.RANK_PRECISION - 1) // board + 1,
        )
        return candidate_count, board, reachable_count

    @classmethod
    def _decode_candidate_index(cls, P, target_id, gene):
        """從 gene 的高位部分解出 CCS candidate index。"""
        _, board, reachable_count = cls._candidate_layout(P, target_id)
        if reachable_count == 0:
            return -1
        schedule_code = (
            int(gene) % cls.TARGET_GENE_DOMAIN
        ) // TargetEncoding.RANK_PRECISION
        return min(schedule_code // board, reachable_count - 1)

    @classmethod
    def _encode_candidate_index(
        cls,
        P,
        target_id,
        candidate_index,
        rank,
    ):
        """將 candidate index 與 priority rank 重新組合成 target gene。"""
        _, board, reachable_count = cls._candidate_layout(P, target_id)
        if reachable_count == 0:
            return int(rank) % TargetEncoding.RANK_PRECISION
        candidate_index = int(
            np.clip(candidate_index, 0, reachable_count - 1)
        )
        schedule_code = min(
            TargetEncoding.RANK_PRECISION - 1,
            candidate_index * board,
        )
        return (
            schedule_code * TargetEncoding.RANK_PRECISION
            + int(rank) % TargetEncoding.RANK_PRECISION
        )

    def create_candidate(self, P, region=None):
        """均勻建立一條 target-oriented chromosome。"""
        candidate = TargetEncoding.random(P.TARGET_NUMBER, rng=self.rng)
        if region is not None:
            self.align_region(P, candidate, region)
        return candidate

    def align_region(self, P, state, region_id):
        """用 region bits 限制關鍵 target 落在 candidate 前半或後半。"""
        # 每個 region bit 把相應關鍵目標限制在候選集合的前半或後半。
        bits = int(region_id)
        for bit_id, target_id in enumerate(self.critical_targets):
            _, _, reachable_count = self._candidate_layout(P, target_id)
            if reachable_count <= 1:
                continue

            split = max(1, reachable_count // 2)
            if bits & (1 << bit_id):
                allowed = list(range(split, reachable_count))
            else:
                allowed = list(range(0, split))
            if not allowed:
                allowed = list(range(reachable_count))

            current_gene = int(state.code[target_id])
            current_candidate = self._decode_candidate_index(
                P,
                target_id,
                current_gene,
            )
            if current_candidate not in allowed:
                current_candidate = self.random.choice(allowed)
                state.code[target_id] = self._encode_candidate_index(
                    P,
                    target_id,
                    current_candidate,
                    current_gene % TargetEncoding.RANK_PRECISION,
                )
        return state

    def _mutate(self, P, state):
        """突變 candidate、priority rank，或直接重抽完整 target gene。"""
        mutation_count = self.random.choice([1] * 90 + [2] * 5 + [3] * 5)
        target_ids = [
            self.random.randrange(P.TARGET_NUMBER)
            for _ in range(mutation_count)
        ]

        for target_id in target_ids:
            current_gene = int(state.code[target_id])
            current_rank = (
                current_gene % TargetEncoding.RANK_PRECISION
            )
            operation = self.random.randint(0, 2)

            if operation == 0:
                # 保留 priority，只改成另一個可表示的 CCS candidate。
                _, _, reachable_count = self._candidate_layout(
                    P,
                    target_id,
                )
                if reachable_count > 1:
                    current_candidate = self._decode_candidate_index(
                        P,
                        target_id,
                        current_gene,
                    )
                    candidates = [
                        candidate_id
                        for candidate_id in range(reachable_count)
                        if candidate_id != current_candidate
                    ]
                    new_candidate = self.random.choice(candidates)
                    state.code[target_id] = self._encode_candidate_index(
                        P,
                        target_id,
                        new_candidate,
                        current_rank,
                    )
                else:
                    state.code[target_id] = self.random.randint(
                        0,
                        self.TARGET_GENE_DOMAIN - 1,
                    )
            elif operation == 1:
                # 保留 schedule-code，只重抽 priority rank。
                schedule_code = (
                    current_gene % self.TARGET_GENE_DOMAIN
                ) // TargetEncoding.RANK_PRECISION
                new_rank = self.random.randint(
                    0,
                    TargetEncoding.RANK_PRECISION - 1,
                )
                state.code[target_id] = (
                    schedule_code * TargetEncoding.RANK_PRECISION
                    + new_rank
                )
            else:
                # 同時改變 candidate 與 priority。
                state.code[target_id] = self.random.randint(
                    0,
                    self.TARGET_GENE_DOMAIN - 1,
                )
        return state

    def invest(self, P, searcher, good):
        """完成 Target 版本的 crossover 與可選 mutation。"""
        difference = 2
        midpoint = self.code_length // 2
        left = self.random.randint(difference, midpoint - difference)
        right = self.random.randint(
            midpoint - difference,
            self.code_length - difference,
        )
        first, second = swap_segment(searcher, good, left, right)
        investment = first if self.random.random() < 0.5 else second
        if self.random.random() < self.mutation_rate:
            investment = self._mutate(P, investment)
        return investment


__all__ = ["SA_SETS_Target"]
