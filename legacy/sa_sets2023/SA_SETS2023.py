"""學長 2023 SA-SETS 原始程式版（codes/2023以恩MWSN/Algorithm/Combine/SA_SETS.py + SETSM.py）。

只用來重現原版，不進正式比較。市場流程、identity sensor、CCS、交配、突變組數、goods 更新、
searcher 選區、期望值、Beta 機率與步長都和正式 SA_SETS 相同，直接繼承；照原版改的有：

1. 解碼：SensorEncoding2023（v1、原版選父節點打分、容量不足寫回染色體）。
2. 軟分區（GoToRegion）：只強制開，且只看基因原值 == 0，開成 randint(1, LEVEL−1)；不強制關。
3. 基因值域：所有基因（含感測等級）都在 0–9 取值（FMAX = max(LEVEL 6, rank_prec 10)），
   初始解（CreateCode）與突變（Transition → RandomValue）都是；解碼時等級 mod 6。
4. 回傳最好的 searcher（Run 回傳 s[best_id]），不是全程評估過的最佳解；
   history 也記錄每回合最好 searcher 的 fitness。

fitness 用 ``--fitness-service v1``（學長 CodingState.Evaluate 的公式）。
亂數產生方式和原版不同（原版子程序的 random 沒設種子、無法重現），邏輯相同。
未照抄的細節：原版不算 goods 初始評估、每回合多算 1 次 best_w 評估、迴圈條件 <=。
"""

from __future__ import annotations

import numpy as np

from Algorithm.se.sets.SA_SETS import SA_SETS
from Algorithm.se.sensor_operators import draw_mutation_count
from SensorEncoding2023 import SensorEncoding2023

# 原版 CodingState.FMAX：每個基因都在 [0, FMAX) 取值。
FMAX = 10


class SA_SETS2023(SA_SETS):
    """學長原始程式版 SA-SETS。"""

    def create_candidate(self, problem, region=None):
        """原版 CreateCode：所有基因 randint(0, 10)，再依 CCS 為每個 target 寫入一組 (sensor, level)。"""
        code = np.asarray(
            self.rng.integers(0, FMAX, problem.SENSOR_NUMBER * 2), dtype=int
        )
        for target_candidates in problem.cover_candidates:
            if target_candidates:
                sensor_id, level = self.random.choice(target_candidates)
                code[int(sensor_id) * 2] = int(level)
        candidate = SensorEncoding2023(code)
        if region is not None:
            self.align_region(problem, candidate, region)
        return candidate

    def align_region(self, problem, candidate, region):
        """原版 GoToRegion：只強制開，只檢查原值 == 0（例如 6 解碼後是 0，但不會被強制開）。"""
        bits = int(region)
        for bit_id, sensor_id in enumerate(self.identity_sensors):
            gene_id = int(sensor_id) * 2
            if (bits >> bit_id) & 1 and int(candidate.code[gene_id]) == 0:
                candidate.code[gene_id] = self.random.randint(
                    1, problem.sensing_option_count(sensor_id) - 1
                )
        return candidate

    def mutatev1(self, problem, candidate):
        """原版 Transition：1–3 組（90/5/5），每組改等級、路由或兩者，新值都是 randint(0, 9)。"""
        for _ in range(draw_mutation_count(self.random)):
            gene_id = self.random.randrange(self.code_length)
            sensing_id = gene_id if gene_id % 2 == 0 else gene_id - 1
            operation = self.random.randint(0, 2)
            if operation in (0, 1):
                candidate.code[sensing_id] = self.random.randrange(FMAX)
            if operation in (0, 2):
                candidate.code[sensing_id + 1] = self.random.randrange(FMAX)
        return candidate

    def update_search_memory(self, problem, evaluation_start):
        """原版 history 記錄每回合最好 searcher 的 fitness。"""
        super().update_search_memory(problem, evaluation_start)
        history_end = min(int(self.evatime), len(self.history))
        if history_end > evaluation_start:
            self.history[evaluation_start:history_end] = np.max(
                self.searcher_fitness
            )

    def search(self, problem, budget, state=None):
        """跑完市場後，回傳最好的 searcher（原版 Run 的 s[best_id]）。"""
        super().search(problem, budget, state)
        if not self.searchers:
            return None

        best_id = int(np.argmax(self.searcher_fitness))
        candidate = self.searchers[best_id].copy()
        best_state = candidate.decode(problem)
        objectives = np.asarray(problem.evaluate_state(best_state), dtype=float)
        best_state.objectives = objectives.copy()

        self.best_state = best_state
        self.best_objectives = objectives.copy()
        self.fitness = float(np.sum(objectives))
        self.best_candidate = candidate.copy()
        self.final_coding = candidate.copy()
        if self.evatime < len(self.history):
            self.history[self.evatime:] = self.fitness
        return best_state.copy()


__all__ = ["SA_SETS2023"]
