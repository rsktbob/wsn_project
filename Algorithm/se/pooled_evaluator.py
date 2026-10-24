"""SE 演算法共用的平行評估：h 個常駐子程序，只做解碼＋評估。

投資、對齊、更新 goods 都在主程序做；這裡只把一張投資表分給子程序評估，
再照原本的順序收回。演算法以 ``self.evaluator = PooledEvaluator()`` 持有它。
"""

from __future__ import annotations

from multiprocessing import Pipe, Process

import numpy as np


def _worker_loop(problem, connection):
    """子程序：每收到一批候選解就解碼、評估並回傳，收到 ``None`` 就結束。"""
    while True:
        candidates = connection.recv()
        if candidates is None:
            break
        results = []
        for candidate in candidates:
            state = candidate.decode(problem)
            objectives = np.asarray(problem.evaluate_state(state), dtype=float)
            state.objectives = objectives.copy()
            results.append((candidate, state, objectives))
        connection.send(results)
    connection.close()


class PooledEvaluator:
    """用 h 個子程序評估投資表，結果依原本的列順序合併。"""

    def __init__(self):
        # 每個子程序一組 (process, connection)；沒開時是 None。
        self._pool = None

    def start(self, problem, worker_count, next_seed):
        """開啟 ``worker_count`` 個評估子程序。

        子程序不用亂數，但每個子程序仍抽一個 ``next_seed()``，讓主程序的亂數
        序列和舊版（每個子程序有自己 seed）相同。
        """
        self._pool = []
        for _ in range(worker_count):
            next_seed()
            parent_connection, child_connection = Pipe()
            process = Process(
                target=_worker_loop, args=(problem, child_connection)
            )
            process.start()
            self._pool.append((process, parent_connection))

    def close(self):
        """搜尋結束或發生例外時關閉評估子程序。"""
        if self._pool is None:
            return
        for _, connection in self._pool:
            try:
                connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
        for process, connection in self._pool:
            process.join()
            connection.close()
        self._pool = None

    def route(self, investments):
        """第 i 列交給子程序 ``i % h``，再依列順序逐列回傳結果。

        每列結果是 ``[(candidate, state, objectives), ...]``。
        """
        worker_count = len(self._pool)
        assignments = [row_id % worker_count for row_id in range(len(investments))]
        for row, worker_id in zip(investments, assignments):
            self._pool[worker_id][1].send(row)
        # 同一個子程序的回應依送出順序回來，所以照分配順序逐一 recv 即可。
        for worker_id in assignments:
            yield self._pool[worker_id][1].recv()

    def evaluate(self, algorithm, problem, investments):
        """評估每一列投資，計入 ``algorithm`` 的評估數與最佳解，回傳 fitness 矩陣。"""
        scores = []
        for row, results in enumerate(self.route(investments)):
            row_scores = []
            for col, (candidate, state, objectives) in enumerate(results):
                investments[row][col] = candidate
                fitness = float(np.sum(objectives))
                algorithm.evatime += 1
                algorithm.update_best(problem, state, objectives, fitness,
                                      candidate=candidate)
                row_scores.append(fitness)
            scores.append(row_scores)
        return np.asarray(scores, dtype=float)


__all__ = ["PooledEvaluator"]
