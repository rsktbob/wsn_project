"""族群演算法可共用的評估執行器。"""

from __future__ import annotations

from multiprocessing import Pipe, Process

import numpy as np


def _run_worker(owner, evaluator_name, problem, iteration, worker_id, connection):
    """子程序評估迴圈；保留舊程式每批回報 evatime 的行為。"""
    print("Slave ", worker_id, "start")
    evaluator = getattr(owner, evaluator_name)
    for _ in range(iteration):
        owner.evatime = 0
        states, stop = connection.recv()
        if stop:
            break

        fitness = [evaluator(problem, state) for state in states]
        connection.send([owner.evatime, fitness])


class BatchEvaluator:
    """以相同介面執行 serial 或 process-based 族群評估。

    這個元件只處理工作分批、程序生命週期與評估次數回收，不知道 GA、
    PSO、EDA 或 GWO 的搜尋流程。
    """

    def __init__(self, owner, evaluator_name, worker_count=4, label=""):
        self.owner = owner
        self.evaluator_name = evaluator_name
        self.worker_count = worker_count
        self.label = label

    def evaluate(self, problem, states, connections=None):
        """依是否提供 worker connections 選擇平行或循序評估。"""
        evaluator = getattr(self.owner, self.evaluator_name)
        if self.worker_count > 0 and connections is not None:
            batches = np.array_split(np.arange(len(states)), self.worker_count)
            for worker_id, indexes in enumerate(batches):
                connections[worker_id].send(
                    [[states[index] for index in indexes], False]
                )

            fitness = []
            for worker_id in range(self.worker_count):
                evaluations, worker_fitness = connections[worker_id].recv()
                fitness.extend(worker_fitness)
                self.owner.evatime += evaluations
            return fitness

        return [evaluator(problem, state) for state in states]

    def run_worker(self, problem, iteration, worker_id, connection):
        """執行一個 family base 的批次評估 worker。"""
        return _run_worker(
            self.owner,
            self.evaluator_name,
            problem,
            iteration,
            worker_id,
            connection,
        )

    def start(self, problem, iteration):
        """建立固定數量的評估子程序。"""
        processes = []
        connections = []
        print(f"run start {self.label} r")
        for worker_id in range(self.worker_count):
            parent_connection, child_connection = Pipe()
            process = Process(
                target=_run_worker,
                args=(
                    self.owner,
                    self.evaluator_name,
                    problem,
                    iteration,
                    worker_id,
                    child_connection,
                ),
            )
            process.start()
            processes.append(process)
            connections.append(parent_connection)
        return processes, connections

    def close(self, processes, connections):
        """停止並回收所有評估子程序。"""
        for worker_id in range(self.worker_count):
            connections[worker_id].send([[], True])
            processes[worker_id].join()
            connections[worker_id].close()
