import math
import random
import numpy as np
import time
from multiprocessing import Pipe, Process

from Algorithm.core.Algorithm import Algorithm
from Algorithm.core.budget import can_start_iteration
from State.SensorEncoding import SensorEncoding


class CS(Algorithm):
    """Cuckoo Search over positions decoded through ``SensorEncoding``."""

    def __init__(
        self,
        P,
        n=50,
        length=100,
        fmax=1.0,
        fmin=0,
        step_size=0.01,
        Lmbda=1.5,
        pa=0.25,
    ):
        super().__init__()
        self.N = n
        self.fmax = fmax
        self.fmin = fmin
        self.step_size = step_size
        self.Lmbda = Lmbda
        self.pa = pa
        self.length = P.SENSOR_NUMBER * 2
        self.gene_domain = max(P.LEVEL, SensorEncoding.RANK_PRECISION)
        self.evatime = 0
        self.h = 4
        self.name = self.__class__.__name__ + "_" + str(n) + "_" + str(self.pa)
        P.prepare_coding_cache()

    def create_population(self, P, sch_s=None):
        return [self._create_state(P, sch_s) for _ in range(self.N)]

    def levy_flight(self, Lambda):
        sigma1 = np.power(
            (
                math.gamma(1 + Lambda)
                * np.sin((np.pi * Lambda) / 2)
                / math.gamma((1 + Lambda) / 2)
                * np.power(2, (Lambda - 1) / 2)
            ),
            1 / Lambda,
        )
        u = np.random.normal(0, sigma1, size=(self.N, self.length))
        v = np.random.normal(0, 1, size=(self.N, self.length))
        return u / np.power(np.fabs(v), 1 / Lambda)

    def update_position(self, P, states):
        step = self.step_size * self.levy_flight(self.Lmbda)
        for index in range(self.N):
            gene_id = random.randint(0, self.length - 1)
            states[index].code[gene_id] += step[index][gene_id]
            states[index].code[states[index].code > self.fmax] = self.fmax
            states[index].code[states[index].code < self.fmin] = self.fmin
        return states

    def select_individuals(self, P, states, fitness, alpha=0.7, sch_s=None):
        rank_ids = np.argsort(fitness)[::-1]
        not_selected_ids = rank_ids[int(len(rank_ids) * alpha) :]
        for index in not_selected_ids:
            states[index] = self._create_state(P, sch_s)
        return states

    def run_worker(self, P, iteration, worker_id, conn):
        print("Slave ", worker_id, "start")
        for _ in range(iteration):
            self.evatime = 0
            states, stop = conn.recv()
            if stop:
                break
            fitness = [
                np.sum(self._evaluate_state(P, state))
                for state in states
            ]
            conn.send([self.evatime, fitness])

    def evaluate_population(self, P, states, conn):
        if self.h > 0:
            batches = np.array_split(np.arange(len(states)), self.h)
            for worker_id, indexes in enumerate(batches):
                conn[worker_id].send(
                    [[states[index] for index in indexes], False]
                )

            fitness = []
            for worker_id in range(self.h):
                worker_evatime, worker_fitness = conn[worker_id].recv()
                fitness += worker_fitness
                self.evatime += worker_evatime
            return np.array(fitness)

        return np.array(
            [np.sum(self._evaluate_state(P, state)) for state in states]
        )

    def create_workers(self, P, iteration):
        threads = []
        conn = []
        print("run start CS")
        for worker_id in range(self.h):
            parent_conn, child_conn = Pipe()
            process = Process(
                target=self.run_worker,
                args=(P, iteration, worker_id, child_conn),
            )
            process.start()
            threads.append(process)
            conn.append(parent_conn)
        return threads, conn

    def close_workers(self, threads, conn):
        for worker_id in range(self.h):
            conn[worker_id].send([[], True])
            threads[worker_id].join()
            conn[worker_id].close()

    def _create_state(self, P, sch_s):
        coding = SensorEncoding.random(P.SENSOR_NUMBER, self.gene_domain)
        coding.code = coding.code / self.gene_domain
        return coding

    def _decode_particle(self, P, coding):
        discrete = SensorEncoding(self._discretize_code(coding))
        return discrete.decode(P)

    def _discretize_code(self, state):
        span = self.fmax - self.fmin
        if span <= 0:
            span = 1.0
        normalized = (np.clip(state.code, self.fmin, self.fmax) - self.fmin) / span
        code = np.floor(normalized * self.gene_domain).astype("int")
        return np.clip(code, 0, self.gene_domain - 1)

    def _to_discrete_state(self, P, coding):
        if np.issubdtype(np.asarray(coding.code).dtype, np.floating):
            code = self._discretize_code(coding)
        else:
            code = np.asarray(coding.code).astype("int")
        return SensorEncoding(code)

    def _evaluate_state(self, P, coding):
        if np.issubdtype(np.asarray(coding.code).dtype, np.floating):
            state = self._decode_particle(P, coding)
        else:
            state = coding.decode(P)
        self.evatime += 1
        return P.evaluate_state(state)

    def search(self, P, budget, state=None):
        evaluate = max(1, int(budget))
        run = 1
        sch_s = state
        self.history = np.zeros(evaluate)
        global_state = None

        for run_id in range(run):
            start_time = time.time()
            threads, conn = self.create_workers(P, evaluate)
            self.evatime = 0

            states = self.create_population(P, sch_s)
            fitness = self.evaluate_population(P, states, conn)
            best_id = int(np.argmax(fitness))
            global_state = states[best_id].copy()
            global_best = self._evaluate_state(P, states[best_id])
            iteration_id = 0

            while can_start_iteration(self.evatime, evaluate):
                previous_evatime = self.evatime
                states = self.update_position(P, states)
                fitness = self.evaluate_population(P, states, conn)
                states = self.select_individuals(
                    P,
                    states,
                    fitness,
                    alpha=self.pa,
                )

                best_id = int(np.argmax(fitness))
                best_value = self._evaluate_state(P, states[best_id])
                if np.sum(best_value) > np.sum(global_best):
                    global_state = states[best_id].copy()
                    global_best = best_value
                else:
                    states[best_id] = global_state.copy()
                    best_value = global_best

                same_ratio = np.sum(fitness == np.sum(best_value)) / self.N
                if iteration_id % 10 == 0:
                    print(
                        self.evatime,
                        "best",
                        best_value,
                        "same_ratio:",
                        same_ratio,
                        np.sum(best_value),
                        "time",
                        time.time() - start_time,
                    )

                self.on_iteration_finish(
                    problem=P,
                    state=states[best_id],
                    iteration=iteration_id,
                    run=run_id,
                    Name="Test",
                    time_cost=None,
                    stop=False,
                    scale=3,
                    best_value=best_value,
                    fitness=np.sum(best_value),
                )
                self.history[previous_evatime : self.evatime] += np.sum(global_best)
                iteration_id += 1

            self.close_workers(threads, conn)

        self.history /= run
        if global_state is None:
            return None
        return self._to_discrete_state(P, global_state).decode(P)
