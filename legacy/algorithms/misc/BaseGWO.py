import random
import time

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from Algorithm.core.budget import can_start_iteration
from Algorithm.core.evaluation import BatchEvaluator


class BaseGWO(Algorithm):
    """Template for GWO-style WSN optimizers."""

    def __init__(self, P, n, lb=-1, ub=1):
        super().__init__()
        self.lb = lb
        self.ub = ub
        self.N = n
        self.name = self.__class__.__name__ + "_" + str(n)
        self.evatime = 0
        self.h = 4
        self.history = []
        self.batch_evaluator = BatchEvaluator(
            self,
            "_evaluate_state",
            worker_count=self.h,
            label="GWO",
        )

    def _create_state(self, P, sch_s):
        """Create one wolf state."""
        raise NotImplementedError

    def _evaluate_state(self, P, state):
        """Evaluate one wolf state."""
        raise NotImplementedError

    @staticmethod
    def _copy_candidate(candidate):
        """Copy new coding objects while retaining legacy family support."""
        copier = getattr(candidate, "copy", None)
        if copier is not None:
            return copier()
        return candidate.Copy()

    def run_slave(self, P, iteration, worker_id, conn):
        return self.batch_evaluator.run_worker(P, iteration, worker_id, conn)

    def evaluate_population(self, P, states, conn=None):
        """Evaluate wolves, optionally through worker processes."""
        return self.batch_evaluator.evaluate(P, states, conn)

    def create_slaves(self, P, iteration):
        """Create worker processes."""
        self.batch_evaluator.worker_count = self.h
        return self.batch_evaluator.start(P, iteration)

    def close_slaves(self, threads, conn):
        """Stop worker processes."""
        return self.batch_evaluator.close(threads, conn)

    def calculate_fitness(self, P, states):
        """Evaluate states without worker processes."""
        return [self._evaluate_state(P, state) for state in states]

    def clamp_state_codes(self, states):
        """Clamp wolf code values into the search bounds."""
        for index in range(self.N):
            states[index].code = np.clip(states[index].code, self.lb, self.ub)
        return states

    def _decode_if_available(self, P, state):
        decode = getattr(self, "Decode", None)
        if decode is not None:
            return decode(P, state)
        return state

    def _update_leaders(
        self,
        P,
        positions,
        fitness_values,
        alpha_pos,
        beta_pos,
        delta_pos,
        alpha_score,
        beta_score,
        delta_score,
    ):
        for index in range(self.N):
            fitness = fitness_values[index]

            if np.sum(fitness) > np.sum(alpha_score):
                delta_score = beta_score
                delta_pos = self._copy_candidate(beta_pos)
                beta_score = alpha_score
                beta_pos = self._copy_candidate(alpha_pos)
                alpha_score = fitness
                alpha_pos = self._copy_candidate(positions[index])
                alpha_pos = self._decode_if_available(P, alpha_pos)
                continue

            if np.sum(fitness) < np.sum(alpha_score) and np.sum(fitness) > np.sum(
                beta_score
            ):
                delta_score = beta_score
                delta_pos = self._copy_candidate(beta_pos)
                beta_score = fitness
                beta_pos = self._copy_candidate(positions[index])
                continue

            if (
                np.sum(fitness) < np.sum(alpha_score)
                and np.sum(fitness) < np.sum(beta_score)
                and np.sum(fitness) > np.sum(delta_score)
            ):
                delta_score = fitness
                delta_pos = self._copy_candidate(positions[index])

        return alpha_pos, beta_pos, delta_pos, alpha_score, beta_score, delta_score

    def _move_wolves(self, positions, alpha_pos, beta_pos, delta_pos, a):
        for index in range(self.N):
            r1 = random.random()
            r2 = random.random()
            a1 = 2 * a * r1 - a
            c1 = 2 * r2
            d_alpha = np.abs(c1 * alpha_pos.code - positions[index].code)
            x1 = alpha_pos.code - a1 * d_alpha

            r1 = random.random()
            r2 = random.random()
            a2 = 2 * a * r1 - a
            c2 = 2 * r2
            d_beta = np.abs(c2 * beta_pos.code - positions[index].code)
            x2 = beta_pos.code - a2 * d_beta

            r1 = random.random()
            r2 = random.random()
            a3 = 2 * a * r1 - a
            c3 = 2 * r2
            d_delta = np.abs(c3 * delta_pos.code - positions[index].code)
            x3 = delta_pos.code - a3 * d_delta

            positions[index].code = (x1 + x2 + x3) / 3

        return positions

    def run_by_evaluation_limit(self, P, run=1, evaluate=300, sch_s=None, DEBUG=False):
        self.history = np.zeros((evaluate))

        for run_id in range(run):
            print("GWO :", self.lb, self.ub, self.N)

            self.evatime = 0
            alpha_pos = self._create_state(P, sch_s)
            beta_pos = self._create_state(P, sch_s)
            delta_pos = self._create_state(P, sch_s)
            alpha_score = self._evaluate_state(P, alpha_pos)
            beta_score = self._evaluate_state(P, beta_pos)
            delta_score = self._evaluate_state(P, delta_pos)
            self.history[: min(self.evatime, evaluate)] += np.sum(alpha_score)
            positions = [self._create_state(P, sch_s) for _ in range(self.N)]

            timer_start = time.time()
            iteration_id = 0
            threads, conn = self.create_slaves(P, evaluate)

            while can_start_iteration(self.evatime, evaluate):
                previous_evatime = self.evatime
                start_time = time.time()
                positions = self.clamp_state_codes(positions)

                if DEBUG:
                    print(time.time() - start_time)

                fitness_values = self.evaluate_population(P, positions, conn)

                if DEBUG:
                    print(time.time() - start_time)

                (
                    alpha_pos,
                    beta_pos,
                    delta_pos,
                    alpha_score,
                    beta_score,
                    delta_score,
                ) = self._update_leaders(
                    P,
                    positions,
                    fitness_values,
                    alpha_pos,
                    beta_pos,
                    delta_pos,
                    alpha_score,
                    beta_score,
                    delta_score,
                )

                a = 2 - 2 * (self.evatime / evaluate)
                positions = self._move_wolves(positions, alpha_pos, beta_pos, delta_pos, a)

                self.history[previous_evatime:self.evatime] += np.sum(alpha_score)

                self.on_iteration_finish(
                    problem=P,
                    state=alpha_pos,
                    iteration=iteration_id,
                    run=run_id,
                    Name="Test",
                    time_cost=None,
                    stop=False,
                    scale=3,
                    best_value=alpha_score,
                    fitness=np.sum(alpha_score),
                )

                if iteration_id % 10 == 0:
                    print(
                        self.evatime,
                        "best",
                        alpha_score,
                        np.sum(alpha_score),
                        np.sum(beta_score),
                        np.sum(delta_score),
                        "time:",
                        time.time() - timer_start,
                    )
                iteration_id += 1

            self.close_slaves(threads, conn)
            print("final time:", time.time() - timer_start)

        self.history /= run
        return alpha_pos

    def _optimize(self, problem, budget, state=None):
        return self.run_by_evaluation_limit(
            problem,
            run=1,
            evaluate=budget,
            sch_s=state,
        )

    def search(self, problem, budget, state=None):
        coding = self._optimize(problem, budget, state)
        if coding is None:
            return None
        decoder = getattr(coding, "decode", None)
        return decoder(problem) if decoder is not None else coding
