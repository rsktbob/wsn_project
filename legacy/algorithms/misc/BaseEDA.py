import random
import time
from collections import Counter

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from Algorithm.core.budget import can_start_iteration
from Algorithm.core.evaluation import BatchEvaluator


class BaseEDA(Algorithm):
    """Template for EDA-style WSN optimizers."""

    def __init__(self, n=50, fmax=5, length=100, alpha=0.8):
        super().__init__()
        self.N = n
        self.name = self.__class__.__name__ + "_" + str(n) + "_" + str(alpha)
        self.path = None
        self.history = []
        self.fmax = fmax
        self.length = length
        self.alpha = alpha
        self.evatime = 0
        self.h = 4
        self.batch_evaluator = BatchEvaluator(
            self,
            "_evaluate_state",
            worker_count=self.h,
            label="EDA",
        )

    def _build_states(self, P, codes):
        raise NotImplementedError

    def _evaluate_state(self, P, state):
        raise NotImplementedError

    @staticmethod
    def _copy_candidate(candidate):
        """Copy either a new coding object or a legacy family state."""
        copier = getattr(candidate, "copy", None)
        if copier is not None:
            return copier()
        return candidate.Copy()

    def create_probability_vector(self):
        """Create a uniform probability vector for each gene."""
        vector = np.zeros((self.length, self.fmax))
        vector[:, :] = 1 / self.fmax
        return vector

    def create_generation(self, vector, sch_s=None):
        """Sample a generation of chromosome codes from the probability vector."""
        codes = []
        for gene_id in range(self.length):
            values = np.random.choice(
                np.arange(0, self.fmax), self.N, p=vector[gene_id]
            )
            codes.append(values)
        return np.transpose(np.array(codes))

    def run_slave(self, P, iteration, worker_id, conn):
        return self.batch_evaluator.run_worker(P, iteration, worker_id, conn)

    def evaluate_population(self, P, states, conn=None):
        """Evaluate states, optionally through worker processes."""
        fitness = self.batch_evaluator.evaluate(P, states, conn)
        return np.asarray([np.sum(value) for value in fitness])

    def create_slaves(self, P, iteration):
        """Create worker processes."""
        self.batch_evaluator.worker_count = self.h
        return self.batch_evaluator.start(P, iteration)

    def close_slaves(self, threads, conn):
        """Stop worker processes."""
        return self.batch_evaluator.close(threads, conn)

    def select_individuals(self, codes, fitness, alpha=0.7):
        """Select high-fitness codes to update the probability vector."""
        rank_ids = np.argsort(fitness)[::-1]
        selected_count = max(1, int(len(rank_ids) * alpha))
        selected_ids = rank_ids[:selected_count]
        return codes[selected_ids]

    def tournament_select_individuals(self, codes, fitness, alpha=0.7, player=4):
        """Alternative tournament-based code selection."""
        fitness = np.array(fitness)
        selected_codes = []
        player = min(player, len(codes))

        for _ in range(max(1, int(self.N * alpha))):
            selected_ids = random.sample(range(len(codes)), player)
            best_id = selected_ids[np.argmax(fitness[selected_ids])]
            selected_codes.append(codes[best_id].copy())

        return np.array(selected_codes)

    def update_probability_vector(self, selected_codes, vector):
        """Update gene probabilities from selected codes."""
        for gene_id in range(self.length):
            counts = Counter(selected_codes[:, gene_id])
            for value in range(self.fmax):
                vector[gene_id][value] = counts[value] / len(selected_codes)
        return vector

    def run_by_evaluation_limit(self, P, run=1, evaluate=300, sch_s=None):
        self.history = np.zeros((evaluate))
        global_state = None

        for run_id in range(run):
            print("EDA:", self.alpha)
            start_time = time.time()
            global_best = [0, 0, 0]
            threads, conn = self.create_slaves(P, evaluate)
            vector = self.create_probability_vector()
            iteration_id = 0
            self.evatime = 0

            while can_start_iteration(self.evatime, evaluate):
                previous_evatime = self.evatime
                codes = self.create_generation(vector, sch_s)
                states = self._build_states(P, codes)
                fitness = self.evaluate_population(P, states, conn)
                selected_codes = self.select_individuals(codes, fitness, alpha=self.alpha)
                vector = self.update_probability_vector(selected_codes, vector)

                best_id = np.argmax(fitness)
                best_value = self._evaluate_state(P, states[best_id])
                if global_state is None or np.sum(best_value) > np.sum(global_best):
                    global_state = self._copy_candidate(states[best_id])
                    global_best = best_value

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
                self.history[previous_evatime:self.evatime] += np.sum(global_best)
                iteration_id += 1

            self.close_slaves(threads, conn)

        self.history /= run
        return global_state

    def run_iterations(self, P, run=1, iteration=300, sch_s=None):
        return self.run_by_evaluation_limit(P, run=run, evaluate=iteration, sch_s=sch_s)

    def search(self, problem, budget, state=None):
        coding = self.run_by_evaluation_limit(
            problem,
            run=1,
            evaluate=budget,
            sch_s=state,
        )
        if coding is None:
            return None
        decoder = getattr(coding, "decode", None)
        return decoder(problem) if decoder is not None else coding
