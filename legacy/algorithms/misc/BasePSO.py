import random
import time

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from Algorithm.core.budget import can_start_iteration
from Algorithm.core.evaluation import BatchEvaluator


class BasePSO(Algorithm):
    """Template for PSO-style WSN optimizers."""

    def __init__(
        self, n=50, w=0.7968, end_w=0.4, vmax=0.2, vmin=-0.2, fmax=1.0, fmin=0
    ):
        super().__init__()
        self.N = n
        self.name = self.__class__.__name__ + "_" + str(n)
        self.path = None
        self.vmax = vmax
        self.vmin = vmin
        self.fmax = fmax
        self.fmin = fmin
        self.w = w
        self.end_w = end_w
        self.c1 = self.c2 = 1.4962
        self.history = []
        self.evatime = 0
        self.h = 4
        self.batch_evaluator = BatchEvaluator(
            self,
            "_evaluate_state",
            worker_count=self.h,
            label="PSO",
        )

    def _create_state(self, P, sch_s=None):
        raise NotImplementedError

    def _evaluate_state(self, P, state):
        raise NotImplementedError

    @staticmethod
    def _copy_candidate(candidate):
        """Copy new coding objects while retaining legacy family support."""
        copier = getattr(candidate, "copy", None)
        if copier is not None:
            return copier()
        return candidate.Copy()

    def create_velocity(self, state):
        """Create initial particle velocities."""
        return np.random.uniform(self.vmin, self.vmax, (self.N, len(state.code)))

    def create_swarm(self, P, sch_s=None):
        """Create particles, velocities, personal bests, and global best."""
        particles = []
        personal_bests = []
        global_best = self._create_state(P, sch_s)
        velocities = self.create_velocity(global_best)

        for _ in range(self.N):
            particle = self._create_state(P, sch_s)
            particles.append(particle)
            personal_bests.append(self._copy_candidate(particle))

        return particles, velocities, personal_bests, global_best

    def update_personal_best(self, P, particles, fitness, personal_bests, best_fitness):
        """Update each particle's personal best."""
        for index in range(self.N):
            if fitness[index] > best_fitness[index]:
                personal_bests[index] = self._copy_candidate(particles[index])
                best_fitness[index] = fitness[index]
        return personal_bests, best_fitness

    def update_global_best(self, P, particles, fitness, global_best):
        """Update the swarm global best."""
        best_id = np.argmax(fitness)
        if fitness[best_id] > np.sum(self._evaluate_state(P, global_best)):
            global_best = self._copy_candidate(particles[best_id])
        return global_best

    def update_velocity(self, particles, velocities, personal_bests, global_best):
        """Update particle velocities."""
        r1 = random.uniform(0, 1)
        r2 = random.uniform(0, 1)

        for index in range(self.N):
            velocities[index] = (
                self.w * velocities[index]
                + self.c1 * r1 * (personal_bests[index].code - particles[index].code)
                + self.c2 * r2 * (global_best.code - particles[index].code)
            )
            velocities[index][velocities[index] > self.vmax] = self.vmax
            velocities[index][velocities[index] < self.vmin] = self.vmin

        return velocities

    def wrap_code(self, code):
        """Wrap code values into the allowed numeric range."""
        return code % (self.fmax - self.fmin) + self.fmin

    def update_position(self, P, particles, velocities):
        """Move particles and clamp code values."""
        for index in range(self.N):
            particles[index].code = particles[index].code + velocities[index]
            particles[index].code[particles[index].code > self.fmax] = self.fmax
            particles[index].code[particles[index].code < self.fmin] = self.fmin
        return particles

    def run_slave(self, P, iteration, worker_id, conn):
        return self.batch_evaluator.run_worker(P, iteration, worker_id, conn)

    def evaluate_population(self, P, states, conn=None):
        """Evaluate particles, optionally through worker processes."""
        fitness = self.batch_evaluator.evaluate(P, states, conn)
        return np.asarray([np.sum(value) for value in fitness])

    def create_slaves(self, P, iteration):
        """Create worker processes."""
        self.batch_evaluator.worker_count = self.h
        return self.batch_evaluator.start(P, iteration)

    def close_slaves(self, threads, conn):
        """Stop worker processes."""
        return self.batch_evaluator.close(threads, conn)

    def run_iterations(self, P, run=1, iteration=300, sch_s=None):
        self.history = np.zeros((iteration))

        for run_id in range(run):
            particles, velocities, personal_bests, global_best = self.create_swarm(P, sch_s)
            personal_best_fitness = self.evaluate_population(P, personal_bests)
            start_time = time.time()

            for iteration_id in range(iteration):
                fitness = self.evaluate_population(P, particles)
                personal_bests, personal_best_fitness = self.update_personal_best(
                    P, particles, fitness, personal_bests, personal_best_fitness
                )
                global_best = self.update_global_best(P, particles, fitness, global_best)
                velocities = self.update_velocity(
                    particles, velocities, personal_bests, global_best
                )
                particles = self.update_position(P, particles, velocities)

                best_value = self._evaluate_state(P, global_best)
                if iteration_id % 10 == 0:
                    print(
                        iteration_id,
                        "best",
                        best_value,
                        np.sum(best_value),
                        "time:",
                        time.time() - start_time,
                    )

                self.on_iteration_finish(
                    problem=P,
                    state=global_best,
                    iteration=iteration_id,
                    run=run_id,
                    Name="Test",
                    time_cost=None,
                    stop=False,
                    scale=3,
                    best_value=best_value,
                    fitness=np.sum(best_value),
                )
                self.history[iteration_id] += np.sum(best_value)

        self.history /= run
        return global_best

    def run_by_evaluation_limit(self, P, run=1, evaluate=300, sch_s=None):
        self.history = np.zeros((evaluate))

        for run_id in range(run):
            print(str(run_id) + "_PSO:", self.vmax, self.vmin)
            start_time = time.time()
            self.evatime = 0
            particles, velocities, personal_bests, global_best = self.create_swarm(P, sch_s)
            threads, conn = self.create_slaves(P, evaluate)
            personal_best_fitness = self.evaluate_population(P, personal_bests, conn)
            self.history[: min(self.evatime, evaluate)] += np.max(
                personal_best_fitness
            )
            iteration_id = 0

            while can_start_iteration(self.evatime, evaluate):
                previous_evatime = self.evatime
                fitness = self.evaluate_population(P, particles, conn)
                personal_bests, personal_best_fitness = self.update_personal_best(
                    P, particles, fitness, personal_bests, personal_best_fitness
                )
                global_best = self.update_global_best(P, particles, fitness, global_best)
                velocities = self.update_velocity(
                    particles, velocities, personal_bests, global_best
                )
                particles = self.update_position(P, particles, velocities)

                best_value = self._evaluate_state(P, global_best)
                if iteration_id % 10 == 0:
                    print(
                        str(self.evatime),
                        "best",
                        best_value,
                        np.sum(best_value),
                        "time:",
                        time.time() - start_time,
                    )

                self.on_iteration_finish(
                    problem=P,
                    state=global_best,
                    iteration=iteration_id,
                    run=run_id,
                    Name="Test",
                    time_cost=None,
                    stop=False,
                    scale=3,
                    best_value=best_value,
                    fitness=np.sum(best_value),
                )
                self.history[previous_evatime:self.evatime] += np.sum(best_value)
                iteration_id += 1

            self.close_slaves(threads, conn)

        self.history /= run
        return global_best

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
