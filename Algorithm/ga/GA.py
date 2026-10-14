import time

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from State.SensorEncoding import SensorEncoding


class GA(Algorithm):
    """Generational GA on SensorEncoding.

    Each generation: tournament selection -> two-point crossover (rate ``cu``)
    -> mutation (2023 thesis: each chromosome with probability ``mu``,
    default 0.1, redraws one random gene) -> evaluate.  Fitness is always the
    decode of the current chromosome.
    """

    TOURNAMENT_SIZE = 4
    # 2023 thesis Section 4.1: mutation = 0.1
    DEFAULT_MUTATION_RATE = 0.1

    def __init__(self, DP, n=50, cu=0.9, mu=None, seed=None):
        super().__init__(seed=seed)
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if not np.isfinite(cu) or not 0 <= cu <= 1:
            raise ValueError("cu must be a probability")
        if mu is not None and (not np.isfinite(mu) or not 0 <= mu <= 1):
            raise ValueError("mu must be a probability")
        self.N = n
        self.cu = float(cu)
        self.mu = self.DEFAULT_MUTATION_RATE if mu is None else float(mu)
        DP.prepare_coding_cache()

    @staticmethod
    def _gene_domains(P):
        """Value range of every gene: [schedule, priority] per sensor."""
        domains = np.full(P.SENSOR_NUMBER * 2, SensorEncoding.RANK_PRECISION)
        domains[0::2] = P.radius_option_counts
        return domains

    def _create_state(self, P):
        coding = SensorEncoding.random(P.SENSOR_NUMBER, P.radius_option_counts, rng=self.rng)
        # Seed coverage like SA-SETS: each target turns on one valid (sensor, level);
        # later targets may overwrite an earlier choice for the same sensor.
        for target_candidates in P.cover_candidates:
            if target_candidates:
                sensor_id, level = self.random.choice(target_candidates)
                coding.code[int(sensor_id) * 2] = int(level)
        return coding

    def _select(self, states, fitness):
        player = min(self.TOURNAMENT_SIZE, len(states))
        selected = []
        for _ in range(self.N):
            ids = self.random.sample(range(len(states)), player)
            selected.append(states[ids[np.argmax(fitness[ids])]].copy())
        return selected

    def _crossover_pair(self, first, second):
        """Swap one sensor-aligned two-point segment between two chromosomes."""
        sensor_count = first.length // 2
        if sensor_count < 3:
            return
        start, end = sorted(self.random.sample(range(1, sensor_count), 2))
        segment = slice(2 * start, 2 * end)
        f, m = first.code, second.code
        f[segment], m[segment] = m[segment].copy(), f[segment].copy()

    def _crossover(self, states):
        """Apply the fixed crossover probability to randomly paired states."""
        if states[0].length // 2 < 3:
            return
        order = self.random.sample(range(len(states)), len(states))
        for a, b in zip(order[0::2], order[1::2]):
            if self.random.random() < self.cu:
                self._crossover_pair(states[a], states[b])

    def _mutate_with_rates(self, P, states, rates):
        """Apply one per-gene probability to each chromosome."""
        domains = self._gene_domains(P)
        for state, rate in zip(states, rates):
            for gene_id in np.flatnonzero(self.rng.random(len(domains)) < rate):
                if domains[gene_id] > 1:
                    value = self.random.randrange(domains[gene_id] - 1)
                    state.code[gene_id] = value + (value >= state.code[gene_id])

    def _mutate(self, P, states):
        """2023 thesis mutation (Algorithm/GA.py Mutation + JGA.Transition).

        Each chromosome mutates with probability mu; one gene drawn uniformly
        from the whole chromosome is redrawn uniformly from its own domain
        (it may keep its value, as in the thesis).
        """
        domains = self._gene_domains(P)
        for state in states:
            if self.random.random() < self.mu:
                gene_id = self.random.randrange(len(domains))
                state.code[gene_id] = self.random.randrange(int(domains[gene_id]))

    def _make_offspring(self, problem, population, fitness):
        offspring = self._select(population, fitness)
        self._crossover(offspring)
        self._mutate(problem, offspring)
        return offspring

    def search(self, problem, budget, state=None):
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("budget must be a positive integer")
        self.evatime = 0
        self.history = np.zeros(budget)
        population = [self._create_state(problem) for _ in range(min(self.N, budget))]
        objectives, fitness = self.evaluate_batch(problem, population)
        self.history[:self.evatime] = self.fitness
        start_time = time.time()
        generation = 0

        while self.evatime < budget:
            previous_evatime = self.evatime
            offspring = self._make_offspring(problem, population, fitness)
            population = offspring[:budget - self.evatime]
            objectives, fitness = self.evaluate_batch(problem, population)
            self.history[previous_evatime:self.evatime] = self.fitness

            best_id = int(np.argmax(fitness))
            if generation % 10 == 0:
                same_ratio = round(float(np.mean(fitness == fitness[best_id])), 2)
                print(self.evatime, "best", objectives[best_id], "same_ratio:",
                      same_ratio, fitness[best_id], "time:", time.time() - start_time)
            self.on_iteration_finish(
                problem=problem, state=population[best_id], iteration=generation,
                run=0, best_value=objectives[best_id], fitness=fitness[best_id],
            )
            generation += 1

        return self.best_state.copy()
