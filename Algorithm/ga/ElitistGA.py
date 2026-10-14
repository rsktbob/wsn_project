"""Generational GA with unchanged, cached elites carried into each generation."""

import numpy as np

from Algorithm.ga.GA import GA


class ElitistGA(GA):
    def __init__(self, DP, n=30, cu=0.8, mu=None, elite_count=2, seed=None):
        if not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        if not isinstance(elite_count, int) or not 0 <= elite_count < n:
            raise ValueError("elite_count must be an integer in [0, n)")
        if not 0 <= cu <= 1 or (mu is not None and not 0 <= mu <= 1):
            raise ValueError("cu and mu must be probabilities")
        super().__init__(DP, n=n, cu=cu, mu=mu, seed=seed)
        self.elite_count = elite_count
        self.generation_stats = []

    def _prepare_generation(self, P, population, stagnant_generations):
        return {}

    def search(self, P, budget, state=None):
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("budget must be a positive integer")
        self.reset_best()
        self.evatime = 0
        self.iteration = 0
        self.history = np.zeros(budget)
        self.generation_stats = []
        # Match the original GA's coverage-seeded initialization.
        population = [self._create_state(P) for _ in range(min(self.N, budget))]
        objectives, fitness = self.evaluate_batch(P, population, record_history=True)
        stagnant_generations = 0
        generation = 0
        while self.evatime < budget:
            previous_best = self.fitness
            diagnostics = self._prepare_generation(P, population, stagnant_generations)
            elite_ids = np.argsort(-fitness, kind="stable")[:self.elite_count]
            elites = [population[i].copy() for i in elite_ids]
            elite_objectives = [objectives[i].copy() for i in elite_ids]
            child_count = min(self.N - len(elites), budget - self.evatime)
            offspring = self._select(population, fitness)[:child_count]
            self._crossover(offspring)
            self._mutate(P, offspring)
            child_objectives, child_fitness = self.evaluate_batch(P, offspring, record_history=True)
            # No re-evaluation: the problem is fixed within a search call.
            population = elites + offspring
            objectives = elite_objectives + child_objectives
            fitness = np.concatenate((fitness[elite_ids], child_fitness))
            stagnant_generations = (
                0 if self.fitness > previous_best else stagnant_generations + 1
            )
            best_id = int(np.argmax(fitness))
            self.generation_stats.append({
                "generation": generation,
                "evaluations": self.evatime,
                "elite_count": len(elites),
                "offspring_count": child_count,
                "population_best": float(fitness[best_id]),
                **diagnostics,
            })
            self.on_iteration_finish(
                problem=P, state=population[best_id], iteration=generation,
                run=0, best_value=objectives[best_id], fitness=fitness[best_id],
                **diagnostics,
            )
            generation += 1
        return self.best_state.copy()
