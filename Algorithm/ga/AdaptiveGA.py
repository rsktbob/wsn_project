"""Srinivas–Patnaik (1994) adaptive operator rates on SensorEncoding."""

import numpy as np

from Algorithm.ga.GA import GA


class AdaptiveGA(GA):
    """Use the paper's fitness-dependent crossover and mutation probabilities.

    The rate equations and published k values follow the 1994 AGA. The WSN
    chromosome remains integer SensorEncoding, so this is a problem adaptation
    rather than a reproduction of the paper's binary benchmark experiments.
    """

    def __init__(self, DP, n=30, k1=1.0, k2=0.5, k3=1.0, k4=0.5,
                 base_mutation=0.005, seed=None):
        for name, value in (("k1", k1), ("k2", k2), ("k3", k3),
                            ("k4", k4), ("base_mutation", base_mutation)):
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be a probability")
        super().__init__(DP, n=n, cu=1.0, mu=None, seed=seed)
        self.k1, self.k2, self.k3, self.k4 = map(float, (k1, k2, k3, k4))
        self.base_mutation = float(base_mutation)

    @staticmethod
    def _roulette_probabilities(fitness):
        """Use proportional selection; shift nonpositive WSN scores to weights."""
        scores = np.asarray(fitness, dtype=float)
        if np.any(~np.isfinite(scores)):
            raise ValueError("fitness must be finite for proportional selection")
        weights = scores - min(0.0, float(np.min(scores)))
        total = float(np.sum(weights))
        if total <= 0:
            return np.full(len(scores), 1.0 / len(scores))
        return weights / total

    def _crossover_probability(self, parent_fitness, average, maximum):
        """Paper equations (5)–(6), with zero-spread fallback."""
        better_parent = max(parent_fitness)
        if better_parent < average or maximum <= average:
            return self.k3
        return self.k1 * (maximum - better_parent) / (maximum - average)

    def _mutation_probability(self, fitness, average, maximum):
        """Paper equations (7)–(8), with the stated 0.005 baseline floor."""
        if fitness < average or maximum <= average:
            adaptive = self.k4
        else:
            adaptive = self.k2 * (maximum - fitness) / (maximum - average)
        return max(self.base_mutation, adaptive)

    def _make_offspring(self, problem, population, fitness):
        """Select proportionally; adapt crossover by pair and mutation by parent."""
        probabilities = self._roulette_probabilities(fitness)
        selected_ids = self.rng.choice(len(population), size=self.N, p=probabilities)
        offspring = [population[int(i)].copy() for i in selected_ids]
        selected_fitness = np.asarray(fitness)[selected_ids]
        average = float(np.mean(fitness))
        maximum = float(np.max(fitness))

        order = self.random.sample(range(len(offspring)), len(offspring))
        for a, b in zip(order[0::2], order[1::2]):
            rate = self._crossover_probability(
                (selected_fitness[a], selected_fitness[b]), average, maximum,
            )
            if self.random.random() < rate:
                self._crossover_pair(offspring[a], offspring[b])

        mutation_rates = [
            self._mutation_probability(value, average, maximum)
            for value in selected_fitness
        ]
        self._mutate_with_rates(problem, offspring, mutation_rates)
        return offspring
