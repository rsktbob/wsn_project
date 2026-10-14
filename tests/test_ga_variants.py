"""Behavioral checks for elitism, adaptation, budgets and runner registration."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Algorithm.core.Algorithm import Algorithm
from Algorithm.ga.GA import GA
from Algorithm.ga.ElitistGA import ElitistGA
from Algorithm.ga.AdaptiveGA import AdaptiveGA
from State.SensorEncoding import SensorEncoding
from Problem.Problem import Problem
from experiments.builder import build_algorithm


class TinyProblem:
    SENSOR_NUMBER = 3
    radius_option_counts = np.array([2, 1, 3])
    cover_candidates = []

    def prepare_coding_cache(self):
        pass


class TestVariants(unittest.TestCase):
    def test_paper_probabilities_and_domains(self):
        p = TinyProblem()
        a = AdaptiveGA(p, n=4, seed=7)
        self.assertIsInstance(a, GA)
        self.assertEqual(GA.__bases__, (Algorithm,))
        # Paper k1=k3=1, k2=k4=.5, with 0.005 default mutation.
        self.assertEqual(a._crossover_probability((1, 2), 2, 4), 1)
        self.assertEqual(a._crossover_probability((2, 3), 2, 4), 0.5)
        self.assertEqual(a._crossover_probability((2, 4), 2, 4), 0)
        self.assertEqual(a._mutation_probability(1, 2, 4), 0.5)
        self.assertEqual(a._mutation_probability(3, 2, 4), 0.25)
        self.assertEqual(a._mutation_probability(4, 2, 4), 0.005)
        # The paper's denominator is zero for an identical-fitness population.
        self.assertEqual(a._crossover_probability((2, 2), 2, 2), 1)
        self.assertEqual(a._mutation_probability(2, 2, 2), 0.5)
        np.testing.assert_allclose(a._roulette_probabilities([-2, 0, 2]), [0, 1/3, 2/3])
        np.testing.assert_allclose(a._roulette_probabilities([0, 0]), [.5, .5])
        states = [SensorEncoding(np.array([0, 0, 0, 0, 0, 0])) for _ in range(4)]
        offspring = a._make_offspring(p, states, np.array([1., 2., 3., 4.]))
        domains = a._gene_domains(p)
        self.assertEqual(len(offspring), 4)
        for state in offspring:
            self.assertTrue(np.all((state.code >= 0) & (state.code < domains)))

    def test_elites_survive_destructive_offspring(self):
        p = TinyProblem()
        class DestructiveGA(ElitistGA):
            def _create_state(self, P):
                return SensorEncoding(np.ones(6, dtype=int))

            def _mutate(self, P, states):
                for state in states:
                    state.code[:] = 0

            def evaluate(self, P, candidate):
                # Deliberately simple objective: every offspring is worse.
                self.evatime += 1
                value = float(np.sum(candidate.code))
                if value > self.fitness:
                    self.fitness = value
                    self.best_state = candidate.copy()
                return np.array([value])

            def _select(self, states, fitness):
                if self.evatime > self.N:
                    self.assert_population(states, fitness)
                return super()._select(states, fitness)

            @staticmethod
            def assert_population(states, fitness):
                assert fitness[0] == 6
                assert np.all(states[0].code == 1)

        a = DestructiveGA(p, n=4, elite_count=1, seed=1)
        a.search(p, 11)
        self.assertEqual(a.evatime, 11)
        self.assertEqual([s['offspring_count'] for s in a.generation_stats], [3, 3, 1])
        self.assertTrue(all(s['population_best'] == 6 for s in a.generation_stats))
        np.testing.assert_array_equal(a.history, np.full(11, 6))

    def test_real_problem_budget_reproducibility_and_registration(self):
        for name in ('ga_elitist', 'ga_adaptive'):
            for budget in (1, 4, 5, 11):
                runs = []
                for _ in range(2):
                    np.random.seed(7)
                    p = Problem(B=50, S=12, T=4, F=100, FILE=None)
                    a = build_algorithm(name, p, SimpleNamespace(), seed=7)
                    a.N = 4
                    result = a.run(p, budget=budget)
                    self.assertEqual(result.evaluations, budget)
                    self.assertEqual(len(result.history), budget)
                    self.assertTrue(np.isfinite(result.history).all())
                    self.assertTrue(np.all(np.diff(result.history) >= 0))
                    np.testing.assert_allclose(p.evaluate_state(result.best_state), result.best_fitness)
                    runs.append((a.best_candidate.code.copy(), result.history))
                for left, right in zip(*runs):
                    np.testing.assert_array_equal(left, right)

    def test_invalid_parameters(self):
        p = TinyProblem()
        for kwargs in ({'n': 0}, {'n': 2, 'elite_count': 2}, {'mu': -1}, {'cu': float('nan')}):
            with self.assertRaises(ValueError):
                ElitistGA(p, **kwargs)
        for kwargs in ({'k1': float('inf')}, {'k2': -0.1}, {'base_mutation': 2}):
            with self.assertRaises(ValueError):
                AdaptiveGA(p, **kwargs)


if __name__ == '__main__':
    unittest.main()
