"""PSO integration checks for gene domains and evaluation accounting."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Algorithm.pso.PSO import PSO
from Problem.Problem import Problem
from State.SensorEncoding import SensorEncoding
from experiments.builder import build_algorithm


class HeterogeneousProblem:
    SENSOR_NUMBER = 3
    radius_option_counts = np.array([2, 1, 4])

    def prepare_coding_cache(self):
        pass


class TestPSO(unittest.TestCase):
    def test_per_sensor_domains_and_boundaries(self):
        algorithm = PSO(HeterogeneousProblem(), seed=7)
        np.testing.assert_array_equal(algorithm.domains, [2, 10, 1, 10, 4, 10])
        np.testing.assert_array_equal(algorithm._candidate(np.ones(6)).code, [1, 9, 0, 9, 3, 9])
        np.testing.assert_array_equal(algorithm._candidate(np.zeros(6)).code, np.zeros(6))
        self.assertIsInstance(algorithm._candidate(np.full(6, 0.5)), SensorEncoding)

    def test_runner_budget_and_seed(self):
        for budget in (1, 5, 11):
            runs = []
            for _ in range(2):
                np.random.seed(13)
                problem = Problem(B=50, S=12, T=4, F=100, FILE=None)
                algorithm = build_algorithm('pso', problem, SimpleNamespace(), seed=7)
                algorithm.N = 4
                result = algorithm.run(problem, budget=budget)
                self.assertEqual(result.evaluations, budget)
                self.assertEqual(len(result.history), budget)
                self.assertTrue(np.all(np.isfinite(result.history)))
                self.assertTrue(np.all(np.diff(result.history) >= 0))
                self.assertTrue(np.all((algorithm.best_candidate.code >= 0) &
                                       (algorithm.best_candidate.code < algorithm.domains)))
                np.testing.assert_allclose(problem.evaluate_state(result.best_state), result.best_fitness)
                runs.append((algorithm.best_candidate.code.copy(), result.history.copy()))
            for left, right in zip(*runs):
                np.testing.assert_array_equal(left, right)


if __name__ == '__main__':
    unittest.main()
