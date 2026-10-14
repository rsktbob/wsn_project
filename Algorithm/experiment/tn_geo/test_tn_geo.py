"""Numerical and WSN integration checks (run with the existing wsnenv Python)."""
import itertools
import os
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace
import numpy as np

HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
os.environ.setdefault('NUMBA_CACHE_DIR', str(HERE / '.numba_cache'))
sys.path.insert(0, str(HERE.parents[2]))
from Algorithm.experiment.tn_geo.models import BornMPS, IndependentCategorical
from Algorithm.experiment.tn_geo.optimizer import TNGEO
from experiments.maps import load_map_specs
from experiments.problem_setup import build_problem


class Models(unittest.TestCase):
    def test_probabilities_and_sampling(self):
        model = BornMPS([2, 3, 2], 2, np.random.default_rng(5))
        states = np.array(list(itertools.product(range(2), range(3), range(2))))
        dense = np.array([(model.cores[0][:, x[0], :] @ model.cores[1][:, x[1], :]
                          @ model.cores[2][:, x[2], :]).item()**2 for x in states])
        p = dense/dense.sum()
        np.testing.assert_allclose(np.exp(model.log_prob(states)), p, atol=1e-12)
        samples = model.sample(12000, np.random.default_rng(1))
        empirical = np.array([np.mean(np.all(samples == x, axis=1)) for x in states])
        np.testing.assert_allclose(empirical, p, atol=.018)

    def test_analytic_gradient(self):
        model = BornMPS([2, 3, 2], 2, np.random.default_rng(6))
        x = np.array([[0, 1, 1], [1, 2, 0]])
        w = np.array([.3, .7])
        loss, grads = model.loss_grad(x, w)
        self.assertAlmostEqual(loss, -np.dot(w, model.log_prob(x)), places=10)
        for g, grad in zip(model.cores, grads):
            for idx in np.ndindex(g.shape):
                old = g[idx]
                g[idx] = old + 1e-6
                plus = -np.dot(w, model.log_prob(x))
                g[idx] = old - 1e-6
                minus = -np.dot(w, model.log_prob(x))
                g[idx] = old
                self.assertAlmostEqual(grad[idx], (plus-minus)/2e-6, delta=2e-5)

    def test_learn_complementarity(self):
        x, w = np.array([[0, 2], [2, 0]]), np.array([.5, .5])
        model = BornMPS([3, 3], 2, np.random.default_rng(7))
        before = -np.dot(w, model.log_prob(x))
        after = model.fit(x, w, steps=250, learning_rate=.02)
        self.assertLess(after, before)
        self.assertGreater(np.exp(model.log_prob(x)).sum(), .98)
        independent = IndependentCategorical([3, 3], 1, None)
        independent.fit(x, w)
        self.assertAlmostEqual(np.exp(independent.log_prob(x)).sum(), .5, places=8)

    def test_long_chain_finite(self):
        model = BornMPS([3, 2]*100, 2, np.random.default_rng(2))
        x = model.sample(2, np.random.default_rng(3))
        self.assertTrue(np.isfinite(model.log_prob(x)).all())
        loss, grads = model.loss_grad(x, np.array([.5, .5]))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(g).all() for g in grads))


class WSN(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = load_map_specs('maps/maps_100100100.json')[0]

    def problem(self, mode='discrete'):
        return build_problem(SimpleNamespace(_current_map=self.spec, sensing_mode=mode,
                                            fitness_service='v2'), 7)

    def test_budget_reproducibility_and_reset(self):
        p = self.problem()
        energy = p.energy.copy()
        config = dict(seed=9, rank=2, initial_size=4, batch_size=4, train_size=4, train_steps=1)
        a = TNGEO(**config)
        result = a.run(p, budget=11)
        again = TNGEO(**config).run(p, budget=11)
        self.assertEqual(result.evaluations, 11)
        self.assertEqual(result.metadata['best_code'], again.metadata['best_code'])
        np.testing.assert_array_equal(result.history, again.history)
        np.testing.assert_array_equal(energy, p.energy)
        self.assertTrue(np.all(np.diff(result.history) >= 0))
        np.testing.assert_allclose(result.best_fitness, p.evaluate_state(result.best_state))
        self.assertEqual(sum(result.metadata['source_evaluations'].values()), 11)
        repeated = a.run(p, budget=3)
        self.assertEqual(repeated.evaluations, 3)
        self.assertEqual(len(repeated.history), 3)

    def test_matched_initialization(self):
        p = self.problem('bucketed')
        results = [TNGEO(seed=8, model=m, order=o, initial_size=5).run(p, budget=5)
                   for m, o in [('mps', 'coverage'), ('mps', 'random'), ('independent', 'identity')]]
        for result in results[1:]:
            np.testing.assert_array_equal(result.history, results[0].history)
        result = TNGEO(seed=3, model='independent', initial_size=3, batch_size=3,
                       train_size=4).run(p, budget=8)
        self.assertEqual(result.evaluations, 8)
        code = np.array(result.metadata['best_code'])
        self.assertTrue(np.all(code[::2] < p.radius_option_counts))
        self.assertTrue(np.all(code[1::2] < 10))

    def test_validation(self):
        for kwargs in [dict(rank=0), dict(exploration=0), dict(temperature=0),
                       dict(mutation=.95), dict(learning_rate=float('nan'))]:
            with self.assertRaises(ValueError):
                TNGEO(**kwargs)


if __name__ == '__main__':
    unittest.main()
