"""Use MEALPY's unmodified BaseGA operators with the project's WSN contract."""
import math
from importlib.metadata import version
import numpy as np
from Algorithm.core.Algorithm import Algorithm
from State.SensorEncoding import SensorEncoding


class _BudgetReached(Exception):
    """Stop before an extra real WSN objective evaluation."""


class MealpyGA(Algorithm):
    def __init__(self, problem, n=30, pc=0.95, pm=0.025, seed=None):
        super().__init__(seed=seed)
        # BaseGA creates pairs; its 20% tournament needs at least two parents.
        if int(n) != n or n < 10 or n > 10000 or n % 2:
            raise ValueError('n must be even and between 10 and 10000')
        if not 0 < pc < 1 or (pm != "1/d" and not 0 < pm < 1):
            raise ValueError('pc must be in (0,1); pm must be in (0,1) or 1/d')
        try:
            from mealpy import GA, IntegerVar, Problem
        except ImportError as error:
            raise ImportError('Install Algorithm/experiment/mealpy_ga/requirements.txt in its isolated venv') from error
        if version('mealpy') != '3.0.3':
            raise RuntimeError('This adapter is verified with mealpy==3.0.3')
        self._GA, self._IntegerVar, self._Problem = GA, IntegerVar, Problem
        self.n, self.pc = int(n), float(pc)
        self.pm = "1/d" if pm == "1/d" else float(pm)
        self.name = 'MEALPY BaseGA'

    def search(self, problem, budget, state=None):
        # Fresh GA population on each lifetime reoptimization, persistent seed stream.
        problem.prepare_coding_cache()
        self.reset_best()
        self.evatime = self.iteration = 0
        history = []
        domains = np.empty(2 * problem.SENSOR_NUMBER, dtype=int)
        domains[0::2] = problem.radius_option_counts
        domains[1::2] = SensorEncoding.RANK_PRECISION
        self.effective_pm = 1.0 / len(domains) if self.pm == '1/d' else self.pm
        bounds = self._IntegerVar(lb=np.zeros(len(domains), dtype=int),
                                  ub=domains - 1, name='sensor_code')

        def objective(solution):
            if self.evatime >= budget:
                raise _BudgetReached
            code = bounds.decode(solution)
            if np.any(code < 0) or np.any(code >= domains):
                raise ValueError('MEALPY produced an out-of-domain chromosome')
            score = self.score(problem, SensorEncoding(code))
            history.append(self.fitness)
            if self.evatime > self.n and (self.evatime - self.n) % self.n == 0:
                self.on_iteration_finish(problem=problem, state=self.best_state.copy(),
                                         iteration=self.iteration, best_value=self.best_objectives.copy(),
                                         fitness=self.fitness)
            return score

        # Pinning scalar objective metadata avoids MEALPY's uncounted shape probe.
        meal_problem = self._Problem(bounds=bounds, minmax='max', obj_func=objective,
                                     _n_objs=1, obj_weights=np.ones(1), log_to=None)
        epochs = min(100000, self.max_iterations,
                     max(1, math.ceil((budget - self.n) / self.n)))
        self.model = self._GA.BaseGA(epoch=epochs, pop_size=self.n, pc=self.pc, pm=self.effective_pm,
                                    selection='tournament', k_way=0.2, crossover='uniform',
                                    mutation='flip', mutation_multipoints=True)
        self.last_mealpy_seed = self.next_seed()
        try:
            self.model.solve(meal_problem, mode='single', seed=self.last_mealpy_seed)
        except _BudgetReached:
            # A partial generation/initialization is allowed. Return our best
            # actually evaluated WSN state, not a possibly stale model.g_best.
            pass
        self.history = np.asarray(history)
        self.iterations_completed = self.iteration
        self.early_stopped = self.evatime < budget
        return self.best_state.copy()
