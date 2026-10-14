"""Experimental GEO-inspired optimizer; shared WSN encoding/fitness stay unchanged."""
import math
import time
import numpy as np

from Algorithm.core.Algorithm import Algorithm
from State.SensorEncoding import SensorEncoding
from Algorithm.experiment.tn_geo.models import BornMPS, IndependentCategorical


class TNGEO(Algorithm):
    def __init__(self, seed=7, model='mps', order='coverage', rank=4,
                 batch_size=32, initial_size=64, train_size=64, train_steps=3,
                 learning_rate=.02, temperature=1., exploration=.1, mutation=.2,
                 archive_size=512):
        super().__init__(seed=seed)
        if model not in ('mps', 'independent') or order not in ('identity', 'random', 'coverage'):
            raise ValueError('unknown model or ordering')
        for value in (rank, batch_size, initial_size, train_size, train_steps, archive_size):
            if int(value) != value or value < 1:
                raise ValueError('sizes, rank and steps must be positive integers')
        if not np.isfinite([learning_rate, temperature, exploration, mutation]).all():
            raise ValueError('hyperparameters must be finite')
        if learning_rate <= 0 or temperature <= 0 or exploration <= 0 or mutation < 0 or exploration+mutation > 1:
            raise ValueError('positive learning rate/temperature/exploration; exploration+mutation <= 1')
        self.config = dict(model=model, order=order, rank=rank, batch_size=batch_size,
                           initial_size=initial_size, train_size=train_size, train_steps=train_steps,
                           learning_rate=learning_rate, temperature=temperature,
                           exploration=exploration, mutation=mutation, archive_size=archive_size)

    def run(self, problem, *, budget=10000, max_iteration=800, state=None):
        if int(budget) != budget or budget < 1:
            raise ValueError('budget must be a positive integer')
        result = super().run(problem, budget=budget, max_iteration=max_iteration, state=state)
        result.metadata.update(self.diagnostics)
        return result

    def _order(self, problem, rng):
        count = problem.SENSOR_NUMBER
        if self.config['order'] == 'random':
            sensors = rng.permutation(count)
        elif self.config['order'] == 'coverage':
            cover = np.any(problem.coverage_table, axis=2).T.astype(np.int64)
            overlap = cover @ cover.T
            remaining = set(range(count))
            current = int(np.argmax(overlap.sum(axis=1)))
            sensors = []
            while remaining:
                sensors.append(current)
                remaining.remove(current)
                if remaining:
                    current = max(remaining, key=lambda j: (overlap[current, j], -j))
            sensors = np.asarray(sensors)
        else:
            sensors = np.arange(count)
        # Keep each sensor's level and priority next to each other.
        return np.column_stack((2*sensors, 2*sensors+1)).ravel()

    def search(self, problem, budget, state=None):
        if state is not None:
            raise ValueError('warm starts are not supported; each run uses current energy only')
        start = time.perf_counter()
        cfg = self.config
        problem.prepare_coding_cache()
        self.evatime = self.iteration = 0
        self.history, trace = [], []
        self.reset_best()
        # Separate random streams: identical initialization for all model/order arms.
        run_seed = self.next_seed()
        init_rng = np.random.default_rng(run_seed)
        sample_rng = np.random.default_rng(run_seed + 1)
        model_rng = np.random.default_rng(run_seed + 2)
        permutation = self._order(problem, np.random.default_rng(run_seed + 3))
        domains = np.empty(problem.SENSOR_NUMBER*2, dtype=int)
        domains[::2] = problem.radius_option_counts
        domains[1::2] = SensorEncoding.RANK_PRECISION
        model_type = BornMPS if cfg['model'] == 'mps' else IndependentCategorical
        model = model_type(domains[permutation], cfg['rank'], model_rng)
        seen, physical = {}, set()
        duplicates = feasible = proposals = 0
        training_seconds = 0.
        sources = dict(initial=0, model=0, uniform=0, mutation=0, fallback=0)
        space_size = math.prod(map(int, domains))

        def evaluate(code, source):
            nonlocal duplicates, feasible, proposals
            proposals += 1
            key = tuple(map(int, code))
            if key in seen:
                duplicates += 1
                return False
            candidate = SensorEncoding(np.asarray(code, dtype=int).copy())
            decoded = candidate.decode(problem)
            objectives = self.evaluate(problem, decoded, source_candidate=candidate)
            if not np.isfinite(objectives).all():
                raise FloatingPointError('WSN fitness must be finite')
            score = float(objectives.sum())
            seen[key] = score
            physical.add((tuple(decoded.levels), tuple(decoded.next_hops), tuple(decoded.tx_load)))
            cost = problem.calculate_total_cost(decoded)
            feasible += int(problem.LifeCheck(decoded, cost))
            sources[source] += 1
            self.history.append(self.fitness)
            return True

        def uniform(rng):
            return rng.integers(0, domains)

        def fill_uniform(target):
            # Bounded retries; finite tiny spaces terminate instead of looping forever.
            for _ in range(max(100, 20*(target-self.evatime))):
                if self.evatime >= target or len(seen) == space_size:
                    break
                evaluate(uniform(sample_rng), 'fallback')

        initial_target = min(cfg['initial_size'], budget)
        for _ in range(initial_target):
            code = uniform(init_rng)
            # Match existing GA/SETS CCS policy (including overwrite behavior).
            for choices in problem.cover_candidates:
                if choices:
                    sensor, level = choices[int(init_rng.integers(len(choices)))]
                    code[2*int(sensor)] = int(level)
            evaluate(code, 'initial')
        fill_uniform(initial_target)
        reason = 'budget'
        while self.evatime < budget:
            if len(seen) == space_size:
                reason = 'space_exhausted'
                break
            if self.iteration >= self.max_iterations:
                reason = 'iteration_limit'
                break
            before = self.evatime
            archive = sorted(seen, key=seen.get, reverse=True)[:cfg['archive_size']]
            scores = np.array([seen[k] for k in archive])
            scale = max(float(scores.std()), 1e-12) * cfg['temperature']
            weights = np.exp(np.maximum((scores-scores.max())/scale, -700))
            weights /= weights.sum()
            ids = sample_rng.choice(len(archive), cfg['train_size'], p=weights)
            training = np.asarray([archive[i] for i in ids])[:, permutation]
            t0 = time.perf_counter()
            loss = model.fit(training, np.ones(len(training))/len(training),
                             cfg['train_steps'], cfg['learning_rate'])
            training_seconds += time.perf_counter()-t0
            target = min(budget, self.evatime+cfg['batch_size'])
            samples = model.sample(target-self.evatime, sample_rng)
            for sample in samples:
                source = 'model'
                draw = sample_rng.random()
                if draw < cfg['exploration']:
                    code, source = uniform(sample_rng), 'uniform'
                elif draw < cfg['exploration']+cfg['mutation']:
                    code = np.array(archive[int(sample_rng.choice(len(archive), p=weights))])
                    # Change a sensor pair (both level and priority), optionally two sensors.
                    for sensor in sample_rng.choice(problem.SENSOR_NUMBER,
                                                     min(2, problem.SENSOR_NUMBER), replace=False):
                        for gene in (2*sensor, 2*sensor+1):
                            if domains[gene] > 1:
                                code[gene] = (code[gene]+sample_rng.integers(1, domains[gene])) % domains[gene]
                    source = 'mutation'
                else:
                    code = np.empty_like(sample)
                    code[permutation] = sample
                evaluate(code, source)
            fill_uniform(target)
            self.iteration += 1
            trace.append(dict(iteration=self.iteration, evaluations=self.evatime,
                              best_score=self.fitness, training_nll=loss,
                              feasible_evaluations=feasible, unique_decoded=len(physical)))
            if self.evatime == before:
                reason = 'novelty_stalled'
                break
        self.diagnostics = dict(config=cfg.copy(), run_seed=run_seed,
            gene_permutation=permutation.tolist(), stop_reason=reason,
            early_stopped=self.evatime < budget, proposals=proposals,
            duplicate_proposals=duplicates, unique_codes=len(seen),
            unique_decoded=len(physical), feasible_evaluations=feasible,
            source_evaluations=sources, training_seconds=training_seconds,
            elapsed_seconds=time.perf_counter()-start, trace=trace,
            best_code=self.best_candidate.code.tolist() if self.best_candidate else None)
        return self.best_state
