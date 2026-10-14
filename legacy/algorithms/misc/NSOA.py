"""Natural Selection Optimization Algorithm (NSOA) for the coded WSN model.

This is a direct adaptation of Liu, Wang and Fang (2025),
``A novel metaheuristic algorithm with applications in parameter estimation
and engineering problems``, RAIRO Operations Research 59, 2051-2085,
https://doi.org/10.1051/ro/2025017.

The paper's search space is continuous and its objective is minimised.  The
project's WSN state is discrete, so positions are represented in [0, 1] and
decoded through ``SensorEncoding`` before every fitness evaluation.  The NSOA
movement and reproduction equations are kept intact.  By default, selection
uses a WSN feasibility-first ordering so an energy-efficient but infeasible
state cannot replace a feasible state.
"""

import math

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from State.SensorEncoding import SensorEncoding


class NSOA(Algorithm):
    """Paper-faithful NSOA using the project's combined WSN encoding.

    NSOA has two steps in every generation:

    1. Every individual moves in its search direction towards the global best
       with an element-wise ``N(0.85, 0.5)`` coefficient (Equations 1-2).
    2. The worst individuals are eliminated and regenerated around the global
       best.  Their normal distribution follows Equations 5-7, where its
       standard deviation is controlled by the paper's decreasing sigmoid.

    ``initial_elimination_probability`` is the paper's reported
    ``P_elim^0=0.5``.  With its default value the dynamic elimination formula
    is exactly Equation 4; changing it only rescales that probability for
    sensitivity experiments.
    """

    REFERENCE_NORMAL_MEAN = 0.85
    REFERENCE_NORMAL_STD = 0.5
    REFERENCE_INITIAL_ELIMINATION_PROBABILITY = 0.5
    REFERENCE_SIGMOID_DELTA = 30.0
    REFERENCE_SIGMOID_NU = 0.3

    def __init__(
        self,
        P,
        n=30,
        normal_mean=REFERENCE_NORMAL_MEAN,
        normal_std=REFERENCE_NORMAL_STD,
        initial_elimination_probability=REFERENCE_INITIAL_ELIMINATION_PROBABILITY,
        sigmoid_delta=REFERENCE_SIGMOID_DELTA,
        sigmoid_nu=REFERENCE_SIGMOID_NU,
        feasibility_first=True,
        coverage_seeded_initialization=True,
        seed=None,
    ):
        super().__init__()
        if n < 1:
            raise ValueError("n must be at least 1")
        if normal_std <= 0:
            raise ValueError("normal_std must be positive")
        if not 0 < initial_elimination_probability <= 1:
            raise ValueError("initial_elimination_probability must be in (0, 1]")
        if sigmoid_delta <= 0:
            raise ValueError("sigmoid_delta must be positive")
        if not 0 <= sigmoid_nu <= 1:
            raise ValueError("sigmoid_nu must be in [0, 1]")

        P.prepare_coding_cache()
        self.n = int(n)
        self.normal_mean = float(normal_mean)
        self.normal_std = float(normal_std)
        self.initial_elimination_probability = float(initial_elimination_probability)
        self.sigmoid_delta = float(sigmoid_delta)
        self.sigmoid_nu = float(sigmoid_nu)
        self.feasibility_first = bool(feasibility_first)
        self.coverage_seeded_initialization = bool(coverage_seeded_initialization)
        self.name = "NSOA_" + str(self.n)
        self.rng = np.random.default_rng(seed)
        self.coding_length = P.SENSOR_NUMBER * 2
        self.gene_domain = max(P.LEVEL, SensorEncoding.RANK_PRECISION)

        self.evatime = 0
        self.history = np.array([])
        self.elimination_history = []
        self.last_elimination_probability = 0.0
        self.last_elimination_count = 0
        self.last_reproduction_mean = 0.0
        self.last_reproduction_std = 0.0

    def search(self, P, budget, state=None):
        """Optimise a sensor chromosome with a strict evaluation budget."""
        evaluate = max(1, int(budget))
        run = 1
        self.history = np.zeros(evaluate, dtype=float)
        global_best_state = None
        global_best_score = -float("inf")
        global_best_rank = None

        for run_id in range(run):
            self._reset_run_statistics()
            self.evatime = 0
            population_size = min(self.n, evaluate)
            positions = self._create_initial_population(P, population_size)
            scores, ranks, states, objectives = self._evaluate_population(P, positions)

            best_index = self._best_index(ranks)
            best_score = float(scores[best_index])
            best_rank = ranks[best_index]
            best_state = states[best_index].copy()
            best_objectives = objectives[best_index].copy()
            history_end = self.evatime
            self.history[:history_end] += best_score

            iteration = 0

            while self.evatime < evaluate:
                iteration += 1
                # The paper uses (t + 1) / MaxIter.  This project stops by a
                # fitness-evaluation (FE) budget and each NSOA generation uses
                # N + M evaluations, so generation_count cannot be derived by
                # dividing the budget by N.  FE progress is the exact analogue
                # that reaches one when the requested budget is exhausted.
                progress_step = min(self.evatime, evaluate - 1)

                # Equations 1-2: every individual follows its own direction
                # towards the current global optimum.  NSOA does not use a
                # greedy per-individual acceptance rule in this phase.
                moved_positions = self._natural_selection_step(positions, best_state, P)
                movement_count = min(population_size, evaluate - self.evatime)
                moved_scores, moved_ranks, moved_states, moved_objectives = (
                    self._evaluate_population(P, moved_positions[:movement_count])
                )
                positions[:movement_count] = moved_positions[:movement_count]
                scores[:movement_count] = moved_scores
                ranks[:movement_count] = moved_ranks
                states[:movement_count] = moved_states
                objectives[:movement_count] = moved_objectives
                best_score, best_rank, best_state, best_objectives = (
                    self._update_global_best(
                    best_score,
                    best_rank,
                    best_state,
                    best_objectives,
                    moved_scores,
                    moved_ranks,
                    moved_states,
                    moved_objectives,
                    )
                )
                history_end = self._append_history(history_end, best_score)

                # Equations 3-7: rank by WSN fitness (descending, because
                # this project maximises) and regenerate the least fit states
                # around the best solution.  A partial final batch is allowed
                # so evatime always equals the requested budget.
                elimination_probability = self.EliminationProbability(
                    progress_step, evaluate
                )
                requested_eliminations = int(
                    math.floor(elimination_probability * population_size)
                )
                reproduction_count = min(
                    requested_eliminations, evaluate - self.evatime
                )
                self.last_elimination_probability = elimination_probability
                self.last_elimination_count = reproduction_count
                self.last_reproduction_mean = 0.0
                self.last_reproduction_std = 0.0

                if reproduction_count > 0:
                    worst_indexes = self._worst_indexes(ranks, reproduction_count)
                    offspring, reproduction_mean, reproduction_std = (
                        self._reproduce_near_best(
                            self.CodeToPosition(best_state.code, self.gene_domain),
                            progress_step,
                            evaluate,
                            reproduction_count,
                        )
                    )
                    (
                        offspring_scores,
                        offspring_ranks,
                        offspring_states,
                        offspring_objectives,
                    ) = self._evaluate_population(P, offspring)
                    positions[worst_indexes] = offspring
                    scores[worst_indexes] = offspring_scores
                    for target_index, child_rank, child_state in zip(
                        worst_indexes, offspring_ranks, offspring_states
                    ):
                        ranks[int(target_index)] = child_rank
                        states[int(target_index)] = child_state
                    objectives[worst_indexes] = offspring_objectives
                    best_score, best_rank, best_state, best_objectives = (
                        self._update_global_best(
                        best_score,
                        best_rank,
                        best_state,
                        best_objectives,
                        offspring_scores,
                        offspring_ranks,
                        offspring_states,
                        offspring_objectives,
                        )
                    )
                    self.last_reproduction_mean = reproduction_mean
                    self.last_reproduction_std = reproduction_std
                    history_end = self._append_history(history_end, best_score)

                self.elimination_history.append(
                    {
                        "iteration": iteration,
                        "probability": float(elimination_probability),
                        "count": int(reproduction_count),
                        "reproduction_mean": float(self.last_reproduction_mean),
                        "reproduction_std": float(self.last_reproduction_std),
                    }
                )
                self.on_iteration_finish(
                    problem=P,
                    state=best_state,
                    iteration=iteration - 1,
                    run=run_id,
                    Name=self.name,
                    time_cost=None,
                    stop=False,
                    scale=3,
                    best_value=best_objectives.copy(),
                    fitness=best_score,
                    elimination_probability=elimination_probability,
                    elimination_count=reproduction_count,
                    reproduction_mean=self.last_reproduction_mean,
                    reproduction_std=self.last_reproduction_std,
                )

            if global_best_rank is None or best_rank >= global_best_rank:
                global_best_score = best_score
                global_best_rank = best_rank
                global_best_state = best_state.copy()

        self.history /= run
        return (
            None
            if global_best_state is None
            else global_best_state.decode(P)
        )

    def _evaluate_state(self, P, coding):
        """Decode and evaluate one discrete sensor chromosome."""
        state = coding.decode(P)
        self.evatime += 1
        return P.evaluate_state(state)

    # ------------------------------------------------------------------
    # Paper equations
    # ------------------------------------------------------------------
    def NaturalSelectionStep(self, positions, best_position):
        """Apply Equations 1-2 in the continuous normalised encoding."""
        search_direction = best_position - positions  # Equation 1
        coefficient = self.rng.normal(
            loc=self.normal_mean, scale=self.normal_std, size=positions.shape
        )
        return self._clip_positions(positions + coefficient * search_direction)

    def EliminationProbability(self, zero_based_t, total_iterations):
        """Return Equation 4's decreasing random elimination probability."""
        step = min(max(1, int(zero_based_t) + 1), max(1, int(total_iterations)))
        progress = step / max(1, int(total_iterations))
        equation_four = self.rng.random() * (progress**2 - 2.0 * progress + 1.0)
        # The paper's listed P_elim^0 is 0.5.  At the default this multiplier
        # is one, so Equation 4 is used without alteration.
        multiplier = (
            self.initial_elimination_probability
            / self.REFERENCE_INITIAL_ELIMINATION_PROBABILITY
        )
        return float(np.clip(multiplier * equation_four, 0.0, 1.0))

    def ReproductionDistribution(self, zero_based_t, total_iterations):
        """Return ``ave_i`` and ``std_i`` from Equations 6-7."""
        step = min(max(1, int(zero_based_t) + 1), max(1, int(total_iterations)))
        progress = step / max(1, int(total_iterations))
        exponent = self.sigmoid_delta * (progress - self.sigmoid_nu)
        exponent = float(np.clip(exponent, -700.0, 700.0))
        std_i = 1.0 / (1.0 + math.exp(exponent))
        # Equation 5 explicitly fixes ave_i at 0.85.  Only std_i is adapted by
        # Equations 6-7; the mean must not be derived from the sigmoid.
        ave_i = self.REFERENCE_NORMAL_MEAN
        return float(ave_i), float(std_i)

    # ------------------------------------------------------------------
    # WSN encoding bridge
    # ------------------------------------------------------------------
    def _create_initial_population(self, P, population_size):
        # Paper Part 1: initialize positions with N(0.85, 0.5).
        positions = self._clip_positions(
            self.rng.normal(
                loc=self.normal_mean,
                scale=self.normal_std,
                size=(population_size, self.coding_length),
            )
        )
        if not self.coverage_seeded_initialization:
            return positions

        # SensorEncoding initialization uses the same project-specific idea: seed one
        # covering sensor/level from every cover-candidate set.  Only those
        # scheduling coordinates are overwritten; all other scheduling and
        # routing coordinates retain the paper's Normal initialization.
        fmax = self.gene_domain
        for row in range(population_size):
            for candidates in P.cover_candidates:
                if len(candidates) == 0:
                    continue
                sensor, level = candidates[int(self.rng.integers(len(candidates)))]
                positions[row, int(sensor) * 2] = (float(level) + 0.5) / fmax
        return self._clip_positions(positions)

    def _natural_selection_step(self, positions, best_state, P):
        best_position = self.CodeToPosition(best_state.code, self.gene_domain)
        return self.NaturalSelectionStep(positions, best_position)

    def _reproduce_near_best(
        self, best_position, zero_based_t, total_iterations, count
    ):
        ave_i, std_i = self.ReproductionDistribution(
            zero_based_t, total_iterations
        )
        coefficient = self.rng.normal(
            loc=ave_i, scale=std_i, size=(count, len(best_position))
        )
        # Equation 5: P_i(t+1) = nr_i(ave_i, std_i) o P_O(t+1).
        return (
            self._clip_positions(coefficient * best_position),
            ave_i,
            std_i,
        )

    def _evaluate_population(self, P, positions):
        scores = np.zeros(len(positions), dtype=float)
        ranks = []
        codings = []
        objectives = np.zeros(
            (len(positions), len(P.fitness_service.objective_names)),
            dtype=float,
        )
        for index, position in enumerate(positions):
            score, rank, coding, values = self.EvaluatePosition(P, position)
            scores[index] = score
            ranks.append(rank)
            codings.append(coding)
            objectives[index] = values
        return scores, ranks, codings, objectives

    def EvaluatePosition(self, P, position):
        coding = self.PositionToCoding(P, position)
        state = coding.decode(P)
        values = np.asarray(P.evaluate_state(state), dtype=float)
        values = np.nan_to_num(values, nan=-5.0, neginf=-5.0, posinf=5.0)
        score = float(np.sum(values))
        rank = self._selection_rank(P, state, score)
        self.evatime += 1
        return score, rank, coding, values

    @staticmethod
    def CodeToPosition(code, fmax):
        return np.asarray(code, dtype=float) / max(1, int(fmax))

    def PositionToCoding(self, P, position):
        position = np.clip(np.asarray(position, dtype=float), 0.0, 1.0 - 1e-12)
        code = np.floor(position * self.gene_domain).astype(int)
        code = np.clip(code, 0, self.gene_domain - 1)
        return SensorEncoding(code)

    def PositionToState(self, P, position):
        """Decode a continuous NSOA position for external inspection."""
        return self.PositionToCoding(P, position).decode(P)

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _reset_run_statistics(self):
        self.elimination_history = []
        self.last_elimination_probability = 0.0
        self.last_elimination_count = 0
        self.last_reproduction_mean = 0.0
        self.last_reproduction_std = 0.0

    def _update_global_best(
        self,
        best_score,
        best_rank,
        best_state,
        best_objectives,
        candidate_scores,
        candidate_ranks,
        candidate_states,
        candidate_objectives,
    ):
        if len(candidate_scores) == 0:
            return best_score, best_rank, best_state, best_objectives
        candidate_index = self._best_index(candidate_ranks)
        candidate_rank = candidate_ranks[candidate_index]
        if candidate_rank >= best_rank:
            return (
                float(candidate_scores[candidate_index]),
                candidate_rank,
                candidate_states[candidate_index].copy(),
                candidate_objectives[candidate_index].copy(),
            )
        return best_score, best_rank, best_state, best_objectives

    def _selection_rank(self, P, state, score):
        """Build a lexicographic rank; larger tuples are always preferred."""
        if not self.feasibility_first:
            return (float(score),)

        cost = P.calculate_total_cost(state)
        energy_failures = len(P.find_energy_failed_sensors(state, cost))
        disconnected = len(P.find_disconnected(state))
        uncovered = len(P.find_uncovered_targets(state.levels))
        total_violations = energy_failures + disconnected + uncovered
        feasible = int(total_violations == 0)
        return (
            feasible,
            -total_violations,
            -uncovered,
            -disconnected,
            -energy_failures,
            float(score),
        )

    @staticmethod
    def _best_index(ranks):
        return max(range(len(ranks)), key=lambda index: ranks[index])

    @staticmethod
    def _worst_indexes(ranks, count):
        ordered = sorted(range(len(ranks)), key=lambda index: ranks[index])
        return np.asarray(ordered[:count], dtype=int)

    def _append_history(self, history_end, best_score):
        new_end = self.evatime
        self.history[history_end:new_end] += best_score
        return new_end

    @staticmethod
    def _clip_positions(positions):
        return np.clip(np.asarray(positions, dtype=float), 0.0, 1.0)
