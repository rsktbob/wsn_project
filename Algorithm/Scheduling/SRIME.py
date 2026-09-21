import random
import time

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from Algorithm.core.budget import iterations_to_reach_budget
from State.TargetCodingState import TargetCodingState


class SRIME(Algorithm):
    """Scheduling RIME with target coding and sensing-radius repair."""

    def __init__(
        self,
        n=50,
        generation=100,
        soft_rate=0.7,
        hard_rate=0.25,
        mutation_rate=None,
        route_selector=None,
    ):
        super().__init__()
        self.N = n
        self.generation = generation
        self.soft_rate = soft_rate
        self.hard_rate = hard_rate
        self.mutation_rate = mutation_rate
        self.violation_penalty = 10.0
        self.history = []
        self.population = []
        self.best_state = None
        self.evatime = 0
        self.route_selector = route_selector
        self.name = "SRIME_" + str(n)

    def _create_state(self, P):
        state = TargetCodingState(P, route_selector=self.route_selector)
        state.code = np.full(P.TARGET_NUMBER, -1).astype(float)
        for target_id in range(P.TARGET_NUMBER):
            state.code[target_id] = self._random_gene_value(P, target_id)
        return state

    def _random_gene_value(self, P, target_id):
        candidate_count = len(P.cover_candidates[target_id])
        if candidate_count == 0:
            return -1
        board = TargetCodingState.RANK_PRECISION // candidate_count + 1
        return random.uniform(
            0,
            max(
                0.000001,
                candidate_count * board * TargetCodingState.RANK_PRECISION
                - 0.000001,
            ),
        )

    def normalize_state(self, P, state):
        for target_id in range(P.TARGET_NUMBER):
            candidate_count = len(P.cover_candidates[target_id])
            if candidate_count == 0:
                state.code[target_id] = -1
                continue

            board = TargetCodingState.RANK_PRECISION // candidate_count + 1
            code_range = candidate_count * board * TargetCodingState.RANK_PRECISION
            state.code[target_id] = state.code[target_id] % code_range
            upper_bound = code_range - 0.000001
            if state.code[target_id] >= code_range:
                state.code[target_id] = upper_bound
            if state.code[target_id] < 0:
                state.code[target_id] = 0
        return state

    def decode_state(self, P, state):
        state.Decode(P)
        return state

    def _constraint_violation(self, P, state):
        cost = P.calculate_total_cost(state)
        uncovered_count = len(P.find_uncovered_targets(state.levels))
        energy_failed_count = len(P.find_energy_failed_sensors(state, cost))
        disconnected_count = len(P.find_disconnected(state))
        return uncovered_count + energy_failed_count + disconnected_count

    def _evaluate_state(self, P, state):
        self.normalize_state(P, state)
        self.decode_state(P, state)
        self.repair_state(P, state)
        objectives = np.array(state.Evaluate(P)).astype(float)
        state.objectives = objectives
        state.constraint_violation = self._constraint_violation(P, state)
        state.value = np.sum(objectives) - (
            self.violation_penalty * state.constraint_violation
        )
        self.evatime += 1
        return objectives

    def _create_population(self, P):
        population = [self._create_state(P) for _ in range(self.N)]
        for state in population:
            self._evaluate_state(P, state)
        return population

    def _is_better(self, left, right):
        if right is None:
            return True
        if left.constraint_violation != right.constraint_violation:
            return left.constraint_violation < right.constraint_violation
        return left.value > right.value

    def _select_best(self, population):
        best = None
        for state in population:
            if self._is_better(state, best):
                best = state
        return best

    def _update_state(self, P, state, best_state, progress):
        child = state.Copy()
        if child.code is None or best_state.code is None:
            return child

        mutation_rate = self.mutation_rate
        if mutation_rate is None:
            mutation_rate = 1 / max(1, child.length)

        exploration_scale = max(0.05, 1.0 - progress)
        for gene_id in range(child.length):
            if random.random() < self.hard_rate * progress:
                child.code[gene_id] = best_state.code[gene_id]
                continue

            if random.random() < self.soft_rate:
                direction = best_state.code[gene_id] - child.code[gene_id]
                noise = random.uniform(-1.0, 1.0) * exploration_scale
                child.code[gene_id] = (
                    child.code[gene_id] + direction * random.random() + noise
                )

            if random.random() < mutation_rate:
                child.code[gene_id] = self._random_gene_value(P, gene_id)

        self.normalize_state(P, child)
        return child

    def _target_sensor_distance(self, P):
        distance = getattr(P, "target_sensor_distance", None)
        if distance is None:
            distance = P.G.CalDistance(P.target, P.sensor)
            P.target_sensor_distance = distance
        return distance

    def _radius_to_level(self, P, sensor_id, radius):
        return P.level_for_radius(sensor_id, radius)

    def _build_open_sensors(self, P, state):
        active_sensors = np.where(state.levels > 0)[0]
        if len(active_sensors) == 0:
            return []

        priority = P.proximity_score + P.energy_score
        sorted_sensors = active_sensors[np.argsort(priority[active_sensors])[::-1]]
        return [
            [sensor_id, state.levels[sensor_id]]
            for sensor_id in sorted_sensors
        ]

    def _repair_sensing_radius(self, P, state):
        if state.target_assignment is None:
            return state

        distance = self._target_sensor_distance(P)
        state.sensing_radii = np.zeros(P.SENSOR_NUMBER).astype(float)
        state.use_continuous_radius = P.sensing_mode != "discrete"

        for target_id, sensor_id in enumerate(state.target_assignment):
            if sensor_id < 0:
                continue
            sensor_id = int(sensor_id)
            state.sensing_radii[sensor_id] = max(
                state.sensing_radii[sensor_id],
                float(distance[target_id, sensor_id]),
            )

        for sensor_id in range(P.SENSOR_NUMBER):
            if state.sensing_radii[sensor_id] <= 0:
                state.levels[sensor_id] = 0
                continue
            state.levels[sensor_id] = self._radius_to_level(
                P,
                sensor_id,
                state.sensing_radii[sensor_id],
            )

        state.next_hops = np.repeat(-1, P.SENSOR_NUMBER)
        state.remaining_capacity = np.repeat(
            -1, P.DEVICE_NUMBER
        ).astype(float)
        state.tx_load = np.zeros(P.SENSOR_NUMBER).astype(int)
        state.RouDecoding(P, self._build_open_sensors(P, state))
        P.resolve_state_radius(state)
        return state

    def repair_state(self, P, state):
        return self._repair_sensing_radius(P, state)

    def run_generations(self, P, run=1, iteration=None, sch_s=None):
        generation = self.generation if iteration is None else iteration
        self.history = np.zeros(generation)
        best_state = None

        for run_id in range(run):
            start_time = time.time()
            self.evatime = 0
            population = self._create_population(P)
            current_best = self._select_best(population)

            for generation_id in range(generation):
                progress = (generation_id + 1) / max(1, generation)
                next_population = [current_best.Copy()]

                while len(next_population) < self.N:
                    parent = random.choice(population)
                    child = self._update_state(
                        P,
                        parent,
                        current_best,
                        progress,
                    )
                    self._evaluate_state(P, child)
                    next_population.append(child)

                population = next_population
                current_best = self._select_best(population)
                self.history[generation_id] += current_best.value

                if generation_id % 10 == 0:
                    print(
                        generation_id,
                        "best",
                        current_best.objectives,
                        "violation",
                        current_best.constraint_violation,
                        "time",
                        round(time.time() - start_time, 2),
                    )

                self.on_iteration_finish(
                    problem=P,
                    state=current_best,
                    iteration=generation_id,
                    run=run_id,
                    Name="Test",
                    time_cost=None,
                    stop=False,
                    scale=3,
                    best_value=current_best.objectives,
                    fitness=current_best.value,
                )

            if self._is_better(current_best, best_state):
                best_state = current_best.Copy()

            self.population = population
            self.best_state = best_state.Copy()

        self.history /= max(1, run)
        return best_state

    def search(self, P, budget, state=None):
        generation = iterations_to_reach_budget(
            budget,
            initialization_evaluations=self.N,
            evaluations_per_iteration=max(1, self.N - 1),
        )
        return self.run_generations(
            P,
            run=1,
            iteration=generation,
            sch_s=None,
        )
