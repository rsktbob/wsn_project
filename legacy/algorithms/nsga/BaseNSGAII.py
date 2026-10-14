"""Stable NSGA-II flow with coding, decoded state, and search metadata split."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from Algorithm.core.budget import iterations_to_reach_budget
from Algorithm.core.initialization import sample_target_mutation_gene
from State.TargetEncoding import TargetEncoding


@dataclass
class _Individual:
    """NSGA-II-only pairing of a chromosome and its evaluated state."""

    coding: Any
    state: Any | None = None
    objectives: np.ndarray | None = None
    value: float = 0.0
    constraint_violation: float = 0.0
    rank: int = 0
    crowding_distance: float = 0.0

    def copy(self):
        return _Individual(
            coding=self.coding.copy(),
            state=None if self.state is None else self.state.copy(),
            objectives=(
                None if self.objectives is None else self.objectives.copy()
            ),
            value=float(self.value),
            constraint_violation=float(self.constraint_violation),
            rank=int(self.rank),
            crowding_distance=float(self.crowding_distance),
        )


class BaseNSGAII(Algorithm):
    """Family flow shared by sensor-coded and target-coded NSGA-II."""

    def __init__(
        self,
        n=50,
        generation=100,
        cu=0.9,
        mu=None,
        tournament_size=2,
        coding_cls=TargetEncoding,
        coding_kwargs=None,
    ):
        super().__init__()
        self.N = n
        self.generation = generation
        self.cu = cu
        self.mu = mu
        self.tournament_size = tournament_size
        self.coding_cls = coding_cls
        self.coding_kwargs = coding_kwargs or {}
        self.name = self.__class__.__name__ + "_" + str(n)
        self.history = []
        self.evatime = 0
        self.population = []
        self.pareto_front = []
        self.pareto_codings = []
        self._best_individual = None

    def _create_coding(self, problem):
        return TargetEncoding.random(
            problem.TARGET_NUMBER,
            route_selector=self.coding_kwargs.get("route_selector"),
        )

    def _create_individual(self, problem):
        individual = _Individual(self._create_coding(problem))
        self._evaluate_individual(problem, individual)
        return individual

    def _initialize_population(self, problem):
        return [self._create_individual(problem) for _ in range(self.N)]

    def _evaluate_individual(self, problem, individual):
        state = individual.coding.decode(problem)
        objectives = np.asarray(problem.evaluate_state(state), dtype=float)
        individual.state = state
        individual.objectives = objectives
        individual.value = float(np.sum(objectives))
        individual.constraint_violation = self._constraint_violation(
            problem,
            state,
        )
        state.objectives = objectives.copy()
        state.value = individual.value
        state.constraint_violation = individual.constraint_violation
        self.evatime += 1
        return objectives

    def _constraint_violation(self, problem, state):
        cost = problem.calculate_total_cost(state)
        uncovered_count = len(problem.find_uncovered_targets(state.levels))
        energy_failed_count = len(
            problem.find_energy_failed_sensors(state, cost)
        )
        disconnected_count = len(problem.find_disconnected(state))
        return uncovered_count + energy_failed_count + disconnected_count

    def dominates(self, left, right):
        if left.constraint_violation == 0 and right.constraint_violation > 0:
            return True
        if left.constraint_violation > 0 and right.constraint_violation == 0:
            return False
        if left.constraint_violation != right.constraint_violation:
            return left.constraint_violation < right.constraint_violation

        return np.all(left.objectives >= right.objectives) and np.any(
            left.objectives > right.objectives
        )

    def sort_fronts(self, population):
        dominates = {index: [] for index in range(len(population))}
        dominated_count = np.zeros(len(population), dtype=int)
        fronts = [[]]

        for left_index, left in enumerate(population):
            for right_index, right in enumerate(population):
                if left_index == right_index:
                    continue
                if self.dominates(left, right):
                    dominates[left_index].append(right_index)
                elif self.dominates(right, left):
                    dominated_count[left_index] += 1

            if dominated_count[left_index] == 0:
                left.rank = 0
                fronts[0].append(left_index)

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for left_index in fronts[current_front]:
                for right_index in dominates[left_index]:
                    dominated_count[right_index] -= 1
                    if dominated_count[right_index] == 0:
                        population[right_index].rank = current_front + 1
                        next_front.append(right_index)
            current_front += 1
            fronts.append(next_front)

        return [
            [population[index] for index in front]
            for front in fronts
            if front
        ]

    @staticmethod
    def assign_crowding_distance(front):
        if not front:
            return

        for individual in front:
            individual.crowding_distance = 0.0

        objective_count = len(front[0].objectives)
        for objective_id in range(objective_count):
            front.sort(
                key=lambda individual: individual.objectives[objective_id]
            )
            front[0].crowding_distance = math.inf
            front[-1].crowding_distance = math.inf

            min_objective = front[0].objectives[objective_id]
            max_objective = front[-1].objectives[objective_id]
            if max_objective == min_objective:
                continue

            for index in range(1, len(front) - 1):
                distance = (
                    front[index + 1].objectives[objective_id]
                    - front[index - 1].objectives[objective_id]
                ) / (max_objective - min_objective)
                front[index].crowding_distance += distance

    def rank_population(self, population):
        fronts = self.sort_fronts(population)
        for front in fronts:
            self.assign_crowding_distance(front)
        return fronts

    @staticmethod
    def _is_better(left, right):
        if left.rank != right.rank:
            return left.rank < right.rank
        if left.crowding_distance != right.crowding_distance:
            return left.crowding_distance > right.crowding_distance
        return left.value > right.value

    def _select_parent(self, population):
        player_count = min(len(population), self.tournament_size)
        selected = random.sample(population, player_count)
        best = selected[0]
        for individual in selected[1:]:
            if self._is_better(individual, best):
                best = individual
        return best.copy()

    def _crossover(self, parent_a, parent_b):
        child_a = parent_a.copy()
        child_b = parent_b.copy()

        if random.random() >= self.cu:
            return child_a, child_b

        mask = np.random.random(parent_a.coding.length) < 0.5
        child_a.coding.code[mask] = parent_b.coding.code[mask]
        child_b.coding.code[mask] = parent_a.coding.code[mask]
        return child_a, child_b

    def _mutate(self, problem, individual):
        coding = individual.coding
        mutation_rate = self.mu
        if mutation_rate is None:
            mutation_rate = 1.0 / max(1, coding.length)

        for gene_id in range(coding.length):
            if random.random() >= mutation_rate:
                continue

            current_value = coding.code[gene_id]
            next_value = self._random_gene_value(problem, coding, gene_id)
            retry_count = 0
            while next_value == current_value and retry_count < 5:
                next_value = self._random_gene_value(problem, coding, gene_id)
                retry_count += 1
            coding.code[gene_id] = next_value
        return individual

    def _random_gene_value(self, problem, coding, gene_id):
        return sample_target_mutation_gene(problem)

    def _create_offspring(self, problem, population):
        offspring = []
        while len(offspring) < self.N:
            parent_a = self._select_parent(population)
            parent_b = self._select_parent(population)
            child_a, child_b = self._crossover(parent_a, parent_b)
            child_a = self._mutate(problem, child_a)
            child_b = self._mutate(problem, child_b)
            self._evaluate_individual(problem, child_a)
            offspring.append(child_a)
            if len(offspring) < self.N:
                self._evaluate_individual(problem, child_b)
                offspring.append(child_b)
        return offspring

    def _select_survivors(self, population):
        next_population = []
        fronts = self.rank_population(population)

        for front in fronts:
            if len(next_population) + len(front) <= self.N:
                next_population.extend(front)
                continue

            front.sort(
                key=lambda individual: individual.crowding_distance,
                reverse=True,
            )
            remaining_count = self.N - len(next_population)
            next_population.extend(front[:remaining_count])
            break

        return next_population

    def _get_pareto_front(self, population):
        fronts = self.rank_population(population)
        return [] if not fronts else fronts[0]

    @staticmethod
    def _select_best(individuals):
        if not individuals:
            return None
        feasible = [
            individual
            for individual in individuals
            if individual.constraint_violation == 0
        ]
        candidates = feasible if feasible else individuals
        return max(
            candidates,
            key=lambda individual: (
                individual.value,
                individual.crowding_distance,
            ),
        )

    def run_generations(self, problem, run=1, iteration=None, sch_s=None):
        generation = self.generation if iteration is None else iteration
        best_individual = None
        self.history = np.zeros(generation)

        for run_id in range(run):
            start_time = time.time()
            self.evatime = 0
            population = self._initialize_population(problem)
            self.rank_population(population)

            for generation_id in range(generation):
                offspring = self._create_offspring(problem, population)
                population = self._select_survivors(population + offspring)
                pareto_front = self._get_pareto_front(population)
                current_best = self._select_best(pareto_front)
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
                    problem=problem,
                    state=current_best.state,
                    iteration=generation_id,
                    run=run_id,
                    Name="Test",
                    time_cost=None,
                    stop=False,
                    scale=3,
                    best_value=current_best.objectives,
                    fitness=current_best.value,
                    pareto_front=[item.state for item in pareto_front],
                )

            pareto_front = self._get_pareto_front(population)
            current_best = self._select_best(pareto_front)
            if best_individual is None or self._is_better(
                current_best,
                best_individual,
            ):
                best_individual = current_best.copy()

            self.population = [item.copy() for item in population]
            self.pareto_front = [item.state.copy() for item in pareto_front]
            self.pareto_codings = [
                item.coding.copy() for item in pareto_front
            ]

        self.history /= max(1, run)
        self._best_individual = best_individual.copy()
        return best_individual.coding.copy()

    def run_by_evaluation_limit(
        self,
        problem,
        run=1,
        evaluate=300,
        sch_s=None,
    ):
        generation = iterations_to_reach_budget(
            evaluate,
            initialization_evaluations=self.N,
            evaluations_per_iteration=self.N,
        )
        return self.run_generations(
            problem,
            run=run,
            iteration=generation,
            sch_s=sch_s,
        )

    def search(self, problem, budget, state=None):
        self.run_by_evaluation_limit(
            problem,
            run=1,
            evaluate=budget,
            sch_s=None,
        )
        individual = self._best_individual
        if individual is None:
            return None

        state = individual.state.copy()
        state.objectives = individual.objectives.copy()
        state.value = float(individual.value)
        state.constraint_violation = float(individual.constraint_violation)
        state.rank = int(individual.rank)
        state.crowding_distance = float(individual.crowding_distance)
        return state


__all__ = ["BaseNSGAII"]
