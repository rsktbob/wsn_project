import time

import numpy as np

from Algorithm.core.Algorithm import Algorithm
from Algorithm.core.budget import can_start_iteration


class BaseGA(Algorithm):
    """GA 家族共用且穩定的族群搜尋流程。"""

    def __init__(self, n=50, l=100, cu=0.9, mu=0.4, seed=None):
        super().__init__(seed=seed)
        self.N = n
        self.cu = cu
        self.mu = mu
        self.name = self.__class__.__name__ + "_" + str(n) + str(self.mu) + str(self.cu)
        self.path = None
        self.history = []
        self.length = l
        self.evatime = 0

    # GA、SGA、RGA 僅實作 state、fitness 與兩個 genetic operators。
    def _create_state(self, problem, initial_state=None):
        raise NotImplementedError

    def _crossover_states(self, father, mother, start, end):
        raise NotImplementedError

    def _mutate_state(self, problem, state):
        raise NotImplementedError

    def _copy_state(self, state):
        """Copy one GA search object; subclasses may also copy snapshots."""
        return state.copy()

    def create_population(self, P, sch_s=None, agent=None):
        """Create the initial population."""
        return [self._create_state(P, sch_s) for _ in range(self.N)]

    def evaluate_population(self, P, states):
        """Evaluate every state and return scalar fitness values."""
        fitness = np.zeros(len(states))
        for index, state in enumerate(states):
            fitness[index] = np.sum(self.evaluate(P, state))
        return fitness

    def tournament_select(self, states, fitness, player, number):
        """Select states by tournament selection."""
        fitness = np.array(fitness)
        selected_states = []
        player = min(player, len(states))

        for _ in range(number):
            selected_ids = self.random.sample(range(len(states)), player)
            best_id = selected_ids[np.argmax(fitness[selected_ids])]
            selected_states.append(self._copy_state(states[best_id]))

        return selected_states

    def crossover_segment(self, father_code, mother_code, start, end):
        """Swap one segment between two code arrays."""
        child_a = np.concatenate(
            (father_code[0:start], mother_code[start:end], father_code[end:])
        )
        child_b = np.concatenate(
            (mother_code[0:start], father_code[start:end], mother_code[end:])
        )
        return child_a, child_b

    def crossover_population(self, P, states, rate):
        """Apply crossover to a shuffled population."""
        if len(states) < 2 or self.length <= 1:
            return states

        diff = 2
        sequence = self.random.sample(range(len(states)), len(states))
        for index in range(0, len(sequence) - 1, 2):
            if self.random.random() >= rate:
                continue

            half = max(1, self.length // 2)
            left_high = max(diff, half - diff)
            right_low = min(self.length - 1, half + diff)
            right_high = max(right_low, self.length - diff)
            start = self.random.randint(diff, left_high)
            end = self.random.randint(right_low, right_high)

            left_id = sequence[index]
            right_id = sequence[index + 1]
            states[left_id], states[right_id] = self._crossover_states(
                states[left_id], states[right_id], start, end
            )

        return states

    def mutate_population(self, P, states, rate):
        """Mutate states by applying the subclass transition operator."""
        for index in range(len(states)):
            if self.random.random() < rate:
                states[index] = self._mutate_state(
                    P,
                    self._copy_state(states[index]),
                )
        return states

    def print_state(self, P, state):
        state.PrintState(P)
        print("best", self.evaluate(P, state))

    def run_iterations(self, P, run=1, iteration=300, sch_s=None):
        """以固定迭代數執行原本的 GA 流程。"""
        start_time = time.time()
        self.history = np.zeros((iteration))

        for run_id in range(run):
            states = self.create_population(P, sch_s)
            fitness = self.evaluate_population(P, states)
            best_id = np.argmax(fitness)

            for iteration_id in range(iteration):
                selected = self.tournament_select(states, fitness, 4, self.N)
                children = self.crossover_population(P, selected, self.cu)
                states = self.mutate_population(P, children, self.mu)
                fitness = self.evaluate_population(P, states)

                best_id = np.argmax(fitness)
                best_value = self.evaluate(P, states[best_id])
                same_ratio = np.sum(fitness == np.sum(best_value)) / self.N

                if iteration_id % 10 == 0:
                    print(
                        iteration_id,
                        "best",
                        best_value,
                        "same_ratio:",
                        same_ratio,
                        np.sum(best_value),
                        "time:",
                        time.time() - start_time,
                    )

                self.on_iteration_finish(
                    problem=P,
                    state=states[best_id],
                    iteration=iteration_id,
                    run=run_id,
                    Name="Test",
                    time_cost=None,
                    stop=False,
                    scale=3,
                    best_value=best_value,
                    fitness=np.sum(best_value),
                )
                self.history[iteration_id] += np.sum(best_value)

        self.history /= run
        return states[best_id]

    def run_by_evaluation_limit(self, P, run=1, evaluate=300, sch_s=None, agent=None):
        self.history = np.zeros((evaluate))

        for run_id in range(run):
            print(str(run_id) + "_GA:", self.N, self.mu, self.cu)
            self.evatime = 0
            states = self.create_population(P, sch_s, agent)
            fitness = self.evaluate_population(P, states)
            best_id = np.argmax(fitness)
            self.history[: min(self.evatime, evaluate)] += fitness[best_id]
            start_time = time.time()
            iteration_id = 0

            while can_start_iteration(self.evatime, evaluate):
                previous_evatime = self.evatime
                selected = self.tournament_select(states, fitness, 4, self.N)
                children = self.crossover_population(P, selected, self.cu)
                states = self.mutate_population(P, children, self.mu)
                fitness = self.evaluate_population(P, states)

                best_id = np.argmax(fitness)
                best_value = self.evaluate(P, states[best_id])
                same_ratio = np.sum(fitness == np.sum(best_value)) / self.N

                if iteration_id % 10 == 0:
                    print(
                        self.evatime,
                        "best",
                        best_value,
                        "same_ratio:",
                        round(same_ratio, 2),
                        np.sum(best_value),
                        "time:",
                        time.time() - start_time,
                    )

                self.on_iteration_finish(
                    problem=P,
                    state=states[best_id],
                    iteration=iteration_id,
                    run=run_id,
                    Name="Test",
                    time_cost=None,
                    stop=False,
                    scale=3,
                    best_value=best_value,
                    fitness=np.sum(best_value),
                )
                self.history[previous_evatime:self.evatime] += np.sum(best_value)
                iteration_id += 1

        self.history /= run
        return states[best_id]

    def search(self, problem, budget, state=None):
        """Run GA and return the decoded state of the final chromosome."""
        coding = self.run_by_evaluation_limit(
            problem,
            run=1,
            evaluate=budget,
            sch_s=state,
        )
        if coding is None:
            return None
        decoder = getattr(coding, "decode", None)
        return decoder(problem) if decoder is not None else coding
