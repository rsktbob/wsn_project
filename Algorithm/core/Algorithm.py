"""All optimizers share this small execution contract."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from Algorithm.core.result import AlgorithmResult
from Problem.services.fitness_service import FitnessService


class Algorithm:
    """Base contract for one optimizer.

    The class owns run state, deterministic random streams, common evaluation
    and best-result tracking, the public search contract, and result packaging.
    Candidate creation and search operators remain family-specific.
    """

    def __init__(self, seed=None):
        self.Fname = list(FitnessService.OBJECTIVE_NAMES)
        self.iteration = 0
        self.evatime = 0
        self.history = np.array([], dtype=float)
        self.max_iterations = 800

        # Every evaluated solution can be compared through the same decoded
        # WSN state, even when optimizers use different Encoding subclasses.
        self.best_state = None
        self.best_candidate = None
        self.best_objectives = np.array([], dtype=float)
        self.fitness = -np.inf
        self.coverage = -np.inf

        self.iteration_callback = None
        self.iteration_listeners = []
        self.iteration_report_every = None

        self._seed_random_streams(seed)

    def _seed_random_streams(self, seed):
        """Create local NumPy and Python streams from the same seed."""
        if seed is None:
            seed = int(
                np.random.SeedSequence().generate_state(1, dtype=np.uint64)[0]
            )
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.random = random.Random(self.seed)

    def next_seed(self):
        """Return a deterministic child seed for a worker or child search."""
        return int(self.rng.integers(0, 2**32, dtype=np.uint64))

    def reset_best(self):
        """Clear the best solution tracked during one optimizer run."""
        self.best_state = None
        self.best_candidate = None
        self.best_objectives = np.array([], dtype=float)
        self.fitness = -np.inf
        self.coverage = -np.inf

    def decode_candidate(self, problem, candidate):
        """Return a decoded State; optimizers may override decode caching."""
        decoder = getattr(candidate, "decode", None)
        return decoder(problem) if callable(decoder) else candidate

    def evaluate(self, problem, candidate, *, source_candidate=None):
        """Evaluate one State or Encoding and update the common best result.

        Encoding implementations share the ``decode(problem)`` contract. A
        decoded State is accepted directly.  Only this method consumes one
        unit of the optimizer's evaluation budget.
        """
        state = self.decode_candidate(problem, candidate)
        if not all(
            hasattr(state, name)
            for name in ("levels", "next_hops", "tx_load")
        ):
            raise TypeError(
                "evaluate() requires a decoded State or an Encoding with "
                "decode(problem)"
            )

        self.evatime += 1
        objectives = np.asarray(
            problem.evaluate_state(state),
            dtype=float,
        )
        state.objectives = objectives.copy()
        fitness = float(np.sum(objectives))
        self.update_best(
            problem,
            state,
            objectives,
            fitness,
            candidate=(
                candidate if source_candidate is None else source_candidate
            ),
        )
        return objectives

    def score(self, problem, candidate):
        """Evaluate one candidate and return its summed scalar fitness."""
        return float(np.sum(self.evaluate(problem, candidate)))

    def evaluate_many(self, problem, candidates):
        """Evaluate candidates and return their scalar fitness values."""
        return np.asarray(
            [self.score(problem, candidate) for candidate in candidates],
            dtype=float,
        )

    def update_best(
        self,
        problem,
        state,
        objectives,
        fitness,
        coverage=None,
        *,
        candidate=None,
    ):
        """Keep the evaluated result with the greatest scalar fitness.

        Coverage remains available as result metadata, but the common
        algorithm contract does not apply a second, hidden ordering rule on
        top of the fitness service. Algorithms with a genuinely different
        meaning of "best" may replace this method.
        """
        fitness = float(fitness)
        if self.best_state is not None and fitness <= self.fitness:
            return False

        self.best_state = state.copy()
        self.best_objectives = np.asarray(objectives, dtype=float).copy()
        self.best_state.objectives = self.best_objectives.copy()
        self.fitness = fitness
        self.coverage = None if coverage is None else float(coverage)
        copier = getattr(candidate, "copy", None)
        self.best_candidate = copier() if callable(copier) else None
        return True

    def run(
        self,
        problem,
        *,
        budget=10000,
        max_iteration=800,
        state=None,
    ):
        """Run one optimization and return a uniform :class:`AlgorithmResult`."""
        budget = max(1, int(budget))
        self.max_iterations = max(1, int(max_iteration))
        self.iteration = 0
        self.reset_best()

        # Every family uses the same search contract. The iteration limit is
        # execution state, not another positional protocol.
        searched_state = self.search(problem, budget, state)
        best_state = (
            self.best_state.copy()
            if self.best_state is not None
            else searched_state
        )
        if best_state is not None and not all(
            hasattr(best_state, name)
            for name in ("levels", "next_hops", "tx_load")
        ):
            raise TypeError(
                f"{self.__class__.__name__}.search() must return a decoded State"
            )
        best_fitness = self._read_result_fitness(problem, best_state)

        metadata = {"algorithm": self.__class__.__name__, "seed": self.seed}
        if hasattr(self, "early_stopped"):
            metadata["early_stopped"] = bool(self.early_stopped)
        if hasattr(self, "iterations_completed"):
            metadata["iterations_completed"] = int(self.iterations_completed)

        return AlgorithmResult(
            best_state=best_state,
            best_fitness=best_fitness,
            evaluations=int(self.evatime),
            iterations=int(self.iteration),
            history=np.asarray(self.history, dtype=float).copy(),
            metadata=metadata,
        )

    def search(self, problem, budget, state=None):
        """Search for and return one decoded WSN State.

        Encoding objects are private implementation details of concrete
        optimizers and must not cross this boundary.
        """
        raise NotImplementedError

    @staticmethod
    def _read_result_fitness(problem, state):
        """Read final fitness without consuming another evaluation."""
        if state is None:
            return np.asarray([], dtype=float)

        objectives = getattr(state, "objectives", None)
        if objectives is not None:
            return np.asarray(objectives, dtype=float).copy()

        if all(
            hasattr(state, name)
            for name in ("levels", "next_hops", "tx_load")
        ):
            return np.asarray(problem.evaluate_state(state), dtype=float)

        evaluator = getattr(state, "Evaluate", None)
        if evaluator is not None:
            return np.asarray(evaluator(problem), dtype=float)
        return np.asarray([], dtype=float)

    def on_iteration_finish(
        self,
        problem=None,
        state=None,
        iteration=None,
        run=None,
        **details,
    ):
        """Publish one iteration event to optional observers."""
        self.iteration += 1
        event = {
            "algorithm": self,
            "problem": problem,
            "state": state,
            "iteration": iteration,
            "run": run,
            **details,
        }
        if self.iteration_callback is not None:
            self.iteration_callback(**event)
        for listener in self.iteration_listeners:
            listener(**event)
        self._print_iteration_event(event)
        return event

    def set_iteration_callback(self, callback):
        self.iteration_callback = callback
        return self

    def add_iteration_listener(self, listener):
        self.iteration_listeners.append(listener)
        return self

    def clear_iteration_listeners(self):
        self.iteration_listeners.clear()
        return self

    def enable_iteration_reporting(self, every=1):
        self.iteration_report_every = max(1, int(every))
        return self

    def disable_iteration_reporting(self):
        self.iteration_report_every = None
        return self

    def _print_iteration_event(self, event: dict[str, Any]):
        every = self.iteration_report_every
        iteration = event.get("iteration")
        if every is None or (
            iteration is not None and int(iteration) % every != 0
        ):
            return

        values = event.get("best_value")
        if values is None:
            return
        values = np.asarray(values, dtype=float).reshape(-1)
        labels = [
            f"{name}={value:.6f}"
            for name, value in zip(self.Fname, values)
        ]
        fields = [f"[{self.__class__.__name__}]", f"iteration={iteration}"]
        if event.get("run") is not None:
            fields.append(f"run={event['run']}")
        fields.extend(labels)
        if len(values):
            fields.append(f"total={float(np.sum(values)):.6f}")
        print(" ".join(fields))


__all__ = ["Algorithm", "AlgorithmResult"]
