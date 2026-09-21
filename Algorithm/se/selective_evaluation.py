"""SI-SETS 子程序只處理解碼與評估，候選順序由主程序維持。"""

import numpy as np


def initialize_problem(problem):
    global _problem
    _problem = problem


def evaluate_batch(candidates):
    results = []
    for candidate in candidates:
        state = candidate.decode(_problem)
        objectives = np.asarray(_problem.evaluate_state(state), dtype=float)
        state.objectives = objectives.copy()
        results.append((candidate, state, objectives,
                        getattr(_problem, "target_red", None)))
    return results


def build_evaluate_handler(problem, worker_id, seed):
    """PersistentWorkerPool handler factory: bind this worker's problem once.

    ``worker_id``/``seed`` are unused -- evaluation is stateless and needs
    no per-worker identity or RNG -- but PersistentWorkerPool always passes
    them, so every handler factory accepts the same signature.
    """
    initialize_problem(problem)
    return evaluate_batch
