"""Evaluation-budget rules shared by batch-based optimizers."""

from __future__ import annotations

import math


def can_start_iteration(evaluations: int, limit: int) -> bool:
    """Return whether a new complete iteration may start.

    The caller checks the budget before starting an atomic iteration. Once an
    iteration starts, it may finish and therefore exceed ``limit`` slightly.
    """

    return int(evaluations) < max(1, int(limit))


def iterations_to_reach_budget(
    limit: int,
    initialization_evaluations: int,
    evaluations_per_iteration: int,
) -> int:
    """Count complete iterations needed to reach an evaluation limit."""

    per_iteration = int(evaluations_per_iteration)
    if per_iteration <= 0:
        raise ValueError("evaluations_per_iteration must be positive")

    remaining = max(0, max(1, int(limit)) - int(initialization_evaluations))
    if remaining == 0:
        return 0
    return int(math.ceil(remaining / per_iteration))
