"""Shared scalar mutation samplers.

Uniform chromosome creation belongs to ``SensorEncoding.random`` and
``TargetEncoding.random``.  Algorithm-specific initialization stays in the
algorithm that needs it.
"""

import random

from State.SensorEncoding import SensorEncoding


def sample_sensor_gene(problem):
    """Sample one legacy sensor chromosome value."""
    upper_bound = max(problem.LEVEL, SensorEncoding.RANK_PRECISION)
    return random.randint(0, upper_bound - 1)


def sample_target_mutation_gene(problem):
    """Sample the target mutation range used by the former implementation."""
    return random.randint(0, (problem.SENSOR_NUMBER + 100) * 4 - 1)


__all__ = [
    "sample_sensor_gene",
    "sample_target_mutation_gene",
]
