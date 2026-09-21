"""Shared environment defaults and checkpoint compatibility for RL experiments."""

import hashlib
import inspect
from pathlib import Path


DEFAULT_FITNESS_SERVICE = "v2"
DEFAULT_ROUTING_SERVICE = "v1"
DEFAULT_SENSING_MODE = "discrete"
DEFAULT_SENSOR_ENCODING = "v3"


def environment_contract(problem):
    """Record actual implementations, including changes within a version."""
    from State.SensorEncoding import SensorEncoding
    from State.State import State

    implementations = {
        "fitness": type(problem.fitness_service),
        "routing": type(problem.routing_service),
        "decode": SensorEncoding,
        "problem": type(problem),
        "state": State,
        "energy": type(problem.energy_service),
        "coverage": type(problem.coverage_service),
        "geometry": type(problem.G),
    }
    # Hash inherited service methods too, not only each leaf class file.
    sources = {
        f"{base.__module__}.{base.__name__}":
            hashlib.sha256(Path(inspect.getfile(base)).read_bytes()).hexdigest()
        for implementation in implementations.values()
        for base in implementation.__mro__ if base is not object
    }
    return {
        "sensing_mode": problem.sensing_mode,
        "sensor_encoding": DEFAULT_SENSOR_ENCODING,
        "implementations": {
            key: f"{value.__module__}.{value.__name__}"
            for key, value in implementations.items()
        },
        "source_sha256": sources,
    }


def check_environment_contract(saved, problem):
    if saved != environment_contract(problem):
        raise ValueError(
            "RL checkpoint environment differs from this run (fitness, routing, "
            "decode or sensing mode). Use the training configuration or retrain."
        )
