"""Turning a preset into a constructed algorithm instance.

Almost every algorithm is built as ``AlgorithmType(problem, seed=seed,
**params)``, so that is the default and needs no table entry. Only the handful
of constructors that deviate -- no ``seed``, no ``problem``, a generation count
derived from the evaluation budget, or a post-construction assignment -- get a
builder of their own below.
"""

import copy
from pathlib import Path

import numpy as np

from Algorithm.core import iterations_to_reach_budget

from experiments.config import evaluate_budget
from experiments.presets import ALGORITHM_PRESETS
from experiments.registry import RL_SETS_ALGORITHMS, load_algorithm_type


def nsga_generation(evaluate, population_size):
    population_size = max(1, int(population_size))
    return iterations_to_reach_budget(
        evaluate,
        initialization_evaluations=population_size,
        evaluations_per_iteration=population_size,
    )


def srime_generation(evaluate, population_size):
    population_size = max(1, int(population_size))
    return iterations_to_reach_budget(
        evaluate,
        initialization_evaluations=population_size,
        evaluations_per_iteration=max(1, population_size - 1),
    )


RANDOM_POLICY_ALGORITHMS = frozenset(("rl_setsv4", "rl_setsv5"))


def configured_algorithm_params(name, args):
    """The preset parameters, with the run's command line applied."""
    params = copy.deepcopy(ALGORITHM_PRESETS[name]["params"])
    if name in RL_SETS_ALGORITHMS:
        # Formal experiments are always frozen. rl_setsv4/v5 also permit an
        # explicit no-checkpoint uniform-random policy baseline.
        params["training"] = False
        params["model_path"] = getattr(args, "rl_model", None)
        if name in RANDOM_POLICY_ALGORITHMS:
            params["random_policy"] = not bool(params["model_path"])
    return params


def algorithm_label(name):
    return ALGORITHM_PRESETS[name]["label"]


def algorithm_params(name, args):
    """The parameters as they should appear in the experiment report."""
    params = configured_algorithm_params(name, args)
    if name in ("nsga", "snsga_rqlearning"):
        params["generation"] = nsga_generation(
            evaluate_budget(args), params["n"]
        )
    if name == "srime":
        params["generation"] = srime_generation(
            evaluate_budget(args), params["n"]
        )
    return params


def _build_rl_sets(algorithm_type, problem, params, args, seed):
    if not params["model_path"]:
        raise ValueError(
            "rl_sets requires --rl-model pointing to a trained checkpoint"
        )
    model_path = Path(params["model_path"]).expanduser()
    with np.load(model_path, allow_pickle=False) as archive:
        if "v2_metadata_json" in archive:
            raise ValueError(
                "this checkpoint uses the retired RL_SETSv2 format and is "
                "not compatible with the current rl_sets checkpoint schema"
            )
    return algorithm_type(problem, seed=seed, **params)


def _build_pso(algorithm_type, problem, params, args, seed):
    # PSO takes its acceleration coefficients as attributes, not arguments.
    algorithm = algorithm_type(
        problem,
        **{key: value for key, value in params.items() if key not in ("c1", "c2")},
    )
    algorithm.c1 = params["c1"]
    algorithm.c2 = params["c2"]
    return algorithm


def _build_nsga(algorithm_type, problem, params, args, seed):
    return algorithm_type(
        problem,
        generation=nsga_generation(evaluate_budget(args), params["n"]),
        **params,
    )


def _build_snsga_rqlearning(algorithm_type, problem, params, args, seed):
    rqlearning_params = dict(params.pop("rqlearning"))
    rqlearning_params["seed"] = seed
    # SNSGAII builds its own problem internally and takes no problem argument.
    return algorithm_type(
        generation=nsga_generation(evaluate_budget(args), params["n"]),
        rqlearning_kwargs=rqlearning_params,
        **params,
    )


def _build_srime(algorithm_type, problem, params, args, seed):
    # SRIME builds its own problem internally and takes no problem argument.
    return algorithm_type(
        generation=srime_generation(evaluate_budget(args), params["n"]),
        **params,
    )


SPECIAL_BUILDERS = {
    "rl_sets": _build_rl_sets,
    "pso": _build_pso,
    "nsga": _build_nsga,
    "snsga_rqlearning": _build_snsga_rqlearning,
    "srime": _build_srime,
}

# Constructors that do not accept a seed argument.
UNSEEDED_ALGORITHMS = frozenset(("eda", "cs"))


def build_algorithm(name, problem, args, seed):
    params = configured_algorithm_params(name, args)
    algorithm_type = load_algorithm_type(name)

    special = SPECIAL_BUILDERS.get(name)
    if special is not None:
        return special(algorithm_type, problem, params, args, seed)
    if (
        name in RL_SETS_ALGORITHMS
        and name not in RANDOM_POLICY_ALGORITHMS
        and not params["model_path"]
    ):
        raise ValueError(
            f"{name} requires --rl-model pointing to a trained checkpoint"
        )
    if name in UNSEEDED_ALGORITHMS:
        return algorithm_type(problem, **params)
    return algorithm_type(problem, seed=seed, **params)
