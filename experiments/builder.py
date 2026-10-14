"""Turning a preset into a constructed algorithm instance.

Registered algorithms are constructed from their documented presets.
The RL-SETS variants additionally receive a frozen checkpoint path.
"""

import copy


from experiments.presets import ALGORITHM_PRESETS
from experiments.registry import RL_SETS_ALGORITHMS, load_algorithm_type


def configured_algorithm_params(name, args):
    """The preset parameters, with the run's command line applied."""
    params = copy.deepcopy(ALGORITHM_PRESETS[name]["params"])
    if name in RL_SETS_ALGORITHMS:
        # Formal experiments are always frozen.
        params["training"] = False
        params["model_path"] = getattr(args, "rl_model", None)
    return params


def algorithm_label(name):
    return ALGORITHM_PRESETS[name]["label"]


def algorithm_params(name, args):
    """The parameters as they should appear in the experiment report."""
    return configured_algorithm_params(name, args)


def build_algorithm(name, problem, args, seed):
    params = configured_algorithm_params(name, args)
    algorithm_type = load_algorithm_type(name)

    if name in RL_SETS_ALGORITHMS and not params["model_path"]:
        raise ValueError(
            f"{name} requires --rl-model pointing to a trained checkpoint"
        )
    return algorithm_type(problem, seed=seed, **params)
