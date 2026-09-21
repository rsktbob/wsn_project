"""Which algorithms the experiment runner knows about, and where they live.

Adding an algorithm means adding one row to ``ALGORITHM_IMPORTS``, one preset
in :mod:`experiments.presets` and -- only when its constructor does not simply
take the preset parameters -- one row in :mod:`experiments.builder`.
"""

import importlib

ALGORITHM_IMPORTS = {
    "alns": ("Algorithm.misc.ALNS", "ALNS"),
    "eopt": ("Algorithm.misc.EOPT", "EOPT"),
    "sa_sets": ("Algorithm.se.SA_SETS", "SA_SETS"),
    "exp3_sa_sets": ("Algorithm.se.EXP3_SA_SETS", "EXP3_SA_SETS"),
    "linucb_sa_sets": ("Algorithm.se.LinUCB_SA_SETS", "LinUCB_SA_SETS"),
    "thompson_sa_sets": ("Algorithm.se.Thompson_SA_SETS", "Thompson_SA_SETS"),
    "contextual_thompson_ig_sa_sets": (
        "Algorithm.se.ContextualThompsonIG_SA_SETS",
        "ContextualThompsonIG_SA_SETS",
    ),
    "sa_setsv3": ("Algorithm.se.SA_SETSv3", "SA_SETSv3"),
    "sa_setsv4": ("Algorithm.se.SA_SETSv4", "SA_SETSv4"),
    "rl_sets": ("Algorithm.se.RL_SETS", "RL_SETS"),
    "rl_setsv2": ("Algorithm.se.RL_SETSv2", "RL_SETSv2"),
    "rl_setsv3": ("Algorithm.se.RL_SETSv3", "RL_SETSv3"),
    "rl_setsv4": ("Algorithm.se.RL_SETSv4", "RL_SETSv4"),
    "rl_setsv5": ("Algorithm.se.RL_SETSv5", "RL_SETSv5"),
    "si_sets": ("Algorithm.se.SI_SETS", "SI_SETS"),
    "si_setsv2": ("Algorithm.se.SI_SETSv2", "SI_SETSv2"),
    "sa_sets_target": ("Algorithm.se.SA_SETS_Target", "SA_SETS_Target"),
    "gi_gomea": ("Algorithm.gomea.GI_GOMEA", "GI_GOMEA"),
    "gpu_gomea": ("Algorithm.gomea.GPU_GOMEA", "GPU_GOMEA"),
    "gi_gomea_target": ("Algorithm.gomea.GI_GOMEA_Target", "GI_GOMEA_Target"),
    "cma_mae": ("Algorithm.map_elites.CMA_MAE", "CMA_MAE"),
    "differential_map_elites": (
        "Algorithm.map_elites.Differential_MAP_Elites",
        "Differential_MAP_Elites",
    ),
    "setsv1": ("Algorithm.se.SETSv1", "SETSv1"),
    "setsv2": ("Algorithm.se.SETSv2", "SETSv2"),
    "sa_setsv2": ("Algorithm.se.SA_SETSv2", "SA_SETSv2"),
    "codingse": ("Algorithm.se.CodingSE", "CodingSE"),
    "codingsev2": ("Algorithm.se.CodingSEv2", "CodingSEv2"),
    "ga": ("Algorithm.ga.GA", "GA"),
    "pso": ("Algorithm.misc.PSO", "PSO"),
    "eda": ("Algorithm.misc.EDA", "EDA"),
    "cs": ("Algorithm.misc.CS", "CS"),
    "nsoa": ("Algorithm.misc.NSOA", "NSOA"),
    "cpo": ("Algorithm.misc.CPO", "CPO"),
    "cpo_v2": ("Algorithm.misc.CPOv2", "CPOv2"),
    "nsga": ("Algorithm.nsga.NSGAII", "NSGAII"),
    "snsga_rqlearning": ("Algorithm.Scheduling.SNSGAII", "SNSGAII"),
    "rime": ("Algorithm.misc.RIME", "RIME"),
    "srime": ("Algorithm.Scheduling.SRIME", "SRIME"),
}

ALGORITHM_NAMES = tuple(ALGORITHM_IMPORTS)
ALGORITHM_CHOICES = ("all",) + ALGORITHM_NAMES

# RL-SETS versions read a frozen .npz policy, so they only run when the caller
# supplies --rl-model. They are therefore excluded from --algorithm all.
RL_SETS_ALGORITHMS = frozenset(
    ("rl_sets", "rl_setsv2", "rl_setsv3", "rl_setsv4", "rl_setsv5")
)
ALL_RUNNABLE_ALGORITHMS = tuple(
    name for name in ALGORITHM_NAMES if name not in RL_SETS_ALGORITHMS
)


def load_algorithm_type(name):
    """只在建立指定演算法時載入對應模組，降低 Windows worker 啟動成本。"""
    try:
        module_name, class_name = ALGORITHM_IMPORTS[name]
    except KeyError as error:
        raise ValueError("unknown algorithm: " + str(name)) from error
    module = importlib.import_module(module_name)
    return getattr(module, class_name)
