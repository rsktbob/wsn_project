"""Which algorithms the experiment runner knows about, and where they live.

Adding an algorithm means adding one row to ``ALGORITHM_IMPORTS``, one preset
in :mod:`experiments.presets` and -- only when its constructor does not simply
take the preset parameters -- one row in :mod:`experiments.builder`.
"""

import importlib

ALGORITHM_IMPORTS = {
    "qea3": ("Algorithm.qea.QEA3", "QEA3"),
    "qea": ("Algorithm.qea.QEA", "QEA"),
    "qea_random": ("Algorithm.qea.QEA", "QEA"),
    "qea_classical": ("Algorithm.qea.QEA", "QEA"),
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
    "ring_sets": ("Algorithm.se.Ring_SETS", "Ring_SETS"),
    "ring_setspriority": ("Algorithm.se.Ring_SETSpriority", "Ring_SETSpriority"),
    "linucb_ring_sets": ("Algorithm.se.LinUCB_Ring_SETS", "LinUCB_Ring_SETS"),
    "linucb_ring_setspriority": ("Algorithm.se.LinUCB_Ring_SETSpriority", "LinUCB_Ring_SETSpriority"),
    "rl_ring_sets": ("Algorithm.se.RL_Ring_SETS", "RL_Ring_SETS"),
    "rl_ring_setspriority": ("Algorithm.se.RL_Ring_SETSpriority", "RL_Ring_SETSpriority"),
    "linucb_ring_setsv2": ("Algorithm.se.LinUCB_Ring_SETSv2", "LinUCB_Ring_SETSv2"),
    "linucb_ring_setspriorityv2": ("Algorithm.se.LinUCB_Ring_SETSpriorityv2", "LinUCB_Ring_SETSpriorityv2"),
    "rl_ring_setsv2": ("Algorithm.se.RL_Ring_SETSv2", "RL_Ring_SETSv2"),
    "rl_ring_setspriorityv2": ("Algorithm.se.RL_Ring_SETSpriorityv2", "RL_Ring_SETSpriorityv2"),
    "si_sets": ("Algorithm.se.SI_SETS", "SI_SETS"),
    "sa_sets_target": ("Algorithm.se.SA_SETS_Target", "SA_SETS_Target"),
    "gi_gomea": ("Algorithm.gomea.GI_GOMEA", "GI_GOMEA"),
    "gpu_gomea": ("Algorithm.gomea.GPU_GOMEA", "GPU_GOMEA"),
    "gi_gomea_target": ("Algorithm.gomea.GI_GOMEA_Target", "GI_GOMEA_Target"),
    "cma_mae": ("Algorithm.map_elites.CMA_MAE", "CMA_MAE"),
    "differential_map_elites": (
        "Algorithm.map_elites.Differential_MAP_Elites",
        "Differential_MAP_Elites",
    ),
    "sets": ("Algorithm.se.SETS", "SETS"),
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

# RL algorithms read a frozen .npz policy, so they only run when the caller
# supplies --rl-model. They are therefore excluded from --algorithm all.
# (The old RL-SETS v1–v5 series was moved to legacy/ on 2026-10-04.)
RL_SETS_ALGORITHMS = frozenset((
    "rl_ring_sets", "rl_ring_setspriority", "rl_ring_setsv2", "rl_ring_setspriorityv2",
))
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
