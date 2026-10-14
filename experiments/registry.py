"""Which algorithms the experiment runner knows about, and where they live.

Adding an algorithm means adding one row to ``ALGORITHM_IMPORTS``, one preset
in :mod:`experiments.presets` and -- only when its constructor does not simply
take the preset parameters -- one row in :mod:`experiments.builder`.
"""

import importlib

ALGORITHM_IMPORTS = {
    "qea": ("Algorithm.quantum.QEA", "QEA"),
    "qring_se": ("Algorithm.quantum.QRing_SE", "QRing_SE"),
    "qsi_sets": ("Algorithm.quantum.QSI_SETS", "QSI_SETS"),
    "qts": ("Algorithm.quantum.QTS", "QTS"),
    "sa_sets": ("Algorithm.se.sets.SA_SETS", "SA_SETS"),
    "sa_setspri": ("Algorithm.se.sets.SA_SETSpri", "SA_SETSpri"),
    "ring_se": ("Algorithm.se.ring_se.Ring_SE", "Ring_SE"),
    "ring_sepri": ("Algorithm.se.ring_se.Ring_SEpri", "Ring_SEpri"),
    "linucb_ring_se": ("Algorithm.se.ring_se.LinUCB_Ring_SE", "LinUCB_Ring_SE"),
    "linucb_ring_sepri": ("Algorithm.se.ring_se.LinUCB_Ring_SEpri", "LinUCB_Ring_SEpri"),
    "si_sets": ("Algorithm.se.si_sets.SI_SETS", "SI_SETS"),
    "si_setspri": ("Algorithm.se.si_sets.SI_SETSpri", "SI_SETSpri"),
    "sa_sets_target": ("Algorithm.se.sets.SA_SETS_Target", "SA_SETS_Target"),
    "sets": ("Algorithm.se.sets.SETS", "SETS"),
    "codingse": ("Algorithm.se.sets.CodingSE", "CodingSE"),
    "ga": ("Algorithm.ga.GA", "GA"),
    "pso": ("Algorithm.pso.PSO", "PSO"),
    "ga_elitist": ("Algorithm.ga.ElitistGA", "ElitistGA"),
    "ga_adaptive": ("Algorithm.ga.AdaptiveGA", "AdaptiveGA"),
}

ALGORITHM_NAMES = tuple(ALGORITHM_IMPORTS)
ALGORITHM_CHOICES = ("all",) + ALGORITHM_NAMES

# RL algorithms read a frozen .npz policy, so they only run when the caller
# supplies --rl-model. They are therefore excluded from --algorithm all.
# (The old RL-SETS v1–v5 series was moved to legacy/ on 2026-10-04; the
# RL-Ring-SETS series to legacy/rl_ring/ on 2026-10-13.  No RL algorithm is
# registered right now; the --rl-model hook stays for the next one.)
RL_SETS_ALGORITHMS = frozenset()
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
