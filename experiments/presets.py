"""Per-algorithm labels and hyper-parameters.

``params`` holds only what the experiment runner actually passes to a
constructor. Two things deliberately live outside it:

* ``parameter_basis`` -- the prose justification quoted in the experiment
  report. It is documentation, not configuration, so it is stored in
  ``parameter_basis.json`` next to this module.
* RL-SETS training hyper-parameters (replay buffer, batch size, learning rate,
  epsilon schedule, ...). This runner always evaluates a frozen policy, so
  those values changed nothing here while still appearing in the report as if
  they were experiment settings. ``train_rl_sets.py`` owns them and already
  defines its own defaults for every one of them.
"""

import copy
import json

from experiments import PROJECT_ROOT

ALGORITHM_PRESETS = {
    'alns': {
        "label": 'ALNS',
        "params": {
            'initial_samples': 8,
            'min_destroy_ratio': 0.05,
            'max_destroy_ratio': 0.2,
            'reaction_factor': 0.2,
            'segment_length': 25,
            'initial_temperature': 0.05,
            'final_temperature': 0.001,
            'score_global_best': 10.0,
            'score_improved': 5.0,
            'score_accepted': 1.0,
            'feasibility_first': True,
        },
    },
    'eopt': {
        "label": 'EOPT',
        "params": {
            'tau': 1.5,
            'initial_samples': 32,
            'depletion_weight': 0.55,
            'routing_weight': 0.25,
            'redundancy_weight': 0.2,
            'replacement_pool_size': 5,
        },
    },
    'sa_sets': {
        "label": 'SA-SETS',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
        },
    },
    'exp3_sa_sets': {
        "label": 'EXP3-SA-SETS',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
        },
    },
    'linucb_sa_sets': {
        "label": 'LinUCB-SA-SETS',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
        },
    },
    'thompson_sa_sets': {
        "label": 'Thompson-SA-SETS',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
        },
    },
    'contextual_thompson_ig_sa_sets': {
        "label": 'Contextual-Thompson + I×G SA-SETS',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
        },
    },
    'sa_setsv3': {
        "label": 'SA-SETSv3 (Ring-local Elitist Investment)',
        "encoding": 'N sensing levels + N routing-priority ranks',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
        },
    },
    'sa_setsv4': {
        "label": 'SA-SETSv4 (Shared-Pool Ring-Segment Operator Selection)',
        "encoding": 'N sensing levels + N routing-priority ranks',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
        },
    },
    'rl_sets': {
        "label": 'RL-SETS (Selective Investment)',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
            # Network width has to match the checkpoint that is loaded, so it
            # stays here even though the rest of the D3QN settings do not.
            'hidden_size': 64,
        },
    },
    'si_sets': {
        "label": 'SI-SETS',
        "params": {
            'n': 8,
            'h': 4,
            'w': 8,
            'mu': 0.4,
        },
    },
    'si_setsv2': {
        "label": 'SI-SETSv2 (Elitist Role-separated Update)',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
        },
    },
    'sa_sets_target': {
        "label": 'SA-SETS-TARGET',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
        },
    },
    'gi_gomea': {
        "label": 'GI-GOMEA-WSN',
        "params": {
            'population_size': 24,
            'elite_fraction': 0.5,
            'statistical_weight': 0.7,
            'graph_weight': 0.3,
            'max_linkage_size': 16,
            'max_linkage_sets': 160,
        },
    },
    'gpu_gomea': {
        "label": 'GPU-GOMEA-WSN',
        "params": {
            'population_size': 24,
            'elite_fraction': 0.5,
            'statistical_weight': 0.7,
            'graph_weight': 0.3,
            'max_linkage_size': 16,
            'max_linkage_sets': 160,
            'cuda_device': 0,
            'require_cuda': False,
        },
    },
    'gi_gomea_target': {
        "label": 'GI-GOMEA-TARGET',
        "params": {
            'population_size': 16,
            'elite_fraction': 0.5,
            'statistical_weight': 0.7,
            'graph_weight': 0.3,
            'max_linkage_size': 16,
            'max_linkage_sets': 120,
        },
    },
    'cma_mae': {
        "label": 'CMA-MAE',
        "params": {
            'num_emitters': 15,
            'batch_size': 36,
            'sigma': 0.2,
            'archive_dims': (100, 100),
            'learning_rate': 0.01,
            'threshold_min': 0.0,
        },
    },
    'differential_map_elites': {
        "label": 'Differential-MAP-Elites',
        "params": {
            'centroid_count': 25000,
            'initial_sample_ratio': 0.01,
            'minimum_initial_samples': 100,
            'scaling_factor': 0.5,
            'crossover_rate': 0.9,
            'cvt_sample_count': 100000,
            'cvt_iterations': 5,
        },
    },
    'setsv1': {
        "label": 'SETSv1',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
        },
    },
    'setsv2': {
        "label": 'SETSv2 (2022 paper)',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'player': 2,
            'crossover_rate': 1.0,
            'mutation_rate': 0.4,
            'adaptive_constant': 0.001,
        },
    },
    'sa_setsv2': {
        "label": 'SA-SETSv2 (2023 paper)',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'player': 2,
            'crossover_rate': 1.0,
            'mutation_rate': 0.4,
            'adaptive_constant': 0.001,
            'energy_weight': 0.5,
            'distance_weight': 0.5,
        },
    },
    'codingse': {
        "label": 'CodingSE',
        "params": {
            'n': 4,
            'h': 4,
            'w': 8,
            'mu': 0.2,
        },
    },
    'codingsev2': {
        "label": 'CodingSEv2',
        "params": {
            'n': 5,
            'h': 4,
            'w': 2,
            'mu': 1.0,
        },
    },
    'ga': {
        "label": 'GA',
        "params": {
            'n': 30,
            'cu': 0.8,
            'mu': 0.1,
        },
    },
    'pso': {
        "label": 'PSO',
        "params": {
            'n': 60,
            'w': 0.7968,
            'end_w': 0.7968,
            'c1': 1.4962,
            'c2': 1.4962,
            'vmax': 0.2,
            'vmin': -0.2,
            'fmax': 1.0,
            'fmin': 0,
        },
    },
    'eda': {
        "label": 'EDA',
        "params": {
            'n': 50,
            'alpha': 0.5,
        },
    },
    'cs': {
        "label": 'CS',
        "params": {
            'n': 15,
            'fmax': 1.0,
            'fmin': 0,
            'step_size': 0.01,
            'Lmbda': 1.5,
            'pa': 0.75,
        },
    },
    'nsoa': {
        "label": 'NSOA',
        "params": {
            'n': 30,
            'normal_mean': 0.85,
            'normal_std': 0.5,
            'initial_elimination_probability': 0.5,
            'sigmoid_delta': 30.0,
            'sigmoid_nu': 0.3,
            'feasibility_first': True,
            'coverage_seeded_initialization': True,
        },
    },
    'cpo': {
        "label": 'CPO',
        "params": {
            'n': 30,
            'min_population': 10,
            'cycles': 2,
            'alpha': 0.2,
            'tradeoff': 0.8,
        },
    },
    'cpo_v2': {
        "label": 'CPOv2 (Discrete Adaptive)',
        "params": {
            'n': 30,
            'min_population': 10,
            'cycles': 2,
            'adapt_interval': 300,
            'adaptive_weight': 0.5,
            'minimum_probability': 0.05,
            'stagnation_limit': 1000,
            'restart_fraction': 0.2,
        },
    },
    'nsga': {
        "label": 'NSGA-II',
        "params": {
            'n': 50,
            'cu': 0.9,
            'mu': None,
            'tournament_size': 2,
        },
    },
    'snsga_rqlearning': {
        "label": 'SNSGAII + RQLearning',
        "params": {
            'n': 50,
            'cu': 0.9,
            'mu': None,
            'tournament_size': 2,
            'rqlearning': {
                'alpha': 0.2,
                'gamma': 0.2,
                'epsilon': 0.1,
                'heuristic_weight': 0.35,
                'learn': True,
            },
        },
    },
    'rime': {
        "label": 'RIME',
        "params": {
            'n': 50,
            'w': 5,
        },
    },
    'srime': {
        "label": 'SRIME',
        "params": {
            'n': 50,
            'soft_rate': 0.7,
            'hard_rate': 0.25,
            'mutation_rate': None,
        },
    },
}


# RL-SETS versions share one market and differ only in policy details, so they
# are derived from the rl_sets preset instead of being repeated in full.
def _derive_rl_preset(name, label, **overrides):
    preset = copy.deepcopy(ALGORITHM_PRESETS["rl_sets"])
    preset["label"] = label
    preset["params"].update(overrides)
    ALGORITHM_PRESETS[name] = preset


_derive_rl_preset("rl_setsv2", "RL-SETSv2 (Role-separated Investment)")
_derive_rl_preset("rl_setsv3", "RL-SETSv3 (Elitist Lévy Selection)")
_derive_rl_preset("rl_setsv4", "RL-SETSv4 (Lifetime-aware Scale Selection)")
_derive_rl_preset(
    # RL_SETSv5 stores lifetime_normalization in its checkpoint metadata and
    # refuses a model trained under a different value, so it stays an
    # evaluation parameter rather than a training one.
    "rl_setsv5",
    "RL-SETSv5 (Factorized SA-centred Selection)",
    lifetime_normalization=5000.0,
)

_BASIS_PATH = PROJECT_ROOT / "experiments" / "parameter_basis.json"
with _BASIS_PATH.open("r", encoding="utf-8") as _basis_file:
    PARAMETER_BASIS = json.load(_basis_file)

for _name, _preset in ALGORITHM_PRESETS.items():
    _preset["parameter_basis"] = PARAMETER_BASIS.get(_name, "No recorded basis.")
