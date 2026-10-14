"""Per-algorithm labels and hyper-parameters.

``params`` holds only what the experiment runner actually passes to a
constructor. Two things deliberately live outside it:

* ``parameter_basis`` -- the prose justification quoted in the experiment
  report. It is documentation, not configuration, so it is stored in
  ``parameter_basis.json`` next to this module.
* RL training hyper-parameters (replay buffer, batch size, learning rate,
  epsilon schedule, ...). This runner always evaluates a frozen policy, so
  those values changed nothing here while still appearing in the report as if
  they were experiment settings. The training script owns them (the old
  ``train_rl_ring_sets.py`` now lives in ``legacy/rl_ring/``).
"""

import json

from experiments import PROJECT_ROOT

ALGORITHM_PRESETS = {
    'qea': {'label': 'Global-QEA',
     'encoding': 'SensorEncoding: sensing levels + routing-priority ranks',
     'params': {'n': 30,
                'rotation': 0.031415926535897934,
                'probability_floor': 0.01,
                'update_rule': 'rotation',
                'learning_rate': 0.05}},

    'qts': {'label': 'QTS',
     'encoding': 'SensorEncoding: sensing levels + routing-priority ranks',
     'params': {'n': 10,
                'rotation': 0.031415926535897934,
                'probability_floor': 0.01,
                'init': 'uniform'}},

    'qring_se': {
        "label": 'Q-Ring-SE (Ring-SE with Q-bit goods)',
        "encoding": 'N sensing levels + N routing-priority ranks',
        "params": {
            'n': 8,
            'h': 4,
            'w': 8,
            'mu': 0.4,
            'rotation': 0.15707963267948966,
            'probability_floor': 0.01,
            'init_confidence': 0.0,
        },
    },
    'qsi_sets': {
        "label": 'Q-SI-SETS (SI-SETS with Q-bit goods)',
        "encoding": 'N sensing levels + N routing-priority ranks',
        "params": {
            'n': 8,
            'h': 4,
            'w': 8,
            'mu': 0.4,
            'rotation': 0.15707963267948966,
            'probability_floor': 0.01,
            'init_confidence': 0.0,
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
    'sa_setspri': {
        "label": 'SA-SETSpri (SA-SETS + PriorityEncoding, no partition)',
        "encoding": 'N activation priorities + N routing priorities (energy activation, energy + farness routing)',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
        },
    },
    'ring_se': {
        "label": 'Ring-SE (Shared-Pool Ring-Segment Operator Selection)',
        "encoding": 'N sensing levels + N routing-priority ranks',
        "params": {
            'n': 8,
            'h': 4,
            'w': 8,
            'mu': 0.4,
        },
    },
    'ring_sepri': {
        "label": 'Ring-SEpri (Ring-SE + PriorityEncoding)',
        "encoding": 'N activation priorities + N routing priorities (energy activation, energy + farness routing)',
        "params": {
            'n': 8,
            'h': 4,
            'w': 8,
            'mu': 0.4,
        },
    },
    'linucb_ring_se': {
        "label": 'LinUCB-Ring-SE (per-pair LinUCB segment choice)',
        "encoding": 'N sensing levels + N routing-priority ranks',
        "params": {
            'n': 8,
            'h': 4,
            'w': 8,
            'mu': 0.4,
        },
    },
    'linucb_ring_sepri': {
        "label": 'LinUCB-Ring-SEpri (per-pair LinUCB segment choice)',
        "encoding": 'N activation priorities + N routing priorities (energy activation, energy + farness routing)',
        "params": {
            'n': 8,
            'h': 4,
            'w': 8,
            'mu': 0.4,
        },
    },
    'si_sets': {
        "label": 'SI-SETS (Role-separated EN Update)',
        "params": {
            'n': 8,
            'h': 4,
            'w': 8,
            'mu': 0.4,
        },
    },
    'si_setspri': {
        "label": 'SI-SETSpri (SI-SETS + PriorityEncoding, no partition)',
        "encoding": 'N activation priorities + N routing priorities (energy activation, energy + farness routing)',
        "params": {
            'n': 8,
            'h': 4,
            'w': 8,
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
    'sets': {
        "label": 'SETS',
        "params": {
            'n': 8,
            'h': 4,
            'w': 2,
            'mu': 0.4,
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
    'ga_elitist': {
        "label": 'Elitist GA',
        "params": {'n': 30, 'cu': 0.8, 'elite_count': 2},
    },
    'ga_adaptive': {
        "label": 'Srinivas-Patnaik Adaptive GA (SensorEncoding)',
        "params": {
            'n': 30, 'k1': 1.0, 'k2': 0.5,
            'k3': 1.0, 'k4': 0.5, 'base_mutation': 0.005,
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
        "encoding": 'SensorEncoding: sensing choices + routing-priority ranks',
        "params": {'n': 30, 'w': 0.7, 'c1': 1.5, 'c2': 1.5, 'vmax': 0.2},
    },
}


_BASIS_PATH = PROJECT_ROOT / "experiments" / "parameter_basis.json"
with _BASIS_PATH.open("r", encoding="utf-8") as _basis_file:
    PARAMETER_BASIS = json.load(_basis_file)

for _name, _preset in ALGORITHM_PRESETS.items():
    _preset["parameter_basis"] = PARAMETER_BASIS.get(_name, "No recorded basis.")
