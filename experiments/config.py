"""Experiment-wide defaults, service tables and small shared helpers."""

import argparse
import random

import numpy as np

from Problem.services import (
    FitnessService,
    FitnessServiceV2,
    FitnessServiceV3,
)

IMPLEMENTATION_NAME = "algorithm"

FITNESS_SERVICES = {
    "v1": FitnessService,
    "v2": FitnessServiceV2,
    "v3": FitnessServiceV3,
}
ROUTING_SERVICES = ("v1",)

EXPERIMENT_PRESET = {
    "boundary": 100,
    "sensors": 100,
    "targets": 100,
    "full_energy": 10,
    "file": "A2",
    "evaluate": 10000,
    "runs": 5,
    "max_rounds": 2000,
    "segment_slot_limit": 5000,
}

# rl_setsv4/v5 observe lifetime-scaled features and were tuned at a smaller
# budget, so they run at this budget unless --evaluate says otherwise.
EVALUATE_OVERRIDES = {
    "rl_setsv4": 5000,
    "rl_setsv5": 5000,
}

MOVING_TARGET_ENERGY_THRESHOLD = 10.0


def parse_bool(value):
    """解析命令列的 true/false 布林值。"""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes", "y", "on"):
        return True
    if normalized in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError("請輸入 true 或 false。")


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)


def evaluate_budget(args):
    if getattr(args, "evaluate", None) is not None:
        return int(args.evaluate)
    return int(EXPERIMENT_PRESET["evaluate"])


def effective_evaluate_budget(algorithm, args, after_movement=False):
    """讓支援動態額度的演算法決定本次實際搜尋預算。

    目前只有 ``Algorithm2`` 的 optimizer 實作 ``recommended_evaluation_budget``；
    ``Algorithm`` 底下的演算法沒有這個方法，會直接沿用固定預算。
    """
    base_budget = evaluate_budget(args)
    recommender = getattr(algorithm, "recommended_evaluation_budget", None)
    if recommender is None:
        return base_budget
    return int(recommender(base_budget, after_movement=after_movement))


def lifetime_cap(args):
    value = getattr(args, "max_lifetime", None)
    if value is None:
        return None
    return int(value)


def moving_enabled(args):
    return bool(getattr(args, "moving", False))
