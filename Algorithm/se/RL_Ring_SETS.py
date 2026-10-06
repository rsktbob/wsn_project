"""RL-Ring-SETS：Ring_SETS（SensorEncoding），每組 searcher-good 用 D3QN 選區段。

網路的優勢分支在各區段間共用權重；事先用 ``train_rl_ring_sets.py`` 訓練，
正式實驗以 ``--rl-model`` 載入並凍結。細節見 :mod:`ring_segment_policy`。
"""

from __future__ import annotations

from Algorithm.se.Ring_SETS import Ring_SETS
from Algorithm.se.ring_segment_policy import D3QNSegmentPolicy
from State.SensorEncoding import SensorEncoding


class RL_Ring_SETS(D3QNSegmentPolicy, Ring_SETS):
    """Ring_SETS with a per-pair D3QN segment choice."""

    ENCODING_TYPE = SensorEncoding


__all__ = ["RL_Ring_SETS"]
