"""RL-Ring-SETSv2：Ring_SETS（SensorEncoding）的 v2 逐組區段選擇（D3QN，γ = 0）。

v2 的特徵、分級回饋見 :mod:`ring_segment_policy`；用 ``train_rl_ring_sets.py`` 訓練，
正式實驗以 ``--rl-model`` 載入並凍結。
"""

from __future__ import annotations

from Algorithm.se.Ring_SETS import Ring_SETS
from Algorithm.se.ring_segment_policy import D3QNSegmentPolicyV2
from State.SensorEncoding import SensorEncoding


class RL_Ring_SETSv2(D3QNSegmentPolicyV2, Ring_SETS):
    """Ring_SETS with the v2 per-pair D3QN segment choice."""

    ENCODING_TYPE = SensorEncoding


__all__ = ["RL_Ring_SETSv2"]
