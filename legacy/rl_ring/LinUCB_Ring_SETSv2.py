"""LinUCB-Ring-SETSv2：Ring_SETS（SensorEncoding）的 v2 逐組區段選擇（LinUCB）。

v2 的特徵（含最慘 target／全網成本占比）與分級回饋見 :mod:`ring_segment_policy`。
"""

from __future__ import annotations

from Algorithm.se.Ring_SETS import Ring_SETS
from Algorithm.se.ring_segment_policy import LinUCBSegmentPolicyV2


class LinUCB_Ring_SETSv2(LinUCBSegmentPolicyV2, Ring_SETS):
    """Ring_SETS with the v2 per-pair LinUCB segment choice."""


__all__ = ["LinUCB_Ring_SETSv2"]
