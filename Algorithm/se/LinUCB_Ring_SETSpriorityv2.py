"""LinUCB-Ring-SETSpriorityv2：Ring_SETSpriority（C4 PriorityEncoding）的 v2 逐組區段選擇（LinUCB）。"""

from __future__ import annotations

from Algorithm.se.Ring_SETSpriority import Ring_SETSpriority
from Algorithm.se.ring_segment_policy import LinUCBSegmentPolicyV2


class LinUCB_Ring_SETSpriorityv2(LinUCBSegmentPolicyV2, Ring_SETSpriority):
    """Ring_SETSpriority with the v2 per-pair LinUCB segment choice."""


__all__ = ["LinUCB_Ring_SETSpriorityv2"]
