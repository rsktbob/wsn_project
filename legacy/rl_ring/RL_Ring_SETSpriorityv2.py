"""RL-Ring-SETSpriorityv2：Ring_SETSpriority（C4 PriorityEncoding）的 v2 逐組區段選擇（D3QN，γ = 0）。"""

from __future__ import annotations

from Algorithm.se.Ring_SETSpriority import PriorityEncodingC4, Ring_SETSpriority
from Algorithm.se.ring_segment_policy import D3QNSegmentPolicyV2


class RL_Ring_SETSpriorityv2(D3QNSegmentPolicyV2, Ring_SETSpriority):
    """Ring_SETSpriority with the v2 per-pair D3QN segment choice."""

    ENCODING_TYPE = PriorityEncodingC4


__all__ = ["RL_Ring_SETSpriorityv2"]
