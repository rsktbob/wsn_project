"""RL-Ring-SETSpriority：Ring_SETSpriority（C4 PriorityEncoding），每組 searcher-good 用 D3QN 選區段。"""

from __future__ import annotations

from Algorithm.se.Ring_SETSpriority import PriorityEncodingC4, Ring_SETSpriority
from Algorithm.se.ring_segment_policy import D3QNSegmentPolicy


class RL_Ring_SETSpriority(D3QNSegmentPolicy, Ring_SETSpriority):
    """Ring_SETSpriority with a per-pair D3QN segment choice."""

    ENCODING_TYPE = PriorityEncodingC4


__all__ = ["RL_Ring_SETSpriority"]
